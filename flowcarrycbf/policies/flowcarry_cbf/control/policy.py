"""FlowCarryCBF direct-action policy with perception gating and whole-body safety."""

from __future__ import annotations

import time

import numpy as np
import torch

from .expert import BimanualExpertPlanner, expert_condition
from .kinematics import FullBodyKinematics
from .model import SafeFlowActionModel
from .safety import WholeBodyCBFQP
from .schema import HORIZON
from .stage_one import StageOneStaticExpert, StageTwoDynamicExpert


class FlowCarryCBFPolicy:
    # Keep a buffer above the mathematical zero of the local projection. The
    # execution task also requires an 8 cm truth margin, and a moving sphere can
    # consume a few centimeters between RGB-D frames.
    EMERGENCY_CLEARANCE_TRIGGER = 0.05

    # All profiles are still passed through the whole-body CBF-QP. The forward
    # profiles let the expert keep making progress when a certified side-step
    # is available, instead of making emergency mode intrinsically retreat.
    BASE_ESCAPE_PROFILES = (
        (0.70, 0.0, 0.0),
        (0.70, 0.45, 0.0),
        (0.70, -0.45, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.45, 0.0),
        (-1.0, -0.45, 0.0),
        (0.0, 0.45, 0.0),
        (0.0, -0.45, 0.0),
    )

    def __init__(self, *, checkpoint: str | None = None, device: str = "cpu", mode: str = "fm", arm_locked: bool = False, seed: int = 0) -> None:
        self.device = torch.device(device)
        self.mode = mode
        self.arm_locked = bool(arm_locked)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._torch_generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.expert = BimanualExpertPlanner()
        self.kinematics = self.expert.kinematics
        self.safety = WholeBodyCBFQP(self.kinematics)
        self.stage1 = StageOneStaticExpert()
        self.stage2 = StageTwoDynamicExpert(
            self.expert._region_targets,
            self.kinematics,
            self.safety,
        )
        self.model = None
        self.normalization = {}
        if checkpoint:
            self.model, payload = SafeFlowActionModel.load_checkpoint(checkpoint, self.device)
            self.normalization = payload.get("normalization", {})
        self._previous = np.zeros(17, dtype=np.float32)
        self._certified_tail: np.ndarray | None = None
        self._last_obstacles = np.zeros((5, 15), dtype=np.float32)
        self._tail_age = 0
        self._emergency_stops = 0
        self._certified_tail_source = "none"
        self._fm_candidate_total = 0
        self._fm_raw_feasible_total = 0
        self._fm_action_steps = 0
        self._fallback_steps = 0
        self._expert_fallback_steps = 0
        self._emergency_fallback_steps = 0
        self._perception_fallback_steps = 0
        self.last_metrics: dict[str, float | str | int] = {}
        self.last_sequence: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._torch_generator.manual_seed(self.seed)
        self.expert.reset()
        self.stage1.reset()
        self.stage2.reset()
        self._previous.fill(0.0)
        self._certified_tail = None
        self._last_obstacles.fill(0.0)
        self._tail_age = 0
        self._emergency_stops = 0
        self._certified_tail_source = "none"
        self._fm_candidate_total = 0
        self._fm_raw_feasible_total = 0
        self._fm_action_steps = 0
        self._fallback_steps = 0
        self._expert_fallback_steps = 0
        self._emergency_fallback_steps = 0
        self._perception_fallback_steps = 0
        self.last_metrics = {}
        self.last_sequence = None

    def _record_action_source(self, source: str) -> dict[str, float | str | int]:
        source = str(source)
        if self.mode in {"fm", "hybrid"}:
            if source == "fm":
                self._fm_action_steps += 1
            else:
                self._fallback_steps += 1
                if source == "expert":
                    self._expert_fallback_steps += 1
                elif source.startswith("perception"):
                    self._perception_fallback_steps += 1
                else:
                    self._emergency_fallback_steps += 1
        return {
            "action_source": source,
            "fallback_used": int(self._fallback_steps > 0),
            "fallback_steps": int(self._fallback_steps),
            "expert_fallback_steps": int(self._expert_fallback_steps),
            "emergency_fallback_steps": int(self._emergency_fallback_steps),
            "perception_fallback_steps": int(self._perception_fallback_steps),
            "fm_action_steps": int(self._fm_action_steps),
            "fm_candidate_total": int(self._fm_candidate_total),
            "fm_raw_feasible_total": int(self._fm_raw_feasible_total),
            "raw_fm_feasible_rate": float(
                self._fm_raw_feasible_total / max(self._fm_candidate_total, 1)
            ),
        }

    def _emergency_action(self, state: np.ndarray, obstacles: np.ndarray) -> tuple[np.ndarray, object]:
        """Choose a certified 1 s whole-body escape, then execute its first step."""
        state = np.asarray(state, dtype=np.float32).reshape(60)
        obstacles = np.asarray(obstacles, dtype=np.float32).reshape(5, 15)
        raw_candidates: list[tuple[float, np.ndarray]] = []

        # First compare all arm/payload grid seeds while braking the base. This
        # is the important escape degree of freedom when a sphere crosses the
        # carried box rather than the mobile base.
        vertical_region = self.expert._dynamic_vertical_region(state, obstacles)
        for _, _, _, _, sequence in self.expert._ranked_grid(state, obstacles, vertical_region):
            candidate = np.asarray(sequence, dtype=np.float32).copy()
            candidate[:10, :3] = 0.0
            rollout = self.kinematics.rollout(state, candidate[:10])
            clearance = self.kinematics.minimum_clearance(rollout, obstacles, certified_steps=10)
            raw_candidates.append((float(clearance), candidate))

        # Apply base escape profiles to the best arm motion. During emergency
        # mode, preserve the arm part of the previously certified tail first;
        # selecting a fresh grid seed can abruptly move both graspers and slip
        # the payload even when the base action itself is safe.
        grid_raw_candidates = raw_candidates
        if self._certified_tail is not None:
            best_arm = np.vstack((self._certified_tail[1:], self._certified_tail[-1:])).astype(np.float32, copy=True)
            primary_raw_candidates: list[tuple[float, np.ndarray]] = []
        else:
            best_arm = max(grid_raw_candidates, key=lambda item: item[0])[1]
            primary_raw_candidates = grid_raw_candidates
        profile_candidates: list[tuple[tuple[float, float, float], float, np.ndarray]] = []
        for profile in self.BASE_ESCAPE_PROFILES:
            candidate = best_arm.copy()
            candidate[:10, :3] = np.asarray(profile, dtype=np.float32)
            rollout = self.kinematics.rollout(state, candidate[:10])
            clearance = self.kinematics.minimum_clearance(rollout, obstacles, certified_steps=10)
            profile_candidates.append((profile, float(clearance), candidate))

        certified = []
        # A global top-k can discard the only lateral/forward escape before the
        # hard QP sees it. Keep the best raw seeds and one candidate per profile.
        preferred_profiles = list(profile_candidates)
        preferred_raw = [(clearance, candidate) for _, clearance, candidate in preferred_profiles]
        selected_candidates = sorted(primary_raw_candidates + preferred_raw, key=lambda item: item[0], reverse=True)[:4]
        selected_candidates.extend(preferred_raw)
        seen: set[int] = set()
        for _, candidate in selected_candidates:
            marker = id(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            result = self.safety.project(candidate, state, obstacles, previous_action=self._previous)
            if result.feasible:
                certified.append((float(result.minimum_clearance), result))
        # If the goal-directed side is fully blocked, retry the opposite side
        # as a genuine safety escape rather than silently stopping.
        if not certified and len(preferred_profiles) < len(profile_candidates):
            for _, _, candidate in profile_candidates:
                result = self.safety.project(candidate, state, obstacles, previous_action=self._previous)
                if result.feasible:
                    certified.append((float(result.minimum_clearance), result))
        # A stale tail can become impossible as an obstacle moves. Only then
        # permit a fresh arm seed, and still keep it behind the preferred
        # profile candidates above.
        if not certified and self._certified_tail is not None:
            fallback = sorted(grid_raw_candidates, key=lambda item: item[0], reverse=True)[:4]
            for _, candidate in fallback:
                result = self.safety.project(candidate, state, obstacles, previous_action=self._previous)
                if result.feasible:
                    certified.append((float(result.minimum_clearance), result))
        if certified:
            # Hard feasibility is decided above. Among feasible escapes, retain
            # clearance as a secondary term but prefer motion that reduces the
            # current world-frame goal distance; otherwise a stationary profile
            # can win every emergency comparison and strand the carry.
            c, s = np.cos(float(state[2])), np.sin(float(state[2]))
            relative_goal = state[53:55]
            world_goal = state[:2] + np.asarray(
                [c * relative_goal[0] - s * relative_goal[1], s * relative_goal[0] + c * relative_goal[1]],
                dtype=np.float32,
            )
            initial_distance = float(np.linalg.norm(world_goal - state[:2]))

            def emergency_score(item: tuple[float, object]) -> float:
                clearance, projection = item
                rollout = self.kinematics.rollout(state, np.asarray(projection.actions, dtype=np.float32)[:10])
                progress = initial_distance - float(np.linalg.norm(world_goal - rollout.base[-1, :2]))
                # All entries here have already passed the hard dynamic
                # clearance constraints. Progress therefore breaks ties;
                # clearance only provides a small stability preference.
                return float(progress + 0.05 * clearance)

            result = max(certified, key=emergency_score)[1]
            return np.asarray(result.actions[0], dtype=np.float32), result

        # A dynamic obstacle can make every 1 s sequence infeasible while the
        # immediate control step is still recoverable. Certify one step from
        # every distinct escape and execute the action with the largest next-
        # frame margin. Replanning on the next RGB-D frame then extends the
        # escape without pretending the unavailable 1 s certificate exists.
        one_step_candidates: list[tuple[float, object]] = []
        seen_actions: set[bytes] = set()
        for _, candidate in grid_raw_candidates:
            first = np.asarray(candidate[0], dtype=np.float32)
            marker = np.round(first, 5).tobytes()
            if marker in seen_actions:
                continue
            seen_actions.add(marker)
            shield = self.safety.shield(first, state, obstacles, previous_action=self._previous)
            if shield.feasible:
                one_step_candidates.append((float(shield.minimum_clearance), shield))
        for _, _, candidate in profile_candidates:
            first = np.asarray(candidate[0], dtype=np.float32)
            marker = np.round(first, 5).tobytes()
            if marker in seen_actions:
                continue
            seen_actions.add(marker)
            shield = self.safety.shield(first, state, obstacles, previous_action=self._previous)
            if shield.feasible:
                one_step_candidates.append((float(shield.minimum_clearance), shield))
        if one_step_candidates:
            result = max(one_step_candidates, key=lambda item: item[0])[1]
            return np.asarray(result.actions, dtype=np.float32), result

        result = self.safety.shield(np.zeros(17, dtype=np.float32), state, obstacles, previous_action=self._previous)
        return np.zeros(17, dtype=np.float32), result

    def _condition(self, state: np.ndarray, obstacles: np.ndarray) -> torch.Tensor:
        value = expert_condition(state, obstacles)
        mean = np.asarray(self.normalization.get("condition_mean", np.zeros(135)), dtype=np.float32)
        scale = np.asarray(self.normalization.get("condition_scale", np.ones(135)), dtype=np.float32)
        value = (value - mean) / np.maximum(scale, 1.0e-5)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _cached_safety_obstacles(
        self,
        obstacles: np.ndarray,
        dynamic_missing_time: float,
    ) -> np.ndarray:
        value = np.asarray(obstacles, dtype=np.float32).reshape(5, 15).copy()
        valid = value[:, 14] > 0.5
        self._last_obstacles[valid] = value[valid]
        missing = ~valid & (self._last_obstacles[:, 14] > 0.5)
        if not np.any(missing):
            return value
        value[missing] = self._last_obstacles[missing]
        # Tracks remain valid and predicted for the first 0.5 s. The cache is
        # therefore already at that time; advance only the unobserved excess.
        elapsed = max(0.0, float(dynamic_missing_time) - 0.5)
        value[missing, :3] += value[missing, 3:6] * elapsed + 0.5 * value[missing, 6:9] * elapsed**2
        value[missing, 13] += 0.5 * 0.45 * float(dynamic_missing_time) ** 2
        return value

    def _fm_candidates(self, state: np.ndarray, obstacles: np.ndarray) -> list[np.ndarray]:
        if self.model is None:
            return []
        condition = self._condition(state, obstacles).repeat(4, 1)
        with torch.no_grad():
            samples = self.model.sample(condition, generator=self._torch_generator, prediction_steps=8).detach().cpu().numpy()
        mean = np.asarray(self.normalization.get("action_mean", np.zeros(17)), dtype=np.float32)
        scale = np.asarray(self.normalization.get("action_scale", np.ones(17)), dtype=np.float32)
        return [np.clip(sample * scale + mean, -1.0, 1.0).astype(np.float32) for sample in samples]

    def _candidate_score(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
        candidate: np.ndarray,
    ) -> tuple[int, int, float, float]:
        rollout = self.kinematics.rollout(state, candidate)
        grasp = self.kinematics.grasp_consistent(
            rollout,
            reference_vector=self.kinematics.grasp_vector(state),
        )
        clearance = self.kinematics.minimum_clearance(
            rollout,
            obstacles,
            certified_steps=10,
        )
        state = np.asarray(state, dtype=np.float32).reshape(60)
        c, s = np.cos(float(state[2])), np.sin(float(state[2]))
        relative_goal = state[53:55]
        world_goal = state[:2] + np.asarray(
            [c * relative_goal[0] - s * relative_goal[1], s * relative_goal[0] + c * relative_goal[1]],
            dtype=np.float32,
        )
        initial_distance = float(np.linalg.norm(world_goal - state[:2]))
        final_distance = float(np.linalg.norm(world_goal - rollout.base[-1, :2]))
        return int(grasp), int(clearance >= 0.0), initial_distance - final_distance, float(clearance)

    def act(
        self,
        observation: dict[str, np.ndarray],
        *,
        perception_accepts_new_plan: bool = True,
        dynamic_missing_time: float = 0.0,
        track_missing_times: np.ndarray | list[float] | None = None,
        expert_region: str | None = None,
    ) -> np.ndarray:
        started = time.perf_counter()
        self.last_sequence = None
        if expert_region is not None and self.mode != "expert":
            raise ValueError("expert_region is only valid in expert mode")
        state = np.asarray(observation["state"], dtype=np.float32).reshape(60)
        obstacles = np.asarray(observation["obstacles"], dtype=np.float32).reshape(5, 15)
        if self.mode == "stage1":
            action, route_status = self.stage1.act(state)
            self._previous = action.copy()
            self.last_metrics = {
                "qp_status": route_status,
                "cbf_intervention": 0.0,
                "minimum_clearance": float("inf"),
                "filter_latency_ms": 0.0,
                "planning_latency_ms": float(1000.0 * (time.perf_counter() - started)),
                "region": route_status,
                "candidate_count": 0,
                "raw_fm_feasible": 0,
                "emergency_stops": 0,
                "new_plan": 0,
                **self._record_action_source("stage1"),
            }
            return action
        if self.mode == "stage2":
            started_stage2 = time.perf_counter()
            safety_obstacles = obstacles
            if not perception_accepts_new_plan:
                safety_obstacles = self._cached_safety_obstacles(obstacles, dynamic_missing_time)
            action, route_status = self.stage2.act(
                state,
                safety_obstacles,
                observed_obstacles=obstacles,
                tracks_fresh=perception_accepts_new_plan,
                dynamic_missing_time=dynamic_missing_time,
                track_missing_times=track_missing_times,
                previous_action=self._previous,
            )
            self._previous = np.asarray(action, dtype=np.float32).copy()
            stage2_metrics = dict(self.stage2.last_metrics)
            self.last_metrics = {
                "qp_status": route_status,
                "cbf_intervention": float(stage2_metrics.get("cbf_intervention", 0.0)),
                "minimum_clearance": float(stage2_metrics.get("minimum_clearance", float("inf"))),
                "filter_latency_ms": float(stage2_metrics.get("filter_latency_ms", 0.0)),
                "planning_latency_ms": float(1000.0 * (time.perf_counter() - started_stage2)),
                "region": str(stage2_metrics.get("selected_arm_region", "nominal")),
                "candidate_count": int(stage2_metrics.get("candidate_count", 0)),
                "raw_fm_feasible": 0,
                **stage2_metrics,
                **self._record_action_source("stage2"),
            }
            return action
        planning_region = expert_region
        if self.mode == "expert" and planning_region is None and np.linalg.norm(state[53:55]) < 0.35:
            planning_region = "nominal"
        safety_obstacles = obstacles
        if not perception_accepts_new_plan:
            safety_obstacles = self._cached_safety_obstacles(obstacles, dynamic_missing_time)
        else:
            self._last_obstacles[obstacles[:, 14] > 0.5] = obstacles[obstacles[:, 14] > 0.5]
        forced_selection = None
        if not perception_accepts_new_plan and self._certified_tail is not None and self._tail_age < HORIZON:
            shifted = np.vstack((self._certified_tail[1:], self._certified_tail[-1:]))
            recertified = self.safety.project(shifted, state, safety_obstacles, previous_action=self._previous)
            if recertified.feasible:
                forced_selection = (recertified.actions, recertified)
        if not perception_accepts_new_plan and forced_selection is None:
            self._emergency_stops += 1
            yaw_error = float(state[55])
            if abs(yaw_error) > np.deg2rad(6.0):
                reacquire = np.zeros(17, dtype=np.float32)
                reacquire[2] = np.clip(yaw_error / float(self.kinematics.base_limits[2]), -0.60, 0.60)
                reacquire_result = self.safety.shield(reacquire, state, safety_obstacles, previous_action=self._previous)
                if reacquire_result.feasible:
                    action = np.asarray(reacquire_result.actions, dtype=np.float32)
                    self._previous = action.copy()
                    self.last_metrics = {
                        "qp_status": "perception_lost_reacquire",
                        "emergency_stops": self._emergency_stops,
                        "planning_latency_ms": 1000.0 * (time.perf_counter() - started),
                        "filter_latency_ms": reacquire_result.latency_ms,
                        "cbf_intervention": reacquire_result.intervention_rms,
                        "minimum_clearance": reacquire_result.minimum_clearance,
                        "new_plan": 0,
                        **self._record_action_source("perception_reacquire"),
                    }
                    return action
            action, emergency = self._emergency_action(state, safety_obstacles)
            if not emergency.feasible:
                action = np.zeros(17, dtype=np.float32)
            self._previous = np.asarray(action, dtype=np.float32).copy()
            self.last_metrics = {"qp_status": "perception_lost_emergency" if emergency.feasible else "perception_lost_stop", "emergency_stops": self._emergency_stops, "planning_latency_ms": 1000.0 * (time.perf_counter() - started), "filter_latency_ms": emergency.latency_ms, "cbf_intervention": emergency.intervention_rms, "minimum_clearance": emergency.minimum_clearance, "new_plan": 0, **self._record_action_source("perception_emergency")}
            return np.asarray(action, dtype=np.float32)
        new_plan_selected = False
        if forced_selection is not None:
            candidates = []
            region, projection = "recertified_old", forced_selection[1]
            action_source = self._certified_tail_source
            self._tail_age += 1
            preprojected = None
        else:
            self._tail_age = 0
            candidates = self._fm_candidates(state, obstacles) if self.mode in {"fm", "hybrid"} else []
            if not candidates:
                action_source = "expert"
                if planning_region is None:
                    sequence, region, projection = self.expert.plan(state, obstacles)
                else:
                    sequence, region, projection = self.expert.plan_region(state, obstacles, planning_region)
                    if not projection.feasible:
                        sequence, region, projection = self.expert.plan(state, obstacles)
                candidates = [sequence]
                preprojected = (sequence, projection) if projection.feasible else None
                new_plan_selected = bool(projection.feasible)
            else:
                action_source = "fm"
                region, projection = "fm", None
                preprojected = None
        selected = forced_selection if forced_selection is not None else preprojected
        if selected is not None and float(selected[1].minimum_clearance) < self.EMERGENCY_CLEARANCE_TRIGGER:
            selected = None
            new_plan_selected = False
        projections = [preprojected[1]] if preprojected is not None else []
        scored_candidates = [
            (self._candidate_score(state, obstacles, candidate), candidate)
            for candidate in candidates
        ]
        ranked_candidates = [
            candidate
            for _, candidate in sorted(scored_candidates, key=lambda item: item[0], reverse=True)
        ]
        if action_source == "fm":
            self._fm_candidate_total += len(scored_candidates)
            self._fm_raw_feasible_total += sum(score[1] for score, _ in scored_candidates)
        for candidate in (() if forced_selection is not None or preprojected is not None else ranked_candidates):
            result = self.safety.project(candidate, state, obstacles, previous_action=self._previous)
            projections.append(result)
            if result.feasible:
                selected = (result.actions, result)
                new_plan_selected = True
                break
        if selected is None and self._certified_tail is not None:
            shifted = np.vstack((self._certified_tail[1:], self._certified_tail[-1:]))
            result = self.safety.project(shifted, state, obstacles, previous_action=self._previous)
            if result.feasible:
                selected = (result.actions, result)
                new_plan_selected = False
                region = "recertified_old"
                action_source = self._certified_tail_source
        if selected is None:
            self._emergency_stops += 1
            action, emergency = self._emergency_action(state, safety_obstacles)
            if not emergency.feasible:
                action = np.zeros(17, dtype=np.float32)
            elif np.asarray(getattr(emergency, "actions", action)).ndim == 2:
                self._certified_tail = np.asarray(emergency.actions, dtype=np.float32).copy()
                self._certified_tail_source = "emergency"
            self._previous = np.asarray(action, dtype=np.float32).copy()
            self.last_metrics = {
                "qp_status": "candidate_emergency" if emergency.feasible else "candidate_stop",
                "cbf_intervention": float(emergency.intervention_rms),
                "minimum_clearance": float(emergency.minimum_clearance),
                "filter_latency_ms": float(emergency.latency_ms),
                "planning_latency_ms": float(1000.0 * (time.perf_counter() - started)),
                "region": "emergency",
                "candidate_count": len(candidates),
                "raw_fm_feasible": sum(int(item.feasible) for item in projections),
                "emergency_stops": self._emergency_stops,
                "new_plan": 0,
                **self._record_action_source("emergency"),
            }
            return np.asarray(action, dtype=np.float32)
        sequence, result = selected
        action = np.asarray(sequence[0], dtype=np.float32).copy()
        if self.arm_locked:
            action[3:] = 0.0
        shield = self.safety.shield(action, state, obstacles, previous_action=self._previous)
        if shield.feasible:
            action = np.asarray(shield.actions, dtype=np.float32)
        else:
            # A zero delta keeps both arm position targets unchanged. Never
            # execute unshielded arm components after the final CBF-QP fails.
            action = np.zeros(17, dtype=np.float32)
        self._previous = action.copy()
        if result.feasible:
            self._certified_tail = np.asarray(sequence, dtype=np.float32)
            self._certified_tail_source = action_source
            if new_plan_selected:
                self.last_sequence = np.asarray(sequence, dtype=np.float32).copy()
        self.last_metrics = {
            "qp_status": str(result.status),
            "cbf_intervention": float(result.intervention_rms),
            "minimum_clearance": float(result.minimum_clearance),
            "filter_latency_ms": float(shield.latency_ms),
            "planning_latency_ms": float(1000.0 * (time.perf_counter() - started)),
            "region": str(region),
            "candidate_count": len(candidates),
            "raw_fm_feasible": sum(int(item.feasible) for item in projections),
            "emergency_stops": self._emergency_stops,
            "new_plan": int(self.last_sequence is not None),
            **self._record_action_source(action_source),
        }
        return np.clip(action, -1.0, 1.0)


# Compatibility for code written against the pre-release controller name.
TiagoDualSafeCarryV2Policy = FlowCarryCBFPolicy

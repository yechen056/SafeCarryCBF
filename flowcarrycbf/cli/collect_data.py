#!/usr/bin/env python3
"""Prescreen and record Phase 3 RGB demonstrations with a no-CBF oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np

from flowcarrycbf.policies.flowcarry_cbf.schema import (
    CAMERA_COUNT,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DATASET_FORMAT_VERSION,
    controlled_positions,
    validate_episode,
)
from flowcarrycbf.policies.flowcarry_cbf.control.schema import TYPE_INDEX, VALID_INDEX
from flowcarrycbf.policies.flowcarry_cbf.control.scene import (
    QUICK_TEMPLATE_ALIASES,
    QUICK_TEMPLATE_IDS,
    QUICK_TEMPLATE_TOPOLOGY,
)


TEACHER_NAME = "phase3_local_oracle_no_cbf"
QUICK_SCENARIO = "phase3_quick"
REPLAY_STATE_ERROR_TOLERANCE = 5.0e-3
PLAN_FORMAT_VERSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prescreen", "record", "all"), default="all")
    parser.add_argument(
        "--scenario", choices=(QUICK_SCENARIO,), default=QUICK_SCENARIO,
        help="The only supported RGB collection profile.",
    )
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--max-attempts", type=int, default=0,
        help="Optional emergency cap across all templates; 0 means keep resampling until quotas are met.",
    )
    parser.add_argument(
        "--max-trials-per-template", type=int, default=0,
        help="Optional diagnostic cap per template; 0 means no cap.",
    )
    parser.add_argument(
        "--no-success-stop", type=int, default=30,
        help="Stop a template after this many attempts with zero accepted episodes; 0 disables the watchdog.",
    )
    parser.add_argument(
        "--template-id", default=None,
        help="Collect only this fixed quick behavior template (useful for smoke tests).",
    )
    parser.add_argument(
        "--template-ids", default=None,
        help="Comma-separated quick templates to collect with an even quota split.",
    )
    parser.add_argument("--worker-threads", type=int, default=4)
    parser.add_argument("--physics-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/raw/phase3_quick"),
    )
    parser.add_argument("--plans-dir", type=Path, default=None)
    parser.add_argument(
        "--require-topology",
        choices=("2ball_no_cube", "1ball_cube"),
        default=None,
        help="Restrict prescreen to one topology for a visual approval sample.",
    )
    parser.add_argument(
        "--require-arm-avoidance", action="store_true",
        help="Require at least one dynamic_arm_avoidance step in each accepted plan.",
    )
    parser.add_argument(
        "--require-cube-bypass", action="store_true",
        help="Require the static cube route in each accepted plan.",
    )
    parser.add_argument(
        "--prescreen-video", action="store_true",
        help="Temporarily record diagnostic cameras for prescreen attempts; no HDF5 is written.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _checksum(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def _topology(task: Any) -> tuple[int, int]:
    models = list(getattr(task, "_v2_models", []))
    balls = sum(str(model.get("shape")) == "sphere" for model in models)
    cubes = sum(str(model.get("shape")) == "cube" for model in models)
    return int(balls), int(cubes > 0)


def topology_key(balls: int, has_cube: int | bool) -> str:
    return f"{int(balls)}ball_{'cube' if has_cube else 'no_cube'}"


def topology_quotas(episodes: int) -> dict[str, int]:
    """Return deterministic 50/50 quotas for the reduced layouts."""

    if episodes <= 0:
        raise ValueError("--episodes must be positive")
    cube = int(round(0.50 * episodes))
    no_cube = episodes - cube
    return {
        "2ball_no_cube": no_cube,
        "1ball_cube": cube,
    }


def _failure_reason(info: dict[str, Any], *, exhausted: bool = False) -> str:
    if bool(info.get("table_collision", False)):
        name = str(info.get("table_collision_name", "table"))
        entity = str(info.get("table_collision_entity", "robot"))
        return f"table_collision:{name}:{entity}"
    if bool(info.get("obstacle_collision", False)):
        entity = str(info.get("obstacle_collision_entity", "robot"))
        return f"random_obstacle_collision:{entity}"
    if not bool(info.get("continuous_handle_contact", True)):
        return "handle_contact_lost"
    if exhausted:
        return "max_steps_without_success"
    return str(info.get("failure_reason", "")) or "task_failed"


def _is_valid(info: dict[str, Any]) -> bool:
    return bool(
        info.get("success", False)
        and not info.get("table_collision", False)
        and not info.get("obstacle_collision", False)
        and info.get("continuous_handle_contact", False)
        and info.get("left_handle_contact", False)
        and info.get("right_handle_contact", False)
    )


def _early_failure(info: dict[str, Any]) -> bool:
    return bool(
        info.get("table_collision", False)
        or info.get("obstacle_collision", False)
        or not info.get("continuous_handle_contact", True)
    )


def _hard_clearance_frames(info: dict[str, Any], *, missing_default: int) -> int:
    return int(
        info.get(
            "clearance_below_hard_limit_frames",
            info.get("clearance_below_8cm_frames", missing_default),
        )
    )


def _minimum_true_clearance(task: Any) -> float:
    value = float(getattr(task, "_v2_minimum_clearance", float("inf")))
    return value if np.isfinite(value) else float("nan")


def _quick_preflight(task: Any, template_id: str) -> tuple[bool, str]:
    """Reject only structurally impossible templates before expensive planning."""
    template_id = QUICK_TEMPLATE_ALIASES.get(str(template_id), str(template_id))
    models = list(getattr(task, "_v2_models", []))
    if not models:
        return False, "no_obstacles"
    expected_cube = template_id.startswith("cube_")
    if sum(str(model.get("shape")) == "sphere" for model in models) != 2 - int(expected_cube):
        return False, "template_sphere_count"
    if bool(any(str(model.get("shape")) == "cube" for model in models)) != expected_cube:
        return False, "template_cube_count"
    positions = [np.asarray(model["position"], dtype=np.float32) for model in models]
    if len(positions) == 2 and abs(float(positions[0][0] - positions[1][0])) < 1.0 - 1e-6:
        return False, "x_separation_below_1m"
    for model in models:
        if str(model.get("shape")) == "cube" and float(model["position"][1]) <= 0.0:
            return False, "cube_not_on_plus_y"
    return True, ""


def _template_behavior_failure(
    template_id: str, phase_counts: dict[str, int], route: str,
    states: list[np.ndarray] | None = None,
    kinematics: Any | None = None,
) -> str:
    arm_steps = sum(
        count for phase, count in phase_counts.items()
        if str(phase).startswith("dynamic_arm_avoidance")
    )
    base_steps = int(phase_counts.get("dynamic_emergency_base", 0))
    cube_steps = int(phase_counts.get("preturn_side_shift", 0)) + int(phase_counts.get("traverse", 0))
    lateral_motion = 0.0
    arm_motion = 0.0
    if states:
        values = np.asarray(states, dtype=np.float32)
        if values.ndim == 2 and values.shape[1] > 1:
            lateral_motion = float(np.max(np.abs(values[:, 1] - values[0, 1])))
        # The quick state contract is [base(3), left arm(7), right arm(7)].
        if values.ndim == 2 and values.shape[1] >= 17:
            if kinematics is not None:
                hand_y = []
                for value in values:
                    spheres = kinematics._spheres(
                        value[:3], value[:3], value[3:10], value[10:17]
                    )
                    hand_y.append([float(spheres[-3].center[1]), float(spheres[-2].center[1])])
                hand_values = np.asarray(hand_y, dtype=np.float32)
                arm_motion = float(np.max(np.abs(hand_values - hand_values[0])))
            else:
                arm_motion = float(np.max(np.abs(values[:, 3:17] - values[0, 3:17])))
    if template_id.endswith("_arm") and arm_steps <= 0:
        return "template_missing_arm_response"
    if template_id.endswith("_arm") and arm_motion < 0.08:
        return "template_arm_motion_too_small"
    if template_id.endswith("_joint") and (
        arm_steps <= 0 or arm_motion < 0.08
        or (base_steps + cube_steps) <= 0 or lateral_motion < 0.15
    ):
        return (
            "template_missing_joint_response:"
            f"arm_steps={arm_steps},arm_motion={arm_motion:.3f},"
            f"base_steps={base_steps},cube_steps={cube_steps},"
            f"lateral_motion={lateral_motion:.3f}"
        )
    if template_id == "cube_arm_easy" and route != "cube_bypass":
        return "template_missing_cube_bypass"
    return ""


def _resolved_failure(
    oracle_failure_reason: str,
    diagnostic_replay_reason: str,
    *,
    diagnostic_only: bool,
) -> str:
    """Keep Oracle rejection authoritative over an optional debug replay."""

    if oracle_failure_reason:
        return str(oracle_failure_reason)
    if diagnostic_only and diagnostic_replay_reason:
        return f"diagnostic_replay_failed:{diagnostic_replay_reason}"
    return str(diagnostic_replay_reason)


def _template_statistics(
    template_ids: list[str], accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]], attempts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for template in template_ids:
        accepted_count = sum(str(row.get("template_id", "")) == template for row in accepted)
        rejected_rows = [row for row in rejected if str(row.get("template_id", "")) == template]
        preflight = sum(str(row.get("failure_stage", "")) == "preflight" for row in rejected_rows)
        oracle = sum(str(row.get("failure_stage", "")) == "oracle" for row in rejected_rows)
        physx = sum(str(row.get("failure_stage", "")) == "physx" for row in rejected_rows)
        total = int(attempts.get(template, 0))
        result[template] = {
            "attempts": total,
            "preflight_rejected": preflight,
            "oracle_rejected": oracle,
            "physx_rejected": physx,
            "accepted": accepted_count,
            "acceptance_rate": float(accepted_count / total) if total else 0.0,
            "plan_success_rate": float(sum(bool(row.get("oracle_plan_success", False)) for row in rejected_rows) + accepted_count) / total if total else 0.0,
            "replay_success_rate": float(sum(bool(row.get("physx_check_success", False)) for row in rejected_rows) + accepted_count) / total if total else 0.0,
        }
    return result


def _make_env(args: argparse.Namespace, mode: str):
    from flowcarrycbf.envs.tiago_flow_carry_env import FlowCarryCBFEnv

    return FlowCarryCBFEnv(
        headless=True,
        render=mode in {"record", "prescreen_video"},
        randomize=True,
        obstacle_scenario=getattr(args, "scenario", QUICK_SCENARIO),
        lightweight_info=True,
        phase3_visuals=True,
        enable_eval_video_cameras=mode in {"record", "prescreen_video"},
        observation_mode="record" if mode == "prescreen_video" else mode,
        worker_threads=args.worker_threads,
        physics_device=args.physics_device,
    )


def _base_manifest(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    template_ids = _requested_template_ids(args)
    template_quotas = _template_quotas(args, template_ids)
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "plan_format_version": PLAN_FORMAT_VERSION,
        "teacher": TEACHER_NAME,
        "scenario": str(getattr(args, "scenario", QUICK_SCENARIO)),
        "oracle_profile": "quick",
        # Clearance is reported for analysis; it is not a collection gate.
        "hard_clearance_threshold_m": 0.0,
        "stage": stage,
        "status": "collecting",
        "episodes_requested": int(args.episodes),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "worker_threads": int(args.worker_threads),
        "physics_device": str(args.physics_device),
        "topology_quotas": topology_quotas(int(args.episodes)),
        "template_ids": template_ids,
        "template_quotas": template_quotas,
        "max_trials_per_template": int(getattr(args, "max_trials_per_template", 0)),
        "no_success_stop": int(getattr(args, "no_success_stop", 30)),
        "template_id": str(getattr(args, "template_id", "") or ""),
        "episodes": [],
        "rejected": [],
        "lifecycle_stage": "manifest_created",
        "lifecycle_seed": None,
    }


def _load_or_create_manifest(path: Path, args: argparse.Namespace, stage: str) -> dict[str, Any]:
    if args.resume and path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if int(result.get("episodes_requested", -1)) != args.episodes:
            raise ValueError("resume episode count differs from manifest")
        if int(result.get("seed", -1)) != args.seed:
            raise ValueError("resume seed differs from manifest")
        expected_scenario = str(getattr(args, "scenario", QUICK_SCENARIO))
        if str(result.get("scenario", "")) != expected_scenario:
            raise ValueError("resume scenario differs from phase3_quick")
        if str(result.get("oracle_profile", "")) != "quick":
            raise ValueError("resume oracle profile is not quick")
        expected_templates = _requested_template_ids(args)
        saved_templates = result.get("template_ids")
        if saved_templates is None:
            saved_template = str(result.get("template_id", "") or "")
            saved_templates = [saved_template] if saved_template else list(QUICK_TEMPLATE_IDS)
        saved_templates = [
            QUICK_TEMPLATE_ALIASES.get(str(template), str(template))
            for template in saved_templates
        ]
        if list(saved_templates) != expected_templates:
            raise ValueError("resume templates differ from manifest")
        result["status"] = "collecting"
        return result
    return _base_manifest(args, stage)


def _requested_template_ids(args: argparse.Namespace) -> list[str]:
    single = str(getattr(args, "template_id", "") or "")
    raw = getattr(args, "template_ids", None)
    if single and raw:
        raise ValueError("use --template-id or --template-ids, not both")
    if single:
        single = QUICK_TEMPLATE_ALIASES.get(single, single)
        if single not in QUICK_TEMPLATE_IDS:
            raise ValueError(f"unsupported quick template ID: {single}")
        return [single]
    if raw:
        values = [
            QUICK_TEMPLATE_ALIASES.get(item.strip(), item.strip())
            for item in str(raw).split(",") if item.strip()
        ]
        if not values or len(set(values)) != len(values):
            raise ValueError("--template-ids must contain unique template IDs")
        unsupported = [item for item in values if item not in QUICK_TEMPLATE_IDS]
        if unsupported:
            raise ValueError(f"unsupported quick template IDs: {unsupported}")
        return values
    return list(QUICK_TEMPLATE_IDS)


def _template_quotas(args: argparse.Namespace, template_ids: list[str]) -> dict[str, int]:
    episodes = int(args.episodes)
    if getattr(args, "template_id", None):
        return {template_ids[0]: episodes}
    if getattr(args, "template_ids", None):
        if episodes <= 0 or episodes % len(template_ids) != 0:
            raise ValueError("--episodes must divide evenly across --template-ids")
        return {template: episodes // len(template_ids) for template in template_ids}
    if episodes == len(template_ids):
        return {template: 1 for template in template_ids}
    if episodes == 3 * len(template_ids):
        return {template: 3 for template in template_ids}
    raise ValueError(
        f"quick collection requires --episodes {len(template_ids)} "
        f"(one per template) or {3 * len(template_ids)} "
        "(three per template)"
    )


def _save_plan(
    path: Path,
    *,
    seed: int,
    actions: np.ndarray,
    states: np.ndarray,
    schedule: np.ndarray,
    template_id: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(PLAN_FORMAT_VERSION, dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.float32),
        states=np.asarray(states, dtype=np.float32),
        obstacle_schedule=np.asarray(schedule, dtype=np.float32),
        template_id=np.asarray(str(template_id)),
    )


def _write_progress(
    message: str, *, newline: bool = False, green: bool = False,
) -> None:
    """Replace the current terminal line without leaving stale characters."""
    rendered = f"\033[32m{message}\033[0m" if green else message
    if os.environ.get("SAFE_CARRY_VERBOSE_PROGRESS", "0") == "1":
        sys.stdout.write(f"{rendered}\n")
        sys.stdout.flush()
        return
    line_ending = "\n" if newline else ""
    sys.stdout.write(f"\r\033[2K{rendered}{line_ending}")
    sys.stdout.flush()


def _mark_lifecycle(
    manifest_path: Path, manifest: dict[str, Any], stage: str, **fields: Any,
) -> None:
    """Persist the last collector stage before entering Isaac work."""

    manifest["lifecycle_stage"] = str(stage)
    manifest["lifecycle_stage_started_at"] = time.time()
    manifest.update(fields)
    _atomic_json(manifest_path, manifest)


def _prescreen(args: argparse.Namespace, plans_dir: Path) -> int:
    from flowcarrycbf.policies.flowcarry_cbf.oracle_expert import Phase3OracleTrajectoryExpert

    plans_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plans_dir / "manifest.json"
    manifest = _load_or_create_manifest(manifest_path, args, "prescreen")
    template_ids = _requested_template_ids(args)
    quotas = _template_quotas(args, template_ids)
    manifest["template_quotas"] = dict(quotas)
    manifest["topology_quotas"] = topology_quotas(sum(quotas.values()))
    _atomic_json(manifest_path, manifest)
    if args.require_topology is not None:
        # Approval samples intentionally request a topology that may have a
        # zero quota in the normal 30% cube split (for example one 3-ball cube
        # episode).  Formal multi-episode collection keeps the normal quotas.
        template_ids = [template for template in template_ids if QUICK_TEMPLATE_TOPOLOGY[template] == args.require_topology]
        if not template_ids:
            raise ValueError(f"no quick template matches topology {args.require_topology}")
        if args.template_id:
            quotas = {template_ids[0]: int(args.episodes)}
        elif args.template_ids:
            requested = _requested_template_ids(args)
            if int(args.episodes) % len(requested) != 0:
                raise ValueError("--episodes must divide evenly across --template-ids")
            per_template = int(args.episodes) // len(requested)
            quotas = {template: per_template for template in template_ids}
        else:
            quotas = {template: 3 for template in template_ids}
        manifest["topology_quotas"] = {args.require_topology: int(args.episodes)}
        manifest["template_quotas"] = dict(quotas)
        _atomic_json(manifest_path, manifest)
    accepted = list(manifest.get("episodes", []))
    rejected = list(manifest.get("rejected", []))
    counts = {key: 0 for key in quotas}
    for row in accepted:
        template = str(row.get("template_id", ""))
        if template in counts:
            counts[template] += 1
    attempts = int(manifest.get("attempts", len(accepted) + len(rejected)))
    elapsed_before_run = float(manifest.get("elapsed_seconds", 0.0))
    started = time.perf_counter()
    env = None
    try:
        _mark_lifecycle(manifest_path, manifest, "env_create")
        env = _make_env(args, "prescreen_video" if args.prescreen_video else "truth")
        expert = Phase3OracleTrajectoryExpert(dt=0.1, profile="quick")
        max_attempts = int(getattr(args, "max_attempts", 0))
        max_trials_per_template = int(getattr(args, "max_trials_per_template", 0))
        no_success_stop = int(getattr(args, "no_success_stop", 30))
        while len(accepted) < sum(quotas.values()) and (max_attempts <= 0 or attempts < max_attempts):
            available = [
                template for template in template_ids
                if counts[template] < quotas[template]
                and (max_trials_per_template <= 0 or int(manifest.get("template_attempts", {}).get(template, 0)) < max_trials_per_template)
                and (no_success_stop <= 0 or counts[template] > 0 or int(manifest.get("template_attempts", {}).get(template, 0)) < no_success_stop)
            ]
            if not available:
                stopped = [
                    template for template in template_ids
                    if counts[template] == 0
                    and no_success_stop > 0
                    and int(manifest.get("template_attempts", {}).get(template, 0)) >= no_success_stop
                ]
                if stopped:
                    manifest.setdefault("diagnostic_stops", []).extend(
                        template for template in stopped
                        if template not in manifest.get("diagnostic_stops", [])
                    )
                    _write_progress(
                        "Stopping templates with no accepted episode after "
                        f"{no_success_stop} attempts: {', '.join(stopped)}",
                        newline=True,
                    )
                break
            # Interleave templates deterministically instead of exhausting the
            # first template.  Deriving the choice from the global attempt
            # count keeps resumed collections on the same sequence.
            selection_key = f"{int(args.seed)}:{attempts}".encode("utf-8")
            selection_digest = hashlib.blake2b(selection_key, digest_size=8).digest()
            selection_index = int.from_bytes(selection_digest, "big") % len(available)
            template_id = available[selection_index]
            template_attempts = dict(manifest.get("template_attempts", {}))
            attempt_index = int(template_attempts.get(template_id, 0))
            template_attempts[template_id] = attempt_index + 1
            manifest["template_attempts"] = template_attempts
            if (attempt_index + 1) % 10 == 0 and counts[template_id] == 0:
                alert = (
                    f"Template {template_id} has no accepted episode after "
                    f"{attempt_index + 1} attempts; inspect planner/layout diagnostics."
                )
                _write_progress(alert, newline=True)
                manifest.setdefault("diagnostic_alerts", []).append({
                    "template_id": template_id,
                    "attempts": attempt_index + 1,
                    "accepted": counts[template_id],
                    "message": alert,
                })
            episode_seed = args.seed + template_ids.index(template_id) * 100000 + attempt_index
            attempts += 1
            attempt_started = time.perf_counter()
            _mark_lifecycle(
                manifest_path,
                manifest,
                "reset",
                lifecycle_seed=int(episode_seed),
                lifecycle_template_id=template_id,
            )
            try:
                observation, info = env.reset(
                    seed=episode_seed,
                    options={
                        "obstacle_scenario": getattr(args, "scenario", QUICK_SCENARIO),
                        "obstacle_template_id": template_id,
                        "randomize": True,
                    },
                )
            except BaseException as exc:
                manifest.update({
                    "status": "interrupted",
                    "interrupted_stage": "reset",
                    "interrupted_seed": int(episode_seed),
                    "interrupted_template_id": template_id,
                    "interrupted_exception_type": type(exc).__name__,
                    "interrupted_exception": str(exc) or repr(exc),
                })
                _atomic_json(manifest_path, manifest)
                raise
            _mark_lifecycle(
                manifest_path,
                manifest,
                "preflight",
                lifecycle_seed=int(episode_seed),
                lifecycle_template_id=template_id,
            )
            balls, has_cube = _topology(env.task)
            key = topology_key(balls, has_cube)
            expected_key = QUICK_TEMPLATE_TOPOLOGY[template_id]
            if key != expected_key:
                rejected.append({
                    "seed": episode_seed,
                    "template_id": template_id,
                    "topology": key,
                    "attempt_index": attempt_index,
                    "failure_stage": "preflight",
                    "reason": "template_topology_mismatch",
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                })
                manifest.update({"attempts": attempts, "rejected": rejected})
                manifest["rejected_attempts"] = len(rejected)
                _atomic_json(manifest_path, manifest)
                continue

            if args.require_topology is not None and key != args.require_topology:
                rejected.append({
                    "seed": episode_seed,
                    "template_id": template_id,
                    "attempt_index": attempt_index,
                    "topology": key,
                    "reason": "approval_topology_mismatch",
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                })
                manifest.update({"attempts": attempts, "rejected": rejected})
                manifest["rejected_attempts"] = len(rejected)
                _atomic_json(manifest_path, manifest)
                continue

            schedule = env.get_oracle_obstacle_schedule(args.max_steps)
            preflight_success, preflight_reason = _quick_preflight(env.task, template_id)
            if not preflight_success:
                rejected.append({
                    "seed": episode_seed,
                    "template_id": template_id,
                    "template_variant": template_id,
                    "attempt_index": attempt_index,
                    "topology": key,
                    "reason": f"preflight:{preflight_reason}",
                    "failure_stage": "preflight",
                    "preflight_success": False,
                    "oracle_plan_success": False,
                    "physx_check_success": False,
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                })
                manifest.update({"attempts": attempts, "rejected": rejected, "rejected_attempts": len(rejected)})
                _atomic_json(manifest_path, manifest)
                continue
            expert.reset()
            actions: list[np.ndarray] = []
            states: list[np.ndarray] = [controlled_positions(observation["state"])]
            debug_states: list[np.ndarray] = [controlled_positions(observation["state"])]
            debug_frames: list[list[np.ndarray]] = [[], []]
            debug_eval_frames: list[list[np.ndarray]] = [[], []]
            terminated = False
            truncated = False
            predicted_clearance = float("inf")
            phase_counts: dict[str, int] = {}
            oracle_seconds = 0.0
            oracle_failure_reason = ""
            diagnostic_replay_reason = ""
            diagnostic_only = False
            diagnostic_replay_steps = 0
            oracle_started = time.perf_counter()
            _write_progress(
                f"Oracle planning seed {episode_seed} | step 0/{args.max_steps} | beam 0",
                newline=True,
            )

            def _planning_progress(step: int, total: int, beam_size: int) -> None:
                if step == 0 or step % 10 == 0 or step + 1 == total:
                    _write_progress(
                        f"Oracle planning seed {episode_seed} | "
                        f"step {step}/{total} | beam {beam_size}",
                        newline=True,
                    )

            _mark_lifecycle(
                manifest_path,
                manifest,
                "oracle",
                lifecycle_seed=int(episode_seed),
                lifecycle_template_id=template_id,
            )
            oracle_plan = expert.plan(
                observation["state"], schedule, max_steps=args.max_steps,
                progress_callback=_planning_progress,
                required_arm_response=template_id.endswith(("_arm", "_joint")),
                required_joint_response=template_id.endswith("_joint"),
            )
            oracle_seconds = time.perf_counter() - oracle_started
            if oracle_plan is None:
                oracle_failure = str(
                    getattr(expert, "_last_plan_failure", "")
                    or "unknown"
                )
                oracle_failure_reason = f"oracle_no_complete_safe_trajectory:{oracle_failure}"
                partial_actions = np.asarray(
                    getattr(expert, "_last_plan_actions", np.empty((0, 17), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1, 17)
                partial_phases = tuple(getattr(expert, "_last_plan_phases", ()))
                # A rejected seed's test video must show the actual Oracle
                # branch up to failure. Nominal is only a last-resort fallback
                # when the planner produced no action at all.
                if args.prescreen_video and len(partial_actions):
                    planned_actions = partial_actions
                    planned_phases = partial_phases
                elif args.prescreen_video:
                    planned_actions = expert.nominal_debug_plan(
                        observation["state"], schedule, max_steps=args.max_steps
                    )
                    planned_phases = tuple("nominal_debug_fallback" for _ in planned_actions)
                else:
                    planned_actions = np.empty((0, 17), dtype=np.float32)
                    planned_phases = ()
                diagnostic_only = bool(args.prescreen_video)
            else:
                diagnostic_only = False
                predicted_clearance = float(oracle_plan.clearance)
                planned_actions = oracle_plan.actions
                planned_phases = oracle_plan.phases
                _write_progress(
                    f"Oracle plan ready seed {episode_seed} | "
                    f"steps {len(planned_actions)} | {oracle_seconds:.2f}s",
                    newline=True,
                )
                for phase in planned_phases:
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
                arm_avoidance_steps = sum(
                    count for phase, count in phase_counts.items()
                    if str(phase).startswith("dynamic_arm_avoidance")
                )
                if args.require_arm_avoidance and arm_avoidance_steps == 0:
                    oracle_failure_reason = "approval_missing_dynamic_arm_avoidance"
                if args.require_cube_bypass and oracle_plan.route != "cube_bypass":
                    oracle_failure_reason = "approval_missing_cube_bypass"
            if oracle_plan is None:
                _write_progress(
                    f"Oracle failed seed {episode_seed} | "
                    f"partial steps {len(planned_actions)} | {oracle_seconds:.2f}s",
                    newline=True,
                )
            _write_progress(
                f"Prescreen replay seed {episode_seed} | "
                f"steps {len(planned_actions)} | diagnostic={diagnostic_only}",
                newline=True,
            )
            _mark_lifecycle(
                manifest_path,
                manifest,
                "replay",
                lifecycle_seed=int(episode_seed),
                lifecycle_template_id=template_id,
            )
            for step_index, planned_action in enumerate(planned_actions):
                if oracle_failure_reason and not diagnostic_only:
                    break
                _write_progress(
                    f"Prescreen {len(accepted) + 1}/{args.episodes} | "
                    f"seed {episode_seed} | step {step_index}/{len(planned_actions)}",
                    green=True,
                )
                if args.prescreen_video:
                    views = env.rgb_views
                    video_views = env.eval_video_views
                    if len(views) == CAMERA_COUNT and len(video_views) == 2:
                        for camera_index, view in enumerate(views):
                            debug_frames[camera_index].append(np.asarray(view, dtype=np.uint8))
                        for camera_index, view in enumerate(video_views):
                            debug_eval_frames[camera_index].append(np.asarray(view, dtype=np.uint8))
                action = np.asarray(planned_action, dtype=np.float32)
                if not diagnostic_only:
                    actions.append(action)
                else:
                    diagnostic_replay_steps += 1
                observation, _, terminated, truncated, info = env.step(action)
                debug_states.append(controlled_positions(observation["state"]))
                if not diagnostic_only:
                    states.append(controlled_positions(observation["state"]))
                if _early_failure(info):
                    diagnostic_replay_reason = _failure_reason(info)
                    break
                if terminated or truncated:
                    break

            if not diagnostic_replay_reason and not _is_valid(info):
                diagnostic_replay_reason = _failure_reason(
                    info, exhausted=not (terminated or truncated)
                )
            if not diagnostic_replay_reason and oracle_plan is not None:
                behavior_failure = _template_behavior_failure(
                    template_id, phase_counts, str(oracle_plan.route), states,
                    expert.kinematics,
                )
                if behavior_failure:
                    diagnostic_replay_reason = behavior_failure
            failure = _resolved_failure(
                oracle_failure_reason,
                diagnostic_replay_reason,
                diagnostic_only=diagnostic_only,
            )
            if failure:
                if args.prescreen_video and debug_frames[0]:
                    debug_dir = plans_dir.parent / "prescreen_debug"
                    obstacle_debug = []
                    for model in list(getattr(env.task, "_v2_models", [])):
                        obstacle_debug.append({
                            "shape": str(model.get("shape", "")),
                            "position": np.asarray(model.get("position", (0, 0, 0)), dtype=np.float32).tolist(),
                            "half_extents": np.asarray(model.get("half_extents", (0, 0, 0)), dtype=np.float32).tolist(),
                            "interaction_candidate": bool(model.get("interaction_candidate", False)),
                        })
                    _atomic_json(
                        debug_dir / "obstacles" / f"seed_{episode_seed:06d}.json",
                        {
                            "seed": int(episode_seed),
                            "failure": failure,
                            "oracle_failure_reason": oracle_failure_reason,
                            "diagnostic_replay_reason": diagnostic_replay_reason,
                            "oracle_planned_steps": int(len(planned_actions)),
                            "oracle_planned_phases": list(planned_phases),
                            "diagnostic_replay_steps": int(diagnostic_replay_steps),
                            "diagnostic_only": diagnostic_only,
                            "models_at_failure": obstacle_debug,
                            "initial_schedule": np.asarray(schedule[0], dtype=np.float32).tolist(),
                            "state_at_failure": np.asarray(observation["state"], dtype=np.float32).tolist(),
                            "failure_info": {str(k): _json_safe(v) for k, v in dict(info).items()},
                        },
                    )
                    _write_video(
                        debug_dir / "videos" / f"seed_{episode_seed:06d}_overview.mp4",
                        np.stack(debug_eval_frames[1]),
                    )
                behavior_failure = str(diagnostic_replay_reason).startswith("template_")
                rejected.append({
                    "seed": episode_seed,
                    "template_id": template_id,
                    "template_variant": template_id,
                    "attempt_index": attempt_index,
                    "topology": key,
                    "reason": failure,
                    "failure_stage": (
                        "oracle" if oracle_failure_reason
                        else "behavior" if behavior_failure
                        else "physx"
                    ),
                    "preflight_success": True,
                    "oracle_plan_success": bool(oracle_plan is not None and not oracle_failure_reason),
                    "physx_check_success": bool(not diagnostic_replay_reason and oracle_plan is not None),
                    "steps": len(actions) if not diagnostic_only else diagnostic_replay_steps,
                    "oracle_failure_reason": oracle_failure_reason,
                    "diagnostic_replay_reason": diagnostic_replay_reason,
                    "oracle_planned_steps": int(len(planned_actions)),
                    "oracle_planned_phases": list(planned_phases),
                    "diagnostic_replay_steps": int(diagnostic_replay_steps),
                    "diagnostic_only": diagnostic_only,
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                })
                manifest.update({"attempts": attempts, "rejected": rejected})
                manifest["rejected_attempts"] = len(rejected)
                _atomic_json(manifest_path, manifest)
                _write_progress(f"Rejected seed {episode_seed}: {failure}", newline=True)
                continue

            episode_index = len(accepted)
            plan_name = f"plan_{episode_index:06d}.npz"
            state_values = np.asarray(states, dtype=np.float32)
            base_lateral_displacement = float(
                np.max(np.abs(state_values[:, 1] - state_values[0, 1]))
            ) if state_values.ndim == 2 and state_values.shape[1] > 1 else 0.0
            arm_lateral_displacement = 0.0
            if state_values.ndim == 2 and state_values.shape[1] >= 17:
                hand_y = []
                for value in state_values:
                    spheres = expert.kinematics._spheres(
                        value[:3], value[:3], value[3:10], value[10:17]
                    )
                    hand_y.append([float(spheres[-3].center[1]), float(spheres[-2].center[1])])
                hand_values = np.asarray(hand_y, dtype=np.float32)
                arm_lateral_displacement = float(np.max(np.abs(hand_values - hand_values[0])))
            _save_plan(
                plans_dir / plan_name,
                seed=episode_seed,
                actions=np.stack(actions),
                states=np.stack(states),
                schedule=schedule,
                template_id=template_id,
            )
            row = {
                "episode": episode_index,
                "seed": episode_seed,
                "template_id": template_id,
                "template_variant": template_id,
                "attempt_index": attempt_index,
                "plan": plan_name,
                "teacher": TEACHER_NAME,
                "topology": key,
                "dynamic_spheres": balls,
                "static_cubes": has_cube,
                "steps": len(actions),
                "obstacle_schedule_sha256": _checksum(schedule),
                "minimum_predicted_clearance": float(predicted_clearance),
                "minimum_true_clearance": _minimum_true_clearance(env.task),
                "minimum_table_clearance": float(getattr(env.task, "_v2_minimum_table_clearance", float("inf"))),
                "minimum_table_clearance_entity": str(getattr(env.task, "_v2_minimum_table_clearance_entity", "")),
                "phase_counts": phase_counts,
                "route": str(oracle_plan.route),
                "arm_region": str(oracle_plan.arm_region),
                "wait_steps": int(oracle_plan.wait_steps),
                "dynamic_arm_avoidance_steps": int(sum(
                    count for phase, count in phase_counts.items()
                    if str(phase).startswith("dynamic_arm_avoidance")
                )),
                "dynamic_emergency_base_steps": int(phase_counts.get("dynamic_emergency_base", 0)),
                "base_lateral_displacement_m": base_lateral_displacement,
                "base_return_error_m": float(abs(state_values[-1, 1] - state_values[0, 1])) if state_values.ndim == 2 else 0.0,
                "arm_lateral_displacement_m": arm_lateral_displacement,
                "behavior_phase_order": list(phase_counts),
                "cube_bypass_steps": int(phase_counts.get("preturn_side_shift", 0) + phase_counts.get("traverse", 0)) if oracle_plan.route == "cube_bypass" else 0,
                "oracle_candidates": int(oracle_plan.candidate_count),
                "timing_seconds": {
                    "oracle": oracle_seconds,
                    "physics": max(0.0, time.perf_counter() - attempt_started - oracle_seconds),
                    "total": time.perf_counter() - attempt_started,
                },
                "scenario": QUICK_SCENARIO,
                "oracle_profile": "quick",
                "oracle_failure_reason": "",
                "diagnostic_replay_reason": "",
                "preflight_success": True,
                "oracle_plan_success": True,
                "physx_check_success": True,
                "oracle_planned_steps": int(len(planned_actions)),
                "diagnostic_replay_steps": 0,
                "diagnostic_only": False,
            }
            accepted.append(row)
            counts[template_id] += 1
            manifest.update({
                "episodes": accepted,
                "accepted_episodes": len(accepted),
                "attempts": attempts,
                "rejected": rejected,
                "rejected_attempts": len(rejected),
                "elapsed_seconds": elapsed_before_run + time.perf_counter() - started,
            })
            _atomic_json(manifest_path, manifest)
            _write_progress(
                f"Prescreen {len(accepted)}/{sum(quotas.values())} complete | "
                f"seed {episode_seed} | steps {len(actions)}",
                newline=True,
            )

        manifest.update({
            "episodes": accepted,
            "accepted_episodes": len(accepted),
            "attempts": attempts,
            "rejected": rejected,
            "rejected_attempts": len(rejected),
            "template_statistics": _template_statistics(
                template_ids, accepted, rejected,
                dict(manifest.get("template_attempts", {})),
            ),
            "status": "complete" if len(accepted) == sum(quotas.values()) else "incomplete",
            "lifecycle_stage": "complete" if len(accepted) == sum(quotas.values()) else "incomplete",
        })
        _atomic_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["interrupted_stage"] = str(manifest.get("lifecycle_stage", "unknown"))
        manifest["interrupted_exception_type"] = type(exc).__name__
        manifest["interrupted_exception"] = str(exc) or repr(exc)
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        if env is not None:
            env.close()
    if len(accepted) != sum(quotas.values()):
        print(f"Prescreen incomplete: {len(accepted)}/{sum(quotas.values())}", flush=True)
        return 3
    return 0


def _write_video(path: Path, frames: np.ndarray, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _write_debug_plot(path: Path, states: np.ndarray, schedule: np.ndarray) -> None:
    """Write a compact XY/arm-response diagnostic for approval episodes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(states, dtype=np.float32)
    obstacles = np.asarray(schedule, dtype=np.float32)[: len(values)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(values[:, 0], values[:, 1], color="#202a44", linewidth=2.2, label="base")
    for index in range(obstacles.shape[1]):
        valid = obstacles[:, index, VALID_INDEX] > 0.5
        if not np.any(valid):
            continue
        kind = "cube" if obstacles[0, index, TYPE_INDEX] > 0.5 else "ball"
        axes[0].plot(
            obstacles[valid, index, 0], obstacles[valid, index, 1],
            "--", linewidth=1.2, label=f"{kind} {index}",
        )
    axes[0].set_title("Base and obstacle paths")
    axes[0].set_xlabel("world x (m)")
    axes[0].set_ylabel("world y (m)")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    arm_delta = np.linalg.norm(values[:, 8:22] - values[0, 8:22], axis=1)
    axes[1].plot(np.arange(len(arm_delta)) * 0.1, arm_delta, color="#b13b2e", linewidth=2)
    axes[1].set_title("Coordinated arm displacement")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("joint-space norm")
    axes[1].grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_episode(
    path: Path,
    frames: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    validate_episode(frames, proprio, actions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = DATASET_FORMAT_VERSION
        handle.attrs["camera_count"] = CAMERA_COUNT
        handle.attrs["camera_height"] = CAMERA_HEIGHT
        handle.attrs["camera_width"] = CAMERA_WIDTH
        handle.attrs["fps"] = 10
        handle.attrs["scenario"] = QUICK_SCENARIO
        handle.attrs["teacher"] = TEACHER_NAME
        handle.create_dataset("rgb", data=frames, dtype="u1", compression="lzf", chunks=True)
        handle.create_dataset("proprio", data=proprio, dtype="f4", compression="lzf", chunks=True)
        handle.create_dataset("actions", data=actions, dtype="f4", compression="lzf", chunks=True)
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                handle.attrs[key] = value


def _record(args: argparse.Namespace, plans_dir: Path, output: Path) -> int:
    plans_manifest_path = plans_dir / "manifest.json"
    if not plans_manifest_path.exists():
        raise FileNotFoundError(f"missing prescreen manifest: {plans_manifest_path}")
    plans_manifest = json.loads(plans_manifest_path.read_text(encoding="utf-8"))
    if str(plans_manifest.get("scenario", "")) != QUICK_SCENARIO:
        raise ValueError("prescreen manifest is not phase3_quick")
    if str(plans_manifest.get("oracle_profile", "")) != "quick":
        raise ValueError("prescreen manifest does not use the quick Oracle")
    if int(plans_manifest.get("plan_format_version", -1)) != PLAN_FORMAT_VERSION:
        raise ValueError("prescreen plan format is incompatible")
    plans = list(plans_manifest.get("episodes", []))
    if len(plans) < args.episodes:
        raise RuntimeError(f"only {len(plans)}/{args.episodes} prescreen plans are available")

    episodes_dir = output / "episodes"
    videos_dir = output / "videos"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = _load_or_create_manifest(manifest_path, args, "record")
    accepted = list(manifest.get("episodes", []))
    completed = {int(row["episode"]) for row in accepted}
    rejected = list(manifest.get("rejected", []))
    elapsed_before_run = float(manifest.get("elapsed_seconds", 0.0))
    started = time.perf_counter()
    env = None
    try:
        _mark_lifecycle(manifest_path, manifest, "env_create")
        env = _make_env(args, "record")
        for plan_row in plans[:args.episodes]:
            episode_index = int(plan_row["episode"])
            if episode_index in completed:
                continue
            plan_path = plans_dir / str(plan_row["plan"])
            with np.load(plan_path) as plan:
                if int(plan["format_version"]) != PLAN_FORMAT_VERSION:
                    raise ValueError(f"incompatible plan format: {plan_path}")
                episode_seed = int(plan["seed"])
                raw_actions = np.asarray(plan["actions"], dtype=np.float32)
                expected_states = np.asarray(plan["states"], dtype=np.float32)
                saved_schedule = np.asarray(plan["obstacle_schedule"], dtype=np.float32)
                saved_template = str(np.asarray(plan.get("template_id", "")).item())
            template_id = str(plan_row.get("template_id", saved_template))
            template_id = QUICK_TEMPLATE_ALIASES.get(template_id, template_id)
            attempt_started = time.perf_counter()
            _mark_lifecycle(
                manifest_path,
                manifest,
                "reset",
                lifecycle_seed=int(episode_seed),
                lifecycle_template_id=template_id,
            )
            try:
                observation, info = env.reset(
                    seed=episode_seed,
                    options={
                        "obstacle_scenario": getattr(args, "scenario", QUICK_SCENARIO),
                        "obstacle_template_id": template_id,
                        "randomize": True,
                    },
                )
            except BaseException as exc:
                manifest.update({
                    "status": "interrupted",
                    "interrupted_stage": "reset",
                    "interrupted_seed": int(episode_seed),
                    "interrupted_template_id": template_id,
                    "interrupted_exception_type": type(exc).__name__,
                    "interrupted_exception": str(exc) or repr(exc),
                })
                _atomic_json(manifest_path, manifest)
                raise
            _mark_lifecycle(
                manifest_path,
                manifest,
                "replay",
                lifecycle_seed=int(episode_seed),
                lifecycle_template_id=template_id,
            )
            if str(getattr(env.task, "_v2_template_id", "")) != template_id:
                raise RuntimeError(f"template reset mismatch for seed {episode_seed}: {template_id}")
            replay_schedule = env.get_oracle_obstacle_schedule(args.max_steps)
            if _checksum(replay_schedule) != _checksum(saved_schedule):
                raise RuntimeError(f"seed {episode_seed} obstacle trajectory is not deterministic")

            frames: list[np.ndarray] = []
            eval_frames: list[list[np.ndarray]] = [[], []]
            proprio: list[np.ndarray] = []
            action_targets: list[np.ndarray] = []
            maximum_state_error = 0.0
            oracle_failure_reason = str(plan_row.get("oracle_failure_reason", ""))
            diagnostic_replay_reason = ""
            failure = ""
            for step_index, raw_action in enumerate(raw_actions):
                _write_progress(
                    f"Record {episode_index + 1}/{args.episodes} | "
                    f"seed {episode_seed} | step {step_index}/{len(raw_actions)}"
                )
                current = controlled_positions(observation["state"])
                maximum_state_error = max(
                    maximum_state_error,
                    float(np.max(np.abs(current - expected_states[step_index]))),
                )
                views = env.rgb_views
                if len(views) != CAMERA_COUNT:
                    raise RuntimeError(f"expected {CAMERA_COUNT} training cameras, got {len(views)}")
                frame = np.stack([np.asarray(value, dtype=np.uint8) for value in views])
                if frame.shape != (CAMERA_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, 3):
                    raise RuntimeError(f"unexpected RGB shape {frame.shape}")
                frames.append(frame)
                video_views = env.eval_video_views
                if len(video_views) != 2:
                    raise RuntimeError(
                        f"expected first-person + overview cameras, got {len(video_views)}"
                    )
                for camera_index, video_view in enumerate(video_views):
                    eval_frames[camera_index].append(video_view)
                proprio.append(current)
                observation, _, terminated, truncated, info = env.step(raw_action)
                action_targets.append(controlled_positions(observation["state"]))
                if _early_failure(info):
                    diagnostic_replay_reason = _failure_reason(info)
                    break
                if terminated or truncated:
                    break

            if maximum_state_error > REPLAY_STATE_ERROR_TOLERANCE and not diagnostic_replay_reason:
                diagnostic_replay_reason = f"replay_state_error:{maximum_state_error:.6g}"
            if not diagnostic_replay_reason and not _is_valid(info):
                diagnostic_replay_reason = _failure_reason(
                    info, exhausted=not (terminated or truncated)
                )
            failure = _resolved_failure(
                oracle_failure_reason,
                diagnostic_replay_reason,
                diagnostic_only=False,
            )
            if failure:
                rejected.append({
                    "episode": episode_index,
                    "seed": episode_seed,
                    "reason": failure,
                    "steps": len(action_targets),
                    "oracle_failure_reason": oracle_failure_reason,
                    "diagnostic_replay_reason": diagnostic_replay_reason,
                    "oracle_planned_steps": int(plan_row.get("steps", len(raw_actions))),
                    "diagnostic_replay_steps": len(action_targets),
                    "diagnostic_only": False,
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                })
                manifest.update({"rejected": rejected, "rejected_attempts": len(rejected)})
                _atomic_json(manifest_path, manifest)
                _write_progress(f"Record rejected seed {episode_seed}: {failure}", newline=True)
                continue

            frame_array = np.stack(frames)
            proprio_array = np.stack(proprio).astype(np.float32)
            action_array = np.stack(action_targets).astype(np.float32)
            episode_name = f"episode_{episode_index:06d}.h5"
            metadata = {
                "seed": episode_seed,
                "steps": len(frame_array),
                "dynamic_spheres": int(plan_row["dynamic_spheres"]),
                "static_cubes": int(plan_row["static_cubes"]),
                "topology": str(plan_row["topology"]),
                "template_id": template_id,
                "success": True,
                "obstacle_collision": False,
                "table_collision": False,
                "clearance_below_8cm_frames": 0,
                "clearance_below_hard_limit_frames": 0,
                "hard_clearance_threshold_m": 0.0,
                "scenario": QUICK_SCENARIO,
                "oracle_profile": "quick",
                "continuous_handle_contact": True,
                "minimum_true_clearance": _minimum_true_clearance(env.task),
                "minimum_table_clearance": float(getattr(env.task, "_v2_minimum_table_clearance", float("inf"))),
                "minimum_table_clearance_entity": str(getattr(env.task, "_v2_minimum_table_clearance_entity", "")),
                "maximum_replay_state_error": maximum_state_error,
                "oracle_failure_reason": oracle_failure_reason,
                "diagnostic_replay_reason": diagnostic_replay_reason,
                "oracle_planned_steps": int(plan_row.get("steps", len(raw_actions))),
                "diagnostic_replay_steps": len(action_targets),
                "diagnostic_only": False,
            }
            _write_episode(episodes_dir / episode_name, frame_array, proprio_array, action_array, metadata)
            # Video is deliberately derived from the HDF5 payload, not encoded in the control loop.
            with h5py.File(episodes_dir / episode_name, "r") as handle:
                stored_frames = np.asarray(handle["rgb"], dtype=np.uint8)
            _write_video(
                videos_dir / f"episode_{episode_index:06d}_overview.mp4",
                np.stack(eval_frames[1]),
            )
            row = {
                "episode": episode_index,
                "seed": episode_seed,
                "file": episode_name,
                "teacher": TEACHER_NAME,
                "topology": str(plan_row["topology"]),
                "template_id": template_id,
                "dynamic_spheres": int(plan_row["dynamic_spheres"]),
                "static_cubes": int(plan_row["static_cubes"]),
                "steps": len(frame_array),
                "minimum_true_clearance": metadata["minimum_true_clearance"],
                "minimum_table_clearance": metadata["minimum_table_clearance"],
                "minimum_table_clearance_entity": metadata["minimum_table_clearance_entity"],
                "maximum_replay_state_error": maximum_state_error,
                "dynamic_arm_avoidance_steps": int(plan_row.get("dynamic_arm_avoidance_steps", 0)),
                "dynamic_emergency_base_steps": int(plan_row.get("dynamic_emergency_base_steps", 0)),
                "cube_bypass_steps": int(plan_row.get("cube_bypass_steps", 0)),
                "timing_seconds": {
                    "record_and_hdf5_video": time.perf_counter() - attempt_started,
                },
            }
            accepted.append(row)
            completed.add(episode_index)
            accepted.sort(key=lambda value: int(value["episode"]))
            manifest.update({
                "episodes": accepted,
                "accepted_episodes": len(accepted),
                "rejected": rejected,
                "rejected_attempts": len(rejected),
                "elapsed_seconds": elapsed_before_run + time.perf_counter() - started,
            })
            _atomic_json(manifest_path, manifest)
            _write_progress(
                f"Record {len(accepted)}/{args.episodes} complete | "
                f"seed {episode_seed} | steps {len(frame_array)}",
                newline=True,
            )

        manifest.update({
            "episodes": accepted,
            "accepted_episodes": len(accepted),
            "rejected": rejected,
            "rejected_attempts": len(rejected),
            "status": "complete" if len(accepted) == args.episodes else "incomplete",
            "lifecycle_stage": "complete" if len(accepted) == args.episodes else "incomplete",
        })
        _atomic_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["interrupted_stage"] = str(manifest.get("lifecycle_stage", "unknown"))
        manifest["interrupted_exception_type"] = type(exc).__name__
        manifest["interrupted_exception"] = str(exc) or repr(exc)
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        if env is not None:
            env.close()
    if len(accepted) != args.episodes:
        print(f"Record incomplete: {len(accepted)}/{args.episodes}", flush=True)
        return 4
    return 0


def _run_substage(args: argparse.Namespace, stage: str, plans_dir: Path, output: Path) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage", stage,
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--max-steps", str(args.max_steps),
        "--max-attempts", str(args.max_attempts),
        "--max-trials-per-template", str(getattr(args, "max_trials_per_template", 0)),
        "--no-success-stop", str(getattr(args, "no_success_stop", 30)),
        "--worker-threads", str(args.worker_threads),
        "--physics-device", args.physics_device,
        "--scenario", getattr(args, "scenario", QUICK_SCENARIO),
        "--output", str(output),
        "--plans-dir", str(plans_dir),
    ]
    if getattr(args, "template_id", None):
        command.extend(["--template-id", args.template_id])
    if getattr(args, "template_ids", None):
        command.extend(["--template-ids", args.template_ids])
    if args.require_topology is not None:
        command.extend(["--require-topology", args.require_topology])
    if args.require_arm_avoidance:
        command.append("--require-arm-avoidance")
    if args.require_cube_bypass:
        command.append("--require-cube-bypass")
    if args.prescreen_video:
        command.append("--prescreen-video")
    if args.resume:
        command.append("--resume")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        return result.returncode
    if stage == "prescreen":
        manifest_path = plans_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 3
        # A clean child exit is not enough: a reset interruption can leave a
        # stale or incomplete manifest while the parent would otherwise start
        # Record and report a misleading "0/N plans" error.
        if (
            str(manifest.get("status", "")) != "complete"
            or int(manifest.get("accepted_episodes", 0)) < int(args.episodes)
        ):
            return 3
    return 0


def main() -> int:
    args = parse_args()
    if args.max_steps <= 0 or args.max_attempts < 0 or args.max_trials_per_template < 0 or args.no_success_stop < 0:
        raise ValueError("--max-steps must be positive; attempt caps must be non-negative")
    output = args.output.resolve()
    plans_dir = (args.plans_dir or (output / "plans")).resolve()
    if args.stage == "prescreen":
        return _prescreen(args, plans_dir)
    if args.stage == "record":
        return _record(args, plans_dir, output)
    prescreen_code = _run_substage(args, "prescreen", plans_dir, output)
    if prescreen_code != 0:
        return prescreen_code
    return _run_substage(args, "record", plans_dir, output)


if __name__ == "__main__":
    raise SystemExit(main())

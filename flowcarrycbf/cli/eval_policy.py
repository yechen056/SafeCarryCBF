#!/usr/bin/env python3
"""Paired RGB Flow and predictive whole-body CBF-QP evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

from flowcarrycbf.policies.flowcarry_cbf.eval_scenarios import (
    EVAL_BASE_TEMPLATE_IDS,
    EVAL_HARD_TEMPLATE_IDS,
    EVAL_STRESS_TEMPLATE_IDS,
    EVAL_TEMPLATE_IDS,
    is_hard_template,
    is_stress_template,
    sample_eval_cases,
)
from flowcarrycbf.policies.flowcarry_cbf.policy import RGBFlowCarryPolicy
from flowcarrycbf.policies.flowcarry_cbf.schema import controlled_positions
from flowcarrycbf.policies.flowcarry_cbf.control.schema import UNCERTAINTY_INDEX, VALID_INDEX


ALIGNED_SCENARIO = "phase3_quick"
EVAL_METHODS = ("flow", "flow_cbf")
ROTATION_TOLERANCE = math.radians(8.0)
ROTATION_TRANSLATION_TOLERANCE = 0.005
ROTATION_ARM_TOLERANCE = 0.005
GOAL_POSITION_TOLERANCE = 0.12
GOAL_YAW_TOLERANCE = math.radians(5.0)
GOAL_XY = np.asarray([-3.0, 0.0], dtype=np.float32)
GOAL_YAW = math.pi


@dataclass(frozen=True)
class EvalCase:
    case_id: int
    seed: int
    template_id: str
    topology: str
    expected_schedule_checksum: str = ""
    adaptive_template: bool = False


def _checksum(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def parse_case_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not indices:
        raise ValueError("--case-indices must not be empty")
    if len(indices) != len(set(indices)) or min(indices) < 0:
        raise ValueError("--case-indices must contain unique non-negative values")
    return indices


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("--methods must contain unique method names")
    unknown = sorted(set(methods) - set(EVAL_METHODS))
    if unknown:
        raise ValueError(f"unsupported evaluation methods: {unknown}")
    return tuple(method for method in EVAL_METHODS if method in methods)


def update_rotation_protocol(protocol: dict[str, Any], state: np.ndarray) -> None:
    controlled = controlled_positions(state)
    if protocol["completed"]:
        return
    protocol["max_translation_m"] = max(
        protocol["max_translation_m"],
        float(np.linalg.norm(controlled[:2] - protocol["initial"][:2])),
    )
    protocol["max_arm_deviation"] = max(
        protocol["max_arm_deviation"],
        float(np.max(np.abs(controlled[3:] - protocol["initial"][3:]))),
    )
    yaw_error = abs((math.pi - float(controlled[2]) + math.pi) % (2.0 * math.pi) - math.pi)
    if yaw_error <= ROTATION_TOLERANCE:
        protocol["completed"] = True
        protocol["completed_step"] = int(protocol["steps"])


def rotation_protocol_result(protocol: dict[str, Any]) -> tuple[bool, str]:
    if not protocol["completed"]:
        return False, "initial_rotation_incomplete"
    if protocol["max_translation_m"] > ROTATION_TRANSLATION_TOLERANCE:
        return False, "translation_before_initial_rotation"
    if protocol["max_arm_deviation"] > ROTATION_ARM_TOLERANCE:
        return False, "arm_motion_before_initial_rotation"
    return True, ""


def aligned_goal_reached(state: np.ndarray) -> bool:
    controlled = controlled_positions(state)
    distance = float(np.linalg.norm(controlled[:2] - GOAL_XY))
    yaw_error = abs(
        (GOAL_YAW - float(controlled[2]) + math.pi) % (2.0 * math.pi) - math.pi
    )
    return bool(distance <= GOAL_POSITION_TOLERANCE and yaw_error <= GOAL_YAW_TOLERANCE)


def aligned_method_summary(
    rows: list[dict[str, Any]], minimum_success_rate: float = 1.0,
) -> dict[str, Any]:
    templates = sorted({str(row["template_id"]) for row in rows})
    groups: dict[str, Any] = {}
    for template in templates:
        selected = [row for row in rows if row["template_id"] == template]
        successes = sum(bool(row["success"]) for row in selected)
        required = int(math.ceil(len(selected) * float(minimum_success_rate) - 1.0e-9))
        groups[template] = {
            "episodes": len(selected),
            "successes": successes,
            "required_successes": required,
            "passed": successes >= required,
            "cbf_intervention_steps": sum(
                int(row.get("cbf_intervention_steps", 0)) for row in selected
            ),
        }
    return {
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "groups": groups,
        "collisions": sum(bool(row["collision"]) for row in rows),
        "contact_loss_episodes": sum(
            not bool(row["continuous_handle_contact"]) for row in rows
        ),
        "clearance_below_8cm_frames": sum(
            int(row["clearance_below_8cm_frames"]) for row in rows
        ),
        "passed": bool(
            rows
            and all(group["passed"] for group in groups.values())
            and not any(bool(row["collision"]) for row in rows)
            and all(bool(row["continuous_handle_contact"]) for row in rows)
            and all(bool(row["schedule_checksum_match"]) for row in rows)
        ),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _finite(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _label_video_paths(paths: list[Path], *, success: bool) -> None:
    """Append the final episode outcome to each completed recording."""
    outcome = "success" if success else "fail"
    for path in paths:
        if path.exists():
            path.replace(path.with_name(f"{path.stem}_{outcome}{path.suffix}"))


def _failure_reason(info: dict[str, Any], exhausted: bool) -> str:
    if bool(info.get("table_collision", False)):
        return (
            f"table_collision:{info.get('table_collision_name', 'table')}:"
            f"{info.get('table_collision_entity', 'robot')}"
        )
    if bool(info.get("obstacle_collision", False)):
        return f"obstacle_collision:{info.get('obstacle_collision_entity', 'robot')}"
    if not bool(info.get("continuous_handle_contact", True)):
        return "handle_contact_lost"
    reason = str(info.get("failure_reason", ""))
    if "drop" in reason.lower():
        return reason
    if exhausted:
        return "timeout"
    return reason or "task_failed"


def _step_diagnostic(
    step: int,
    metrics: dict[str, Any],
    observation: dict[str, Any],
    info: dict[str, Any],
) -> dict[str, Any]:
    obstacles = np.asarray(observation["obstacles"], dtype=np.float32)
    valid = obstacles[:, VALID_INDEX] > 0.5
    estimates = [
        {
            "position": obstacle[:3].astype(float).tolist(),
            "velocity": obstacle[3:6].astype(float).tolist(),
            "acceleration": obstacle[6:9].astype(float).tolist(),
            "half_extents": obstacle[9:12].astype(float).tolist(),
            "type": float(obstacle[12]),
            "uncertainty": float(obstacle[UNCERTAINTY_INDEX]),
        }
        for obstacle in obstacles[valid]
    ]
    return {
        "step": int(step),
        "action_source": str(metrics.get("action_source", "")),
        "cbf_status": str(metrics.get("cbf_status", "not_run")),
        "cbf_risk": float(metrics.get("cbf_risk", 0.0)),
        "cbf_intervention_rms": float(metrics.get("cbf_intervention", 0.0)),
        "hard_minimum_clearance": _finite(
            metrics.get("hard_minimum_clearance", float("nan"))
        ),
        "soft_minimum_clearance": _finite(
            metrics.get("soft_minimum_clearance", float("nan"))
        ),
        "threatened_obstacle": str(metrics.get("threatened_obstacle", "none")),
        "threatened_robot_part": str(metrics.get("threatened_robot_part", "none")),
        "threat_mode": str(metrics.get("threat_mode", "none")),
        "approach_speed": float(metrics.get("approach_speed", 0.0)),
        "time_to_collision": _finite(
            metrics.get("time_to_collision", float("nan"))
        ),
        "avoidance_side": float(metrics.get("avoidance_side", 0.0)),
        "base_lateral_escape_m": float(
            metrics.get("base_lateral_escape_m", 0.0)
        ),
        "arm_task_escape_m": float(metrics.get("arm_task_escape_m", 0.0)),
        "first_risk_step": int(metrics.get("first_risk_step", -1)),
        "first_intervention_step": int(metrics.get("first_intervention_step", -1)),
        "predicted_clearance_margin": _finite(
            metrics.get("predicted_clearance_margin", float("nan"))
        ),
        "escape_mode": str(metrics.get("escape_mode", "none")),
        "certificate_failure_reason": str(
            metrics.get("certificate_failure_reason", "")
        ),
        "recovery_type": str(metrics.get("recovery_type", "none")),
        "cbf_base_correction_rms": float(
            metrics.get("cbf_base_correction_rms", 0.0)
        ),
        "cbf_arm_correction_rms": float(
            metrics.get("cbf_arm_correction_rms", 0.0)
        ),
        "grasp_consistent": bool(metrics.get("grasp_consistent", 1)),
        "arm_escape_within_limit": bool(
            metrics.get("arm_escape_within_limit", 1)
        ),
        "joint_limits_satisfied": bool(metrics.get("joint_limits_satisfied", 1)),
        "self_collision_free": bool(metrics.get("self_collision_free", 1)),
        "estimated_obstacle_count": int(np.count_nonzero(valid)),
        "maximum_obstacle_uncertainty": (
            float(np.max(obstacles[valid, UNCERTAINTY_INDEX]))
            if np.any(valid) else None
        ),
        "estimated_obstacles": estimates,
        "true_minimum_clearance": _finite(
            info.get("minimum_true_clearance", float("nan"))
        ),
        "true_obstacle_collision": bool(info.get("obstacle_collision", False)),
        "base": controlled_positions(observation["state"])[:3].astype(float).tolist(),
    }


def _diagnostic_classification(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    risk_steps = [row["step"] for row in diagnostics if row["cbf_risk"] > 0.0]
    intervention_steps = [
        row["step"] for row in diagnostics
        if row["cbf_intervention_rms"] > 1.0e-6
    ]
    collision_steps = [
        row["step"] for row in diagnostics if row["true_obstacle_collision"]
    ]
    first_risk = min(risk_steps, default=None)
    first_intervention = min(intervention_steps, default=None)
    collision_step = min(collision_steps, default=None)
    lead = (
        collision_step - first_risk
        if collision_step is not None and first_risk is not None else None
    )
    perception_late = bool(
        collision_step is not None and (first_risk is None or lead < 5)
    )
    prediction_underestimate = False
    control_insufficient = False
    if collision_step is not None and first_risk is not None:
        prior = [row for row in diagnostics if first_risk <= row["step"] <= collision_step]
        predicted = [
            row["soft_minimum_clearance"] for row in prior
            if row["soft_minimum_clearance"] is not None
        ]
        prediction_underestimate = bool(predicted and min(predicted) >= 0.0)
        control_insufficient = bool(
            lead >= 5 and any(row["cbf_intervention_rms"] > 1.0e-6 for row in prior)
        )
    labels = [
        label for label, active in (
            ("perception_late", perception_late),
            ("prediction_underestimate", prediction_underestimate),
            ("control_insufficient", control_insufficient),
        ) if active
    ]
    return {
        "first_risk_step": first_risk,
        "first_intervention_step": first_intervention,
        "true_collision_step": collision_step,
        "risk_lead_steps": lead,
        "diagnostic_labels": labels,
    }


def _run_case(
    env,
    policy: RGBFlowCarryPolicy,
    case: EvalCase,
    method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    options: dict[str, Any] = {
        "obstacle_scenario": ALIGNED_SCENARIO,
        "randomize": True,
    }
    if case.adaptive_template:
        options["eval_obstacle_template_id"] = case.template_id
    else:
        options["obstacle_template_id"] = case.template_id
    observation, info = env.reset(seed=case.seed, options=options)
    actual_template = (
        getattr(env.task, "_v2_eval_template_id", None)
        if case.adaptive_template else getattr(env.task, "_v2_template_id", None)
    )
    if str(actual_template or "") != case.template_id:
        raise RuntimeError(
            f"template reset mismatch for case {case.case_id}: {case.template_id}"
        )
    schedule = env.get_oracle_obstacle_schedule(args.max_steps)
    schedule_checksum = _checksum(schedule)
    checksum_match = bool(
        not case.expected_schedule_checksum
        or schedule_checksum == case.expected_schedule_checksum
    )
    policy.safety_mode = {
        "flow": "flow",
        "flow_cbf": "cbf",
    }[method]
    policy.reset(case.seed, template_id=case.template_id)
    initial = controlled_positions(observation["state"])
    protocol: dict[str, Any] = {
        "initial": initial,
        "completed": False,
        "completed_step": None,
        "max_translation_m": 0.0,
        "max_arm_deviation": 0.0,
        "steps": 0,
    }
    update_rotation_protocol(protocol, observation["state"])
    diagnostics: list[dict[str, Any]] = []
    planning_latencies: list[float] = []
    filter_latencies: list[float] = []
    intervention_steps = 0
    safety_hold_steps = 0
    action_correction_rms_values: list[float] = []
    writers = []
    video_paths: list[Path] = []
    stem = f"case_{case.case_id:03d}_{case.template_id}"
    diagnostic_path = args.output_dir / method / "diagnostics" / f"{stem}.json"
    terminated = truncated = False
    goal_reached = False
    goal_step: int | None = None
    progress = tqdm(
        total=args.max_steps,
        desc=f"{method} case={case.case_id} seed={case.seed} {case.template_id}",
        file=sys.stdout,
        leave=False,
        dynamic_ncols=True,
        unit="step",
        mininterval=1.0,
    )
    try:
        if not checksum_match:
            return {
                "method": method,
                "case_id": case.case_id,
                "seed": case.seed,
                "template_id": case.template_id,
                "topology": case.topology,
                "stress": is_stress_template(case.template_id),
                "hard": is_hard_template(case.template_id),
                "cbf_strategy": policy.safety_strategy,
                "schedule_checksum": schedule_checksum,
                "schedule_checksum_match": False,
                "success": False,
                "failure_reason": "schedule_mismatch",
                "steps": 0,
                "collision": False,
                "obstacle_collision": False,
                "table_collision": False,
                "continuous_handle_contact": bool(info.get("continuous_handle_contact", True)),
                "payload_dropped": False,
                "minimum_true_clearance": None,
                "clearance_below_8cm_frames": 0,
                "cbf_intervention_steps": 0,
                "safety_hold_steps": 0,
                "initial_rotation_protocol_satisfied": False,
                "elapsed_seconds": time.perf_counter() - started,
            }
        if args.record_video:
            video_dir = args.output_dir / method / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            for camera_name in ("first_person", "overview"):
                video_path = video_dir / f"{stem}_{camera_name}.mp4"
                video_paths.append(video_path)
                writers.append(imageio.get_writer(
                    video_path,
                    fps=10,
                    macro_block_size=1,
                ))
        for step in range(args.max_steps):
            if writers:
                frames = env.task.get_eval_video_rgb_observations()
                if len(frames) != len(writers):
                    raise RuntimeError("evaluation recording cameras are unavailable")
                for writer, frame in zip(writers, frames):
                    writer.append_data(frame)
            action = policy.act(observation, env.task.get_rgbd_observations())
            env.set_safety_metrics(**policy.metrics)
            intervention_steps += int(
                float(policy.metrics.get("cbf_intervention", 0.0)) > 1.0e-6
            )
            action_correction_rms_values.append(
                float(policy.metrics.get("action_correction_rms", 0.0))
            )
            source = str(policy.metrics.get("action_source", ""))
            safety_hold_steps += int("hold" in source)
            planning_latencies.append(
                float(policy.metrics.get("planning_latency_ms", 0.0))
            )
            filter_latencies.append(
                float(policy.metrics.get("filter_latency_ms", 0.0))
            )
            observation, _, terminated, truncated, info = env.step(action)
            protocol["steps"] = step + 1
            update_rotation_protocol(protocol, observation["state"])
            physical_failure = bool(
                info.get("table_collision", False)
                or info.get("obstacle_collision", False)
                or not info.get("continuous_handle_contact", True)
                or "drop" in str(info.get("failure_reason", "")).lower()
            )
            goal_reached = bool(
                aligned_goal_reached(observation["state"])
                and not physical_failure
                and not truncated
            )
            diagnostics.append(
                _step_diagnostic(step + 1, policy.metrics, observation, info)
            )
            progress.update(1)
            progress.set_postfix_str(
                f"src={source} risk={float(policy.metrics.get('cbf_risk', 0.0)):.2f} "
                f"qp={policy.metrics.get('cbf_status', 'not_run')}",
                refresh=False,
            )
            if goal_reached:
                goal_step = step + 1
            if physical_failure or terminated or truncated or goal_reached:
                break
    finally:
        progress.close()
        for writer in writers:
            writer.close()
    _atomic_json(diagnostic_path, diagnostics)
    collision = bool(
        info.get("obstacle_collision", False) or info.get("table_collision", False)
    )
    contact = bool(info.get("continuous_handle_contact", False))
    payload_dropped = "drop" in str(info.get("failure_reason", "")).lower()
    exhausted = bool(truncated or len(diagnostics) >= args.max_steps)
    # Safely entering the aligned geometric goal is the single evaluation
    # success criterion. ``goal_reached`` already excludes physical failures
    # and truncation.
    success = bool(goal_reached)
    _label_video_paths(video_paths, success=success)
    protocol_ok, protocol_reason = rotation_protocol_result(protocol)
    hard_clearances = [
        float(row["hard_minimum_clearance"])
        for row in diagnostics if row["hard_minimum_clearance"] is not None
    ]
    soft_clearances = [
        float(row["soft_minimum_clearance"])
        for row in diagnostics if row["soft_minimum_clearance"] is not None
    ]
    risks = [float(row["cbf_risk"]) for row in diagnostics]
    base_escapes = [float(row["base_lateral_escape_m"]) for row in diagnostics]
    arm_escapes = [float(row["arm_task_escape_m"]) for row in diagnostics]
    base_corrections = [
        float(row["cbf_base_correction_rms"]) for row in diagnostics
    ]
    arm_corrections = [
        float(row["cbf_arm_correction_rms"]) for row in diagnostics
    ]
    sources = Counter(str(row["action_source"]) for row in diagnostics)
    threat_modes = Counter(str(row["threat_mode"]) for row in diagnostics)
    classification = _diagnostic_classification(diagnostics)
    return {
        "method": method,
        "case_id": case.case_id,
        "seed": case.seed,
        "policy_seed": case.seed,
        "template_id": case.template_id,
        "topology": case.topology,
        "stress": is_stress_template(case.template_id),
        "hard": is_hard_template(case.template_id),
        "cbf_strategy": policy.safety_strategy,
        "schedule_checksum": schedule_checksum,
        "schedule_checksum_match": checksum_match,
        "success": success,
        "failure_reason": "" if success else _failure_reason(info, exhausted),
        "goal_region_reached": goal_reached,
        "goal_region_reached_step": goal_step,
        "steps": len(diagnostics),
        "collision": collision,
        "obstacle_collision": bool(info.get("obstacle_collision", False)),
        "table_collision": bool(info.get("table_collision", False)),
        "continuous_handle_contact": contact,
        "payload_dropped": payload_dropped,
        "minimum_true_clearance": _finite(
            info.get("minimum_true_clearance", float("nan"))
        ),
        "clearance_below_8cm_frames": int(info.get("clearance_below_8cm_frames", 0)),
        "final_goal_distance": float(
            np.linalg.norm(controlled_positions(observation["state"])[:2] - GOAL_XY)
        ),
        "cbf_intervention_steps": intervention_steps,
        "cbf_intervention_ratio": (
            float(intervention_steps) / len(diagnostics) if diagnostics else 0.0
        ),
        "mean_action_correction_rms": (
            float(np.mean(action_correction_rms_values))
            if action_correction_rms_values else 0.0
        ),
        "safety_hold_steps": safety_hold_steps,
        "minimum_hard_predicted_clearance": min(hard_clearances, default=None),
        "minimum_soft_predicted_clearance": min(soft_clearances, default=None),
        "maximum_cbf_risk": max(risks, default=0.0),
        "maximum_base_lateral_escape_m": max(base_escapes, default=0.0),
        "maximum_arm_task_escape_m": max(arm_escapes, default=0.0),
        "maximum_cbf_base_correction_rms": max(base_corrections, default=0.0),
        "maximum_cbf_arm_correction_rms": max(arm_corrections, default=0.0),
        "mean_cbf_base_correction_rms": (
            float(np.mean(base_corrections)) if base_corrections else 0.0
        ),
        "mean_cbf_arm_correction_rms": (
            float(np.mean(arm_corrections)) if arm_corrections else 0.0
        ),
        "action_source_counts": dict(sources),
        "threat_mode_counts": dict(threat_modes),
        **classification,
        "initial_rotation_completed": bool(protocol["completed"]),
        "initial_rotation_completed_step": protocol["completed_step"],
        "initial_rotation_max_translation_m": float(protocol["max_translation_m"]),
        "initial_rotation_max_arm_deviation": float(protocol["max_arm_deviation"]),
        "initial_rotation_protocol_satisfied": protocol_ok,
        "initial_rotation_protocol_violation_reason": protocol_reason,
        "mean_planning_latency_ms": (
            float(np.mean(planning_latencies)) if planning_latencies else 0.0
        ),
        "mean_filter_latency_ms": (
            float(np.mean(filter_latencies)) if filter_latencies else 0.0
        ),
        "p50_filter_latency_ms": (
            float(np.percentile(filter_latencies, 50)) if filter_latencies else 0.0
        ),
        "p95_filter_latency_ms": (
            float(np.percentile(filter_latencies, 95)) if filter_latencies else 0.0
        ),
        "diagnostics": str(diagnostic_path),
        "elapsed_seconds": time.perf_counter() - started,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def paired_summary(
    rows: list[dict[str, Any]],
    category: str,
    methods: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    selected_methods = methods or tuple(
        method for method in EVAL_METHODS
        if any(row["method"] == method for row in rows)
    )
    if not selected_methods:
        selected_methods = ("flow", "flow_cbf")
    method_rows = {
        method: [row for row in rows if row["method"] == method]
        for method in selected_methods
    }
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(int(row["case_id"]), {})[str(row["method"])] = row
    completed_pairs = [
        pair for pair in by_pair.values()
        if set(pair) == set(selected_methods)
    ]
    safety_methods = [method for method in selected_methods if method != "flow"]
    safety_method = safety_methods[0] if len(safety_methods) == 1 else None
    paired_checksums_match = all(
        len({pair[method]["schedule_checksum"] for method in selected_methods}) == 1
        for pair in completed_pairs
    )
    rescues = 0 if safety_method is None else sum(
        not pair["flow"]["success"] and pair[safety_method]["success"]
        for pair in completed_pairs if "flow" in pair
    )
    regressions = 0 if safety_method is None else sum(
        pair["flow"]["success"] and not pair[safety_method]["success"]
        for pair in completed_pairs if "flow" in pair
    )
    summary: dict[str, Any] = {
        "category": category,
        "rows": len(rows),
        "completed_pairs": len(completed_pairs),
        "paired_schedule_checksums_match": paired_checksums_match,
        "rescues": rescues,
        "regressions": regressions,
        "methods": {},
    }
    for method, selected in method_rows.items():
        summary["methods"][method] = {
            "episodes": len(selected),
            "successes": sum(bool(row["success"]) for row in selected),
            "collisions": sum(bool(row["collision"]) for row in selected),
            "contact_losses": sum(
                not bool(row["continuous_handle_contact"]) for row in selected
            ),
            "payload_drops": sum(bool(row["payload_dropped"]) for row in selected),
            "mean_minimum_true_clearance": (
                float(np.mean([
                    float(row["minimum_true_clearance"])
                    for row in selected
                    if row.get("minimum_true_clearance") is not None
                ])) if any(
                    row.get("minimum_true_clearance") is not None
                    for row in selected
                ) else None
            ),
            "mean_cbf_intervention_ratio": (
                float(np.mean([
                    float(row.get("cbf_intervention_ratio", 0.0))
                    for row in selected
                ])) if selected else 0.0
            ),
            "mean_action_correction_rms": (
                float(np.mean([
                    float(row.get("mean_action_correction_rms", 0.0))
                    for row in selected
                ])) if selected else 0.0
            ),
            "intervention_steps": sum(
                int(row["cbf_intervention_steps"]) for row in selected
            ),
        }
    flow_rows = method_rows.get("flow", [])
    flow_stress = [row for row in flow_rows if row["stress"]]
    flow_successes = sum(bool(row["success"]) for row in flow_stress)
    summary["flow_stress"] = {
        "episodes": len(flow_stress),
        "successes": flow_successes,
        "failures": len(flow_stress) - flow_successes,
    }
    flow = summary["methods"].get("flow")
    cbf = summary["methods"].get(safety_method) if safety_method else None
    physical_no_worse = bool(
        flow is not None and cbf is not None
        and cbf["collisions"] <= flow["collisions"]
        and cbf["contact_losses"] <= flow["contact_losses"]
        and cbf["payload_drops"] <= flow["payload_drops"]
    )
    all_pairs_complete = bool(
        len(completed_pairs) == len(by_pair) and paired_checksums_match
    )
    # Evaluation is diagnostic by design: physical failures are reported,
    # but do not stop or invalidate a completed paired experiment.
    passed = all_pairs_complete
    summary["physical_safety_no_worse"] = physical_no_worse
    summary["passed"] = passed
    return summary


def _make_env(*, record_video: bool, worker_threads: int, adaptive: bool):
    from flowcarrycbf.envs.tiago_flow_carry_env import FlowCarryCBFEnv

    return FlowCarryCBFEnv(
        headless=True,
        render=True,
        randomize=True,
        obstacle_scenario=ALIGNED_SCENARIO,
        lightweight_info=True,
        phase3_visuals=True,
        enable_eval_video_cameras=record_video,
        adaptive_eval=adaptive,
        observation_mode="perception",
        worker_threads=int(worker_threads),
        physics_device="cpu",
    )


def _run_cases(
    args: argparse.Namespace,
    cases: list[EvalCase],
    config: dict[str, Any],
) -> int:
    methods = parse_methods(args.methods)
    args.output_dir = args.output_dir.resolve()
    result_path = args.output_dir / "results.json"
    if result_path.exists():
        raise FileExistsError(f"evaluation output already exists: {result_path}")
    _atomic_json(args.output_dir / "config.json", config)
    env = _make_env(
        record_video=args.record_video,
        worker_threads=int(config.get("worker_threads", 8)),
        adaptive=bool(config.get("adaptive", False)),
    )
    policy = RGBFlowCarryPolicy(args.model, safety_mode="flow")
    rows: list[dict[str, Any]] = []
    print("Paired RGB Flow Evaluation", flush=True)
    print(f"Model      : {args.model.name}", flush=True)
    print(f"Category   : {config['category']}", flush=True)
    print(f"Cases      : {len(cases)}", flush=True)
    print(f"Methods    : {', '.join(methods)}", flush=True)
    print("-" * 60, flush=True)
    try:
        for case in cases:
            for method in methods:
                row = _run_case(env, policy, case, method, args)
                rows.append(row)
                summary = paired_summary(
                    rows,
                    str(config["category"]),
                    methods,
                )
                _atomic_json(result_path, {
                    "model": str(args.model.resolve()),
                    "config": config,
                    "summary": summary,
                    "episodes": rows,
                })
                print(
                    f"{method} case={case.case_id} seed={case.seed} "
                    f"template={case.template_id} "
                    f"status={'SUCCESS' if row['success'] else 'FAIL'} "
                    f"steps={row['steps']} reason={row['failure_reason'] or 'none'} "
                    f"interventions={row['cbf_intervention_steps']} "
                    f"holds={row['safety_hold_steps']}",
                    flush=True,
                )
    finally:
        env.close()
    summary = paired_summary(
        rows,
        str(config["category"]),
        methods,
    )
    result = {
        "model": str(args.model.resolve()),
        "config": config,
        "summary": summary,
        "episodes": rows,
    }
    _atomic_json(result_path, result)
    print("-" * 60, flush=True)
    print(f"Result  : {'PASSED' if summary['passed'] else 'FAILED'}", flush=True)
    print("Primary metrics:", flush=True)
    for method, metrics in summary["methods"].items():
        clearance = metrics["mean_minimum_true_clearance"]
        print(
            f"  {method}: success_rate="
            f"{metrics['successes'] / metrics['episodes'] if metrics['episodes'] else 0.0:.3f} "
            f"mean_min_clearance_m={clearance if clearance is not None else float('nan'):.4f} "
            f"cbf_intervention_ratio={metrics['mean_cbf_intervention_ratio']:.3f} "
            f"mean_action_correction_rms={metrics['mean_action_correction_rms']:.4f}",
            flush=True,
        )
    print(f"Metrics : {result_path}", flush=True)
    return 0 if summary["passed"] else 2


def _parallel_worker(
    args: argparse.Namespace,
    cases: list[EvalCase],
    config: dict[str, Any],
) -> None:
    """Run one isolated Isaac Sim shard in a spawned child process."""

    raise SystemExit(_run_cases(args, cases, config))


def _run_parallel_cases(
    args: argparse.Namespace,
    cases: list[EvalCase],
    config: dict[str, Any],
) -> int:
    if args.output_dir.exists():
        raise FileExistsError(
            f"parallel evaluation output already exists: {args.output_dir}"
        )
    worker_count = min(int(args.workers), len(cases))
    chunks = [cases[index::worker_count] for index in range(worker_count)]
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    part_dirs: list[Path] = []
    for index, chunk in enumerate(chunks):
        part_dir = args.output_dir.parent / f"{args.output_dir.name}.part{index:02d}"
        if part_dir.exists():
            raise FileExistsError(f"parallel shard output already exists: {part_dir}")
        child_args = copy.copy(args)
        child_args.output_dir = part_dir
        child_config = dict(config)
        child_config.update({
            "parallel_worker": index,
            "parallel_workers": worker_count,
            "case_indices": [case.case_id for case in chunk],
            "templates": [case.template_id for case in chunk],
        })
        process = context.Process(
            target=_parallel_worker,
            args=(child_args, chunk, child_config),
            name=f"flowcarry-eval-{index:02d}",
        )
        process.start()
        processes.append(process)
        part_dirs.append(part_dir)
    for process in processes:
        process.join()
    failed = [
        (process.name, process.exitcode)
        for process in processes if process.exitcode != 0
    ]
    if failed:
        raise RuntimeError(f"parallel evaluation shard failed: {failed}")

    rows: list[dict[str, Any]] = []
    for part_dir in part_dirs:
        part_result = json.loads((part_dir / "results.json").read_text())
        rows.extend(part_result.get("episodes", []))
    rows.sort(key=lambda row: (int(row["case_id"]), str(row["method"])))
    methods = parse_methods(args.methods)
    summary = paired_summary(rows, str(config["category"]), methods)
    merged_config = dict(config)
    merged_config.update({
        "parallel_workers": worker_count,
        "case_indices": [case.case_id for case in cases],
        "templates": [case.template_id for case in cases],
    })
    result = {
        "model": str(args.model.resolve()),
        "config": merged_config,
        "summary": summary,
        "episodes": rows,
    }
    _atomic_json(args.output_dir / "config.json", merged_config)
    _atomic_json(args.output_dir / "results.json", result)
    print("Parallel evaluation complete", flush=True)
    print(f"Workers    : {worker_count}", flush=True)
    print(f"Cases      : {len(cases)}", flush=True)
    print(f"Episodes   : {len(rows)}", flush=True)
    print(f"Result     : {'PASSED' if summary['passed'] else 'FAILED'}", flush=True)
    for method, metrics in summary["methods"].items():
        clearance = metrics["mean_minimum_true_clearance"]
        print(
            f"  {method}: success_rate="
            f"{metrics['successes'] / metrics['episodes'] if metrics['episodes'] else 0.0:.3f} "
            f"mean_min_clearance_m={clearance if clearance is not None else float('nan'):.4f} "
            f"cbf_intervention_ratio={metrics['mean_cbf_intervention_ratio']:.3f} "
            f"mean_action_correction_rms={metrics['mean_action_correction_rms']:.4f}",
            flush=True,
        )
    print(f"Metrics    : {args.output_dir / 'results.json'}", flush=True)
    return 0 if summary["passed"] else 2


def run_adaptive(args: argparse.Namespace) -> int:
    sampled = sample_eval_cases(
        selection_seed=args.seed,
        runs=args.runs,
    )
    cases = [
        EvalCase(
            index,
            seed,
            template,
            "2ball_no_cube" if template.startswith("dual") else "1ball_cube",
            "",
            True,
        )
        for index, (seed, template) in enumerate(sampled)
    ]
    category_templates = {
        "all": set(EVAL_TEMPLATE_IDS),
        "easy": set(EVAL_BASE_TEMPLATE_IDS),
        "stress": set(EVAL_STRESS_TEMPLATE_IDS),
        "hard": set(EVAL_HARD_TEMPLATE_IDS),
    }
    cases = [
        case for case in cases
        if case.template_id in category_templates[args.category]
    ]
    selected_case_ids = (
        parse_case_indices(args.case_indices)
        if args.case_indices.strip() else ()
    )
    if selected_case_ids:
        requested = set(selected_case_ids)
        cases = [case for case in cases if case.case_id in requested]
        missing = requested - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"adaptive case indices out of range: {sorted(missing)}")
    config = {
        "category": args.category,
        "adaptive": True,
        "seed": args.seed,
        "runs_per_template": args.runs,
        "methods": list(parse_methods(args.methods)),
        "worker_threads": 8,
        "physics_device": "cpu",
        "max_steps": args.max_steps,
        "templates": [case.template_id for case in cases],
        "case_indices": [case.case_id for case in cases],
        "base_templates": list(EVAL_BASE_TEMPLATE_IDS),
        "hard_templates": list(EVAL_HARD_TEMPLATE_IDS),
        "available_templates": list(EVAL_TEMPLATE_IDS),
        "parallel_workers": int(args.workers),
    }
    if int(args.workers) > 1:
        return _run_parallel_cases(args, cases, config)
    return _run_cases(args, cases, config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--category",
        choices=("all", "easy", "stress", "hard"),
        default="all",
    )
    parser.add_argument(
        "--case-indices",
        default="",
        help="optional deterministic case IDs for targeted diagnostics",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--methods", default="flow,flow_cbf")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_final12"),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="number of deterministic cases generated for each selected template",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of isolated Isaac Sim processes (default: 1)",
    )
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"model does not exist: {args.model}")
    if args.max_steps <= 0 or args.runs <= 0 or args.workers <= 0:
        raise ValueError("runs, max steps, and workers must be positive")
    return run_adaptive(args)


if __name__ == "__main__":
    raise SystemExit(main())

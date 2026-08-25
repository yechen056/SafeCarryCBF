"""RGB-D instance geometry extraction and constant-acceleration tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.optimize import least_squares

from .schema import (
    ACCELERATION,
    CUBE_TYPE,
    HALF_EXTENTS,
    INSTANCE_COLORS,
    MAX_OBSTACLES,
    OBSTACLE_FEATURES,
    POSITION,
    SPHERE_TYPE,
    TYPE_INDEX,
    UNCERTAINTY_INDEX,
    VALID_INDEX,
    VELOCITY,
)


@dataclass(frozen=True)
class Detection:
    track_id: int
    position: np.ndarray
    half_extents: np.ndarray
    shape_type: float
    residual: float
    points: int


def backproject_optical(
    rows: np.ndarray,
    columns: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Backproject pixels into Isaac's +X-right, +Y-up, -Z-forward frame."""
    z = np.asarray(depth, dtype=np.float64)
    camera_x = (np.asarray(columns, dtype=np.float64) - float(cx)) * z / float(fx)
    camera_y = -(np.asarray(rows, dtype=np.float64) - float(cy)) * z / float(fy)
    return np.column_stack((camera_x, camera_y, -z, np.ones_like(z)))


def fit_sphere(points: np.ndarray, radius_bounds: tuple[float, float] = (0.08, 0.18)) -> tuple[np.ndarray, float, float]:
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(value) < 8:
        raise ValueError("at least eight points are required for sphere fitting")
    median = np.median(value, axis=0)
    distance = np.linalg.norm(value - median, axis=1)
    cutoff = np.quantile(distance, 0.90)
    value = value[distance <= cutoff]
    design = np.column_stack((2.0 * value, np.ones(len(value))))
    rhs = np.square(value).sum(axis=1)
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    algebraic_center = solution[:3]
    algebraic_radius = math.sqrt(max(float(solution[3] + algebraic_center @ algebraic_center), 0.0))
    maximum_radius = float(radius_bounds[1])
    center_lower = np.min(value, axis=0) - maximum_radius
    center_upper = np.max(value, axis=0) + maximum_radius
    initial = np.append(
        np.clip(algebraic_center, center_lower, center_upper),
        np.clip(algebraic_radius, *radius_bounds),
    )
    lower = np.append(center_lower, float(radius_bounds[0]))
    upper = np.append(center_upper, float(radius_bounds[1]))
    optimized = least_squares(
        lambda parameters: np.linalg.norm(value - parameters[:3], axis=1) - parameters[3],
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.005,
        max_nfev=50,
    )
    center = optimized.x[:3]
    radius = float(optimized.x[3])
    residual = float(np.sqrt(np.mean(np.square(np.linalg.norm(value - center, axis=1) - radius))))
    return center.astype(np.float32), radius, residual


def fit_aabb(points: np.ndarray, size_bounds: tuple[float, float] = (0.30, 0.55)) -> tuple[np.ndarray, np.ndarray, float]:
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(value) < 8:
        raise ValueError("at least eight points are required for AABB fitting")
    lower = np.quantile(value, 0.03, axis=0)
    upper = np.quantile(value, 0.97, axis=0)
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    half = np.clip(half, 0.5 * size_bounds[0], 0.5 * size_bounds[1])
    # Cubes are grounded; the visible upper and side surfaces provide a more
    # stable vertical center than the incomplete depth lower bound.
    side = float(np.median(2.0 * half))
    half[:] = 0.5 * np.clip(side, *size_bounds)
    center[2] = half[2]
    face_distance = np.min(np.abs(np.abs(value - center) - half), axis=1)
    return center.astype(np.float32), half.astype(np.float32), float(np.median(face_distance))


def classify_shape(
    points: np.ndarray,
    maximum_sphere_span: float = 0.30,
) -> float:
    """Classify elevated compact spheres versus larger grounded cubes."""
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(value) < 8:
        raise ValueError("at least eight points are required for shape classification")
    robust_span = np.quantile(value, 0.97, axis=0) - np.quantile(value, 0.03, axis=0)
    # Classification must not depend on a sphere center fitted from a single
    # partial surface. Elevated compact components are spheres; grounded and
    # substantially larger components are cubes in the V2 scene contract.
    if (
        float(np.median(value[:, 2])) < 0.55
        or float(np.max(robust_span)) > float(maximum_sphere_span)
    ):
        return CUBE_TYPE
    return SPHERE_TYPE


class ConstantAccelerationTrack:
    def __init__(self, track_id: int, shape_type: float) -> None:
        self.track_id = int(track_id)
        self.shape_type = float(shape_type)
        self.state = np.zeros(9, dtype=np.float64)
        self.covariance = np.eye(9, dtype=np.float64)
        self.half_extents = np.zeros(3, dtype=np.float64)
        self.initialized = False
        self.missing_time = float("inf")

    def predict(self, dt: float) -> None:
        dt = float(dt)
        transition = np.eye(9)
        for axis in range(3):
            transition[axis, axis + 3] = dt
            transition[axis, axis + 6] = 0.5 * dt * dt
            transition[axis + 3, axis + 6] = dt
        process = np.eye(9) * (2.0e-4 + 5.0e-3 * dt)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.missing_time += dt

    def update(self, detection: Detection) -> None:
        self.shape_type = float(detection.shape_type)
        measurement = np.asarray(detection.position, dtype=np.float64)
        if not self.initialized:
            self.state[:3] = measurement
            self.covariance = np.diag([2.5e-3] * 3 + [0.04] * 3 + [0.16] * 3)
            self.initialized = True
        else:
            observation = np.zeros((3, 9), dtype=np.float64)
            observation[:, :3] = np.eye(3)
            variance = max(2.5e-5, min(2.5e-3, detection.residual**2 + 1.0e-5))
            innovation = measurement - observation @ self.state
            if np.linalg.norm(innovation) > 0.50:
                return
            innovation_covariance = observation @ self.covariance @ observation.T + np.eye(3) * variance
            gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
            self.state += gain @ innovation
            self.covariance = (np.eye(9) - gain @ observation) @ self.covariance
        if self.shape_type == CUBE_TYPE:
            self.state[3:9] = 0.0
        else:
            velocity_norm = float(np.linalg.norm(self.state[3:6]))
            acceleration_norm = float(np.linalg.norm(self.state[6:9]))
            if velocity_norm > 0.35:
                self.state[3:6] *= 0.35 / velocity_norm
            if acceleration_norm > 0.50:
                self.state[6:9] *= 0.50 / acceleration_norm
        self.half_extents = 0.75 * self.half_extents + 0.25 * detection.half_extents if np.any(self.half_extents) else detection.half_extents.copy()
        self.missing_time = 0.0

    def feature(self) -> np.ndarray:
        result = np.zeros(OBSTACLE_FEATURES, dtype=np.float32)
        if not self.initialized or self.missing_time > 0.5:
            return result
        result[POSITION] = self.state[:3]
        result[VELOCITY] = self.state[3:6]
        result[ACCELERATION] = self.state[6:9]
        result[HALF_EXTENTS] = self.half_extents
        result[TYPE_INDEX] = self.shape_type
        result[UNCERTAINTY_INDEX] = float(np.sqrt(max(np.trace(self.covariance[:3, :3]), 0.0)))
        result[VALID_INDEX] = 1.0
        return result


class RGBDObstacleTracker:
    """Tracks color-coded V2 obstacles using RGB and metric depth only."""

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        *,
        minimum_pixels: int = 20,
        sphere_radius_bounds: tuple[float, float] = (0.08, 0.18),
        cube_size_bounds: tuple[float, float] = (0.30, 0.55),
        maximum_sphere_span: float = 0.30,
    ) -> None:
        self.fx, self.fy, self.cx, self.cy = map(float, (fx, fy, cx, cy))
        self.minimum_pixels = int(minimum_pixels)
        self.sphere_radius_bounds = tuple(map(float, sphere_radius_bounds))
        self.cube_size_bounds = tuple(map(float, cube_size_bounds))
        self.maximum_sphere_span = float(maximum_sphere_span)
        self.tracks: list[ConstantAccelerationTrack] = []
        self.shape_types = np.full(MAX_OBSTACLES, -1.0, dtype=np.float32)
        self._type_history: list[deque[int]] = []
        self.last_detections: list[Detection] = []
        self.last_view_detections: list[list[Detection]] = []
        self.last_ground_heights: list[float] = []
        self._last_base_pose = np.zeros(3, dtype=np.float64)
        self.reset()

    def reset(self, shape_types: np.ndarray | None = None) -> None:
        if shape_types is None:
            self.shape_types = np.full(MAX_OBSTACLES, -1.0, dtype=np.float32)
        else:
            value = np.asarray(shape_types, dtype=np.float32).reshape(MAX_OBSTACLES)
            self.shape_types = value.copy()
        self.tracks = [ConstantAccelerationTrack(i, self.shape_types[i]) for i in range(MAX_OBSTACLES)]
        self._type_history = [deque(maxlen=5) for _ in range(MAX_OBSTACLES)]
        self.last_detections = []
        self.last_view_detections = []
        self.last_ground_heights = []
        self._last_base_pose = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _base_transform(base_pose: np.ndarray) -> np.ndarray:
        x, y, yaw = np.asarray(base_pose, dtype=np.float64)[:3]
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        world_from_base = np.asarray([[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        return np.linalg.inv(world_from_base)

    def _masks(self, rgb: np.ndarray) -> list[np.ndarray]:
        image = np.asarray(rgb, dtype=np.uint8)[..., :3]
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        prototype = cv2.cvtColor((INSTANCE_COLORS[None] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)[0].astype(np.float32)
        masks = []
        for color in prototype:
            hue = np.abs(image_hsv[..., 0] - color[0])
            hue = np.minimum(hue, 180.0 - hue)
            mask = (hue <= 8.0) & (image_hsv[..., 1] >= 75.0) & (image_hsv[..., 2] >= 35.0)
            masks.append(mask)
        return masks

    @staticmethod
    def _inside_obstacle_workspace(position: np.ndarray) -> bool:
        """Reject similarly colored robot/background geometry before tracking."""

        x, y, z = np.asarray(position, dtype=np.float64).reshape(3)
        return -3.0 <= x <= -0.80 and abs(y) <= 1.20 and 0.05 <= z <= 1.35

    def _estimate_ground_height(self, depth_image: np.ndarray, camera_transform: np.ndarray) -> float:
        """Estimate the dominant horizontal ground height from RGB-D alone."""

        rows, columns = np.indices(depth_image.shape)
        rows = rows[::8, ::8].reshape(-1)
        columns = columns[::8, ::8].reshape(-1)
        values = depth_image[rows, columns].astype(np.float64)
        valid = np.isfinite(values) & (values > 0.05) & (values < 8.0)
        if int(np.count_nonzero(valid)) < 50:
            return 0.0
        points_camera = backproject_optical(
            rows[valid], columns[valid], values[valid],
            self.fx, self.fy, self.cx, self.cy,
        )
        world_z = (camera_transform @ points_camera.T).T[:, 2]
        possible_ground = world_z[(world_z > -0.20) & (world_z < 0.30)]
        if len(possible_ground) < 50:
            return 0.0
        histogram, edges = np.histogram(possible_ground, bins=np.arange(-0.20, 0.305, 0.005))
        peak = int(np.argmax(histogram))
        in_peak = possible_ground[(possible_ground >= edges[peak]) & (possible_ground < edges[peak + 1])]
        return float(np.median(in_peak)) if len(in_peak) else float(0.5 * (edges[peak] + edges[peak + 1]))

    def _height_clusters(self, points: np.ndarray) -> list[np.ndarray]:
        """Split image-connected color regions that lie on separate 3-D surfaces."""

        value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(value) < self.minimum_pixels:
            return []
        order = np.argsort(value[:, 2])
        sorted_points = value[order]
        boundaries = np.flatnonzero(np.diff(sorted_points[:, 2]) > 0.08) + 1
        return [
            cluster
            for cluster in np.split(sorted_points, boundaries)
            if len(cluster) >= self.minimum_pixels
        ]

    @staticmethod
    def _select_detection_candidate(candidates: list[Detection]) -> Detection:
        """Prefer the elevated body over smoother, larger color bleed on ground."""

        return max(
            candidates,
            key=lambda item: (
                float(item.position[2]),
                float(item.points) / max(float(item.residual), 1.0e-3),
            ),
        )

    def _vote_shape_type(self, detection: Detection) -> Detection:
        track_id = int(detection.track_id)
        inferred = int(float(detection.shape_type) >= 0.5)
        history = self._type_history[track_id]
        history.append(inferred)
        if len(history) >= 3:
            sphere_votes = sum(value == 0 for value in history)
            cube_votes = len(history) - sphere_votes
            winner = 0 if sphere_votes > cube_votes else 1
            if max(sphere_votes, cube_votes) >= 3:
                self.shape_types[track_id] = float(winner)
        resolved = float(self.shape_types[track_id])
        if resolved < 0.0:
            resolved = float(detection.shape_type)
        return Detection(
            detection.track_id,
            detection.position,
            detection.half_extents,
            resolved,
            detection.residual,
            detection.points,
        )

    def detect(self, rgb: np.ndarray, depth: np.ndarray, world_from_camera: np.ndarray, base_pose: np.ndarray) -> list[Detection]:
        depth_image = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_image.ndim != 2:
            return []
        camera_transform = np.asarray(world_from_camera, dtype=np.float64).reshape(4, 4)
        ground_height = self._estimate_ground_height(depth_image, camera_transform)
        self.last_ground_heights.append(ground_height)
        detections: list[Detection] = []
        for track_id, color_mask in enumerate(self._masks(rgb)):
            usable = color_mask & np.isfinite(depth_image) & (depth_image > 0.05) & (depth_image < 8.0)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(usable.astype(np.uint8), 8)
            candidates: list[Detection] = []
            component_ids = sorted(
                range(1, count),
                key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
                reverse=True,
            )
            for component_id in component_ids:
                if int(stats[component_id, cv2.CC_STAT_AREA]) < self.minimum_pixels:
                    continue
                rows, columns = np.nonzero(labels == component_id)
                stride = max(1, len(rows) // 2000)
                rows, columns = rows[::stride], columns[::stride]
                z = depth_image[rows, columns].astype(np.float64)
                # Isaac cameras look along optical -Z. Their local +X is image
                # right and +Y is image up, matching the USD extrinsic matrix.
                points_camera = backproject_optical(rows, columns, z, self.fx, self.fy, self.cx, self.cy)
                points_world = (camera_transform @ points_camera.T).T[:, :3]
                points_world = points_world[points_world[:, 2] > ground_height + 0.03]
                for cluster in self._height_clusters(points_world):
                    try:
                        shape_type = classify_shape(
                            cluster, self.maximum_sphere_span,
                        )
                        if shape_type == SPHERE_TYPE:
                            center, radius, residual = fit_sphere(
                                cluster, self.sphere_radius_bounds,
                            )
                            half_extents = np.full(3, radius, dtype=np.float32)
                        else:
                            center, half_extents, residual = fit_aabb(
                                cluster, self.cube_size_bounds,
                            )
                        if not self._inside_obstacle_workspace(center):
                            continue
                        candidates.append(
                            Detection(track_id, center, half_extents, shape_type, residual, len(cluster))
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        continue
            if candidates:
                detections.append(self._select_detection_candidate(candidates))
        return detections

    def update(self, rgb: np.ndarray, depth: np.ndarray, world_from_camera: np.ndarray, base_pose: np.ndarray, *, dt: float) -> np.ndarray:
        return self.update_views(
            [(rgb, depth, world_from_camera)],
            base_pose,
            dt=dt,
        )

    @staticmethod
    def _fuse_detections(detections: list[Detection]) -> Detection:
        """Fuse simultaneous same-color detections without double-updating a track."""

        if len(detections) == 1:
            return detections[0]
        types = {int(float(item.shape_type) >= 0.5) for item in detections}
        if len(types) > 1:
            preferred = max(detections, key=lambda item: float(item.position[2]))
            detections = [
                item for item in detections
                if int(float(item.shape_type) >= 0.5) == int(float(preferred.shape_type) >= 0.5)
            ]
            if len(detections) == 1:
                return detections[0]
        residuals = np.asarray([max(float(item.residual), 1.0e-3) for item in detections])
        points = np.asarray([max(int(item.points), 1) for item in detections], dtype=np.float64)
        weights = points / np.square(residuals)
        weights /= weights.sum()
        positions = np.stack([item.position for item in detections]).astype(np.float64)
        extents = np.stack([item.half_extents for item in detections]).astype(np.float64)
        position = np.sum(weights[:, None] * positions, axis=0)
        half_extents = np.sum(weights[:, None] * extents, axis=0)
        disagreement = np.sum(weights * np.sum(np.square(positions - position), axis=1))
        residual = math.sqrt(float(np.sum(weights * np.square(residuals)) + disagreement))
        exemplar = detections[int(np.argmax(weights))]
        return Detection(
            exemplar.track_id,
            position.astype(np.float32),
            half_extents.astype(np.float32),
            exemplar.shape_type,
            residual,
            int(points.sum()),
        )

    def update_views(
        self,
        views: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        base_pose: np.ndarray,
        *,
        dt: float,
    ) -> np.ndarray:
        """Predict once, then fuse all synchronous RGB-D views in world space."""

        self._last_base_pose = np.asarray(base_pose, dtype=np.float64).reshape(3).copy()
        for track in self.tracks:
            track.predict(dt)
        self.last_ground_heights = []
        self.last_view_detections = [
            self.detect(rgb, depth, world_from_camera, base_pose)
            for rgb, depth, world_from_camera in views
        ]
        grouped: dict[int, list[Detection]] = {}
        for view_detections in self.last_view_detections:
            for detection in view_detections:
                grouped.setdefault(detection.track_id, []).append(detection)
        detections = [
            self._vote_shape_type(self._fuse_detections(grouped[index]))
            for index in sorted(grouped)
        ]
        for detection in detections:
            self.tracks[detection.track_id].update(detection)
        self.last_detections = detections
        return self.observation(self._last_base_pose)

    def observation(self, base_pose: np.ndarray | None = None) -> np.ndarray:
        pose = self._last_base_pose if base_pose is None else np.asarray(base_pose, dtype=np.float64).reshape(3)
        base_from_world = self._base_transform(pose)
        rotation = base_from_world[:3, :3]
        result = []
        for track in self.tracks:
            feature = track.feature()
            if feature[VALID_INDEX] > 0.5:
                position = np.append(feature[POSITION].astype(np.float64), 1.0)
                feature[POSITION] = (base_from_world @ position)[:3]
                feature[VELOCITY] = rotation @ feature[VELOCITY]
                feature[ACCELERATION] = rotation @ feature[ACCELERATION]
            result.append(feature)
        return np.stack(result).astype(np.float32)

    @property
    def dynamic_missing_time(self) -> float:
        dynamic = [track.missing_time for track in self.tracks if track.shape_type == SPHERE_TYPE and track.initialized]
        return max(dynamic, default=0.0)

    @property
    def track_missing_times(self) -> list[float]:
        """Per-slot age of the last RGB-D observation for diagnostics."""
        return [float(track.missing_time) for track in self.tracks]

    @property
    def detection_pixel_counts(self) -> list[int]:
        """Combined pixel counts for detections in the latest camera set."""
        counts = [0] * len(self.tracks)
        for detection in self.last_detections:
            counts[int(detection.track_id)] = int(detection.points)
        return counts

    @property
    def detection_pixel_counts_by_camera(self) -> list[list[int]]:
        """Per-camera visible pixels, retained to diagnose view occlusion."""

        result: list[list[int]] = []
        for detections in self.last_view_detections:
            counts = [0] * len(self.tracks)
            for detection in detections:
                counts[int(detection.track_id)] = int(detection.points)
            result.append(counts)
        return result

    @property
    def accepts_new_plan(self) -> bool:
        dynamic_seen = any(
            track.shape_type == SPHERE_TYPE and track.initialized
            for track in self.tracks
        )
        return dynamic_seen and self.dynamic_missing_time <= 0.2

#!/usr/bin/env python3
"""Convert Phase 3 HDF5 episodes to a training-optimized 320x240 Zarr."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from flowcarrycbf.policies.flowcarry_cbf.schema import (
    ACTION_DIMENSION,
    CAMERA_COUNT,
    PROPRIO_DIMENSION,
    TRAIN_CAMERA_HEIGHT,
    TRAIN_CAMERA_WIDTH,
    ZARR_FORMAT_VERSION,
)


def _dependencies():
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as error:
        raise RuntimeError("conversion requires `pip install 'zarr<3'`") from error
    return zarr, Blosc


def episode_paths(root: Path) -> list[Path]:
    paths = sorted((root / "episodes").glob("episode_*.h5"))
    if not paths:
        raise FileNotFoundError(f"no HDF5 episodes under {root / 'episodes'}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    paths = episode_paths(args.source.resolve())
    if len(paths) != args.episodes:
        raise ValueError(f"expected {args.episodes} episodes, found {len(paths)}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    lengths = []
    camera_count = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            if camera_count is None:
                camera_count = int(handle["rgb"].shape[1])
            if int(handle["rgb"].shape[1]) != camera_count:
                raise ValueError("all episodes must have the same camera count")
            lengths.append(int(len(handle["rgb"])))
    camera_count = int(camera_count or CAMERA_COUNT)
    total = int(sum(lengths))
    zarr, Blosc = _dependencies()
    group = zarr.open_group(str(args.output), mode="w")
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    rgb = group.create_dataset(
        "rgb",
        shape=(total, camera_count, TRAIN_CAMERA_HEIGHT, TRAIN_CAMERA_WIDTH, 3),
        chunks=(1, camera_count, TRAIN_CAMERA_HEIGHT, TRAIN_CAMERA_WIDTH, 3),
        dtype="u1",
        compressor=compressor,
    )
    proprio = group.create_dataset(
        "proprio", shape=(total, PROPRIO_DIMENSION), chunks=(1024, PROPRIO_DIMENSION),
        dtype="f4", compressor=compressor,
    )
    actions = group.create_dataset(
        "actions", shape=(total, ACTION_DIMENSION), chunks=(1024, ACTION_DIMENSION),
        dtype="f4", compressor=compressor,
    )
    ends = group.create_dataset("episode_ends", shape=(len(paths),), dtype="i8")
    cursor = 0
    for episode_index, (path, length) in enumerate(zip(paths, lengths)):
        with h5py.File(path, "r") as handle:
            if handle["rgb"].shape[1] != camera_count:
                raise ValueError(f"invalid camera count in {path.name}")
            if handle["proprio"].shape != (length, PROPRIO_DIMENSION):
                raise ValueError(f"invalid proprio shape in {path.name}")
            if handle["actions"].shape != (length, ACTION_DIMENSION):
                raise ValueError(f"invalid action shape in {path.name}")
            proprio[cursor:cursor + length] = handle["proprio"][:]
            actions[cursor:cursor + length] = handle["actions"][:]
            for block_start in range(0, length, 64):
                block_end = min(block_start + 64, length)
                frames = np.asarray(handle["rgb"][block_start:block_end], dtype=np.uint8)
                resized = np.empty(
                    (
                        len(frames), camera_count, TRAIN_CAMERA_HEIGHT,
                        TRAIN_CAMERA_WIDTH, 3,
                    ),
                    dtype=np.uint8,
                )
                for frame_index in range(len(frames)):
                    for camera_index in range(camera_count):
                        resized[frame_index, camera_index] = cv2.resize(
                            frames[frame_index, camera_index],
                            (TRAIN_CAMERA_WIDTH, TRAIN_CAMERA_HEIGHT),
                            interpolation=cv2.INTER_AREA,
                        )
                output_start = cursor + block_start
                rgb[output_start:output_start + len(resized)] = resized
        cursor += length
        ends[episode_index] = cursor
        print(f"Convert episode {episode_index + 1}/{len(paths)} | frames {cursor}/{total}", flush=True)
    group.attrs.update(
        {
            "format_version": ZARR_FORMAT_VERSION,
            "source": str(args.source.resolve()),
            "episodes": len(paths),
            "total_frames": total,
            "resize": "area_no_crop",
            "camera_height": TRAIN_CAMERA_HEIGHT,
            "camera_width": TRAIN_CAMERA_WIDTH,
            "camera_count": camera_count,
            "robot": "tiago",
            "schema_version": 1,
        }
    )
    print(json.dumps({"output": str(args.output.resolve()), "episodes": len(paths), "frames": total}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train the paper-aligned Phase 3 dual-RGB Flow Matching model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from flowcarrycbf.policies.flowcarry_cbf.dataset import (
    RGBFlowZarrDataset,
    fit_minmax_normalization,
    split_episode_indices,
    validate_zarr_dataset,
)
from flowcarrycbf.policies.flowcarry_cbf.model import RGBFlowTrajectoryModel
from flowcarrycbf.policies.flowcarry_cbf.schema import HORIZON


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Converted Phase 3 Zarr dataset")
    parser.add_argument("--config", type=Path, default=None, help="Optional robot YAML metadata")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/flow_seed0.pt"),
        help="Best-validation EMA checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _paths(output: Path) -> dict[str, Path]:
    stem = output.with_suffix("")
    return {
        "best": output,
        "latest": stem.with_name(stem.name + "_latest").with_suffix(".pt"),
        "final": stem.with_name(stem.name + "_final").with_suffix(".pt"),
        "history": stem.with_suffix(".json"),
    }


def _normalization_tensors(
    normalization: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value, device=device, dtype=torch.float32)
        for name, value in normalization.items()
        if name.endswith(("_scale", "_offset"))
    }


def _normalize(
    proprio: torch.Tensor,
    targets: torch.Tensor,
    values: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    proprio = proprio * values["proprio_scale"] + values["proprio_offset"]
    targets = targets * values["actions_scale"] + values["actions_offset"]
    return proprio, targets


@torch.no_grad()
def evaluate(
    model: RGBFlowTrajectoryModel,
    loader: DataLoader,
    device: torch.device,
    normalization: dict[str, torch.Tensor],
    *,
    epoch: int,
) -> float:
    model.eval()
    losses = []
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(100000)
        progress = tqdm(
            loader,
            desc=f"Validation {epoch:04d}",
            leave=False,
            dynamic_ncols=True,
            unit="batch",
            mininterval=1.0,
        )
        for images, proprio, targets in progress:
            proprio, targets = _normalize(
                proprio.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
                normalization,
            )
            loss = float(
                model.cfm_loss(
                    targets, images.to(device, non_blocking=True), proprio
                ).cpu()
            )
            losses.append(loss)
            progress.set_postfix_str(
                f"loss={loss:.5f}  avg={np.mean(losses):.5f}", refresh=False
            )
    return float(np.mean(losses))


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _checkpoint(
    ema: EMAModel,
    ema_model: RGBFlowTrajectoryModel,
    normalization: dict[str, np.ndarray],
    *,
    epoch: int,
    validation_loss: float,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    live_model: RGBFlowTrajectoryModel,
    history: list[dict[str, float]],
    include_training_state: bool,
) -> dict[str, Any]:
    payload = ema_model.checkpoint(
        normalization,
        seed=args.seed,
        epoch=epoch,
        validation_loss=validation_loss,
        dataset=str(args.data.resolve()),
        reference="third_party/flow_matching@516e8e18875b27741bdbdb8a252e904032c90723",
        training_contract={
            "observation_horizon": 1,
            "prediction_horizon": HORIZON,
            "executed_horizon": 1,
            "flow_solver_steps": 8,
            "source_distribution": "normal",
            "ema_power": 0.75,
        },
        robot="tiago",
        action_semantics="tiago_absolute_control_target",
        camera_count=int(getattr(args, "camera_count", 2)),
    )
    if include_training_state:
        ema_state = ema.state_dict()
        ema_state.pop("shadow_params")
        payload["training_state"] = {
            "live_model": live_model.state_dict(),
            "ema_metadata": ema_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
        }
    return payload


def main() -> int:
    args = parse_args()
    if args.seed != 0:
        raise ValueError("formal Phase 3 training uses seed 0")
    if min(
        args.epochs, args.batch_size, args.checkpoint_every,
        args.resume_every, args.validate_every,
    ) <= 0:
        raise ValueError("epochs, batch size, and checkpoint interval must be positive")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("max train batches must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    summary = validate_zarr_dataset(
        args.data,
        expected_robot="tiago",
    )
    args.camera_count = int(summary["rgb_shape"][1])
    train_indices, validation_indices = split_episode_indices(summary["episodes"])
    train_data = RGBFlowZarrDataset(args.data, train_indices, horizon=HORIZON)
    validation_data = RGBFlowZarrDataset(args.data, validation_indices, horizon=HORIZON)
    normalization = fit_minmax_normalization(args.data)
    loader_generator = torch.Generator().manual_seed(args.seed)
    common_loader = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        **common_loader,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        **common_loader,
    )
    device = _device(args.device)
    model = RGBFlowTrajectoryModel(
        horizon=HORIZON, camera_count=args.camera_count,
    ).to(device)
    ema = EMAModel(
        parameters=model.parameters(), power=0.75, use_ema_warmup=True
    )
    ema_model = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(step, args.warmup_steps, total_steps),
    )
    normalization_gpu = _normalization_tensors(normalization, device)
    paths = _paths(args.output.resolve())
    paths["best"].parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best = float("inf")
    history: list[dict[str, float]] = []
    if args.resume:
        if not paths["latest"].exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {paths['latest']}")
        resume = torch.load(paths["latest"], map_location=device, weights_only=False)
        checkpoint_robot = resume.get("metadata", {}).get("robot")
        if checkpoint_robot not in (None, "tiago"):
            raise ValueError("resume checkpoint is not a TIAGo checkpoint")
        state = resume["training_state"]
        model.load_state_dict(state["live_model"])
        ema_state = dict(state["ema_metadata"])
        ema_state["shadow_params"] = [
            resume["model_state"][name] for name, _ in model.named_parameters()
        ]
        ema.load_state_dict(ema_state)
        ema.to(device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        history = list(state.get("history", []))
        best = min(
            (
                float(row["validation_loss"])
                for row in history
                if row.get("validation_loss") is not None
            ),
            default=float("inf"),
        )
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Flow Matching Training", flush=True)
    print(f"Data        : {args.data}", flush=True)
    print(f"Device      : {device_name}", flush=True)
    print(
        f"Episodes    : train {len(train_indices)} | validation {len(validation_indices)}",
        flush=True,
    )
    print(
        f"Windows     : train {len(train_data)} | validation {len(validation_data)}",
        flush=True,
    )
    print(f"Model       : {parameter_count / 1_000_000:.2f} M parameters", flush=True)
    print(f"Checkpoint  : {paths['best']}", flush=True)
    print(f"Start epoch : {start_epoch}", flush=True)
    print("-" * 60, flush=True)
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        loader_generator.manual_seed(args.seed + epoch)
        model.train()
        train_losses = []
        progress_total = len(train_loader)
        if args.max_train_batches is not None:
            progress_total = min(progress_total, args.max_train_batches)
        progress = tqdm(
            train_loader,
            total=progress_total,
            desc=f"Epoch {epoch + 1:04d}/{args.epochs:04d}",
            leave=False,
            dynamic_ncols=True,
            unit="batch",
            mininterval=1.0,
        )
        for batch_index, (images, proprio, targets) in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            proprio, targets = _normalize(
                proprio.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
                normalization_gpu,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.cfm_loss(targets, images.to(device, non_blocking=True), proprio)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.step(model.parameters())
            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            progress.set_postfix_str(
                f"loss={loss_value:.5f}  avg={np.mean(train_losses):.5f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                refresh=False,
            )
        if args.smoke_test:
            print(
                "Smoke test passed  "
                f"batches={len(train_losses)}  "
                f"train_loss={np.mean(train_losses):.6f}  "
                f"optimizer_step={ema.optimization_step}",
                flush=True,
            )
            return 0
        completed_epoch = epoch + 1
        save_resume = completed_epoch % args.resume_every == 0 or completed_epoch == args.epochs
        save_periodic = completed_epoch % args.checkpoint_every == 0
        should_validate = (
            completed_epoch % args.validate_every == 0
            or save_resume
            or save_periodic
            or completed_epoch == args.epochs
        )
        validation_loss = None
        if should_validate:
            ema.copy_to(ema_model.parameters())
            validation_loss = evaluate(
                ema_model,
                validation_loader,
                device,
                normalization_gpu,
                epoch=completed_epoch,
            )
        row = {
            "epoch": epoch + 1,
            "optimizer_step": ema.optimization_step,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "lr": scheduler.get_last_lr()[0],
            "ema_decay": float(ema.cur_decay_value or 0.0),
        }
        history.append(row)
        epoch_seconds = time.perf_counter() - epoch_start
        summary_line = (
            f"Epoch {completed_epoch:04d}/{args.epochs:04d}  "
            f"train_loss={row['train_loss']:.6f}  "
        )
        if validation_loss is not None:
            summary_line += f"val_loss={validation_loss:.6f}  "
        summary_line += f"lr={row['lr']:.2e}  time={epoch_seconds:.1f}s"
        print(summary_line, flush=True)
        if validation_loss is not None and validation_loss < best:
            best = validation_loss
            payload = _checkpoint(
                ema, ema_model, normalization,
                epoch=epoch + 1, validation_loss=validation_loss, args=args,
                optimizer=optimizer, scheduler=scheduler, live_model=model,
                history=history, include_training_state=False,
            )
            torch.save(payload, paths["best"])
            print(f"Saved best   : {paths['best'].name}", flush=True)
        if save_resume or save_periodic:
            payload = _checkpoint(
                ema, ema_model, normalization,
                epoch=epoch + 1, validation_loss=float(validation_loss), args=args,
                optimizer=optimizer, scheduler=scheduler, live_model=model,
                history=history, include_training_state=True,
            )
            if save_resume:
                torch.save(payload, paths["latest"])
                print(f"Saved latest : {paths['latest'].name}", flush=True)
        if save_periodic:
            periodic = paths["best"].with_name(
                f"{paths['best'].stem}_epoch{epoch + 1:04d}{paths['best'].suffix}"
            )
            torch.save(payload, periodic)
            print(f"Saved checkpoint: {periodic.name}", flush=True)
        paths["history"].write_text(
            json.dumps(
                {"data": summary, "best_validation_loss": best, "history": history},
                indent=2,
            ),
            encoding="utf-8",
        )
    final_payload = _checkpoint(
        ema,
        ema_model,
        normalization,
        epoch=args.epochs,
        validation_loss=float(history[-1]["validation_loss"]),
        args=args,
        optimizer=optimizer,
        scheduler=scheduler,
        live_model=model,
        history=history,
        include_training_state=False,
    )
    torch.save(final_payload, paths["final"])
    print("-" * 60, flush=True)
    print("Training complete", flush=True)
    print(f"Best validation loss : {best:.6f}", flush=True)
    print(f"Best checkpoint       : {paths['best']}", flush=True)
    print(f"Final checkpoint      : {paths['final']}", flush=True)
    print(f"History               : {paths['history']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

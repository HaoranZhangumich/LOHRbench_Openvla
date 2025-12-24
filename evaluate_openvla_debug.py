#!/usr/bin/env python3
"""
Evaluate OpenVLA finetuned with LoRA + L1 regression head (finetune.py default).

This script:
- Loads base model from checkpoint dir
- Loads LoRA adapter from ckpt_dir/lora_adapter (optional merge)
- Loads action_head--{step}_checkpoint.pt (CRITICAL for L1 regression training)
- Evaluates on RLDS TFDS episodes

Example:
export OPENVLA_ROOT="/home/haoran-zhang/openvla/LOHRbench_Openvla/openvla-oft"

TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 python evaluate_openvla_lora_l1.py \
  --ckpt /home/haoran-zhang/openvla/LOHRbench_Openvla/openvla-oft/openvla/openvla-7b+lohrbench_rlds+b8+lr-0.0005+lora-r32+dropout-0.0--lohrbench_lora_r32_bs2x1_ga32--50000_chkpt \
  --builder_dir /home/haoran-zhang/data/Lohrbench_rlds/lohrbench_rlds/lohrbench_rlds/0.1.0 \
  --split train \
  --episode_index 371 \
  --num_episodes 1 \
  --max_steps 100 \
  --out_dir ./eval_results_debug \
  --num_images_in_input 2 \
  --merge_lora \
  --step 50000
"""

from __future__ import annotations

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# TF only for TFDS reading; avoid touching GPU
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
import tensorflow_datasets as tfds

from PIL import Image
import matplotlib.pyplot as plt


# =============================================================================
# Locate OPENVLA_ROOT and import repo utilities
# =============================================================================
def find_openvla_root() -> str:
    root = os.environ.get("OPENVLA_ROOT")
    if root and os.path.exists(root):
        return root
    script_path = Path(__file__).resolve()
    cur = script_path.parent
    for _ in range(10):
        if (cur / "experiments" / "robot").exists():
            return str(cur)
        cur = cur.parent
    raise RuntimeError(
        "Could not find openvla-oft repo root. Please set OPENVLA_ROOT=/path/to/openvla-oft"
    )


OPENVLA_ROOT = find_openvla_root()
sys.path.insert(0, OPENVLA_ROOT)
print(f"✓ Added to Python path: {OPENVLA_ROOT}")

# Repo utils (these avoid the processor batching issues you hit)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_vla,
    get_processor,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    set_seed_everywhere,
)
from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402
from prismatic.models.projectors import ProprioProjector  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa: E402
from peft import PeftModel  # noqa: E402


# =============================================================================
# Small helpers
# =============================================================================
def decode_text(x: Any) -> str:
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def uint8_image(x: Any) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def pack_state_from_qpos(qpos_9: Any) -> np.ndarray:
    """9D qpos -> 8D state: [qpos[:7], mean(qpos[7:9])]"""
    qpos_9 = np.asarray(qpos_9, dtype=np.float32)
    if qpos_9.shape[-1] != 9:
        raise ValueError(f"Expected qpos last dim=9, got {qpos_9.shape}")
    joints7 = qpos_9[..., :7]
    gripper1 = np.mean(qpos_9[..., 7:9], axis=-1, keepdims=True)
    return np.concatenate([joints7, gripper1], axis=-1).astype(np.float32)


def summarize_vec(name: str, x: np.ndarray) -> None:
    x = np.asarray(x)
    print(
        f"{name}: shape={x.shape} min={x.min():.4f} max={x.max():.4f} "
        f"mean={x.mean():.4f} std={x.std():.4f}",
        flush=True,
    )


def remove_ddp_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def infer_step_from_ckpt_name(ckpt_dir: Path) -> int | None:
    """
    If ckpt dir name ends with --{step}_chkpt, extract step.
    """
    m = re.search(r"--(\d+)_chkpt$", ckpt_dir.name)
    if m:
        return int(m.group(1))
    return None


def save_images(
    base_frames: List[np.ndarray],
    wrist_frames: List[np.ndarray],
    out_dir: str,
    stride: int = 50,
    max_frames: int = 30,
):
    os.makedirs(out_dir, exist_ok=True)
    T = min(len(base_frames), len(wrist_frames))
    idxs = list(range(0, T, max(1, stride)))[:max_frames]
    for t in idxs:
        Image.fromarray(base_frames[t]).save(os.path.join(out_dir, f"base_t{t:04d}.png"))
        Image.fromarray(wrist_frames[t]).save(os.path.join(out_dir, f"wrist_t{t:04d}.png"))
    print(f"✅ Saved {len(idxs)} image pairs to: {out_dir}")


def plot_debug(gt: np.ndarray, pred: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    T, D = gt.shape
    x = np.arange(T)

    for d in range(D):
        plt.figure(figsize=(10, 5))
        plt.plot(x, gt[:, d], label="GT")
        plt.plot(x, pred[:, d], linestyle="--", label="Pred")
        plt.xlabel("t")
        plt.ylabel(f"dim {d}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dim_{d:02d}.png"), dpi=120)
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x, np.linalg.norm(pred - gt, axis=1))
    plt.xlabel("t")
    plt.ylabel("L2 error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "l2.png"), dpi=120)
    plt.close()


def print_error_summary(gt_arr: np.ndarray, pred_arr: np.ndarray) -> None:
    err = (pred_arr - gt_arr).astype(np.float64)
    abs_err = np.abs(err)
    mse = err ** 2
    mae = abs_err.mean(axis=0)
    rmse = np.sqrt(mse.mean(axis=0))
    print(f"Mean MAE:  {mae.mean():.6f}")
    print(f"Mean RMSE: {rmse.mean():.6f}")
    print(f"Max |err|: {abs_err.max():.6f}")
    for d in range(gt_arr.shape[1]):
        print(f"  dim {d}: MAE {mae[d]:.6f} RMSE {rmse[d]:.6f} Max {abs_err[:, d].max():.6f}")


# =============================================================================
# EvalConfig compatible with get_vla/get_action
# =============================================================================
class EvalConfig:
    def __init__(self, args):
        self.pretrained_checkpoint = args.ckpt
        self.model_family = "openvla"

        self.num_images_in_input = args.num_images_in_input
        self.image_size = 224
        self.use_fused_vision_backbone = True
        self.center_crop = args.center_crop

        self.unnorm_key = args.unnorm_key if args.unnorm_key != "None" else None

        # MATCH finetune.py defaults unless user overrides
        self.use_proprio = args.use_proprio
        self.use_film = args.use_film
        self.use_l1_regression = True          # finetune.py default
        self.use_diffusion = False

        self.load_in_8bit = False
        self.load_in_4bit = False
        self.device = args.device
        self.dtype = args.dtype
        self.attn_implementation = "sdpa"


# =============================================================================
# Load model + LoRA + action head (+ optional proprio projector)
# =============================================================================
def load_model_and_modules(cfg: EvalConfig, step: int, merge_lora: bool):
    print(f"\n🤖 Loading base model via get_vla(cfg) from: {cfg.pretrained_checkpoint}")
    vla = get_vla(cfg)

    # Ensure correct number of images in input (training did this)
    if hasattr(vla, "vision_backbone"):
        try:
            vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
        except Exception:
            pass

    # Attach LoRA if present
    ckpt_dir = Path(cfg.pretrained_checkpoint)
    adapter_dir = ckpt_dir / "lora_adapter"
    if adapter_dir.exists():
        print(f"✅ Found LoRA adapter: {adapter_dir}")
        vla = PeftModel.from_pretrained(vla, str(adapter_dir))
        if merge_lora:
            print("🔧 Merging LoRA (merge_and_unload) ...")
            vla = vla.merge_and_unload()
    else:
        print("ℹ️  No lora_adapter found; assuming ckpt already merged or non-LoRA.")

    vla.eval()

    processor = get_processor(cfg)

    # Load action_head (CRITICAL for your training setup)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.dtype]
    action_head = L1RegressionActionHead(
        input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM
    ).to(cfg.device).to(torch_dtype)
    head_path = ckpt_dir / f"action_head--{step}_checkpoint.pt"
    if not head_path.exists():
        raise FileNotFoundError(
            f"Missing {head_path}\n"
            f"Your finetune.py default use_l1_regression=True, so this file must exist for correct eval."
        )
    sd = torch.load(str(head_path), map_location="cpu")
    if isinstance(sd, dict):
        sd = remove_ddp_prefix(sd)
    action_head.load_state_dict(sd)
    action_head.eval()
    print(f"✅ Loaded action_head from: {head_path}")

    # Optional proprio projector (only if you actually trained with --use_proprio True)
    proprio_projector = None
    if cfg.use_proprio:
        pp_path = ckpt_dir / f"proprio_projector--{step}_checkpoint.pt"
        if not pp_path.exists():
            raise FileNotFoundError(
                f"--use_proprio was set, but missing {pp_path}. "
                f"Train/eval must match: finetune.py default use_proprio=False."
            )
        proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM).to(cfg.device).to(torch_dtype)
        pp_sd = torch.load(str(pp_path), map_location="cpu")
        proprio_projector.load_state_dict(remove_ddp_prefix(pp_sd))
        proprio_projector.eval()
        print(f"✅ Loaded proprio_projector from: {pp_path}")

    return vla, processor, action_head, proprio_projector


# =============================================================================
# Main evaluation
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint directory (--XXXX_chkpt dir)")
    ap.add_argument("--builder_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episode_index", type=int, default=0)
    ap.add_argument("--num_episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--out_dir", default="./eval_results_debug")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    ap.add_argument("--unnorm_key", default="lohrbench_rlds")
    ap.add_argument("--num_images_in_input", type=int, default=2)
    ap.add_argument("--center_crop", action="store_true")

    ap.add_argument("--use_proprio", action="store_true", help="ONLY if you trained with use_proprio=True")
    ap.add_argument("--use_film", action="store_true", help="ONLY if you trained with use_film=True")

    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--img_stride", type=int, default=50)
    ap.add_argument("--img_max_frames", type=int, default=30)

    ap.add_argument("--seed", type=int, default=42)

    # Must match file names action_head--{step}_checkpoint.pt
    ap.add_argument("--step", type=int, default=None, help="Training step (e.g. 50000). If omitted, infer from ckpt dir name.")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)

    ckpt_dir = Path(args.ckpt)
    step = args.step if args.step is not None else infer_step_from_ckpt_name(ckpt_dir)
    if step is None:
        raise ValueError(
            "Could not infer --step from ckpt dir name. Please pass --step 50000 (or correct step)."
        )

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = EvalConfig(args)

    print("=" * 90)
    print("OpenVLA Evaluation (LoRA + L1Regression head)")
    print("=" * 90)
    print(f"ckpt: {args.ckpt}")
    print(f"step: {step}")
    print(f"builder_dir: {args.builder_dir}")
    print(f"split: {args.split}")
    print(f"episodes: {args.episode_index}..{args.episode_index + args.num_episodes - 1}")
    print(f"num_images_in_input: {args.num_images_in_input}")
    print(f"use_proprio: {args.use_proprio} (finetune.py default was False)")
    print(f"use_film: {args.use_film} (finetune.py default was False)")
    print(f"merge_lora: {args.merge_lora}")
    print("=" * 90)

    vla, processor, action_head, proprio_projector = load_model_and_modules(cfg, step=step, merge_lora=args.merge_lora)

    print("\n📂 Loading dataset...")
    builder = tfds.builder_from_directory(args.builder_dir)
    ds = builder.as_dataset(split=args.split, shuffle_files=False)

    all_gt, all_pred = [], []

    for ep_idx in range(args.episode_index, args.episode_index + args.num_episodes):
        print(f"\n{'='*80}\nEpisode {ep_idx}\n{'='*80}")
        ep = next(iter(ds.skip(ep_idx).take(1)))
        steps_list = list(ep["steps"].as_numpy_iterator())
        if len(steps_list) < 2:
            print(f"⚠ Episode too short ({len(steps_list)}). Skipping.")
            continue
        steps_list = steps_list[:-1]  # drop last dummy
        print(f"Episode length: {len(steps_list)}")

        instruction = decode_text(steps_list[0]["language_instruction"])
        print("Instruction:", instruction)

        gt_list, pred_list = [], []
        base_frames, wrist_frames = [], []

        T = min(len(steps_list), args.max_steps)
        for t in range(T):
            step_d = steps_list[t]
            obs_raw = step_d["observation"]

            base_np = uint8_image(obs_raw["base_rgb"])
            wrist_np = uint8_image(obs_raw["hand_rgb"])

            if args.save_images:
                base_frames.append(base_np)
                wrist_frames.append(wrist_np)

            # Proprio/state (only if cfg.use_proprio)
            state = pack_state_from_qpos(obs_raw["qpos"])  # (8,)

            # Ground truth action can be either (8,) or (8,8) flattened
            act = np.asarray(step_d["action"], dtype=np.float32).reshape(-1)
            if act.size == ACTION_DIM:
                gt = act
            elif act.size == NUM_ACTIONS_CHUNK * ACTION_DIM:
                gt = act.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)[0]
            else:
                raise ValueError(f"Unexpected action size {act.size}; expected 8 or 64.")

            obs = {
                "full_image": base_np,
                "wrist_image": wrist_np,
                "state": state,
            }

            # IMPORTANT: pass action_head (+ proprio_projector if used) so we actually use the trained head
            action_chunk = get_action(
                cfg=cfg,
                model=vla,
                obs=obs,
                task_label=instruction,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=None,
                use_film=cfg.use_film,
            )

            # action_chunk can be list or np array
            if isinstance(action_chunk, list):
                pred0 = np.asarray(action_chunk[0], dtype=np.float32).reshape(-1)
            else:
                arr = np.asarray(action_chunk, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == NUM_ACTIONS_CHUNK:
                    pred0 = arr[0]
                else:
                    pred0 = arr.reshape(-1)

            if t == 0:
                print("\nFirst step sanity check:")
                summarize_vec("  gt", gt)
                summarize_vec("  pred", pred0)
                print("  gt:", np.round(gt, 3))
                print("  pr:", np.round(pred0, 3))
                print("  |err|:", np.round(np.abs(pred0 - gt), 3))

            gt_list.append(gt)
            pred_list.append(pred0)

        gt_arr = np.stack(gt_list, 0)
        pred_arr = np.stack(pred_list, 0)

        ep_dir = os.path.join(args.out_dir, f"episode_{ep_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        np.save(os.path.join(ep_dir, "gt.npy"), gt_arr)
        np.save(os.path.join(ep_dir, "pred.npy"), pred_arr)
        plot_debug(gt_arr, pred_arr, os.path.join(ep_dir, "plots"))

        if args.save_images:
            save_images(base_frames, wrist_frames, os.path.join(ep_dir, "images"),
                        stride=args.img_stride, max_frames=args.img_max_frames)

        print("\nEpisode error summary:")
        print_error_summary(gt_arr, pred_arr)

        all_gt.append(gt_arr)
        all_pred.append(pred_arr)

    if all_gt:
        all_gt = np.concatenate(all_gt, 0)
        all_pred = np.concatenate(all_pred, 0)
        print(f"\n{'='*80}\nAGGREGATE ({all_gt.shape[0]} steps)\n{'='*80}")
        print_error_summary(all_gt, all_pred)

    print(f"\n✅ Done. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()

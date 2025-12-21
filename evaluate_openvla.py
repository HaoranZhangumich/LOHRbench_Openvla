#!/usr/bin/env python3
"""
Example:
CUDA_VISIBLE_DEVICES=5 python evaluate_openvla.py \
  --ckpt /home/users/haoran/runs/openvla_oft_lohrbench/openvla-7b+lohrbench_rlds+b16+lr-0.0005+lora-r32+dropout-0.0--lohrbench_lora_r32_bs1x2_ga16--30000_chkpt \
  --builder_dir /home/users/haoran/data/Lohrbench_rlds/lohrbench_rlds/lohrbench_rlds/0.1.0 \
  --split train \
  --num_views 2 \
  --base_key base_rgb \
  --hand_key hand_rgb \
  --attn_impl sdpa \
  --dtype bf16 \
  --unnorm_key lohrbench_rlds \
  --max_steps_per_episode 200 \
  --plot_dir ./plots
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from PIL import Image
import tensorflow_datasets as tfds
from transformers import AutoModelForVision2Seq, AutoProcessor

import matplotlib.pyplot as plt


def make_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction}?\nOut:"


def decode_text(x: Any) -> str:
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def to_pil(img: np.ndarray) -> Image.Image:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


@dataclass
class Metrics:
    n: int
    mae_sum: np.ndarray
    mse_sum: np.ndarray

    def update(self, err: np.ndarray) -> None:
        self.n += 1
        self.mae_sum += np.abs(err)
        self.mse_sum += err ** 2

    def finalize(self) -> Dict[str, Any]:
        mae = self.mae_sum / max(self.n, 1)
        mse = self.mse_sum / max(self.n, 1)
        rmse = np.sqrt(mse)
        return {
            "mae": mae,
            "rmse": rmse,
            "mae_mean": float(mae.mean()),
            "rmse_mean": float(rmse.mean()),
        }


def try_set_num_images_in_input(vla, n: int) -> bool:
    candidates = []
    if hasattr(vla, "model"):
        candidates.append(getattr(vla.model, "vision_backbone", None))
        candidates.append(getattr(vla.model, "vision_tower", None))
    candidates.append(getattr(vla, "vision_backbone", None))
    candidates.append(getattr(vla, "vision_tower", None))

    for vb in candidates:
        if vb is None:
            continue
        if hasattr(vb, "set_num_images_in_input"):
            vb.set_num_images_in_input(n)
            return True
    return False


def to_numpy_action(pred):
    """Convert predict_action output to 1D float32 numpy."""
    if isinstance(pred, dict):
        for k in ["actions", "action", "pred_actions", "pred_action"]:
            if k in pred:
                pred = pred[k]
                break
        else:
            pred = next(iter(pred.values()))

    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    if torch.is_tensor(pred):
        pred = pred.detach().float().cpu().numpy()
    else:
        pred = np.asarray(pred, dtype=np.float32)

    # If action chunk, take first row for per-step comparison
    if pred.ndim >= 2:
        pred = pred[0]

    return pred.astype(np.float32).reshape(-1)


def build_inputs_one_view(
    processor,
    prompt: str,
    img_pil: Image.Image,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    text = processor.tokenizer(prompt, return_tensors="pt")
    pixels = processor.image_processor(img_pil, return_tensors="pt")["pixel_values"]

    inputs: Dict[str, torch.Tensor] = {
        "input_ids": text["input_ids"],
        "attention_mask": text.get("attention_mask", None),
        "pixel_values": pixels,
    }
    inputs = {k: v for k, v in inputs.items() if v is not None}
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    return inputs


def build_inputs_two_views(
    processor,
    prompt: str,
    img_pils: List[Image.Image],
    device: str,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    if len(img_pils) != 2:
        raise ValueError("build_inputs_two_views expects exactly 2 images")

    text = processor.tokenizer(prompt, return_tensors="pt")
    pv0 = processor.image_processor(img_pils[0], return_tensors="pt")["pixel_values"]
    pv1 = processor.image_processor(img_pils[1], return_tensors="pt")["pixel_values"]
    pixel_values = torch.cat([pv0, pv1], dim=1)

    inputs: Dict[str, torch.Tensor] = {
        "input_ids": text["input_ids"],
        "attention_mask": text.get("attention_mask", None),
        "pixel_values": pixel_values,
    }
    inputs = {k: v for k, v in inputs.items() if v is not None}
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    return inputs


def infer_action(
    vla,
    processor,
    instruction: str,
    base_img: Optional[Image.Image],
    hand_img: Optional[Image.Image],
    device: str,
    dtype: torch.dtype,
    unnorm_key: Optional[str],
    num_views: int,
) -> np.ndarray:
    prompt = make_prompt(instruction)

    if num_views == 1:
        assert base_img is not None
        inputs = build_inputs_one_view(processor, prompt, base_img, device, dtype)
    elif num_views == 2:
        assert base_img is not None and hand_img is not None
        inputs = build_inputs_two_views(processor, prompt, [base_img, hand_img], device, dtype)
    else:
        raise ValueError("--num_views must be 1 or 2")

    with torch.no_grad():
        pred = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    return to_numpy_action(pred)


def plot_action_comparison(gt_actions: np.ndarray, pred_actions: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    T = min(gt_actions.shape[0], pred_actions.shape[0])
    D = min(gt_actions.shape[1], pred_actions.shape[1])
    gt = gt_actions[:T, :D]
    pr = pred_actions[:T, :D]
    x = np.arange(T)

    # Per-dim plots
    for d in range(D):
        plt.figure()
        plt.plot(x, gt[:, d], label="gt")
        plt.plot(x, pr[:, d], label="pred")
        plt.xlabel("timestep")
        plt.ylabel(f"action[{d}]")
        plt.title(f"GT vs Pred action dim {d}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"action_dim{d:02d}.png"))
        plt.close()

    # L2 error over time
    l2 = np.linalg.norm(pr - gt, axis=1)
    plt.figure()
    plt.plot(x, l2, label="||pred-gt||2")
    plt.xlabel("timestep")
    plt.ylabel("L2 error")
    plt.title("Per-step action L2 error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "action_l2_error.png"))
    plt.close()

    print(f"✅ Saved plots to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--builder_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_episodes", type=int, default=1)
    ap.add_argument("--max_steps_per_episode", type=int, default=20)

    ap.add_argument("--num_views", type=int, default=2, choices=[1, 2])
    ap.add_argument("--base_key", default="base_rgb")
    ap.add_argument("--hand_key", default="hand_rgb")

    ap.add_argument("--unnorm_key", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--skip_last", action="store_true", default=True)

    # NEW: plotting
    ap.add_argument("--plot_dir", default=None,
                    help="If set, saves GT vs Pred plots (8 dims) for the first episode.")
    ap.add_argument("--plot_max_steps", type=int, default=400,
                    help="Max timesteps to record for plotting.")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if not os.path.isdir(args.builder_dir):
        raise FileNotFoundError(f"builder_dir not found: {args.builder_dir}")

    builder = tfds.builder_from_directory(args.builder_dir)
    ds = builder.as_dataset(split=args.split, shuffle_files=False)

    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
    ).to(args.device)
    vla.eval()

    # confirm torch device (useful)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("model device:", next(vla.parameters()).device)

    if args.num_views == 2:
        ok = try_set_num_images_in_input(vla, 2)
        if not ok:
            print("⚠️  Could not find set_num_images_in_input; checkpoint may only support 1 image.")

    metrics: Optional[Metrics] = None
    total_steps = 0
    used_episodes = 0
    dim_mismatch = 0

    # buffers for plots (first episode only)
    gt_buf: List[np.ndarray] = []
    pr_buf: List[np.ndarray] = []

    for ep_i, ep in enumerate(ds.take(args.max_episodes)):
        used_episodes += 1
        steps_ds = ep["steps"]

        first_step = next(iter(steps_ds.take(1).as_numpy_iterator()))
        instruction = decode_text(first_step["language_instruction"])

        for step in steps_ds.take(args.max_steps_per_episode).as_numpy_iterator():
            if args.skip_last and bool(step["is_last"]):
                continue

            obs = step["observation"]
            base_img = to_pil(obs[args.base_key]) if args.base_key in obs else None
            hand_img = to_pil(obs[args.hand_key]) if args.hand_key in obs else None

            gt = np.asarray(step["action"], dtype=np.float32).reshape(-1)

            pred = infer_action(
                vla=vla,
                processor=processor,
                instruction=instruction,
                base_img=base_img,
                hand_img=hand_img,
                device=args.device,
                dtype=dtype,
                unnorm_key=args.unnorm_key,
                num_views=args.num_views,
            )

            D = min(len(gt), len(pred))
            if len(gt) != len(pred):
                dim_mismatch += 1

            err = (pred[:D] - gt[:D]).astype(np.float64)

            if metrics is None:
                metrics = Metrics(n=0, mae_sum=np.zeros((D,), dtype=np.float64), mse_sum=np.zeros((D,), dtype=np.float64))
            metrics.update(err)
            total_steps += 1

            # record for plotting (first episode only)
            if args.plot_dir is not None and ep_i == 0 and len(gt_buf) < args.plot_max_steps:
                gt_buf.append(gt[:D].copy())
                pr_buf.append(pred[:D].copy())

        # only plot first episode
        if args.plot_dir is not None:
            break

    if metrics is None or total_steps == 0:
        print("No steps evaluated. Check dataset path/split/keys.")
        return

    out = metrics.finalize()

    print("\n==== OpenVLA RLDS Two-View Reproduction Check ====")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Dataset    : {args.builder_dir}  (split={args.split})")
    print(f"Views      : {args.num_views}  (base_key={args.base_key}, hand_key={args.hand_key})")
    print(f"Episodes   : {used_episodes}")
    print(f"Steps eval : {total_steps}")
    print(f"Dim mismatch count: {dim_mismatch} (compared min(D) dims)")

    print("\n-- Overall --")
    print(f"Mean MAE :  {out['mae_mean']:.6f}")
    print(f"Mean RMSE:  {out['rmse_mean']:.6f}")

    mae = out["mae"]
    rmse = out["rmse"]
    show = min(16, mae.shape[0])
    print(f"\n-- Per-dim (first {show}) --")
    for i in range(show):
        print(f"dim[{i:02d}]  MAE={mae[i]:.6f}  RMSE={rmse[i]:.6f}")

    # Save plots
    if args.plot_dir is not None and len(gt_buf) > 0:
        gt_arr = np.stack(gt_buf, axis=0)
        pr_arr = np.stack(pr_buf, axis=0)
        plot_action_comparison(gt_arr, pr_arr, out_dir=args.plot_dir)


if __name__ == "__main__":
    main()

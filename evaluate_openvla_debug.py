#!/usr/bin/env python3
"""
Evaluation script for OpenVLA-OFT model on Lohrbench RLDS dataset.

Usage:
CUDA_VISIBLE_DEVICES=0 python evaluate_openvla_debug.py \
  --ckpt /home/users/haoran/runs/openvla_oft_lohrbench/openvla-7b+lohrbench_rlds+b16+lr-0.0005+lora-r32+dropout-0.0--lohrbench_lora_r32_bs1x2_ga16--40000_chkpt \
  --builder_dir /home/users/haoran/data/Lohrbench_rlds/lohrbench_rlds/lohrbench_rlds/0.1.0 \
  --split train \
  --episode_index 121 \
  --max_steps 100 \
  --out_dir ./eval_results \
  --save_images \
  --num_episodes 1

Key fixes from original code:
1. Correctly packs base_rgb + wrist_image into 6-channel tensor
2. Properly extracts 8D state from 9D qpos (7 joints + mean of 2 fingers)
3. Handles processor that may return 6ch per image (padding secondary)
4. Sets num_images_in_input=1 for packed observation
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
import tensorflow_datasets as tfds
from transformers import AutoModelForVision2Seq, AutoProcessor
import matplotlib.pyplot as plt


# ============================================================================
# Utilities
# ============================================================================

def make_prompt(instruction: str) -> str:
    """Format instruction into VLA prompt."""
    return f"In: What action should the robot take to {instruction}?\nOut:"


def decode_text(x: Any) -> str:
    """Decode bytes/bytearray to string."""
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def uint8_image(x: Any) -> np.ndarray:
    """Ensure image is uint8 numpy array."""
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def pack_state_from_qpos(qpos_9: Any) -> np.ndarray:
    """
    Transform 9D qpos to 8D state matching lohrbench_rlds_dataset_transform:
      state = [qpos[:7], mean(qpos[7:9])]
    
    Args:
        qpos_9: Array with shape (..., 9) containing [7 joints, 2 finger positions]
    
    Returns:
        state: Array with shape (..., 8) containing [7 joints, 1 gripper]
    """
    qpos_9 = np.asarray(qpos_9, dtype=np.float32)
    if qpos_9.shape[-1] != 9:
        raise ValueError(f"Expected qpos last dim=9, got {qpos_9.shape}")
    
    joints7 = qpos_9[..., :7]
    gripper1 = np.mean(qpos_9[..., 7:9], axis=-1, keepdims=True)
    return np.concatenate([joints7, gripper1], axis=-1).astype(np.float32)


def summarize_vec(name: str, x: np.ndarray) -> None:
    """Print statistics for array."""
    x = np.asarray(x)
    print(
        f"{name}: shape={x.shape} "
        f"min={x.min():.4f} max={x.max():.4f} "
        f"mean={x.mean():.4f} std={x.std():.4f}",
        flush=True,
    )


# ============================================================================
# Model utilities
# ============================================================================

def try_set_num_images_in_input(vla, n: int) -> bool:
    """
    Set num_images_in_input on vision backbone if available.
    For packed 6-channel (base+wrist), we want n=1.
    """
    candidates = []
    if hasattr(vla, "model"):
        candidates.append(getattr(vla.model, "vision_backbone", None))
        candidates.append(getattr(vla.model, "vision_tower", None))
    candidates.append(getattr(vla, "vision_backbone", None))
    candidates.append(getattr(vla, "vision_tower", None))

    for vb in candidates:
        if vb is not None and hasattr(vb, "set_num_images_in_input"):
            vb.set_num_images_in_input(n)
            return True
    return False


def to_numpy_action(pred) -> np.ndarray:
    """Convert model prediction to (D,) numpy array."""
    # Handle dict outputs
    if isinstance(pred, dict):
        for k in ["actions", "action", "pred_actions", "pred_action"]:
            if k in pred:
                pred = pred[k]
                break
        else:
            pred = next(iter(pred.values()))

    # Handle tuple/list
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    # Convert to numpy
    if torch.is_tensor(pred):
        pred = pred.detach().float().cpu().numpy()
    else:
        pred = np.asarray(pred, dtype=np.float32)

    # Handle action chunks (T, D) -> take first timestep
    if pred.ndim >= 2:
        pred = pred[0]

    return pred.astype(np.float32).reshape(-1)


def predict_action_with_proprio(
    vla,
    inputs: Dict[str, torch.Tensor],
    proprio_np: Optional[np.ndarray],
    device: str,
    unnorm_key: Optional[str],
    verbose: bool = False,
):
    """
    Call vla.predict_action with proprioception.
    Tries multiple common argument names for proprio.
    """
    kwargs = dict(inputs)
    
    if proprio_np is not None:
        proprio_np = np.asarray(proprio_np, dtype=np.float32).reshape(1, -1)
        proprio_t = torch.from_numpy(proprio_np).to(device)

        # Try common proprio argument names
        for k in ["proprio", "state", "robot_state"]:
            try:
                out = vla.predict_action(
                    **kwargs,
                    **{k: proprio_t},
                    unnorm_key=unnorm_key,
                    do_sample=False
                )
                if verbose:
                    print(f"✓ predict_action accepted proprio key='{k}' shape={tuple(proprio_t.shape)}")
                return out
            except TypeError:
                if verbose:
                    print(f"✗ predict_action rejected proprio key='{k}'")
                continue

        if verbose:
            print("⚠ No proprio key accepted, running without proprio")

    return vla.predict_action(**kwargs, unnorm_key=unnorm_key, do_sample=False)


# ============================================================================
# Image packing (critical fix)
# ============================================================================

def _reshape_pixel_values_to_bchw(pv: torch.Tensor) -> torch.Tensor:
    """
    Reshape pixel_values to (B, C, H, W).
    
    Handles:
      - (B, C, H, W) -> unchanged
      - (B, N, 3, H, W) -> (B, 3N, H, W)
      - (N, 3, H, W) -> (1, 3N, H, W)
    """
    if pv.ndim == 5:
        b, n, c, h, w = pv.shape
        return pv.reshape(b, n * c, h, w)
    
    if pv.ndim == 4:
        # Check if (N, 3, H, W) without batch dim
        if pv.shape[0] in (2, 3) and pv.shape[1] == 3:
            n, c, h, w = pv.shape
            return pv.reshape(1, n * c, h, w)
        return pv
    
    raise ValueError(f"Unexpected pixel_values ndim={pv.ndim}, shape={tuple(pv.shape)}")


def build_inputs_packed_base_wrist(
    processor,
    prompt: str,
    base_pil: Image.Image,
    wrist_pil: Image.Image,  # We keep this arg to avoid breaking the calling loop, but we won't use it
    device: str,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Builds inputs where the Base image is real, but the Wrist image is zeroed out (black).
    This forces the model to evaluate using ONLY the base camera view.
    """
    # Tokenize text
    text = processor.tokenizer(prompt, return_tensors="pt")
    input_ids = text["input_ids"].to(device)
    attention_mask = text.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # 1. Process ONLY the Base image
    pv_base = processor.image_processor(base_pil, return_tensors="pt")["pixel_values"]
    pv_base = _reshape_pixel_values_to_bchw(pv_base) # Shape (1, 3, H, W)
    
    # 2. Extract the base image tensor
    base3 = pv_base[:, :3]   # (1, 3, H, W)
    
    # 3. Create a "Blind" Wrist Tensor (All Zeros)
    # Must match base3 in shape, device, and dtype
    dummy_wrist = torch.zeros_like(base3)

    # 4. Pack them together: [Base RGB | Black RGB]
    # Total shape: (1, 6, H, W)
    pixel_values = torch.cat([base3, dummy_wrist], dim=1)
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    
    # Debug print to confirm it's working
    # print(f"🔍 Packed Blind Input: Base={tuple(base3.shape)} + Wrist(Zeros)={tuple(dummy_wrist.shape)}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }


# ============================================================================
# Metrics and visualization
# ============================================================================

def print_error_summary(gt_arr: np.ndarray, pred_arr: np.ndarray) -> None:
    """Print detailed error statistics."""
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: gt {gt_arr.shape} vs pred {pred_arr.shape}")

    err = (pred_arr - gt_arr).astype(np.float64)
    abs_err = np.abs(err)
    mse = err ** 2

    mae_per_dim = abs_err.mean(axis=0)
    rmse_per_dim = np.sqrt(mse.mean(axis=0))
    max_abs_per_dim = abs_err.max(axis=0)

    print("\n" + "=" * 60)
    print("ERROR SUMMARY (pred - gt)")
    print("=" * 60)
    print(f"Steps: {gt_arr.shape[0]}  |  Dims: {gt_arr.shape[1]}")
    print(f"Mean MAE:  {float(mae_per_dim.mean()):.6f}")
    print(f"Mean RMSE: {float(rmse_per_dim.mean()):.6f}")
    print(f"Max |err|: {float(abs_err.max()):.6f}")

    print("\nPer-dimension errors:")
    print(f"{'Dim':<6} {'MAE':<12} {'RMSE':<12} {'Max|err|':<12}")
    print("-" * 48)
    for d in range(gt_arr.shape[1]):
        print(
            f"{d:<6} {mae_per_dim[d]:<12.6f} "
            f"{rmse_per_dim[d]:<12.6f} {max_abs_per_dim[d]:<12.6f}"
        )


def plot_debug(gt: np.ndarray, pred: np.ndarray, out_dir: str):
    """Generate debug plots comparing GT and predicted actions."""
    os.makedirs(out_dir, exist_ok=True)
    T, D = gt.shape
    x = np.arange(T)

    # Per-dimension plots
    for d in range(D):
        plt.figure(figsize=(10, 6))
        plt.plot(x, gt[:, d], 'b-', label='Ground Truth', linewidth=2)
        plt.plot(x, pred[:, d], 'r--', label='Predicted', linewidth=2)
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel(f'Action Dim {d}', fontsize=12)
        plt.title(f'Action Dimension {d}: GT vs Predicted', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"action_dim{d:02d}.png"), dpi=100)
        plt.close()

    # L2 error plot
    l2_err = np.linalg.norm(pred - gt, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, l2_err, 'r-', linewidth=2)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('L2 Error', fontsize=12)
    plt.title('Per-step L2 Error: ||pred - gt||₂', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "action_l2_error.png"), dpi=100)
    plt.close()

    print(f"✅ Saved {D + 1} debug plots to: {out_dir}")


def save_images(
    base_frames: List[np.ndarray],
    wrist_frames: List[np.ndarray],
    out_dir: str,
    stride: int,
    max_frames: int,
):
    """Save sample frames from episode."""
    os.makedirs(out_dir, exist_ok=True)
    T = min(len(base_frames), len(wrist_frames))
    idxs = list(range(0, T, max(1, stride)))[:max_frames]
    
    for t in idxs:
        Image.fromarray(base_frames[t]).save(
            os.path.join(out_dir, f"base_t{t:04d}.png")
        )
        Image.fromarray(wrist_frames[t]).save(
            os.path.join(out_dir, f"wrist_t{t:04d}.png")
        )
    
    print(f"✅ Saved {len(idxs)} image pairs to: {out_dir}")


# ============================================================================
# Episode evaluation
# ============================================================================

def evaluate_episode(
    vla,
    processor,
    steps_list: List[Dict],
    device: str,
    torch_dtype: torch.dtype,
    unnorm_key: str,
    max_steps: int,
    print_first_k: int,
    save_imgs: bool,
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Evaluate model on a single episode.
    
    Returns:
        gt_arr: Ground truth actions (T, D)
        pred_arr: Predicted actions (T, D)
        base_frames: List of base RGB frames
        wrist_frames: List of wrist RGB frames
    """
    # Get instruction from first step
    instruction = decode_text(steps_list[0]["language_instruction"])
    prompt = make_prompt(instruction)
    print(f"\n📋 Instruction: {instruction}")

    gt_list = []
    pred_list = []
    base_frames = []
    wrist_frames = []

    verbose_accept = True
    T = min(len(steps_list), max_steps)

    for t in range(T):
        step = steps_list[t]
        obs = step["observation"]

        # Get images
        base_np = uint8_image(obs["base_rgb"])
        wrist_np = uint8_image(obs["hand_rgb"])

        if save_imgs:
            base_frames.append(base_np)
            wrist_frames.append(wrist_np)

        # Get state from qpos (9D -> 8D)
        if "qpos" not in obs:
            raise KeyError(f"'qpos' not in observation keys: {list(obs.keys())}")
        state = pack_state_from_qpos(obs["qpos"])  # (8,)

        # Get ground truth action
        gt = np.asarray(step["action"], dtype=np.float32).reshape(-1)

        # Build packed inputs
        inputs = build_inputs_packed_base_wrist(
            processor,
            prompt,
            Image.fromarray(base_np),
            Image.fromarray(wrist_np),
            device,
            torch_dtype,
        )

        # Print shapes on first step
        if t == 0:
            pv = inputs["pixel_values"]
            print(f"\n🔍 First step shapes:")
            print(f"  pixel_values: {tuple(pv.shape)} (expected: (1, 6, H, W))")
            print(f"  state: {state.shape} (expected: (8,))")
            print(f"  action: {gt.shape}")
            summarize_vec("  state", state)
            summarize_vec("  gt_action", gt)

        # Predict action
        with torch.no_grad():
            pred_raw = predict_action_with_proprio(
                vla=vla,
                inputs=inputs,
                proprio_np=state,
                device=device,
                unnorm_key=unnorm_key,
                verbose=verbose_accept,
            )
        verbose_accept = False

        pred = to_numpy_action(pred_raw)

        gt_list.append(gt)
        pred_list.append(pred)

        # Print first few steps
        if t < print_first_k:
            print(f"\n[t={t}]")
            summarize_vec("  gt", gt)
            summarize_vec("  pred", pred)
            print(f"  gt[:8]:   {np.round(gt[:8], 3)}")
            print(f"  pred[:8]: {np.round(pred[:8], 3)}")

    gt_arr = np.stack(gt_list, axis=0)
    pred_arr = np.stack(pred_list, axis=0)

    return gt_arr, pred_arr, base_frames, wrist_frames


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenVLA-OFT model on Lohrbench RLDS dataset"
    )
    
    # Model and data
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--builder_dir", required=True, help="Path to TFDS builder directory")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--episode_index", type=int, default=0, help="Starting episode index")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to evaluate")
    
    # Keys (raw TFDS keys before transform)
    parser.add_argument("--base_key", default="base_rgb", help="Base camera key in TFDS")
    parser.add_argument("--wrist_key", default="hand_rgb", help="Wrist camera key in TFDS")
    parser.add_argument("--qpos_key", default="qpos", help="Joint position key in TFDS")
    
    # Model settings
    parser.add_argument("--unnorm_key", default="lohrbench_rlds", help="Unnormalization key")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_impl", default="sdpa", help="Attention implementation")
    
    # Evaluation settings
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--print_first_k", type=int, default=5, help="Print first K steps")
    
    # Output
    parser.add_argument("--out_dir", default="./eval_results", help="Output directory")
    parser.add_argument("--save_images", action="store_true", help="Save sample images")
    parser.add_argument("--img_stride", type=int, default=5, help="Image save stride")
    parser.add_argument("--img_max_frames", type=int, default=30, help="Max frames to save")

    args = parser.parse_args()

    # Setup
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("OpenVLA-OFT Evaluation on Lohrbench RLDS")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Dataset: {args.builder_dir}")
    print(f"Split: {args.split}")
    print(f"Episodes: {args.episode_index} to {args.episode_index + args.num_episodes - 1}")
    print(f"Device: {args.device} | Dtype: {args.dtype}")
    print(f"Output: {args.out_dir}")
    print("=" * 80)

    # Load dataset
    print("\n📂 Loading dataset...")
    builder = tfds.builder_from_directory(args.builder_dir)
    ds = builder.as_dataset(split=args.split, shuffle_files=False)

    # Load model
    print(f"\n🤖 Loading model from {args.ckpt}...")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
    ).to(args.device)
    vla.eval()

    # Critical: set num_images_in_input=1 for packed 6-channel observation
    ok = try_set_num_images_in_input(vla, 1)
    print(f"✓ set_num_images_in_input(1): {ok}")

    # Evaluate episodes
    all_gt = []
    all_pred = []

    for ep_idx in range(args.episode_index, args.episode_index + args.num_episodes):
        print(f"\n{'=' * 80}")
        print(f"Episode {ep_idx}")
        print(f"{'=' * 80}")

        # Load episode
        ep = next(iter(ds.skip(ep_idx).take(1)))
        steps_list = list(ep["steps"].as_numpy_iterator())

        if len(steps_list) < 2:
            print(f"⚠ Episode {ep_idx} too short ({len(steps_list)} steps), skipping")
            continue

        # Drop last dummy step (matches transform)
        steps_list = steps_list[:-1]
        print(f"📊 Episode length: {len(steps_list)} steps (after dropping last)")

        # Evaluate
        gt_arr, pred_arr, base_frames, wrist_frames = evaluate_episode(
            vla=vla,
            processor=processor,
            steps_list=steps_list,
            device=args.device,
            torch_dtype=torch_dtype,
            unnorm_key=args.unnorm_key,
            max_steps=args.max_steps,
            print_first_k=args.print_first_k,
            save_imgs=args.save_images,
        )

        all_gt.append(gt_arr)
        all_pred.append(pred_arr)

        # Per-episode summary
        print(f"\n📈 Episode {ep_idx} summary:")
        summarize_vec("  GT actions", gt_arr)
        summarize_vec("  Predicted actions", pred_arr)
        print_error_summary(gt_arr, pred_arr)

        # Save episode results
        ep_dir = os.path.join(args.out_dir, f"episode_{ep_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)

        np.save(os.path.join(ep_dir, "gt.npy"), gt_arr)
        np.save(os.path.join(ep_dir, "pred.npy"), pred_arr)

        plot_dir = os.path.join(ep_dir, "plots")
        plot_debug(gt_arr, pred_arr, plot_dir)

        if args.save_images and len(base_frames) > 0:
            img_dir = os.path.join(ep_dir, "images")
            save_images(
                base_frames,
                wrist_frames,
                img_dir,
                stride=args.img_stride,
                max_frames=args.img_max_frames,
            )

    # Aggregate results
    if len(all_gt) > 0:
        print(f"\n{'=' * 80}")
        print(f"AGGREGATE RESULTS ({len(all_gt)} episodes)")
        print(f"{'=' * 80}")

        all_gt_concat = np.concatenate(all_gt, axis=0)
        all_pred_concat = np.concatenate(all_pred, axis=0)

        summarize_vec("All GT actions", all_gt_concat)
        summarize_vec("All predicted actions", all_pred_concat)
        print_error_summary(all_gt_concat, all_pred_concat)

        np.save(os.path.join(args.out_dir, "all_gt.npy"), all_gt_concat)
        np.save(os.path.join(args.out_dir, "all_pred.npy"), all_pred_concat)

    print(f"\n✅ Evaluation complete! Results saved to: {args.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
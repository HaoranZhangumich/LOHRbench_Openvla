#!/usr/bin/env python3
"""
evaluate_openvla_debug.py

Evaluate OpenVLA(-OFT) checkpoint on Lohrbench RLDS TFDS builder directory.

Key behavior:
- If --num_images_in_input 1:
    pixel_values = base image only -> (1,3,H,W)
- If --num_images_in_input 2:
    pixel_values = concat(base,wrist) -> (1,6,H,W)
    optional --blind_wrist will zero the wrist half.

Example (trained on 1 image):
CUDA_VISIBLE_DEVICES=0 python evaluate_openvla.py \
  --ckpt /path/to/ckpt \
  --builder_dir /path/to/tfds/builder_dir \
  --split train --episode_index 0 --num_episodes 1 \
  --num_images_in_input 1 \
  --max_steps 200

Example (trained on 2 images):
CUDA_VISIBLE_DEVICES=0 python evaluate_openvla.py \
  --ckpt /path/to/ckpt \
  --builder_dir /path/to/tfds/builder_dir \
  --split train --episode_index 0 --num_episodes 1 \
  --num_images_in_input 2 \
  --max_steps 200
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoModelForVision2Seq, AutoProcessor


# ------------------------- prompt -------------------------
def make_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction}?\nOut:"


# ------------------------- decode helpers -------------------------
def _decode_any_to_str(x: Any) -> str:
    """
    TFDS as_numpy_iterator yields bytes for tf.string usually.
    Handle bytes / np.bytes_ / scalar numpy / python str.
    """
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray) and x.dtype.type is np.bytes_ and x.shape == ():
        return x.tobytes().decode("utf-8", errors="replace")
    return str(x)


def _ensure_uint8_hwc(img: Any) -> np.ndarray:
    """
    Accept common shapes:
      - (H,W,3)
      - (1,H,W,3)
      - (3,H,W)
    Convert to contiguous uint8 (H,W,3).
    """
    arr = np.asarray(img)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected image (H,W,3), got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        # if normalized, scale
        if float(arr.max()) <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def pack_state_from_qpos(qpos_9: Any) -> np.ndarray:
    """
    qpos (9,) -> state (8,) = [qpos[:7], mean(qpos[7:9])]
    """
    q = np.asarray(qpos_9, dtype=np.float32).reshape(-1)
    if q.shape[0] < 9:
        raise ValueError(f"Expected qpos dim>=9, got {q.shape}")
    joints7 = q[:7]
    gripper1 = np.mean(q[7:9], keepdims=True)
    return np.concatenate([joints7, gripper1], axis=0).astype(np.float32)  # (8,)


# ------------------------- model helpers -------------------------
def try_set_num_images_in_input(vla, n: int) -> bool:
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


def _reshape_pixel_values_to_bchw(pv: torch.Tensor) -> torch.Tensor:
    """
    HF processors sometimes return:
      - (B,C,H,W)
      - (B,N,3,H,W)
    Convert to (B,C,H,W) with C=3N if needed.
    """
    if pv.ndim == 4:
        return pv
    if pv.ndim == 5:
        b, n, c, h, w = pv.shape
        return pv.reshape(b, n * c, h, w)
    raise ValueError(f"Unexpected pixel_values shape {tuple(pv.shape)}")


def build_inputs(
    processor,
    prompt: str,
    base_pil: Image.Image,
    wrist_pil: Optional[Image.Image],
    *,
    device: str,
    dtype: torch.dtype,
    num_images_in_input: int,
    blind_wrist: bool,
) -> Dict[str, torch.Tensor]:
    """
    num_images_in_input=1 -> (1,3,H,W) base only
    num_images_in_input=2 -> (1,6,H,W) base+wrist (or wrist=0 if blind_wrist)
    """
    text = processor.tokenizer(prompt, return_tensors="pt")
    input_ids = text["input_ids"].to(device)
    attention_mask = text.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    pv_base = processor.image_processor(base_pil, return_tensors="pt")["pixel_values"]
    pv_base = _reshape_pixel_values_to_bchw(pv_base)  # (1,3,H,W) typically
    base3 = pv_base[:, :3]

    if num_images_in_input == 1:
        pixel_values = base3.to(device=device, dtype=dtype)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}

    if num_images_in_input == 2:
        if wrist_pil is None:
            raise ValueError("num_images_in_input=2 but wrist_pil is None")

        if blind_wrist:
            wrist3 = torch.zeros_like(base3)
        else:
            pv_wrist = processor.image_processor(wrist_pil, return_tensors="pt")["pixel_values"]
            pv_wrist = _reshape_pixel_values_to_bchw(pv_wrist)
            wrist3 = pv_wrist[:, :3]

        pixel_values = torch.cat([base3, wrist3], dim=1).to(device=device, dtype=dtype)  # (1,6,H,W)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}

    raise ValueError(f"Unsupported num_images_in_input={num_images_in_input}")


def to_numpy_action(pred) -> np.ndarray:
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
    if pred.ndim >= 2:
        pred = pred[0]
    return pred.astype(np.float32).reshape(-1)


def predict_action_with_proprio(
    vla,
    inputs: Dict[str, torch.Tensor],
    proprio_np: Optional[np.ndarray],
    device: str,
    unnorm_key: Optional[str],
):
    kwargs = dict(inputs)
    if proprio_np is not None:
        proprio_t = torch.from_numpy(np.asarray(proprio_np, np.float32).reshape(1, -1)).to(device)
        for k in ["proprio", "state", "robot_state"]:
            try:
                return vla.predict_action(**kwargs, **{k: proprio_t}, unnorm_key=unnorm_key, do_sample=False)
            except TypeError:
                pass
    return vla.predict_action(**kwargs, unnorm_key=unnorm_key, do_sample=False)


# ------------------------- evaluation -------------------------
def eval_one_episode(
    vla,
    processor,
    steps_list: List[Dict[str, Any]],
    *,
    device: str,
    dtype: torch.dtype,
    unnorm_key: str,
    num_images_in_input: int,
    blind_wrist: bool,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    # instruction: try step["language_instruction"], else empty
    instr_raw = steps_list[0].get("language_instruction", b"")
    instruction = _decode_any_to_str(instr_raw)
    prompt = make_prompt(instruction)

    gt_list, pred_list = [], []

    T = min(len(steps_list), max_steps)
    for t in range(T):
        step = steps_list[t]
        obs = step["observation"]

        base_np = _ensure_uint8_hwc(obs["base_rgb"])
        wrist_np = _ensure_uint8_hwc(obs["hand_rgb"])

        qpos = obs["qpos"]
        state8 = pack_state_from_qpos(qpos)  # (8,)

        gt = np.asarray(step["action"], dtype=np.float32).reshape(-1)

        inputs = build_inputs(
            processor,
            prompt,
            Image.fromarray(base_np),
            Image.fromarray(wrist_np),
            device=device,
            dtype=dtype,
            num_images_in_input=num_images_in_input,
            blind_wrist=blind_wrist,
        )

        if t == 0:
            pv = inputs["pixel_values"]
            print(f"[debug] instruction: {instruction}")
            print(f"[debug] num_images_in_input={num_images_in_input} blind_wrist={blind_wrist}")
            print(f"[debug] pixel_values shape = {tuple(pv.shape)}")
            print(f"[debug] state8 shape = {state8.shape}  gt shape = {gt.shape}")

        with torch.inference_mode():
            pred_raw = predict_action_with_proprio(
                vla=vla,
                inputs=inputs,
                proprio_np=state8,
                device=device,
                unnorm_key=unnorm_key,
            )
        pred = to_numpy_action(pred_raw)

        gt_list.append(gt)
        pred_list.append(pred)

    return np.stack(gt_list, 0), np.stack(pred_list, 0), instruction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--builder_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episode_index", type=int, default=0)
    ap.add_argument("--num_episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=200)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--unnorm_key", default="lohrbench_rlds")

    # IMPORTANT: this must match training
    ap.add_argument("--num_images_in_input", type=int, choices=[1, 2], required=True)
    ap.add_argument("--blind_wrist", action="store_true", help="Only meaningful when num_images_in_input=2")

    ap.add_argument("--out_dir", default="./eval_results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] builder_dir={args.builder_dir} split={args.split}")
    print(f"[eval] num_images_in_input={args.num_images_in_input} blind_wrist={args.blind_wrist}")
    print(f"[eval] device={args.device} dtype={args.dtype}")

    # Load dataset (raw TFDS episode dict)
    builder = tfds.builder_from_directory(args.builder_dir)
    ds = builder.as_dataset(split=args.split, shuffle_files=False)

    # Load model
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
    ).to(args.device)
    vla.eval()

    ok = try_set_num_images_in_input(vla, args.num_images_in_input)
    print(f"[eval] set_num_images_in_input({args.num_images_in_input}) -> {ok}")

    all_gt, all_pred = [], []

    for ep_idx in range(args.episode_index, args.episode_index + args.num_episodes):
        ep = next(iter(ds.skip(ep_idx).take(1)))

        # steps are a tf.data.Dataset; convert to numpy dicts
        steps_list = list(ep["steps"].as_numpy_iterator())
        if len(steps_list) < 2:
            print(f"[eval] episode {ep_idx} too short, skipping")
            continue

        # Many RLDS converters append a dummy last step; if yours does, drop it.
        # If you are sure you don't have dummy terminal step, comment this out.
        steps_list = steps_list[:-1]

        gt, pred, instruction = eval_one_episode(
            vla=vla,
            processor=processor,
            steps_list=steps_list,
            device=args.device,
            dtype=torch_dtype,
            unnorm_key=args.unnorm_key,
            num_images_in_input=args.num_images_in_input,
            blind_wrist=args.blind_wrist,
            max_steps=args.max_steps,
        )

        all_gt.append(gt)
        all_pred.append(pred)

        out_ep = os.path.join(args.out_dir, f"episode_{ep_idx:04d}")
        os.makedirs(out_ep, exist_ok=True)
        np.save(os.path.join(out_ep, "gt.npy"), gt)
        np.save(os.path.join(out_ep, "pred.npy"), pred)
        with open(os.path.join(out_ep, "instruction.txt"), "w", encoding="utf-8") as f:
            f.write(instruction + "\n")

        # quick summary
        err = pred - gt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        print(f"[eval] episode {ep_idx}: T={gt.shape[0]} D={gt.shape[1]}  MAE={mae:.6f} RMSE={rmse:.6f}")

    if all_gt:
        gt_all = np.concatenate(all_gt, 0)
        pred_all = np.concatenate(all_pred, 0)
        err = pred_all - gt_all
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        print(f"[eval] ALL: N={gt_all.shape[0]} D={gt_all.shape[1]}  MAE={mae:.6f} RMSE={rmse:.6f}")
        np.save(os.path.join(args.out_dir, "all_gt.npy"), gt_all)
        np.save(os.path.join(args.out_dir, "all_pred.npy"), pred_all)

    print(f"[eval] done. saved to {args.out_dir}")


if __name__ == "__main__":
    main()

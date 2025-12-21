#!/usr/bin/env python3
"""
openvla_server.py

Fixed OpenVLA policy TCP server using JSON for numpy-version compatibility.

CUDA_VISIBLE_DEVICES=5 python openvla_server.py   --ckpt /home/users/haoran/runs/openvla_oft_lohrbench/openvla-7b+lohrbench_rlds+b16+lr-0.0005+lora-r32+dropout-0.0--lohrbench_lora_r32_bs1x2_ga16--30000_chkpt   --host 127.0.0.1   --port 5555   --device cuda:0   --dtype bf16   --attn_impl sdpa
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import json
import base64
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


# ------------------------- wire protocol (JSON-based) -------------------------
def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving.")
        buf += chunk
    return buf


def recv_msg(conn: socket.socket) -> Any:
    """Receive JSON message with numpy arrays encoded as base64"""
    header = _recv_exact(conn, 8)
    (n,) = struct.unpack("!Q", header)
    payload = _recv_exact(conn, n)
    msg_dict = json.loads(payload.decode('utf-8'))
    
    # Decode numpy arrays from base64
    if 'base_rgb' in msg_dict and isinstance(msg_dict['base_rgb'], dict):
        msg_dict['base_rgb'] = _decode_numpy(msg_dict['base_rgb'])
    if 'wrist_rgb' in msg_dict and isinstance(msg_dict['wrist_rgb'], dict):
        msg_dict['wrist_rgb'] = _decode_numpy(msg_dict['wrist_rgb'])
    if 'qpos' in msg_dict and isinstance(msg_dict['qpos'], dict):
        msg_dict['qpos'] = _decode_numpy(msg_dict['qpos'])
    
    return msg_dict


def send_msg(conn: socket.socket, obj: Any) -> None:
    """Send JSON message with numpy arrays encoded as base64"""
    msg_dict = dict(obj)
    
    # Encode numpy arrays to base64
    if 'pred_action' in msg_dict:
        msg_dict['pred_action'] = _encode_numpy(msg_dict['pred_action'])
    
    payload = json.dumps(msg_dict).encode('utf-8')
    header = struct.pack("!Q", len(payload))
    conn.sendall(header + payload)


def _encode_numpy(arr: np.ndarray) -> Dict:
    """Encode numpy array to JSON-serializable dict"""
    return {
        '__numpy__': True,
        'data': base64.b64encode(arr.tobytes()).decode('ascii'),
        'dtype': str(arr.dtype),
        'shape': arr.shape
    }


def _decode_numpy(d: Dict) -> np.ndarray:
    """Decode numpy array from JSON dict"""
    if not d.get('__numpy__'):
        raise ValueError("Not a numpy-encoded dict")
    data = base64.b64decode(d['data'])
    arr = np.frombuffer(data, dtype=d['dtype'])
    return arr.reshape(d['shape'])


# ------------------------- conversion helpers -------------------------
def _get_by_path(d: Any, path: str) -> Any:
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict):
            raise KeyError(f"path '{path}' failed at '{k}', current type={type(cur)}")
        if k not in cur:
            raise KeyError(f"missing key '{k}' in {list(cur.keys())}")
        cur = cur[k]
    return cur


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_hwc_uint8(img: Any) -> np.ndarray:
    arr = _to_numpy(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(arr.max()) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _ensure_qpos_9(qpos: Any) -> np.ndarray:
    arr = _to_numpy(qpos).astype(np.float32).reshape(-1)
    if arr.shape[0] < 9:
        raise ValueError(f"Expected qpos dim>=9, got {arr.shape}")
    return arr[:9]


# ------------------------- OpenVLA helpers -------------------------
def make_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction}?\nOut:"


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


def pack_state_from_qpos(qpos_9: np.ndarray) -> np.ndarray:
    qpos_9 = np.asarray(qpos_9, dtype=np.float32).reshape(-1)
    joints7 = qpos_9[:7]
    gripper1 = np.mean(qpos_9[7:9], keepdims=True)
    return np.concatenate([joints7, gripper1], axis=0).astype(np.float32)


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


def build_inputs_two_view_packed6(
    processor, prompt: str, base_img: Image.Image, wrist_img: Image.Image,
    device: str, dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    text = processor.tokenizer(prompt, return_tensors="pt")
    input_ids = text["input_ids"].to(device)
    attention_mask = text.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    pv_base = processor.image_processor(base_img, return_tensors="pt")["pixel_values"]
    pv_wrist = processor.image_processor(wrist_img, return_tensors="pt")["pixel_values"]

    if pv_base.ndim == 5:
        b, n, c, h, w = pv_base.shape
        pv_base = pv_base.reshape(b, n * c, h, w)
    if pv_wrist.ndim == 5:
        b, n, c, h, w = pv_wrist.shape
        pv_wrist = pv_wrist.reshape(b, n * c, h, w)

    base3 = pv_base[:, :3]
    wrist3 = pv_wrist[:, :3]
    pixel_values = torch.cat([base3, wrist3], dim=1).to(device=device, dtype=dtype)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}


def predict_action_with_proprio(vla, inputs, proprio_np, device: str, unnorm_key):
    kwargs = dict(inputs)
    if proprio_np is not None:
        proprio_t = torch.from_numpy(np.asarray(proprio_np, dtype=np.float32).reshape(1, -1)).to(device)
        try:
            return vla.predict_action(**kwargs, proprio=proprio_t, unnorm_key=unnorm_key, do_sample=False)
        except TypeError:
            for k in ["state", "robot_state"]:
                try:
                    return vla.predict_action(**kwargs, **{k: proprio_t}, unnorm_key=unnorm_key, do_sample=False)
                except TypeError:
                    pass
    return vla.predict_action(**kwargs, unnorm_key=unnorm_key, do_sample=False)


# ------------------------- server -------------------------
class OpenVLAPolicyServer:
    def __init__(self, ckpt: str, device: str, dtype_str: str, attn_impl: str):
        self.device = device
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
        self.request_count = 0

        print(f"[Server] Loading processor from: {ckpt}", flush=True)
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

        print(f"[Server] Loading model from: {ckpt}", flush=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            ckpt,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            attn_implementation=attn_impl,
        ).to(self.device)
        self.vla.eval()

        ok = try_set_num_images_in_input(self.vla, 1)
        print(f"[Server] set_num_images_in_input(1): {ok}", flush=True)
        
        self._predict_lock = threading.Lock()
        print("[Server] ✅ Model loaded and ready!", flush=True)

    @torch.inference_mode()
    def handle_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        self.request_count += 1
        
        # Handle ping/status requests
        if req.get("_type") == "ping":
            return {"status": "ok", "request_count": self.request_count}
        
        # Handle close requests
        if req.get("_type") == "close":
            return {"ok": True}
        
        instruction = str(req["instruction"])
        unnorm_key = req.get("unnorm_key", None)

        # Extract observations
        if "base_rgb" in req and "wrist_rgb" in req and "qpos" in req:
            base_rgb = _ensure_hwc_uint8(req["base_rgb"])
            wrist_rgb = _ensure_hwc_uint8(req["wrist_rgb"])
            qpos = _ensure_qpos_9(req["qpos"])
        elif "env_obs" in req:
            env_obs = req["env_obs"]
            base_path = req.get("base_path", "sensor_data.base_camera.rgb")
            wrist_path = req.get("wrist_path", "sensor_data.hand_camera.rgb")
            qpos_path = req.get("qpos_path", "agent.qpos")
            base_rgb = _ensure_hwc_uint8(_get_by_path(env_obs, base_path))
            wrist_rgb = _ensure_hwc_uint8(_get_by_path(env_obs, wrist_path))
            qpos = _ensure_qpos_9(_get_by_path(env_obs, qpos_path))
        else:
            raise KeyError("Request must contain (base_rgb,wrist_rgb,qpos) or env_obs")

        prompt = make_prompt(instruction)
        base_pil = Image.fromarray(base_rgb)
        wrist_pil = Image.fromarray(wrist_rgb)

        inputs = build_inputs_two_view_packed6(
            self.processor, prompt, base_pil, wrist_pil, self.device, self.dtype
        )
        state8 = pack_state_from_qpos(qpos)

        with self._predict_lock:
            pred_raw = predict_action_with_proprio(
                self.vla, inputs, state8, self.device, unnorm_key
            )

        pred = to_numpy_action(pred_raw)
        return {"pred_action": pred}

    def serve_forever(self, host: str, port: int):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            srv.bind((host, port))
            srv.listen(8)
            print(f"\n{'='*60}")
            print(f"[Server] 🚀 READY - Listening on {host}:{port}")
            print(f"{'='*60}\n")
            print(f"[Server] Using JSON protocol (numpy-version compatible)")
            print(f"[Server] Waiting for client connections...")
            print(f"[Server] Press Ctrl+C to stop\n")
        except OSError as e:
            print(f"[Server] ❌ Failed to bind to {host}:{port}")
            print(f"[Server] Error: {e}")
            print(f"[Server] Make sure the port is not already in use")
            return

        def client_thread(conn: socket.socket, addr):
            print(f"[Server] ✓ Client connected: {addr}", flush=True)
            try:
                while True:
                    req = recv_msg(conn)
                    out = self.handle_request(req)
                    send_msg(conn, out)
                    
                    if req.get("_type") == "close":
                        break
                    
                    if self.request_count % 100 == 0:
                        print(f"[Server] Processed {self.request_count} requests", flush=True)
            except Exception as e:
                print(f"[Server] Client {addr} error: {e}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                print(f"[Server] Client disconnected: {addr}", flush=True)

        try:
            while True:
                conn, addr = srv.accept()
                threading.Thread(target=client_thread, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            print("\n[Server] Shutting down...")
        finally:
            srv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to OpenVLA checkpoint")
    ap.add_argument("--host", default="0.0.0.0", help="Host to bind (0.0.0.0 for all interfaces)")
    ap.add_argument("--port", type=int, default=5555, help="Port to listen on")
    ap.add_argument("--device", default="cuda:0", help="Device for model inference")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--attn_impl", default="sdpa", help="Attention implementation")
    args = ap.parse_args()

    print(f"\n[Server] Starting OpenVLA Policy Server")
    print(f"[Server] Checkpoint: {args.ckpt}")
    print(f"[Server] Device: {args.device}")
    print(f"[Server] Dtype: {args.dtype}\n")

    server = OpenVLAPolicyServer(
        ckpt=args.ckpt,
        device=args.device,
        dtype_str=args.dtype,
        attn_impl=args.attn_impl,
    )
    server.serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()
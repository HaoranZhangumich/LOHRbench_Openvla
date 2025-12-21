#!/usr/bin/env python3
"""
lohrbench_client.py (fixed with JSON protocol)

Run LoHRbench/TAMPBench motion control using an OpenVLA policy served via TCP.
"""

from __future__ import annotations

import argparse
import socket
import struct
import json
import base64
import time
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch

from tampbench.env.tamp_env import TAMPEnvironment


# ------------------------- task instructions -------------------------
TASK_INSTRUCTIONS = {
    "reverse_stack": "reverse 10 stacked cube in reverse order",
    "stack_10_cube": "stack 10 cube together, start with red cube",
    "stack_cube_clutter": "stack 3 cube together , start with red cube",
    "cluttered_packing": "put three cube in to the bowl",
    "pick_active_exploration": "pick up the can, screwdriver and cup out of the drawer",
    "stack_active_exploration": "pick up the cube and stack them together, start with red cube",
    "fruit_placement": "place four starberries into the target position",
    "repackage": "put cube into the bowl and stack the bowl on the plate",
}


# ------------------------- socket helpers (JSON-based) -------------------------
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
    if 'pred_action' in msg_dict and isinstance(msg_dict['pred_action'], dict):
        msg_dict['pred_action'] = _decode_numpy(msg_dict['pred_action'])
    
    return msg_dict


def send_msg(conn: socket.socket, obj: Any) -> None:
    """Send JSON message with numpy arrays encoded as base64"""
    msg_dict = dict(obj)
    
    # Encode numpy arrays to base64
    if 'base_rgb' in msg_dict:
        msg_dict['base_rgb'] = _encode_numpy(msg_dict['base_rgb'])
    if 'wrist_rgb' in msg_dict:
        msg_dict['wrist_rgb'] = _encode_numpy(msg_dict['wrist_rgb'])
    if 'qpos' in msg_dict:
        msg_dict['qpos'] = _encode_numpy(msg_dict['qpos'])
    
    payload = json.dumps(msg_dict).encode('utf-8')
    header = struct.pack("!Q", len(payload))
    conn.sendall(header + payload)


def _encode_numpy(arr: np.ndarray) -> Dict:
    """Encode numpy array to JSON-serializable dict"""
    arr = np.asarray(arr)
    return {
        '__numpy__': True,
        'data': base64.b64encode(arr.tobytes()).decode('ascii'),
        'dtype': str(arr.dtype),
        'shape': list(arr.shape)
    }


def _decode_numpy(d: Dict) -> np.ndarray:
    """Decode numpy array from JSON dict"""
    if not d.get('__numpy__'):
        raise ValueError("Not a numpy-encoded dict")
    data = base64.b64decode(d['data'])
    arr = np.frombuffer(data, dtype=d['dtype'])
    return arr.reshape(d['shape'])


def connect_with_retry(host: str, port: int, max_retries: int = 10, retry_delay: float = 2.0) -> socket.socket:
    """
    Try to connect to server with retries.
    Useful when server takes time to start up.
    """
    print(f"\n🔌 Connecting to policy server at {host}:{port}")
    
    for attempt in range(1, max_retries + 1):
        try:
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.settimeout(5.0)  # 5 second timeout for connection
            conn.connect((host, port))
            conn.settimeout(None)  # Remove timeout after connection
            print(f"✅ Connected on attempt {attempt}/{max_retries}\n")
            return conn
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            if attempt < max_retries:
                print(f"❌ Attempt {attempt}/{max_retries} failed: {e}")
                print(f"   Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"\n❌ Failed to connect after {max_retries} attempts")
                print("\nTroubleshooting:")
                print("1. Is the server running? Start it first:")
                print(f"   CUDA_VISIBLE_DEVICES=0 python openvla_server.py --ckpt /path/to/ckpt --host 0.0.0.0 --port {port}")
                print("2. Check if port is open:")
                print(f"   netstat -tuln | grep {port}")
                print("3. Check firewall settings")
                raise


# ------------------------- benchmark dir helpers -------------------------
def _default_root() -> Path:
    return Path(__file__).resolve().parents[1] / "TAMP" / "LOHRbench" / "TAMPBench" / "benchmark" / "table-top"


def _resolve_benchmark(args) -> Path:
    if args.benchmark:
        return Path(args.benchmark).resolve()
    if not (args.type and args.task):
        raise ValueError("--type and --task required when --benchmark absent")
    root = Path(args.root).resolve() if args.root else _default_root()
    return (root / args.type.lower() / args.task).resolve()


def _detect_problem(dir_path: Path) -> Path:
    cfg_dir = dir_path / "config"
    for where in [cfg_dir, dir_path]:
        if where.is_dir():
            hits = sorted(where.glob("*problem.pddl"))
            if len(hits) == 1:
                return hits[0]
    raise FileNotFoundError(f"No *problem.pddl found under {dir_path} or {dir_path/'config'}")


def _detect_domain(dir_path: Path) -> Path:
    cfg_dir = dir_path / "config"
    for where in [cfg_dir, dir_path]:
        if where.is_dir():
            hits = sorted(where.glob("*domain.pddl"))
            if len(hits) == 1:
                return hits[0]
    legacy = "table_top_default_domain.pddl"
    cur = dir_path.resolve()
    while True:
        cand = cur / legacy
        if cand.exists():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(f"No *domain.pddl found under {dir_path} (or legacy {legacy})")


def _detect_positions(dir_path: Path) -> Path:
    cfg_dir = dir_path / "config"
    if not cfg_dir.is_dir():
        raise FileNotFoundError("Missing config directory in " + str(dir_path))
    hits = sorted(cfg_dir.glob("*position*.*"))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"No position file under {cfg_dir}")
    raise FileExistsError(f"Multiple position files under {cfg_dir}: {hits}")


# ------------------------- obs extraction helpers -------------------------
def _unwrap_reset_or_step(x: Any) -> Any:
    if isinstance(x, tuple) and len(x) >= 1:
        return x[0]
    return x


def _get_by_path(d: Any, path: str) -> Any:
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict):
            raise KeyError(f"path '{path}' failed at '{k}', current type={type(cur)}")
        if k not in cur:
            raise KeyError(f"missing key '{k}' while resolving '{path}', keys={list(cur.keys())}")
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
        raise ValueError(f"Expected image shape (H,W,3); got {arr.shape}")
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


def extract_obs(
    env_obs: Any,
    *,
    base_path: str,
    wrist_path: str,
    qpos_path: str,
    instruction: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    obs = _unwrap_reset_or_step(env_obs)
    if not isinstance(obs, dict):
        raise TypeError(f"env_obs must be dict-like; got {type(obs)}")

    base_raw = _get_by_path(obs, base_path)
    wrist_raw = _get_by_path(obs, wrist_path)
    qpos_raw = _get_by_path(obs, qpos_path)

    base_rgb = _ensure_hwc_uint8(base_raw)
    wrist_rgb = _ensure_hwc_uint8(wrist_raw)
    qpos = _ensure_qpos_9(qpos_raw)

    return base_rgb, wrist_rgb, qpos, instruction


# ------------------------- stepping motion -------------------------
def _normalize_step_output(step_out: Any):
    if not isinstance(step_out, tuple):
        raise TypeError(f"env.step(...) must return tuple; got {type(step_out)}")

    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, reward, bool(done), info

    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated) or bool(truncated)
        return obs, reward, done, info

    raise ValueError(f"Unexpected env.step return length={len(step_out)}; expected 4 or 5")


def step_motion(env: TAMPEnvironment, action_8d: np.ndarray):
    a = np.asarray(action_8d, dtype=np.float32).reshape(-1)

    try:
        out = env.step(a)
        return _normalize_step_output(out)
    except TypeError:
        pass

    if hasattr(env, "task_env") and hasattr(env.task_env, "motion_env"):
        out = env.task_env.motion_env.step(a)
        return _normalize_step_output(out)

    if hasattr(env, "motion_env"):
        out = env.motion_env.step(a)
        return _normalize_step_output(out)

    raise RuntimeError("Could not find a motion step API. Edit step_motion() for your env wrapper.")


# ------------------------- main -------------------------
def main():
    p = argparse.ArgumentParser(description="Run OpenVLA policy over socket inside LoHRbench/TAMPbench env.")
    p.add_argument("-b", "--benchmark", type=str, help="Path to task directory")
    p.add_argument("--type", type=str, help="Task group")
    p.add_argument("--task", type=str, required=True, help="Task folder name")
    p.add_argument("--root", type=str, help="Override root benchmark directory")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-env", type=int, default=1)

    task_only_group = p.add_mutually_exclusive_group()
    task_only_group.add_argument("--task-only", dest="task_only", action="store_true")
    task_only_group.add_argument("--no-task-only", dest="task_only", action="store_false")
    p.set_defaults(task_only=False)

    p.add_argument("--record", action="store_true")
    p.add_argument("--record-dir", type=str, default="rollouts")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--planner-type", type=str, default="maniskill")
    p.add_argument("--render-mode", type=str, choices=["human", "rgb_array"], default="rgb_array")
    p.add_argument("--max-steps-per-video", type=int, default=5000)

    p.add_argument("--server", type=str, default="127.0.0.1:5555")
    p.add_argument("--max-retries", type=int, default=10, help="Max connection retry attempts")
    p.add_argument("--retry-delay", type=float, default=2.0, help="Delay between retries (seconds)")
    p.add_argument("--unnorm_key", type=str, default="lohrbench_rlds")

    p.add_argument("--base-path", type=str, default="sensor_data.base_camera.rgb")
    p.add_argument("--wrist-path", type=str, default="sensor_data.hand_camera.rgb")
    p.add_argument("--qpos-path", type=str, default="agent.qpos")

    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--instruction", type=str, default=None)

    args = p.parse_args()

    task_dir = _resolve_benchmark(args)
    domain_f = _detect_domain(task_dir)
    problem_f = _detect_problem(task_dir)
    position_f = _detect_positions(task_dir)

    instruction = args.instruction
    if instruction is None:
        instruction = TASK_INSTRUCTIONS.get(args.task, args.task)

    print("\n" + "="*60)
    print("LoHRbench OpenVLA Policy Rollout")
    print("="*60)
    print(f"Benchmark : {task_dir}")
    print(f"Domain    : {domain_f}")
    print(f"Problem   : {problem_f}")
    print(f"Position  : {position_f}")
    print(f"task_only : {args.task_only}")
    print(f"server    : {args.server}")
    print(f"instruction: {instruction}")
    print("="*60 + "\n")

    # Connect to server first (with retry)
    host, port_s = args.server.split(":")
    port = int(port_s)
    
    try:
        conn = connect_with_retry(host, port, args.max_retries, args.retry_delay)
    except Exception as e:
        print(f"\n❌ Fatal: Could not connect to server")
        sys.exit(1)

    # Now create environment
    print("Creating environment...")
    env = TAMPEnvironment(
        str(domain_f),
        str(problem_f),
        str(position_f),
        seed=args.seed,
        num_env=args.num_env,
        task_only=args.task_only,
        record=args.record,
        record_dir=str(args.record_dir),
        save_video=args.save_video,
        planner_type=args.planner_type,
        render_mode=args.render_mode,
        max_steps_per_video=args.max_steps_per_video,
        task_name=args.task,
    )

    reset_out = env.reset()
    env_obs = _unwrap_reset_or_step(reset_out)
    total_reward = 0.0

    print(f"\nStarting rollout (max {args.max_steps} steps)...\n")

    try:
        for t in range(args.max_steps):
            base_rgb, wrist_rgb, qpos, instr = extract_obs(
                env_obs,
                base_path=args.base_path,
                wrist_path=args.wrist_path,
                qpos_path=args.qpos_path,
                instruction=instruction,
            )

            req = {
                "instruction": instr,
                "base_rgb": base_rgb,
                "wrist_rgb": wrist_rgb,
                "qpos": qpos,
                "unnorm_key": args.unnorm_key,
            }

            send_msg(conn, req)
            resp = recv_msg(conn)
            action = np.asarray(resp["pred_action"], dtype=np.float32).reshape(-1)

            step_out = step_motion(env, action)
            env_obs, reward, done, info = step_out
            
            # Convert reward to float (handles both tensors and scalars)
            reward_val = float(reward) if hasattr(reward, '__float__') else reward
            total_reward += reward_val

            if t < 5 or t % 100 == 0:
                print(f"[t={t:4d}] reward={reward_val:7.3f} total={total_reward:7.3f} done={done}")

            if bool(done):
                print(f"\n✅ Episode finished at t={t}")
                print(f"   Total reward: {total_reward:.3f}")
                print(f"   Info: {info}")
                break
        else:
            print(f"\n⏱️  Max steps ({args.max_steps}) reached")
            print(f"   Total reward: {total_reward:.3f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during rollout: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            send_msg(conn, {"_type": "close"})
            _ = recv_msg(conn)
        except Exception:
            pass
        conn.close()

        if hasattr(env, "task_env") and hasattr(env.task_env, "motion_env"):
            try:
                env.task_env.motion_env.close()
            except Exception:
                pass

    print("\n✅ Finished.")


if __name__ == "__main__":
    main()
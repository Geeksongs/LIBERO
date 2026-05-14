"""
Asynchronous evaluation script for LIBERO benchmark.
This script uses the AsyncSimulation API from robosuite for real-time evaluation.

Supports two policy types:
1. LIBERO native policies (bc_rnn_policy, bc_transformer_policy, bc_vilt_policy)
2. LeRobot policies (pi0, pi05, act, diffusion, smolvla, etc.)

Usage examples:
    # LIBERO native policy
    python -m libero.lifelong.evaluate_async \\
        --benchmark libero_goal --task_id 0 \\
        --algo base --policy bc_rnn_policy --seed 0 --load_task 2

    # LeRobot PI0.5 policy
    python -m libero.lifelong.evaluate_async \\
        --benchmark libero_goal --task_id 0 \\
        --model_type lerobot \\
        --lerobot_model lerobot/pi05_libero_base
"""
import argparse
import sys
import os

# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import numpy as np
import time
import torch
from pathlib import Path
from functools import partial

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, GroupedTaskDataset
from libero.lifelong.utils import (
    safe_device,
    torch_load_model,
)
from libero.lifelong.main import get_task_embs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

# Import AsyncSimulation from robosuite
from robosuite.environments.async_env import AsyncSimulation

# ======================= LeRobot Integration =======================
# Lazy imports for LeRobot to avoid import errors if not installed
LEROBOT_AVAILABLE = False
try:
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    LEROBOT_AVAILABLE = True
except ImportError:
    pass


benchmark_map = {
    "libero_10": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
}

algo_map = {
    "base": "SingleTask",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}


# ======================= LeRobot Policy Wrapper =======================

def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to axis-angle representation.

    Args:
        quat: Quaternion array of shape (4,) in (x, y, z, w) format

    Returns:
        Axis-angle array of shape (3,)
    """
    x, y, z, w = quat
    w = np.clip(w, -1.0, 1.0)

    den = np.sqrt(max(1.0 - w * w, 0.0))

    if den > 1e-10:
        angle = 2.0 * np.arccos(w)
        axis = np.array([x, y, z]) / den
        return axis * angle
    else:
        return np.zeros(3)


def libero_obs_to_lerobot_obs(obs_data: dict, task_description: str, device: str = "cuda") -> dict:
    """
    Convert LIBERO async observation format to LeRobot format.

    Args:
        obs_data: OrderedDict from AsyncSimulation's Observation.data containing:
            - agentview_image: (H, W, C) uint8
            - robot0_eye_in_hand_image: (H, W, C) uint8
            - robot0_eef_pos: (3,) end-effector position
            - robot0_eef_quat: (4,) end-effector quaternion (x, y, z, w)
            - robot0_gripper_qpos: (2,) gripper joint positions
        task_description: Language description of the task
        device: Target device for tensors

    Returns:
        Dictionary in LeRobot format with:
            - observation.images.image: (1, C, H, W) float32 [0, 1]
            - observation.images.image2: (1, C, H, W) float32 [0, 1]
            - observation.state: (1, 8) float32
            - task: list of task descriptions
    """
    lerobot_obs = {}

    # Process images: HWC uint8 -> BCHW float32 [0, 1] with 180-degree rotation
    for libero_key, lerobot_key in [
        ("agentview_image", "observation.images.image"),
        ("robot0_eye_in_hand_image", "observation.images.image2"),
    ]:
        if libero_key in obs_data:
            img = obs_data[libero_key]  # (H, W, C) uint8

            # Flip both H and W for 180-degree rotation (LIBERO camera convention)
            img = img[::-1, ::-1, :].copy()

            # Convert to tensor: HWC -> CHW
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            # Add batch dimension: (C, H, W) -> (1, C, H, W)
            img_tensor = img_tensor.unsqueeze(0)

            lerobot_obs[lerobot_key] = img_tensor.to(device)

    # Process robot state: construct 8-dim state vector
    # [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    eef_pos = obs_data.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs_data.get("robot0_eef_quat", np.array([0, 0, 0, 1]))
    gripper_qpos = obs_data.get("robot0_gripper_qpos", np.zeros(2))

    # Convert quaternion to axis-angle
    eef_axisangle = quat_to_axisangle(eef_quat)

    # Concatenate state: (8,)
    state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos]).astype(np.float32)

    # Add batch dimension: (8,) -> (1, 8)
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
    lerobot_obs["observation.state"] = state_tensor

    # Add task description
    lerobot_obs["task"] = [task_description]

    return lerobot_obs


class LeRobotPolicyWrapper:
    """
    Wrapper to use LeRobot policies (PI0, PI0.5, ACT, etc.) in LIBERO async evaluation.

    This wrapper handles:
    1. Loading pretrained LeRobot policies from HuggingFace Hub
    2. Converting LIBERO observations to LeRobot format
    3. Processing actions through LeRobot's preprocessor/postprocessor pipeline
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        n_action_steps: int = None,
    ):
        """
        Initialize LeRobot policy wrapper.

        Args:
            model_id: HuggingFace model ID (e.g., "lerobot/pi05_libero_base")
            device: Device to run inference on
            n_action_steps: Number of action steps to use (overrides model config if set)
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "LeRobot is not installed. Please install it with:\n"
                "  pip install lerobot\n"
                "Or for full LIBERO support:\n"
                "  pip install 'lerobot[libero]'"
            )

        self.model_id = model_id
        self.device = device

        print(f"[info] Loading LeRobot policy from: {model_id}")

        # Load config first to determine policy type
        self.config = PreTrainedConfig.from_pretrained(model_id)

        # Override n_action_steps if specified
        if n_action_steps is not None:
            self.config.n_action_steps = n_action_steps
            print(f"[info] Using n_action_steps={n_action_steps}")

        # Set device
        self.config.device = device

        # Load the policy
        policy_cls = get_policy_class(self.config.type)
        self.policy = policy_cls.from_pretrained(model_id, config=self.config)
        self.policy.to(device)
        self.policy.eval()

        # Load preprocessor and postprocessor
        preprocessor_overrides = {
            "device_processor": {"device": device},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.config,
            pretrained_path=model_id,
            preprocessor_overrides=preprocessor_overrides,
        )

        print(f"[info] Loaded {self.config.type} policy successfully")

        # Store task description for observation conversion
        self.task_description = ""

    def set_task(self, task_description: str):
        """Set the current task description for language-conditioned policies."""
        self.task_description = task_description
        print(f"[info] Task set to: {task_description}")

    def reset(self):
        """Reset policy internal state (e.g., action queue for action chunking)."""
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def get_action(self, obs_data: dict) -> np.ndarray:
        """
        Get action from the policy given LIBERO observation.

        Args:
            obs_data: Raw observation dict from LIBERO AsyncSimulation

        Returns:
            Action array of shape (7,) - 6D end-effector delta + 1D gripper
        """
        # Convert LIBERO obs to LeRobot format
        lerobot_obs = libero_obs_to_lerobot_obs(
            obs_data,
            self.task_description,
            self.device
        )

        # Apply preprocessor
        processed_obs = self.preprocessor(lerobot_obs)

        # Get action from policy
        with torch.inference_mode():
            action = self.policy.select_action(processed_obs)

        # Apply postprocessor
        action = self.postprocessor(action)

        # Convert to numpy and squeeze batch dimension
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Squeeze batch dimension if present
        if action.ndim > 1:
            action = action.squeeze(0)

        return action


def async_obs_to_tensor_obs(obs_data, task_emb, cfg):
    """
    Convert async observation (single env) to tensor format for policy input.

    Args:
        obs_data: OrderedDict from AsyncSimulation's Observation.data
        task_emb: Task embedding tensor
        cfg: Configuration object

    Returns:
        Dictionary containing processed observations and task embedding
    """
    data = {
        "obs": {},
        "task_emb": task_emb.unsqueeze(0),  # Add batch dimension for single env
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    # Process single environment observation
    for obs_name in all_obs_keys:
        obs_key = cfg.data.obs_key_mapping[obs_name]
        obs_value = obs_data[obs_key]
        if isinstance(obs_value, np.ndarray):
            obs_tensor = torch.from_numpy(obs_value)
        else:
            obs_tensor = torch.tensor(obs_value)
        processed = ObsUtils.process_obs(obs_tensor, obs_key=obs_name).float()
        data["obs"][obs_name].append(processed)

    # Stack observations (single env, so just unsqueeze)
    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    return data


class LiberoEnvFactory:
    """
    Picklable environment factory for AsyncSimulation.
    """
    def __init__(self, env_args):
        self.env_args = env_args

    def __call__(self):
        return OffScreenRenderEnv(**self.env_args)


def evaluate_single_episode_async(
    sim,
    policy,
    max_steps,
    control_freq,
    task_emb=None,
    cfg=None,
    init_state=None,
    video_writer=None,
    episode_idx=0,
    video_camera_name="frontview_image",
    model_type="libero",
):
    """
    Evaluate a single episode using async simulation.

    Args:
        sim: AsyncSimulation instance
        policy: Policy object (LIBERO algo or LeRobotPolicyWrapper)
        max_steps: Maximum steps per episode (used to compute horizon in seconds)
        control_freq: Control frequency in Hz (used to compute horizon in seconds)
        task_emb: Task embedding (required for model_type='libero')
        cfg: Configuration (required for model_type='libero')
        init_state: Optional initial state to set
        video_writer: Optional VideoWriter for saving videos
        episode_idx: Episode index for video saving
        video_camera_name: Camera name for video recording (e.g., "frontview_image")
        model_type: "libero" or "lerobot"

    Returns:
        bool: Whether the episode was successful

    Note:
        The episode horizon is defined as (max_steps / control_freq) seconds of
        simulation time, ensuring consistency with synchronous evaluation where
        horizon is defined as the number of control steps.
    """
    # Reset the simulation
    sim.reset()

    # Reset policy state
    if model_type == "libero":
        policy.reset()
    else:
        policy.reset()

    # If init_state is provided, we need to set it
    # Note: AsyncSimulation may need modification to support set_init_state
    # For now, we use the default reset behavior

    # Wait for initial observation
    obs = sim.observation_stream.get(timeout=5.0)

    # Warm-up: send zero actions for a few steps
    zero_action = np.zeros(7, dtype=np.float32)
    for _ in range(5):
        sim.control_stream.push(zero_action)
        time.sleep(0.05)

    # Get fresh observation after warm-up
    obs = sim.observation_stream.get(timeout=2.0)

    # Compute horizon in seconds to match synchronous evaluation
    # Sync eval: max_steps control steps = max_steps / control_freq seconds
    horizon_seconds = max_steps / control_freq
    success = False

    with torch.no_grad():
        while obs.time < horizon_seconds and not sim.done():
            # Save video frame if video_writer is provided
            if video_writer is not None and video_writer.save_video:
                if video_camera_name in obs.data:
                    video_writer.append_obs(obs.data, done=False, idx=episode_idx, camera_name=video_camera_name)

            # Get action based on model type
            if model_type == "libero":
                # LIBERO native policy
                data = async_obs_to_tensor_obs(obs.data, task_emb, cfg)
                action = policy.policy.get_action(data)
            else:
                # LeRobot policy
                action = policy.get_action(obs.data)

            # Send action to simulation (action is [1, 7], need to squeeze)
            if hasattr(action, 'shape') and len(action.shape) > 1:
                action = action.squeeze(0)
            sim.control_stream.push(action)

            # Get next observation
            try:
                obs = sim.observation_stream.get(timeout=1.0)
            except TimeoutError:
                print(f"[warning] Timeout waiting for observation at sim_time {obs.time:.2f}s")
                break

            # Check success from observation
            if obs.success:
                success = True
                break

    # Save final frame with success/fail indicator
    if video_writer is not None and video_writer.save_video:
        if video_camera_name in obs.data:
            video_writer.append_obs(obs.data, done=True, idx=episode_idx, camera_name=video_camera_name)

    return success


def parse_args():
    parser = argparse.ArgumentParser(
        description="Async Evaluation Script for LIBERO (supports LIBERO native and LeRobot policies)"
    )

    # === Model type selection ===
    parser.add_argument(
        "--model_type",
        type=str,
        default="libero",
        choices=["libero", "lerobot"],
        help="Policy type: 'libero' for native LIBERO policies, 'lerobot' for LeRobot policies (default: libero)"
    )

    # === LeRobot-specific arguments ===
    parser.add_argument(
        "--lerobot_model",
        type=str,
        default="lerobot/pi05_libero_base",
        help="LeRobot model ID from HuggingFace Hub (e.g., 'lerobot/pi05_libero_base', 'lerobot/pi05_libero_finetuned')"
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Number of action steps for action chunking (LeRobot only, default: use model config)"
    )

    # === Common arguments ===
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, required=True)

    # === LIBERO-native policy arguments (only used when model_type='libero') ===
    parser.add_argument(
        "--algo",
        type=str,
        default="base",
        choices=["base", "er", "ewc", "packnet", "multitask"],
        help="LIBERO algorithm (only for model_type='libero')"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="bc_rnn_policy",
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
        help="LIBERO policy type (only for model_type='libero')"
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (only for model_type='libero')")
    parser.add_argument("--ep", type=int,
                        help="Epoch for multitask model (only for model_type='libero')")
    parser.add_argument("--load_task", type=int,
                        help="Task ID to load checkpoint from (only for model_type='libero')")

    # === Evaluation settings ===
    parser.add_argument("--device_id", type=int,
                        help="CUDA device ID (default: 0)")
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum steps per episode (default: use benchmark default)")
    parser.add_argument("--control_freq", type=float, default=20.0,
                        help="Control frequency for async simulation")
    parser.add_argument("--observation_freq", type=float, default=20.0,
                        help="Observation frequency for async simulation")

    # === Video recording ===
    parser.add_argument("--save-videos", action="store_true",
                        help="Save evaluation videos")
    parser.add_argument("--video-height", type=int, default=480,
                        help="Video frame height (default: 480)")
    parser.add_argument("--video-width", type=int, default=480,
                        help="Video frame width (default: 480)")
    parser.add_argument("--video-camera", type=str, default="frontview",
                        help="Camera for video recording: frontview, agentview, sideview, birdview (default: frontview)")

    # === Simulation settings ===
    parser.add_argument("--real-time-rate", type=float, default=1.0,
                        help="Target real-time rate: 1.0=realtime, 2.0=2x speed, 0=as fast as possible (default: 1.0)")

    args = parser.parse_args()

    # Process device
    args.device_id = "cuda:" + str(args.device_id) if args.device_id is not None else "cuda:0"
    args.save_dir = f"{args.experiment_dir}_saved"

    # Validate LIBERO-specific arguments
    if args.model_type == "libero":
        if args.algo == "multitask":
            assert args.ep in list(range(0, 50, 5)), "[error] ep should be in [0, 5, ..., 50]"
        else:
            assert args.load_task in list(range(10)), "[error] load_task should be in [0, ..., 9]"

    # Validate LeRobot-specific arguments
    if args.model_type == "lerobot":
        if not LEROBOT_AVAILABLE:
            print("[error] LeRobot is not installed. Please install it with:")
            print("  pip install lerobot")
            sys.exit(1)

    return args


def main():
    args = parse_args()

    print(f"[info] Starting async evaluation for benchmark={args.benchmark}, task_id={args.task_id}")
    print(f"[info] Model type: {args.model_type}")

    # Get benchmark info (needed for both model types)
    bddl_folder = get_libero_path("bddl_files")

    # Get benchmark with default task order
    benchmark = get_benchmark(args.benchmark)(0)
    task = benchmark.get_task(args.task_id)
    task_description = task.language

    print(f"[info] Task: {task_description}")

    # ======================= Load Policy =======================
    if args.model_type == "libero":
        # Load LIBERO native policy
        experiment_dir = os.path.join(
            args.experiment_dir,
            f"{benchmark_map[args.benchmark]}/"
            + f"{algo_map[args.algo]}/"
            + f"{policy_map[args.policy]}_seed{args.seed}",
        )

        # Find the checkpoint
        experiment_id = 0
        for path in Path(experiment_dir).glob("run_*"):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split("run_")[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        if experiment_id == 0:
            print(f"[error] cannot find the checkpoint under {experiment_dir}")
            sys.exit(0)

        run_folder = os.path.join(experiment_dir, f"run_{experiment_id:03d}")
        try:
            if args.algo == "multitask":
                model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
                sd, cfg, previous_mask = torch_load_model(
                    model_path, map_location=args.device_id
                )
            else:
                model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")
                sd, cfg, previous_mask = torch_load_model(
                    model_path, map_location=args.device_id
                )
        except Exception as e:
            print(f"[error] cannot find the checkpoint at {str(model_path)}")
            print(f"[error] Exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(0)

        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = bddl_folder
        cfg.init_states_folder = get_libero_path("init_states")

        cfg.device = args.device_id
        policy = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
        policy.policy.previous_mask = previous_mask

        if cfg.lifelong.algo == "PackNet":
            policy.eval()
            for module_idx, module in enumerate(policy.policy.modules()):
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    weight = module.weight.data
                    mask = policy.previous_masks[module_idx].to(cfg.device)
                    weight[mask.eq(0)] = 0.0
                    weight[mask.gt(args.task_id + 1)] = 0.0
                if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                    module.eval()

        policy.policy.load_state_dict(sd)

        if not hasattr(cfg.data, "task_order_index"):
            cfg.data.task_order_index = 0

        # Get task embeddings for LIBERO native policy
        descriptions = [benchmark.get_task(i).language for i in range(10)]
        task_embs = get_task_embs(cfg, descriptions)
        benchmark.set_task_embs(task_embs)
        task_emb = benchmark.get_task_emb(args.task_id)

        # Initialize observation utils
        try:
            dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(args.task_id)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len=cfg.data.seq_len,
            )
        except Exception as e:
            print(
                f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}"
            )
            print(f"[error] Exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(0)

        policy.eval()

        # Get image dimensions from config
        img_h = cfg.data.img_h
        img_w = cfg.data.img_w
        max_steps = args.max_steps if args.max_steps else cfg.eval.max_steps

    else:
        # Load LeRobot policy
        policy = LeRobotPolicyWrapper(
            model_id=args.lerobot_model,
            device=args.device_id,
            n_action_steps=args.n_action_steps,
        )
        policy.set_task(task_description)

        # Set cfg and task_emb to None for LeRobot
        cfg = None
        task_emb = None

        # Default image dimensions for LeRobot (256x256 is common)
        img_h = 256
        img_w = 256

        # Default max steps for different benchmarks
        max_steps_map = {
            "libero_spatial": 280,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
        }
        max_steps = args.max_steps if args.max_steps else max_steps_map.get(args.benchmark, 400)

    # ======================= Setup Save Paths =======================
    if args.model_type == "libero":
        if args.algo == "multitask":
            save_folder = os.path.join(
                args.save_dir,
                f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}_async.stats",
            )
            video_folder = os.path.join(
                args.save_dir,
                f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}_async_videos",
            )
        else:
            save_folder = os.path.join(
                args.save_dir,
                f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_async.stats",
            )
            video_folder = os.path.join(
                args.save_dir,
                f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_async_videos",
            )
    else:
        # LeRobot model naming
        model_name = args.lerobot_model.replace("/", "_")
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_lerobot_{model_name}_on{args.task_id}_async.stats",
        )
        video_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_lerobot_{model_name}_on{args.task_id}_async_videos",
        )

    ### ======================= Start Async Evaluation =======================

    # Determine camera configuration for video recording
    video_camera_name = args.video_camera + "_image"  # e.g., "frontview_image"

    with Timer() as t, VideoWriter(video_folder, args.save_videos, single_video=False) as video_writer:
        # Environment arguments for LIBERO
        # Include video camera in camera_names if saving videos
        camera_names = ["agentview", "robot0_eye_in_hand"]
        if args.save_videos and args.video_camera not in camera_names:
            camera_names.append(args.video_camera)

        # Use higher resolution for video camera if saving videos
        if args.save_videos:
            camera_heights = [img_h, img_h] + [args.video_height] * (len(camera_names) - 2)
            camera_widths = [img_w, img_w] + [args.video_width] * (len(camera_names) - 2)
        else:
            camera_heights = img_h
            camera_widths = img_w

        env_args = {
            "bddl_file_name": os.path.join(
                bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_names": camera_names,
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
        }

        # Create AsyncSimulation with LIBERO environment
        # real_time_rate=0 means run as fast as possible (no real-time constraint)
        real_time_rate = args.real_time_rate if args.real_time_rate > 0 else None
        sim = AsyncSimulation(
            env_factory=LiberoEnvFactory(env_args),
            control_freq=args.control_freq,
            observation_freq=args.observation_freq,
            visualization_freq=None,  # No visualization for evaluation
            target_real_time_rate=real_time_rate,
        )

        num_success = 0
        n_eval = args.n_eval

        rtr_str = f"{args.real_time_rate}x" if args.real_time_rate > 0 else "max speed"
        print(f"[info] Running {n_eval} evaluation episodes with async simulation (real-time rate: {rtr_str})...")
        print(f"[info] Max steps per episode: {max_steps}")

        try:
            sim.start()

            for episode in range(n_eval):
                success = evaluate_single_episode_async(
                    sim=sim,
                    policy=policy,
                    max_steps=max_steps,
                    control_freq=args.control_freq,
                    task_emb=task_emb,
                    cfg=cfg,
                    video_writer=video_writer,
                    episode_idx=episode,
                    video_camera_name=video_camera_name,
                    model_type=args.model_type,
                )

                if success:
                    num_success += 1

                print(f"  Episode {episode + 1}/{n_eval}: {'Success' if success else 'Fail'} "
                      f"(Running success rate: {num_success}/{episode + 1} = {num_success/(episode+1):.2%})")

        finally:
            sim.stop()

        success_rate = num_success / n_eval

        eval_stats = {
            "loss": 0.0,  # Not computed in async mode
            "success_rate": success_rate,
            "n_eval": n_eval,
            "evaluation_mode": "async",
            "model_type": args.model_type,
        }

        if args.model_type == "lerobot":
            eval_stats["lerobot_model"] = args.lerobot_model

        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(eval_stats, save_folder)

    print(f"\n[info] Async evaluation completed in {t.get_elapsed_time():.2f} seconds")
    print(f"[info] Results saved at {save_folder}")
    print(f"[info] Success rate: {num_success}/{n_eval} = {success_rate:.2%}")


if __name__ == "__main__":
    main()

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
import csv
import concurrent.futures
from datetime import datetime
from threading import Lock

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

# Import AsyncSimulation from robosuite
from robosuite.environments.async_env import AsyncSimulation

# ======================= LIBERO Native Policy Imports =======================
# These imports are only needed for --model_type libero
# They are lazy-loaded to avoid robomimic version compatibility issues
LIBERO_NATIVE_AVAILABLE = False
ObsUtils = None
TensorUtils = None

def _load_libero_native_imports():
    """Lazy load LIBERO native policy imports."""
    global LIBERO_NATIVE_AVAILABLE, ObsUtils, TensorUtils
    if LIBERO_NATIVE_AVAILABLE:
        return True
    try:
        from libero.lifelong.datasets import get_dataset, GroupedTaskDataset
        from libero.lifelong.utils import safe_device, torch_load_model
        from libero.lifelong.main import get_task_embs
        import robomimic.utils.obs_utils as _ObsUtils
        import robomimic.utils.tensor_utils as _TensorUtils
        # Import algo classes
        from libero.lifelong.algos.base import Sequential, get_algo_class
        from libero.lifelong.algos.multitask import Multitask
        from libero.lifelong.algos.er import ER
        from libero.lifelong.algos.ewc import EWC
        from libero.lifelong.algos.packnet import PackNet

        globals()['ObsUtils'] = _ObsUtils
        globals()['TensorUtils'] = _TensorUtils
        globals()['get_dataset'] = get_dataset
        globals()['GroupedTaskDataset'] = GroupedTaskDataset
        globals()['safe_device'] = safe_device
        globals()['torch_load_model'] = torch_load_model
        globals()['get_task_embs'] = get_task_embs
        globals()['Sequential'] = Sequential
        globals()['Multitask'] = Multitask
        globals()['ER'] = ER
        globals()['EWC'] = EWC
        globals()['PackNet'] = PackNet
        globals()['SingleTask'] = Sequential  # Alias
        LIBERO_NATIVE_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"[warning] LIBERO native policy imports failed: {e}")
        print("[warning] Only LeRobot policies (--model_type lerobot) will be available")
        return False

# ======================= LeRobot Integration =======================
# Lazy imports for LeRobot to avoid import errors if not installed
LEROBOT_AVAILABLE = False
MOLMOACT2_AVAILABLE = False
try:
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    LEROBOT_AVAILABLE = True
    # Check if MolmoAct2 is available
    try:
        from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
        MOLMOACT2_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

from PIL import Image


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


def libero_obs_to_molmoact2_obs(obs_data: dict, task_description: str) -> dict:
    """
    Convert LIBERO async observation format to MolmoAct2 format.

    MolmoAct2 expects:
    - images: list of 2 PIL Images [agentview, wrist]
    - state: numpy array (8,) float32
    - task: string

    Args:
        obs_data: OrderedDict from AsyncSimulation's Observation.data containing:
            - agentview_image: (H, W, C) uint8
            - robot0_eye_in_hand_image: (H, W, C) uint8
            - robot0_eef_pos: (3,) end-effector position
            - robot0_eef_quat: (4,) end-effector quaternion (x, y, z, w)
            - robot0_gripper_qpos: (2,) gripper joint positions
        task_description: Language description of the task

    Returns:
        Dictionary in MolmoAct2 format with:
            - images: list of 2 PIL Images
            - state: numpy array (8,)
            - task: string
    """
    molmoact2_obs = {}

    # Process images: HWC uint8 -> PIL Image with 180-degree rotation
    images = []
    for libero_key in ["agentview_image", "robot0_eye_in_hand_image"]:
        if libero_key in obs_data:
            img = obs_data[libero_key]  # (H, W, C) uint8

            # Flip both H and W for 180-degree rotation (LIBERO camera convention)
            img = img[::-1, ::-1, :].copy()

            # Convert to PIL Image
            pil_img = Image.fromarray(img, mode="RGB")
            images.append(pil_img)

    molmoact2_obs["images"] = images

    # Process robot state: construct 8-dim state vector
    # [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    eef_pos = obs_data.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs_data.get("robot0_eef_quat", np.array([0, 0, 0, 1]))
    gripper_qpos = obs_data.get("robot0_gripper_qpos", np.zeros(2))

    # Convert quaternion to axis-angle
    eef_axisangle = quat_to_axisangle(eef_quat)

    # Concatenate state: (8,)
    state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos]).astype(np.float32)
    molmoact2_obs["state"] = state

    # Add task description
    molmoact2_obs["task"] = task_description

    return molmoact2_obs


class LeRobotPolicyWrapper:
    """
    Wrapper to use LeRobot policies (PI0, PI0.5, ACT, MolmoAct2, etc.) in LIBERO async evaluation.

    This wrapper handles:
    1. Loading pretrained LeRobot policies from HuggingFace Hub
    2. Converting LIBERO observations to LeRobot format
    3. Processing actions through LeRobot's preprocessor/postprocessor pipeline

    Special handling for MolmoAct2:
    - Uses PIL images directly instead of tensors
    - Uses model's built-in predict_action() method
    - No external preprocessor/postprocessor needed
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
        self.is_molmoact2 = False

        print(f"[info] Loading LeRobot policy from: {model_id}")

        # Check if this is MolmoAct2 (either by model_id or by loading config)
        if "molmoact2" in model_id.lower() or "MolmoAct2" in model_id:
            self._init_molmoact2(model_id, device, n_action_steps)
        else:
            # Try to load config to determine policy type
            try:
                self.config = PreTrainedConfig.from_pretrained(model_id)
                if self.config.type == "molmoact2":
                    self._init_molmoact2(model_id, device, n_action_steps)
                else:
                    self._init_standard_policy(model_id, device, n_action_steps)
            except Exception:
                # If can't load config, try as MolmoAct2 if it's from allenai
                if "allenai" in model_id.lower():
                    self._init_molmoact2(model_id, device, n_action_steps)
                else:
                    raise

        # Store task description for observation conversion
        self.task_description = ""

    def _init_molmoact2(self, model_id: str, device: str, n_action_steps: int = None):
        """Initialize MolmoAct2 policy with direct HuggingFace loading."""
        self.is_molmoact2 = True

        print(f"[info] Detected MolmoAct2 model, using direct HuggingFace loading")

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for MolmoAct2. "
                "Install it with: pip install transformers"
            ) from e

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Determine dtype
        self.dtype = torch.bfloat16

        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(device).eval()

        # MolmoAct2 config
        self.num_steps = n_action_steps if n_action_steps is not None else 10
        self.norm_tag = "libero"
        self.inference_action_mode = "continuous"
        self.enable_cuda_graph = True

        # Action queue for action chunking
        self._action_queue = []

        print(f"[info] Loaded MolmoAct2 policy successfully (num_steps={self.num_steps})")

    def _init_standard_policy(self, model_id: str, device: str, n_action_steps: int = None):
        """Initialize standard LeRobot policy (PI0, PI05, ACT, etc.)."""
        self.is_molmoact2 = False

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

    def set_task(self, task_description: str):
        """Set the current task description for language-conditioned policies."""
        self.task_description = task_description
        print(f"[info] Task set to: {task_description}")

    def reset(self):
        """Reset policy internal state (e.g., action queue for action chunking)."""
        if self.is_molmoact2:
            self._action_queue = []
        elif hasattr(self, 'policy') and hasattr(self.policy, 'reset'):
            self.policy.reset()

    def get_action(self, obs_data: dict) -> np.ndarray:
        """
        Get action from the policy given LIBERO observation.

        Args:
            obs_data: Raw observation dict from LIBERO AsyncSimulation

        Returns:
            Action array of shape (7,) - 6D end-effector delta + 1D gripper
        """
        if self.is_molmoact2:
            return self._get_action_molmoact2(obs_data)
        else:
            return self._get_action_standard(obs_data)

    def _get_action_molmoact2(self, obs_data: dict) -> np.ndarray:
        """Get action from MolmoAct2 policy."""
        # Check if we have actions in the queue
        if len(self._action_queue) > 0:
            return self._action_queue.pop(0)

        # Convert LIBERO obs to MolmoAct2 format
        molmoact2_obs = libero_obs_to_molmoact2_obs(obs_data, self.task_description)

        # Run inference
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            output = self.model.predict_action(
                processor=self.processor,
                images=molmoact2_obs["images"],
                task=molmoact2_obs["task"],
                state=molmoact2_obs["state"],
                norm_tag=self.norm_tag,
                inference_action_mode=self.inference_action_mode,
                enable_depth_reasoning=False,
                num_steps=self.num_steps,
                normalize_language=True,
                enable_cuda_graph=self.enable_cuda_graph,
            )

        # Get actions from output
        actions = output.actions  # Shape: (num_steps, action_dim)

        # Convert to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        # Add all actions to queue except the first one (which we return)
        for i in range(1, len(actions)):
            self._action_queue.append(actions[i])

        return actions[0]

    def _get_action_standard(self, obs_data: dict) -> np.ndarray:
        """Get action from standard LeRobot policy (PI0, PI05, etc.)."""
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


# ======================= CSV Helper Functions =======================

CSV_LOCK = Lock()

def get_csv_header():
    """Get CSV header columns."""
    return [
        "timestamp",
        "benchmark",
        "task_id",
        "task_description",
        "model_type",
        "model_name",
        "episode_idx",
        "success",
        "eval_mode",
        "max_steps",
        "control_freq",
        "real_time_rate",
        "batch_size",
    ]


def append_to_csv(csv_path: str, row_data: dict):
    """
    Append a single row to CSV file. Creates file with header if it doesn't exist.
    Thread-safe with CSV_LOCK.
    """
    with CSV_LOCK:
        file_exists = os.path.exists(csv_path)

        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=get_csv_header())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row_data)


def create_csv_row(
    benchmark: str,
    task_id: int,
    task_description: str,
    model_type: str,
    model_name: str,
    episode_idx: int,
    success: bool,
    max_steps: int,
    control_freq: float,
    real_time_rate: float,
    batch_size: int,
) -> dict:
    """Create a CSV row dictionary."""
    return {
        "timestamp": datetime.now().isoformat(),
        "benchmark": benchmark,
        "task_id": task_id,
        "task_description": task_description,
        "model_type": model_type,
        "model_name": model_name,
        "episode_idx": episode_idx,
        "success": int(success),
        "eval_mode": "async",
        "max_steps": max_steps,
        "control_freq": control_freq,
        "real_time_rate": real_time_rate,
        "batch_size": batch_size,
    }


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

    # === Concurrency settings ===
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of parallel async simulations (default: 1)")

    # === CSV output settings ===
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to CSV file for results (default: auto-generate in output_dir)")

    # === Output directory ===
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for all results (CSV, videos, stats). Default: experiments_saved/")

    args = parser.parse_args()

    # Process device
    args.device_id = "cuda:" + str(args.device_id) if args.device_id is not None else "cuda:0"

    # Set output directory
    if args.output_dir is not None:
        args.save_dir = args.output_dir
    else:
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
        # Load LIBERO native policy imports (lazy loading)
        if not _load_libero_native_imports():
            print("[error] LIBERO native policy requires robomimic and other dependencies.")
            print("[error] Please install: pip install robomimic")
            print("[error] Or use --model_type lerobot for LeRobot policies.")
            sys.exit(1)

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

        # Default max steps for different benchmarks (doubled for async evaluation)
        max_steps_map = {
            "libero_spatial": 560,   # 280 * 2
            "libero_object": 560,    # 280 * 2
            "libero_goal": 600,      # 300 * 2
            "libero_10": 1040,       # 520 * 2
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

        # Determine CSV path
        if args.csv_path:
            csv_path = args.csv_path
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            csv_path = os.path.join(args.save_dir, "async_eval_results.csv")

        # Get model name for CSV
        model_name = args.lerobot_model if args.model_type == "lerobot" else f"{args.algo}_{args.policy}"

        num_success = 0
        n_eval = args.n_eval
        batch_size = args.batch_size

        rtr_str = f"{args.real_time_rate}x" if args.real_time_rate > 0 else "max speed"
        print(f"[info] Running {n_eval} evaluation episodes with async simulation")
        print(f"[info] Real-time rate: {rtr_str}, Batch size: {batch_size}")
        print(f"[info] Max steps per episode: {max_steps}")
        print(f"[info] CSV output: {csv_path}")

        # Create multiple AsyncSimulation instances for parallel evaluation
        sims = []
        for i in range(batch_size):
            sim = AsyncSimulation(
                env_factory=LiberoEnvFactory(env_args),
                control_freq=args.control_freq,
                observation_freq=args.observation_freq,
                visualization_freq=None,  # No visualization for evaluation
                target_real_time_rate=real_time_rate,
            )
            sims.append(sim)

        # Function to run a single episode on a specific simulation
        def run_episode(sim_idx: int, episode_idx: int) -> tuple:
            sim = sims[sim_idx]
            success = evaluate_single_episode_async(
                sim=sim,
                policy=policy,
                max_steps=max_steps,
                control_freq=args.control_freq,
                task_emb=task_emb,
                cfg=cfg,
                video_writer=video_writer if sim_idx == 0 else None,  # Only first sim records video
                episode_idx=episode_idx,
                video_camera_name=video_camera_name,
                model_type=args.model_type,
            )
            return episode_idx, success

        try:
            # Start all simulations
            for sim in sims:
                sim.start()

            if batch_size == 1:
                # Sequential execution (original behavior)
                for episode in range(n_eval):
                    episode_idx, success = run_episode(0, episode)

                    if success:
                        num_success += 1

                    # Write to CSV immediately
                    row = create_csv_row(
                        benchmark=args.benchmark,
                        task_id=args.task_id,
                        task_description=task_description,
                        model_type=args.model_type,
                        model_name=model_name,
                        episode_idx=episode_idx,
                        success=success,
                        max_steps=max_steps,
                        control_freq=args.control_freq,
                        real_time_rate=args.real_time_rate,
                        batch_size=batch_size,
                    )
                    append_to_csv(csv_path, row)

                    print(f"  Episode {episode + 1}/{n_eval}: {'Success' if success else 'Fail'} "
                          f"(Running success rate: {num_success}/{episode + 1} = {num_success/(episode+1):.2%})")
            else:
                # Parallel execution with ThreadPoolExecutor
                completed = 0
                episode_queue = list(range(n_eval))

                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # Submit initial batch
                    futures = {}
                    for sim_idx in range(min(batch_size, n_eval)):
                        episode_idx = episode_queue.pop(0)
                        future = executor.submit(run_episode, sim_idx, episode_idx)
                        futures[future] = sim_idx

                    # Process results and submit new tasks
                    while futures:
                        done, _ = concurrent.futures.wait(
                            futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )

                        for future in done:
                            sim_idx = futures.pop(future)
                            episode_idx, success = future.result()

                            if success:
                                num_success += 1
                            completed += 1

                            # Write to CSV immediately
                            row = create_csv_row(
                                benchmark=args.benchmark,
                                task_id=args.task_id,
                                task_description=task_description,
                                model_type=args.model_type,
                                model_name=model_name,
                                episode_idx=episode_idx,
                                success=success,
                                max_steps=max_steps,
                                control_freq=args.control_freq,
                                real_time_rate=args.real_time_rate,
                                batch_size=batch_size,
                            )
                            append_to_csv(csv_path, row)

                            print(f"  Episode {episode_idx + 1}/{n_eval}: {'Success' if success else 'Fail'} "
                                  f"(Running success rate: {num_success}/{completed} = {num_success/completed:.2%})")

                            # Submit next episode if available
                            if episode_queue:
                                next_episode = episode_queue.pop(0)
                                new_future = executor.submit(run_episode, sim_idx, next_episode)
                                futures[new_future] = sim_idx

        finally:
            # Stop all simulations
            for sim in sims:
                sim.stop()

        success_rate = num_success / n_eval

    print(f"\n[info] Async evaluation completed in {t.get_elapsed_time():.2f} seconds")
    print(f"[info] Results saved to CSV: {csv_path}")
    print(f"[info] Success rate: {num_success}/{n_eval} = {success_rate:.2%}")


if __name__ == "__main__":
    main()

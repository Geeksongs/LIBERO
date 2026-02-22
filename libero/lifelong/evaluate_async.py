"""
Asynchronous evaluation script for LIBERO benchmark.
This script uses the AsyncSimulation API from robosuite for real-time evaluation.
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
    sim, algo, task_emb, cfg, max_steps, init_state=None,
    video_writer=None, episode_idx=0, video_camera_name="frontview_image"
):
    """
    Evaluate a single episode using async simulation.

    Args:
        sim: AsyncSimulation instance
        algo: Policy algorithm
        task_emb: Task embedding
        cfg: Configuration
        max_steps: Maximum steps per episode
        init_state: Optional initial state to set
        video_writer: Optional VideoWriter for saving videos
        episode_idx: Episode index for video saving
        video_camera_name: Camera name for video recording (e.g., "frontview_image")

    Returns:
        bool: Whether the episode was successful
    """
    # Reset the simulation
    sim.reset()
    algo.reset()

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

    steps = 0
    success = False

    with torch.no_grad():
        while steps < max_steps and not sim.done():
            steps += 1

            # Save video frame if video_writer is provided
            if video_writer is not None and video_writer.save_video:
                if video_camera_name in obs.data:
                    video_writer.append_obs(obs.data, done=False, idx=episode_idx, camera_name=video_camera_name)

            # Convert observation to tensor format
            data = async_obs_to_tensor_obs(obs.data, task_emb, cfg)

            # Get action from policy
            action = algo.policy.get_action(data)

            # Send action to simulation (action is [1, 7], need to squeeze)
            action = action.squeeze(0) if len(action.shape) > 1 else action
            sim.control_stream.push(action)

            # Get next observation
            try:
                obs = sim.observation_stream.get(timeout=1.0)
            except TimeoutError:
                print(f"[warning] Timeout waiting for observation at step {steps}")
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
    parser = argparse.ArgumentParser(description="Async Evaluation Script for LIBERO")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--control_freq", type=float, default=20.0,
                        help="Control frequency for async simulation")
    parser.add_argument("--observation_freq", type=float, default=20.0,
                        help="Observation frequency for async simulation")
    parser.add_argument("--save-videos", action="store_true",
                        help="Save evaluation videos")
    parser.add_argument("--video-height", type=int, default=480,
                        help="Video frame height (default: 480)")
    parser.add_argument("--video-width", type=int, default=480,
                        help="Video frame width (default: 480)")
    parser.add_argument("--video-camera", type=str, default="frontview",
                        help="Camera for video recording: frontview, agentview, sideview, birdview (default: frontview)")
    parser.add_argument("--real-time-rate", type=float, default=1.0,
                        help="Target real-time rate: 1.0=realtime, 2.0=2x speed, 0=as fast as possible (default: 1.0)")

    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id) if args.device_id is not None else "cuda:0"
    args.save_dir = f"{args.experiment_dir}_saved"

    if args.algo == "multitask":
        assert args.ep in list(
            range(0, 50, 5)
        ), "[error] ep should be in [0, 5, ..., 50]"
    else:
        assert args.load_task in list(
            range(10)
        ), "[error] load_task should be in [0, ..., 9]"
    return args


def main():
    args = parse_args()

    print(f"[info] Starting async evaluation for benchmark={args.benchmark}, task_id={args.task_id}")

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
    except:
        print(f"[error] cannot find the checkpoint at {str(model_path)}")
        sys.exit(0)

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = args.device_id
    algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask

    if cfg.lifelong.algo == "PackNet":
        algo.eval()
        for module_idx, module in enumerate(algo.policy.modules()):
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                mask = algo.previous_masks[module_idx].to(cfg.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(args.task_id + 1)] = 0.0
            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()

    algo.policy.load_state_dict(sd)

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # Get the benchmark and task
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args. task_id)
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
    except:
        print(
            f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}"
        )
        sys.exit(0)

    algo.eval()

    # Setup save paths
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

    ### ======================= start async evaluation ============================

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
            camera_heights = [cfg.data.img_h, cfg.data.img_h] + [args.video_height] * (len(camera_names) - 2)
            camera_widths = [cfg.data.img_w, cfg.data.img_w] + [args.video_width] * (len(camera_names) - 2)
        else:
            camera_heights = cfg.data.img_h
            camera_widths = cfg.data.img_w

        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
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

        try:
            sim.start()

            for episode in range(n_eval):
                success = evaluate_single_episode_async(
                    sim=sim,
                    algo=algo,
                    task_emb=task_emb,
                    cfg=cfg,
                    max_steps=cfg.eval.max_steps,
                    video_writer=video_writer,
                    episode_idx=episode,
                    video_camera_name=video_camera_name,
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
        }

        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(eval_stats, save_folder)

    print(f"\n[info] Async evaluation completed in {t.get_elapsed_time():.2f} seconds")
    print(f"[info] Results saved at {save_folder}")
    print(f"[info] Success rate: {num_success}/{n_eval} = {success_rate:.2%}")


if __name__ == "__main__":
    main()

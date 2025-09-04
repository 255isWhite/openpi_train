"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch

import numpy as np
import tqdm
import json
import os
from pathlib import Path
from collections import defaultdict

SAVE_JSON = "/home/wzh/openpi/examples/aloha_robotwin/noopt_2.json"

# 确保 JSON 文件可用
if not os.path.exists(SAVE_JSON) or os.path.getsize(SAVE_JSON) == 0:
    with open(SAVE_JSON, "w") as f:
        json.dump([], f, indent=2)

GLOBAL_TRANS = {}

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为彩色图像
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: "LeRobotDataset",
    hdf5_files: list[Path],
    task: str,  # 这里虽然传空，但保留接口
    episodes: list[int] | None = None,
    threshold: float = 1e-3,   # 阈值可调
) -> "LeRobotDataset":
    if episodes is None:
        episodes = range(len(hdf5_files))

    # 统计表，按 instruction 分类
    stats = defaultdict(lambda: {"original": 0, "kept": 0})

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        # load instructions
        dir_path = os.path.dirname(ep_path)
        task_name = dir_path.split('/')[-2].split('-')[0] 
        print(f"dir_path: {dir_path}")
        json_Path = f"{dir_path}/instructions.json"
        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict['instructions']
            instruction = np.random.choice(instructions)  # 用 instruction 作为分类依据
        GLOBAL_TRANS[instruction] = task_name  # 记录 instruction -> task_name 的映射
        kept_count = 0
        noopt_frames = []
        for i in range(num_frames):
            if i > 0:
                diff = np.linalg.norm(action[i] - action[i-1])
                if diff < threshold:
                    continue  # 跳过静止帧
            noopt_frames.append(i)

        for i in range(len(noopt_frames)):
            if i == len(noopt_frames) - 1:
                continue  # 最后一帧不处理
            next_action = action[noopt_frames[i + 1]]
            current_action = action[noopt_frames[i]]
            # next minus current
            delta_action = next_action - current_action
            # for gripper
            delta_action[6] = next_action[6]
            delta_action[13] = next_action[13]
            
            frame = {
                "observation.state": state[i],
                "action": delta_action,
                "task": instruction,
            }
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)
            kept_count += 1

        dataset.save_episode()

        # 打印压缩比 + 实时写 JSON
        ratio = kept_count / num_frames * 100
        msg = f"[Episode {ep_idx}] Instruction: {instruction}, Original: {num_frames}, Kept: {kept_count}, Ratio: {ratio:.2f}%"
        print(msg)

        # 实时写 JSON
        with open(SAVE_JSON, "r") as f:
            log_data = json.load(f)
        log_data.append({
            "episode": int(ep_idx),
            "task_name": task_name,
            "instruction": instruction,
            "original_frames": int(num_frames),
            "kept_frames": int(kept_count),
            "ratio_percent": round(ratio, 2)
        })
        with open(SAVE_JSON, "w") as f:
            json.dump(log_data, f, indent=2)

        # 更新全局统计
        stats[instruction]["original"] += num_frames
        stats[instruction]["kept"] += kept_count

    # 打印 & 写入分 instruction 全局统计
    total_original, total_kept = 0, 0
    for instr, values in stats.items():
        orig = values["original"]
        kept = values["kept"]
        ratio = kept / orig * 100 if orig > 0 else 0
        print(f"[Global] Instruction: {instr}, Original: {orig}, Kept: {kept}, Ratio: {ratio:.2f}%")
        with open(SAVE_JSON, "r") as f:
            log_data = json.load(f)
        log_data.append({
            "instruction": instr,
            "task_name": GLOBAL_TRANS[instr],
            "total_original_frames": int(orig),
            "total_kept_frames": int(kept),
            "global_ratio_percent": round(ratio, 2)
        })
        with open(SAVE_JSON, "w") as f:
            json.dump(log_data, f, indent=2)

        total_original += orig
        total_kept += kept

    # overall 统计
    overall_ratio = total_kept / total_original * 100 if total_original > 0 else 0
    print(f"[Overall] All Instructions, Original: {total_original}, Kept: {total_kept}, Ratio: {overall_ratio:.2f}%")
    with open(SAVE_JSON, "r") as f:
        log_data = json.load(f)
    log_data.append({
        "instruction": "OVERALL",
        "total_original_frames": int(total_original),
        "total_kept_frames": int(total_kept),
        "global_ratio_percent": round(overall_ratio, 2)
    })
    with open(SAVE_JSON, "w") as f:
        json.dump(log_data, f, indent=2)

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)

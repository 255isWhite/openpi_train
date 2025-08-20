"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
from pathlib import Path
import os
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import math
import numpy as np
from tqdm import tqdm
import h5py
import cv2

REPO_NAME = "LIBERO130_10shot"  # Name of the output dataset, also used for the Hugging Face Hub
DATASET_DIR = "/mnt/ssd0/data/libero"
DATASET_DIR = Path(DATASET_DIR)        # 转成 Path 对象
GLOBAL_N = 10


def main():
    # Clean up any existing dataset in the output directory
    # output_path = os.path.join("/mnt/ssd1/data/zh1/pi0/datasets", REPO_NAME)
    # if output_path.exists():
    #     shutil.rmtree(output_path)

    path = "/mnt/ssd1/data/zh1/lerobot/LIBERO130_10shot"

    if os.path.exists(path):
        shutil.rmtree(path)  # 递归删除目录及所有内容
        print(f"已删除 {path}")
    else:
        print(f"{path} 不存在，跳过删除")

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
            
            
    level1_dirs = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    index = 0
    for d1 in level1_dirs:
        # 第二层目录
        level2_dirs = sorted([p for p in d1.iterdir() if p.is_dir()])
        for d2 in level2_dirs:
            # 该叶子目录下的 h5 文件（按字典序）
            h5_files = sorted(list(d2.glob("*.h5")) + list(d2.glob("*.hdf5")))
            h5_files = h5_files[:GLOBAL_N]  # 只取前 n 个

            for h5_path in tqdm(h5_files, desc=f"{d1.name}/{d2.name}", leave=False):
                index += 1
                print(f"Processing file {index}{d1.name}/{d2.name}/: {h5_path.name}")
                with h5py.File(h5_path, "r") as f:
                    print(f"Processing file: {h5_path.name}")
                    # actions = f['action'][()]
                    # proprios = f['proprio'][()]
                    # third_imgs = f['observation']['third_image'][()]
                    # wrist_imgs = f['observation']['wrist_image'][()]
                    # # TODO: 在这里处理数据
                    
                    # proprios = convert_proprios(proprios)  # 转换四元数到 axis-angle
                    # assert actions.shape[0] == third_imgs.shape[0] == wrist_imgs.shape[0] == proprios.shape[0]
                    # # print(f"episode length is {actions.shape[0]}")
                    # print(f"episode length is {actions.shape[0]}")
                    # for i in range(actions.shape[0]):
                    #     # print(f" shape of third_imgs[i]third_imgs[i] is {third_imgs[i]}")
                    #     dataset.add_frame(
                    #         {
                    #             "image": cv2.imdecode(third_imgs[i],  cv2.IMREAD_COLOR),
                    #             "wrist_image": cv2.imdecode(wrist_imgs[i],  cv2.IMREAD_COLOR),
                    #             "state": proprios[i],
                    #             "actions": actions[i],
                    #             "task": f['language_instruction'][()].decode('utf-8'),
                    #         }
                    #     )
                    # dataset.save_episode()
                
        




def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def convert_proprios(proprios):
    xyz = proprios[:, :3]
    quats = proprios[:, 3:7]
    grippers = proprios[:, 7:]

    # 批量转换四元数 -> axis-angle
    axis_angles = np.array([_quat2axisangle(q) for q in quats])  # (n, 3)

    # 拼接新的 (n, 8)
    new_proprios = np.hstack([xyz, axis_angles, grippers])
    return new_proprios

if __name__ == "__main__":
    main()

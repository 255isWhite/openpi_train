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
import os
import fnmatch
REPO_NAME = "Airbot_Cloth"  # Name of the output dataset, also used for the Hugging Face Hub
DATASET_DIR = "/mnt/ssd0/data/airbot_cloth"
DATASET_DIR = Path(DATASET_DIR)        # 转成 Path 对象


def main():
    # Clean up any existing dataset in the output directory
    # output_path = os.path.join("/mnt/ssd1/data/zh1/pi0/datasets", REPO_NAME)
    # if output_path.exists():
    #     shutil.rmtree(output_path)

    path = f"/mnt/ssd1/data/zh1/lerobot/{REPO_NAME}"

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
        robot_type="airbot",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
            
    hdf5_files = []
    count = 0
    for root, _, files in os.walk(DATASET_DIR):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)
    print(f"找到 {len(hdf5_files)} 个 hdf5 文件")
    

    required_keys = [
        "eef_6d",
        "images/top_image",
        "images/wrist_image",
        "language_instruction"
    ]
    missing_files = []  # 用来记录缺 key 的文件
    for index in tqdm(range(len(hdf5_files)), leave=False):
        h5_path = hdf5_files[index]
        count += 1

        try:
            with h5py.File(h5_path, "r") as f:
                # 检查 keys 是否都存在
                has_all_keys = True
                for k in required_keys:
                    parts = k.split("/")
                    g = f
                    for p in parts:
                        if p in g:
                            g = g[p]
                        else:
                            has_all_keys = False
                            break
                    if not has_all_keys:
                        break

                if not has_all_keys:
                    missing_files.append(h5_path)
                    continue  # 跳过这个文件

                # 读取数据
                actions = f["eef_6d"][()]
                proprios = f["eef_6d"][()]
                third_imgs = f["images"]["top_image"][()]
                wrist_imgs = f["images"]["wrist_image"][()]

                assert actions.shape[0] == third_imgs.shape[0] == wrist_imgs.shape[0] == proprios.shape[0]
                print(f"episode length is {actions.shape[0]}")

                for i in range(actions.shape[0]):
                    dataset.add_frame(
                        {
                            "image": cv2.imdecode(third_imgs[i], cv2.IMREAD_COLOR),
                            "wrist_image": cv2.imdecode(wrist_imgs[i], cv2.IMREAD_COLOR),
                            "state": proprios[i],
                            "actions": actions[i],
                            "task": f["language_instruction"][()].decode("utf-8"),
                        }
                    )
                dataset.save_episode()

        except Exception as e:
            print(f"⚠️ Error reading {h5_path}: {e}")
            missing_files.append(h5_path)

    # 遍历完成后打印缺少 key 的文件
    if missing_files:
        print("\n❌ 以下文件缺少必要 key 或读取失败：")
        for p in missing_files:
            print(p)
    else:
        print("\n✅ 所有文件都包含必要 keys")
                
        




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
    grippers = proprios[:, :2]
    xyz = proprios[:, 2:5]
    quats = proprios[:, 5:9]

    # 批量转换四元数 -> axis-angle
    axis_angles = np.array([_quat2axisangle(q) for q in quats])  # (n, 3)

    # 拼接新的 (n, 8)
    new_proprios = np.hstack([xyz, axis_angles, grippers])
    return new_proprios

if __name__ == "__main__":
    main()

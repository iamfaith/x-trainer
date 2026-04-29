import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
import numpy as np

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.array(x)

def merge_datasets(src_roots, out_root, out_repo_id):
    # load first to get canonical features/fps
    ds0 = LeRobotDataset(repo_id=os.path.basename(src_roots[0]), root=src_roots[0], download_videos=False)
    # basic compatibility checks
    for r in src_roots[1:]:
        ds = LeRobotDataset(repo_id=os.path.basename(r), root=r, download_videos=False)
        if ds.fps != ds0.fps:
            raise ValueError("fps mismatch")
        if ds.features != ds0.features:
            raise ValueError("features mismatch (must be identical)")

    # create new dataset
    new_ds = LeRobotDataset.create(
        repo_id=out_repo_id,
        fps=ds0.fps,
        features=ds0.features,
        root=out_root,
        robot_type=ds0.meta.robot_type,
        use_videos=(len(ds0.meta.video_keys) > 0),
    )

    for src in src_roots:
        src_ds = LeRobotDataset(repo_id=os.path.basename(src), root=src, download_videos=False)
        ep_indices = sorted(int(k) for k in src_ds.meta.episodes.keys())
        for ep_idx in ep_indices:
            # load single-episode view to iterate frames
            sub = LeRobotDataset(repo_id=src_ds.repo_id, root=src, episodes=[ep_idx], download_videos=False)
            for i in range(len(sub)):
                item = sub[i]
                # build frame dict with only features keys
                frame = {}
                for key, ft in src_ds.features.items():
                    if ft["dtype"] in ["image", "video"]:
                        frame[key] = to_numpy(item[key])
                    else:
                        frame[key] = to_numpy(item[key])
                task = item["task"]
                new_ds.add_frame(frame=frame, task=task)
            new_ds.save_episode()
            print(f"Appended episode {ep_idx} from {src}")

    print("Merge complete. Merged dataset at:", out_root)

if __name__ == "__main__":
    # Example usage: adjust paths
    src_roots = [
        r"C:\path\to\dataset_A",
        r"C:\path\to\dataset_B",
    ]
    out_root = r"C:\path\to\merged_dataset"
    out_repo_id = "merged_dataset_local"
    merge_datasets(src_roots, out_root, out_repo_id)
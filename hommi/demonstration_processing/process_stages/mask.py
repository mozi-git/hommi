import os
import pathlib
import argparse
import concurrent.futures
import imageio
import numpy as np
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
import multiprocessing


def prompt_user_to_label_first_frame(frame, save_path):
    plt.figure(figsize=(9, 6))
    plt.title(
        "Label Points:\nLeft Click = Positive (Green), Right Click = Negative (Red)\nClose window when done."
    )
    plt.imshow(frame)
    points_list = []
    labels_list = []

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:  # Left click for positive
            points_list.append([x, y])
            labels_list.append(1)
            plt.plot(x, y, "g*", markersize=10)
        elif event.button == 3:  # Right click for negative
            points_list.append([x, y])
            labels_list.append(0)
            plt.plot(x, y, "r*", markersize=10)
        plt.draw()

    cid = plt.gcf().canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    points = np.array(points_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    with open(save_path, "wb") as f:
        pickle.dump({"points": points, "labels": labels}, f)
    return points, labels



def apply_mask_to_frame(frame, mask):
    masked = frame.copy()
    masked[~mask] = 0
    return masked


def process_video_task(args):
    video_path, output_path, sam2_checkpoint_path, reference_pkl_path, device_str = args
    print(f"Processing {video_path}")
    device = torch.device(device_str)

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        os.path.expanduser(sam2_checkpoint_path),
        device=device
    )

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    first_frame = reader.get_data(0)

    if os.path.exists(reference_pkl_path):
        with open(reference_pkl_path, "rb") as f:
            data = pickle.load(f)
            points, labels = data["points"], data["labels"]
    else:
        points, labels = prompt_user_to_label_first_frame(first_frame, reference_pkl_path)

    inference_state = predictor.init_state(video_path=str(video_path))
    predictor.reset_state(inference_state)

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    masks = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        combined_mask = np.zeros_like(mask_logits[0][0].cpu().numpy(), dtype=bool)
        for i in range(len(obj_ids)):
            mask = (mask_logits[i] > 0.0).cpu().numpy()[0]
            combined_mask |= mask
        masks[frame_idx] = combined_mask

    # Re-open reader again for writing
    reader = imageio.get_reader(video_path)
    writer = imageio.get_writer(output_path, fps=fps)

    for idx, frame in enumerate(reader):
        if idx in masks:
            frame = apply_mask_to_frame(frame, ~masks[idx])
        writer.append_data(frame)
    writer.close()

    print(f"Finished {video_path.name}")
    return video_path.name


def main(
    video_dir,
    output_dir,
    sam2_checkpoint_path,
    reference_dir,
    max_workers=4,
    session_name="default_session"
):
    video_dir = pathlib.Path(video_dir).expanduser()
    output_dir = pathlib.Path(output_dir).expanduser()
    reference_dir = pathlib.Path(reference_dir).expanduser()
    reference_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_pkl_path = reference_dir / "default_prompt.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_paths = [p for p in list(video_dir.glob("*_left_rgb.mp4")) + list(video_dir.glob("*_right_rgb.mp4"))
                   if p.name.split("_")[-4] == session_name]

    args_list = [
        (video_path, output_dir / (video_path.stem + "_masked.mp4"), sam2_checkpoint_path, reference_pkl_path, device)
        for video_path in video_paths
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(process_video_task, args_list), total=len(args_list)):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="~/data")
    parser.add_argument("--output_dir", default="~/data")
    parser.add_argument("--sam2_checkpoint_path", default="~/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--reference_dir", default="~/data")
    parser.add_argument("--session_name", required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    main(
        args.video_dir,
        args.output_dir,
        args.sam2_checkpoint_path,
        args.reference_dir,
        args.max_workers,
        args.session_name
    )

import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]  # Add more colors if tracking multiple objects

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    prompts = {}
    for obj_id, line in enumerate(lines):
        x, y, w, h = map(float, line.strip().split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[obj_id] = ((x, y, x + w, y + h), 0)
    return prompts


def determine_model_cfg(model_path):
    # Find the folder where configs actually live
    script_dir = osp.dirname(osp.abspath(__file__))
    cfg_dir = osp.normpath(osp.join(script_dir, "..", "sam2", "sam2", "configs", "samurai"))

    if "large" in model_path:
        return osp.join(cfg_dir, "sam2.1_hiera_l.yaml")
    elif "base_plus" in model_path:
        return osp.join(cfg_dir, "sam2.1_hiera_b+.yaml")
    elif "small" in model_path:
        return osp.join(cfg_dir, "sam2.1_hiera_s.yaml")
    elif "tiny" in model_path:
        return osp.join(cfg_dir, "sam2.1_hiera_t.yaml")
    else:
        raise ValueError("Unknown model size in path!")

# def determine_model_cfg(model_path):
#     if "large" in model_path:
#         return "configs/samurai/sam2.1_hiera_l.yaml"
#     elif "base_plus" in model_path:
#         return "configs/samurai/sam2.1_hiera_b+.yaml"
#     elif "small" in model_path:
#         return "configs/samurai/sam2.1_hiera_s.yaml"
#     elif "tiny" in model_path:
#         return "configs/samurai/sam2.1_hiera_t.yaml"
#     else:
#         raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)

    # Prepare output directory for masks
    sequence_name = osp.basename(args.video_path.rstrip("/\\"))
    output_dir = osp.join(r"D:/Thesis/thesis-temporal-deid/output/samurai/davis", sequence_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load frames if video output requested
    frame_rate = 30
    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.lower().endswith((".jpg", ".jpeg"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

        if len(loaded_frames) == 0:
            raise ValueError("No frames were loaded from the video.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)

        # Initialize all objects with bounding boxes from prompts
        for obj_id, (bbox, _) in prompts.items():
            predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=obj_id)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

                # Save mask per object per frame
                mask_img_uint8 = (mask * 255).astype(np.uint8)
                mask_filename = f"mask_obj{obj_id}_frame{frame_idx:05d}.png"
                mask_path = osp.join(output_dir, mask_filename)
                cv2.imwrite(mask_path, mask_img_uint8)

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                out.write(img)

        if args.save_to_video:
            out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", action="store_true", help="Save results to a video.")  # Boolean flag
    args = parser.parse_args()
    main(args)





# python samurai/scripts/demo.py --video_path "D:/Thesis/thesis-temporal-deid/input/davis2017/JPEGImages/480p/hike" --txt_path "D:/Thesis/thesis-temporal-deid/input/davis2017/bboxes/bbox_hike.txt" --model_path "D:/Thesis/thesis-temporal-deid/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt"



# below working too
# python samurai/scripts/demo.py --video_path "input/davis2017/JPEGImages/480p/hike" --txt_path "input/davis2017/bboxes/bbox_hike.txt" --model_path "samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt"

# python samurai/scripts/demo.py --video_path "input/davis2017/JPEGImages/480p/dog" --txt_path "input/davis2017/bboxes/bbox_dog.txt" --model_path "samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt"

# python samurai/scripts/demo.py --video_path "input/davis2017/JPEGImages/480p/camel" --txt_path "input/davis2017/bboxes/bbox_camel.txt" --model_path "samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
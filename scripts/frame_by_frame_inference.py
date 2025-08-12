import os
import cv2
import torch
import numpy as np
import argparse
import sys
sys.path.append("./sam2")
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

color = [(255, 0, 0)]  # Add more colors if you want for multiple objects

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    prompts = {}
    for obj_id, line in enumerate(lines):
        x, y, w, h = map(float, line.strip().split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[obj_id] = ((x, y, x + w, y + h), 0)
    return prompts

def main(args):
    print(f"Loading model checkpoint: {args.checkpoint}")
    print(f"Using model config: {args.model_cfg}")
    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint))
    
    frames_dir = os.path.join(args.input_base, args.sequence)
    output_dir = os.path.join(args.output_base, args.sequence)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing frames from: {frames_dir}")
    print(f"Saving masks to: {output_dir}")
    
    bbox_file = os.path.join(args.bbox_base, f"bbox_{args.sequence}.txt")
    print(f"Loading bounding box from: {bbox_file}")
    with open(bbox_file, 'r') as f:
        line = f.readline()
        x, y, w, h = map(int, line.strip().split(','))
        box = [x, y, x + w, y + h]
    print(f"Bounding box: {box}")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Found {len(frame_files)} frames to process.")

    for idx, filename in enumerate(frame_files):
        image_path = os.path.join(frames_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {filename}, skipping.")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor.set_image(image_rgb)
        
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        mask = masks[0].cpu().numpy()
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(output_dir, f"mask_{filename.rsplit('.',1)[0]}.png")
        cv2.imwrite(mask_path, mask_uint8)
        
        print(f"[{idx+1}/{len(frame_files)}] Saved mask for {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="Name of the DAVIS sequence, e.g., 'bear'")
    parser.add_argument("--input_base", default=r"D:/Thesis/thesis-temporal-deid/input/davis2017/JPEGImages/480p", help="Base folder of input frames")
    parser.add_argument("--bbox_base", default=r"D:/Thesis/thesis-temporal-deid/input/davis2017/bboxes", help="Base folder of bbox txt files")
    parser.add_argument("--output_base", default=r"D:/Thesis/thesis-temporal-deid/output/samurai/davis", help="Base folder for output masks")
    parser.add_argument("--checkpoint", default=r"D:/Thesis/thesis-temporal-deid/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to SAM 2 checkpoint")
    parser.add_argument("--model_cfg", default="configs/sam2.1_hiera_b+.yaml", help="Path to SAM 2 model config (relative to sam2 repo)")
    args = parser.parse_args()
    main(args)





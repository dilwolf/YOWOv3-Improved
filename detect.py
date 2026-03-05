import os
import cv2
import yaml
import time
import torch
import shutil
import argparse
import numpy as np
from utils import util
from pathlib import Path
from model.TSN.YOWOv3 import build_yowov3

def build_config():
    ucf_config_file = 'utils/best_2falldown.yaml'
    with open(ucf_config_file, "r") as file:
        ucf_config = yaml.load(file, Loader=yaml.SafeLoader) 
    
    return ucf_config

def plot_bboxes(original_frame, bboxes, labels, conf_scores, mapping):
    labels = [int(l) for l in labels]
    for score, cls, bbox in zip(conf_scores, labels, bboxes): # loop over all bboxes
        class_label = mapping[cls] # class name
        label = f"{class_label} : {score*100:0.2f}" # bbox label
        lbl_margin = 3 #label margin
        img = cv2.rectangle(original_frame, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color=(0, 255, 0),
                            thickness=1)
        label_size = cv2.getTextSize(label, # labelsize in pixels 
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                     fontScale=1, thickness=1)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), # plot label background
                             (int(bbox[0])+lbl_w, int(bbox[1])-lbl_h),
                             color=(0, 255, 0), 
                             thickness=-1) # thickness=-1 means filled rectangle
        cv2.putText(img, label, (int(bbox[0])+ lbl_margin, int(bbox[1])-lbl_margin), # write label to the image
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(255, 255, 255 ),
                    thickness=1)
    return img

def scale(coords, shape1, shape2, ratio_pad=None):
    
    if ratio_pad is None:  # calculate from shapes
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])  # old / new
        pad = ((shape1[1] - shape2[1] * gain) / 2,
               (shape1[0] - shape2[0] * gain) / 2)
    else:
        (gain_w, _), (pad_w, pad_h) = ratio_pad
        gain = gain_w
        pad = (pad_w, pad_h)

    coords[:, [0, 2]] -= pad[0]  # remove x padding
    coords[:, [1, 3]] -= pad[1]  # remove y padding
    coords[:, :4] /= gain        # scale back

    coords[:, 0].clamp_(0, shape2[1])  # x1
    coords[:, 1].clamp_(0, shape2[0])  # y1
    coords[:, 2].clamp_(0, shape2[1])  # x2
    coords[:, 3].clamp_(0, shape2[0])  # y2
    return coords

def _letterbox(frames, target_size):
        """
        Transform clip (frames) for model input
        """

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        target_h, target_w = target_size
        h0, w0 = frames[0].shape[:2]  # original shape

        # Scale factor (gain)
        r = min(target_h / h0, target_w / w0)
        # r = min(r, 1.0)

        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))

        # Compute padding
        dw = (target_w - new_w) / 2
        dh = (target_h - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # Resize + pad frames
        processed_frames = []
        for frame in frames:
            # Letterbox
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            frame = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            # Normalize and convert to tensor
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - mean) / std
            processed_frames.append(torch.from_numpy(frame).float())
        
        # Stack: (N, H, W, C) -> (C, N, H, W)
        _frames = torch.stack(processed_frames, dim=0)
        _frames = _frames.permute(3, 0, 1, 2)  # Direct: (N, H, W, C) -> (C, N, H, W)

        # Shapes (orig_shape, (gain, pad))
        shapes = ((h0, w0), ((r, r), (dw, dh)))

        return _frames, shapes

@torch.no_grad()
def detect(config, input_path, model_weight):
    # Initialize the model
    config["pretrain_path"] = str(model_weight)
    model = build_yowov3(config)
    model.to("cuda")
    model.eval()
    util.get_info(config, model)
    class_mapping = config['idx2name']
    clip_length = config['clip_length']
    img_size = config['img_size']
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    # Determine input type: video or image folder
    input_path = str(input_path)
    print(f"input_path: {input_path}, {type(input_path)}")
    if os.path.isfile(input_path) and input_path.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
        is_video = True
    elif os.path.isdir(input_path):
        is_video = False
    else:
        raise ValueError("Input must be a video file or a directory containing image frames.")

    fps = 30  # Manual FPS for output video
    frame_width, frame_height = img_size  # Will be updated if video

    # Generate output filename
    if is_video:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        directory_path = os.path.dirname(input_path)
        output_filename = f"{directory_path}/{base_name}_output.avi"
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        base_name = Path(input_path).name
        output_filename = f"{input_path}/{base_name}_output.avi"
        # Get image list
        img_extensions = ('.png', '.jpg', '.jpeg')
        image_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(img_extensions)]) # file names 05:d{img_ext} format
        if not image_files:
            raise ValueError("No image files found in the directory.")
        # Read first image to get dimensions
        first_img_path = os.path.join(input_path, image_files[0])
        sample_img = cv2.imread(first_img_path)
        if sample_img is None:
            raise ValueError(f"Could not read first image: {first_img_path}")
        frame_height, frame_width = sample_img.shape[:2]

    # Set output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Buffer to store all frames for sliding window
    all_frames = []
    frame_idx = 0
    img_iter = iter(image_files) if not is_video else None

    # Read all frames first
    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            try:
                img_name = next(img_iter)
                img_path = os.path.join(input_path, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Warning: Failed to load image {img_path}, skipping.")
                    continue
            except StopIteration:
                break
        
        all_frames.append(frame)

    if is_video:
        cap.release()

    print(f"Total frames loaded: {len(all_frames)}")

    # Process with sliding window
    for start_idx in range(len(all_frames)):
        # Sample frames: start_idx, start_idx+7, start_idx+14, ...
        frame_list = []
        sampled_indices = []
        
        for i in range(clip_length):
            sample_idx = start_idx + i * 7
            if sample_idx >= len(all_frames):
                break
            frame_list.append(cv2.cvtColor(all_frames[sample_idx], cv2.COLOR_BGR2RGB))
            sampled_indices.append(sample_idx)
        
        # Skip if we don't have enough frames for a full clip
        if len(frame_list) < clip_length:
            break

        start_time = time.perf_counter_ns()

        # Prepare model input
        clip, shapes = _letterbox(frame_list, img_size)
        clip = clip.unsqueeze(0).to("cuda")  # [batch, C, T, H, W]

        # Model Inference
        outputs = model(clip)
        outputs = util.non_max_suppression(outputs, conf_threshold=0.25, iou_threshold=0.7)[0]

        # Use the middle frame or last sampled frame for visualization
        original_frame = all_frames[start_idx].copy()

        if outputs is not None and len(outputs) > 0:
            
            # Scale bounding boxes
            scale(outputs[:, :4], (img_size[0], img_size[1]), shapes[0], shapes[1])
            bboxes = outputs[:, :4]
            original_frame = plot_bboxes(original_frame, bboxes, outputs[:, 5], outputs[:, 4], class_mapping)

        end_time = time.perf_counter_ns()
        execution_time = (end_time - start_time) / 1_000_000
        print(f"Frame {start_idx} - Sampled indices: {sampled_indices} - Execution time: {execution_time:.2f} ms")

        # Write to output video
        out.write(original_frame)

    out.release()
    print(f"Output video saved to: {output_filename}")
            
    # Cleanup
    if is_video:
        cap.release()
    out.release()
    # cv2.destroyAllWindows()

    print(f"Output saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default="test.mp4", type=str, help='input is a video or directory of images')
    parser.add_argument('--weight', default="best.pth", type=str, help='path to weight')
    args = parser.parse_args()
    config = build_config()
    detect(config, args.source , args.weight)
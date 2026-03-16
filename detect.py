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
from collections import deque
from model.TSN.YOWOv3 import build_yowov3

import os

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

def detect_input_type(input_path):
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTS:
            return True, False   # is_video, is_path
        if ext in IMAGE_EXTS:
            return False, False
        raise ValueError("Unsupported file type.")

    if os.path.isdir(input_path):
        image_found = False
        video_found = False
        for f in os.listdir(input_path):
            p = os.path.join(input_path, f)
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                image_found = True
            elif ext in VIDEO_EXTS:
                video_found = True

        if image_found and video_found:
            raise ValueError("Directory contains both image and video files.")
        if image_found:
            return False, True   # image folder
        if video_found:
            return True, True    # video folder
        raise ValueError("Directory contains no supported image or video files.")

    raise ValueError("Input path does not exist.")

def build_config():
    ucf_config_file = 'utils/YAML/falldown_best.yaml'
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
def detect(config, input_path, model_weight, stride):
    # Initialize model
    config["pretrain_path"] = str(model_weight)
    model = build_yowov3(config)
    model.to("cuda")
    model.eval()
    util.get_info(config, model)

    class_mapping = config['idx2name']
    clip_length = config['clip_length']
    img_size = config['img_size']
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    input_path = str(input_path)
    is_video, is_path = detect_input_type(input_path)
    print(f"input_path: {input_path}, is_video: {is_video}, is_path: {is_path}")

    fps = 30
    buffer_size = (clip_length - 1) * 7 + 1

    # Build list of sources to process
    if is_video and is_path:
        # Directory of videos — process each independently
        video_files = sorted([f for f in os.listdir(input_path)
                               if os.path.splitext(f)[1].lower() in VIDEO_EXTS])
        if not video_files:
            raise ValueError("No video files found in the directory.")
        sources = [os.path.join(input_path, f) for f in video_files]
    elif is_video and not is_path:
        # Single video file
        sources = [input_path]
    else:
        # Single image folder — treat as one source
        sources = [input_path]

    for source in sources:
        print(f"\n--- Processing: {source} ---")

        # --- Setup per-source output ---
        if is_video:
            base_name = os.path.splitext(os.path.basename(source))[0]
            directory_path = os.path.dirname(source) or "."
            output_filename = os.path.join(directory_path, f"{base_name}_output.avi")
            cap = cv2.VideoCapture(source)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image_files = None
        else:
            base_name = Path(source).name
            output_filename = os.path.join(source, f"{base_name}_output.avi")
            img_extensions = ('.png', '.jpg', '.jpeg')
            image_files = sorted([f for f in os.listdir(source)
                                   if f.lower().endswith(img_extensions)])
            if not image_files:
                raise ValueError("No image files found in the directory.")
            first_img = cv2.imread(os.path.join(source, image_files[0]))
            if first_img is None:
                raise ValueError(f"Could not read first image: {image_files[0]}")
            frame_height, frame_width = first_img.shape[:2]
            cap = None

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

        # --- Frame reader ---
        img_iter = iter(image_files) if not is_video else None

        def read_next_frame():
            if is_video:
                ret, frame = cap.read()
                return frame if ret else None
            else:
                while True:
                    try:
                        img_name = next(img_iter)
                        frame = cv2.imread(os.path.join(source, img_name))
                        if frame is None:
                            print(f"Warning: skipping unreadable image {img_name}")
                            continue
                        return frame
                    except StopIteration:
                        return None

        # --- Pre-fill buffer ---
        buffer = deque(maxlen=buffer_size)
        frame_idx = 0

        while len(buffer) < buffer_size:
            frame = read_next_frame()
            if frame is None:
                break
            buffer.append(frame)
            frame_idx += 1

        print(f"Buffer pre-filled with {len(buffer)} frames (need {buffer_size})")

        start_idx = 0

        # --- Sliding window inference ---
        while len(buffer) == buffer_size:
            frame_list = [cv2.cvtColor(buffer[i * 7], cv2.COLOR_BGR2RGB)
                          for i in range(clip_length)]

            start_time = time.perf_counter_ns()

            clip, shapes = _letterbox(frame_list, img_size)
            clip = clip.unsqueeze(0).to("cuda")
            outputs = model(clip)
            outputs = util.non_max_suppression(outputs, conf_threshold=0.25, iou_threshold=0.7)[0]

            original_frame = buffer[-1].copy()

            if outputs is not None and len(outputs) > 0:
                scale(outputs[:, :4], (img_size[0], img_size[1]), shapes[0], shapes[1])
                original_frame = plot_bboxes(original_frame, outputs[:, :4],
                                             outputs[:, 5], outputs[:, 4], class_mapping)

            end_time = time.perf_counter_ns()
            print(f"Frame {start_idx} - Time: {(end_time - start_time) / 1_000_000:.2f} ms")

            out.write(original_frame)

            # Slide window by stride
            filled = 0
            for _ in range(stride):
                frame = read_next_frame()
                if frame is None:
                    break
                buffer.append(frame)  # deque auto-drops oldest
                frame_idx += 1
                filled += 1

            if filled < stride:
                break  # source exhausted, can't fill full stride

            start_idx += stride

        out.release()
        if is_video and cap is not None:
            cap.release()

        print(f"Output saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default="test.mp4", type=str, help='input: video file, directory of videos, or directory of images')
    parser.add_argument('--weight', default="best.pth", type=str, help='path to model weight')
    parser.add_argument('--stride', type=int, default=1, help='sliding window stride (1=every frame, 7=non-overlapping clips)')
    args = parser.parse_args()
    config = build_config()
    detect(config, args.source, args.weight, args.stride)
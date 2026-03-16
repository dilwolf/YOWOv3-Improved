import os
import cv2
import yaml
import time
import torch
import argparse
from utils import util
from pathlib import Path
from collections import deque
from model.TSN.YOWOv3 import build_yowov3

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def load_model(config, model_weight):
    """Load either PyTorch or ONNX model based on weight file extension."""
    ext = os.path.splitext(str(model_weight))[1].lower()

    if ext == '.onnx':
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime-gpu")

        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider',
        ]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(str(model_weight), sess_options, providers=providers)

        active = session.get_providers()
        print(f"ONNX Runtime providers active: {active}")
        if 'CUDAExecutionProvider' not in active:
            print("Warning: CUDAExecutionProvider not active, falling back to CPU.")

        return 'onnx', session

    else:
        config["pretrain_path"] = str(model_weight)
        model = build_yowov3(config)
        model.to("cuda")
        model.eval()
        util.get_info(config, model)
        return 'torch', model

def build_config():
    """Build sample config for inference."""
    
    ucf_config_file = 'utils/YAML/sample_config.yaml'
    with open(ucf_config_file, "r") as file:
        ucf_config = yaml.load(file, Loader=yaml.SafeLoader) 
    return ucf_config

@torch.no_grad()
def detect(config, input_path, model_weight, save_sampled_indices, stride):
    # Initialize model
    model_type, model = load_model(config, model_weight)

    class_mapping = config['idx2name']
    clip_length = config['clip_length']
    img_size = config['img_size']
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    input_path = str(input_path)
    is_video, is_path = util.detect_input_type(input_path)
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

            clip, shapes = util.letterbox(frame_list, img_size)
            if model_type == 'torch':
                clip = clip.unsqueeze(0).to("cuda")
            else:
                clip = clip.unsqueeze(0).numpy()
            outputs = util.run_inference(model_type, model, clip)
            outputs = util.non_max_suppression(outputs, conf_threshold=0.25, iou_threshold=0.7)[0]

            original_frame = buffer[-1].copy()

            if outputs is not None and len(outputs) > 0:
                if save_sampled_indices:
                    sampled_indices = [start_idx + i * 7 for i in range(clip_length)]
                    util.save_sampled_frames(buffer, sampled_indices, start_idx,
                                        source, base_name, is_video, image_files)
                util.scale(outputs[:, :4], (img_size[0], img_size[1]), shapes[0], shapes[1])
                original_frame = util.plot_bboxes(original_frame, outputs[:, :4],
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
    parser.add_argument('--save_sampled_indices', action='store_true', help='save sampled frames where object is detected')
    parser.add_argument('--stride', type=int, default=1, help='sliding window stride (1=every frame, 7=non-overlapping clips)')
    args = parser.parse_args()
    config = build_config()
    detect(config, args.source, args.weight, args.save_sampled_indices, args.stride)
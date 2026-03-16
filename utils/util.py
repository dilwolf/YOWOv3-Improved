import os
import cv2
import yaml
import time
import math
import copy
import torch
import shutil
import random
import pickle
import torchvision
import numpy as np
from tqdm import tqdm
from thop import profile
from os import environ
from platform import system

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

def build_config():
    ucf_config_file = 'utils/ucf_config.yaml'
    with open(ucf_config_file, "r") as file:
        ucf_config = yaml.load(file, Loader=yaml.SafeLoader) 
    aug_config = ucf_config['augment']
    
    assert len(ucf_config['idx2name']) == ucf_config['num_classes'], "Mismatch in class configuration!"
    
    return ucf_config, aug_config


def update_mode_list(config):
    root_dir = config['data_root']
    sampling_rate = config['sampling_rate']
    clip_length = config['clip_length']

    VALID_LABEL_EXTENSIONS = ('.txt',)

    for split in ["train", "valid"]:
        labels_root = os.path.join(root_dir, split, "labels")
        cache_dir = os.path.join(root_dir, split, "_cache")
        output_file = os.path.join(root_dir, f"{split}_list.txt")

        file_limit = sampling_rate * (clip_length - 1)

        # Remove cache safely
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Removed cache folder: '{cache_dir}'")

        label_paths = []

        for root, _, files in os.walk(labels_root):

            if 0 < len(files) < file_limit + 1:
                print(f"Skipping - insufficient files: {len(files)} in {root}")
                continue

            files = sorted(files)[file_limit:]

            for file in files:
                if file.lower().endswith(VALID_LABEL_EXTENSIONS) and file[:5].isdigit():

                    full_path = os.path.join(root, file)

                    # Get path starting from split (train/valid)
                    relative_path = os.path.relpath(full_path, root_dir)

                    # Normalize to forward slashes
                    normalized_path = relative_path.replace(os.sep, '/')

                    label_paths.append(normalized_path)

                else:
                    raise ValueError(
                        f"Invalid file format in {root}: '{file}'. "
                        f"Expected {VALID_LABEL_EXTENSIONS} file with 5-digit prefix."
                    )

        # Write to file
        with open(output_file, 'w') as f:
            for path in sorted(label_paths):
                f.write(path + '\n')

    print("\nTrain/valid lists have been updated successfully!\n")


def build_label_cache(config, phase):
    """
    Build label cache once and use later.
    """

    root_path = config['data_root']
    clip_length = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size = config['img_size']
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    data_path = os.path.join(root_path, phase, "images")
    split_path = os.path.join(root_path, f"{phase}_list.txt")

    with open(split_path, 'r') as f:
        lines = [line.strip() for line in sorted(f.readlines())]

    cache_dir = os.path.join(root_path, phase, '_cache')
    os.makedirs(cache_dir, exist_ok=True)

    cache_filename = f"{sampling_rate}x{clip_length}x{img_size[0]}x{img_size[1]}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"Cache already exists: {cache_path}")
        return

    print(f"Creating label cache for {phase} phase at {cache_path}")

    VALID_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

    def get_clip_indices(video_path, key_frame_idx):
        files = os.listdir(video_path)
        file_numbers = []

        for fname in files:
            if fname.lower().endswith(VALID_IMG_EXTENSIONS) and fname[:5].isdigit():
                file_numbers.append(int(fname[:5]))

        if not file_numbers:
            raise FileNotFoundError(f"{video_path} contains no valid image files")

        file_numbers.sort()

        index_pos = file_numbers.index(key_frame_idx)
        clip_indices = sorted(
            file_numbers[:index_pos + 1][::-sampling_rate]
        )[-clip_length:]

        if len(clip_indices) < clip_length:
            clip_indices = [key_frame_idx] * (clip_length - len(clip_indices)) + clip_indices

        return clip_indices

    label_cache = []

    for line in tqdm(lines):

        # Parse key frame path
        key_frame_path = line
        split_parts = key_frame_path.split('/')
        key_frame_idx = int(split_parts[-1].split('.')[-2])
        video_name = '/'.join(split_parts[2:-1])
        video_path = os.path.join(data_path, video_name)

        clip_indices = get_clip_indices(video_path, key_frame_idx)

        ann_paths = [
            os.path.join(video_path.replace("images", "labels"), f"{idx:05d}.txt")
            for idx in clip_indices
        ]

        all_annotations = []

        for ann_file in sorted(ann_paths):

            if not os.path.exists(ann_file):
                raise FileNotFoundError(f"Label file not found: {ann_file}")

            boxes, labels = [], []

            with open(ann_file) as f:
                for row in f:
                    parts = row.strip().split()
                    label_id = int(parts[0])
                    box = [float(x) for x in parts[1:5]]

                    boxes.append(box)
                    labels.append(label_id)

            boxes = (
                np.array(boxes, dtype=np.float32)
                if boxes else np.zeros((0, 4), dtype=np.float32)
            )

            labels = (
                np.array(labels, dtype=np.int64)
                if labels else np.zeros((0,), dtype=np.int64)
            )

            all_annotations.append({
                'boxes': boxes,
                'labels': labels
            })

        label_cache.append(all_annotations)

    with open(cache_path, "wb") as f:
        pickle.dump(label_cache, f)

    print("Cache creation complete.")


class ModelEMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd =  model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    
class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num
            
def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'

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

def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def letterbox(frames, target_size):
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

def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    box1 = box1.T
    box2 = box2.T

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)

def non_max_suppression(prediction, conf_threshold=0.001, iou_threshold=0.7):
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_det = 100  # the maximum number of boxes to keep after NMS
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    start = time.time()
    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index, x in enumerate(prediction):  # image index, image inference
        # Apply constraints

        # [n, 84]
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        # Check shape
        if not x.shape[0]:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections
        outputs[index] = x[i]
        #print(x[i])
        if (time.time() - start) > 0.5 + 0.05 * prediction.shape[0]:
            print(f'WARNING ⚠️ NMS time limit {0.5 + 0.05 * prediction.shape[0]:.3f}s exceeded')
            break  # time limit exceeded

    return outputs

def opacity(img, pt1, pt2):
    x = pt1[0]
    y = pt1[1]
    w = pt2[0] - pt1[0]
    h = pt2[1] - pt1[1] 
    sub_img = img[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    # Putting the image back to its position
    img[y:y+h, x:x+w] = res

# adapted from https://inside-machinelearning.com/en/bounding-boxes-python-function/
# a little modification to suit the purpose
def box_label(image, box, label=None, color=(0, 255, 0), txt_color=(0, 0, 0)):
  """
  :param image : np array [H, W, C] (BGR)
  :param label : text, default = None
  """
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw-1, lineType=cv2.LINE_AA)
  if label is not None:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 10, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    y0, dy = 0,10
    y0 = p1[1] - 2 if outside else p1[1] + h + 2
    for i, line in enumerate(label.split('\n')):
        y = y0 + i*dy
        x = p1[0]
        text_size, _ = cv2.getTextSize(line, cv2.LINE_AA, lw/5, tf)
        text_width, text_height = text_size
        opacity(image, (x, y), (x + text_width + 1, y + text_height + 1))
        cv2.putText(image,
                line, (x, y + text_height),
                0,
                lw / 5,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

  
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    box1 = box1.T
    box2 = box2.T

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)

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

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    ap = np.zeros((nc, tp.shape[1]))
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))

            # Integrate area under curve
            ap[ci, j] = np.trapz(m_pre, m_rec)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    # m_pre, m_rec = p.mean(), r.mean()
    m_rec = r.mean()
    map50, map50_90 = ap50.mean(), ap.mean()
    return tp, fp, m_rec, map50, map50_90

def get_info(config, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    video_clip = torch.randn(1, 3, config['clip_length'], config['img_size'][0], config['img_size'][1]).to(device)

    # set eval mode
    model.trainable = False
    model.eval()

    flops, params = profile(model, inputs=(video_clip, ), verbose=False)

    print('==============================')
    print('FLOPs : {:.2f} G'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))
    print('==============================')

    model.trainable = True

def strip_optimizer(filename):
    x = torch.load(filename, weights_only=True, map_location="cuda")
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, f=filename)

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

def save_sampled_frames(all_frames, sampled_indices, start_idx, input_path, base_name, is_video, image_files):
    """
    Save sampled frames to a folder when object is detected
    """
    
    # Create folder for saving frames
    if is_video:
        video_base_name = os.path.splitext(os.path.basename(input_path))[0]
        save_folder = os.path.join(os.path.dirname(input_path), f"{video_base_name}")
    else:
        save_folder = os.path.join(input_path, f"{base_name}")
    
    os.makedirs(save_folder, exist_ok=True)
    
    # Save sampled frames
    if is_video:
        # For video, we must save from memory
        for i, idx in enumerate(sampled_indices):
            frame_to_save = all_frames[idx]
            save_path = os.path.join(save_folder, f"{idx:05d}.png")
            cv2.imwrite(save_path, frame_to_save)
    else:
        end_idx = sampled_indices[-1]
        for idx in range(start_idx, end_idx + 1):
            list_idx = idx - 1  # Convert frame number to list index
            if list_idx < len(image_files):
                original_file = os.path.join(input_path, image_files[list_idx])
                file_ext = os.path.splitext(image_files[list_idx])[1]
                save_path = os.path.join(save_folder, f"{idx:05d}{file_ext}")
                shutil.copy2(original_file, save_path)
    
    print(f"Saved {len(sampled_indices)} frames to {save_folder}")

def run_inference(model_type, model, model_input):
    if model_type == 'torch':
        with torch.no_grad():
            return model(model_input)
    else:
        input_name  = model.get_inputs()[0].name
        outputs  = model.run(None, {input_name: model_input})
        
        return torch.from_numpy(outputs[0])
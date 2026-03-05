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

def build_config():
    config_file = 'utils/config.yaml'
    with open(config_file, "r") as file:
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

def draw_bounding_box(image, bboxes, labels, confs, map_labels):
    """
    Vẽ bouding box, đầu và image có thể là tensor hoặc np array tuỳ tình huống, đã thiết kế để xử lý cả 2 trường hợp
    
    :param image, tensor [1, 3, H, W] (RGB) hoặc numpy array [H, W, 3] (BGR)
    :param bboxes, tensor [nbox, 4]
    :param labels, tensor [nbox]
    :param confs, tensor [nbox]
    :param map_labels, dict, do labels là số nên cần map thành nhãn (string)
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)[:, :, (2, 1, 0)].contiguous()
        image = image.detach().cpu().numpy()
    
    H, W, C = image.shape 

    pre_box   = []
    meta_data = []

    if bboxes is not None:
        for box, label, conf in zip(bboxes, labels, confs):
            box = box.clone().detach()
            text    = str(map_labels[int(label.item())] + " : " + str(round(conf.item()*100, 2)))
            pre_box.append(box)
            meta_data.append([text, H, W])

        res_box = []
        res_meta_data = []

        for idx1, box1 in enumerate(pre_box):
            flag = 1
            for idx2, box2 in enumerate(res_box):
                iou = box_iou(box1.unsqueeze(0), box2.unsqueeze(0))[0, 0]
                if (iou >= 0.9):
                    res_meta_data[idx2].append(meta_data[idx1])
                    flag = 0
                    break
            if flag:
                res_box.append(box1)
                res_meta_data.append([meta_data[idx1]])

        for box, meta in zip(res_box, res_meta_data):
            text = ''
            for sub_meta in meta:
                if text == '':
                    text = sub_meta[0]
                else:
                    text += '\n' + sub_meta[0]

            H = meta[0][1]
            W = meta[0][2]
            
            bbox = []

            bbox.append(int(max(0, box[0])))
            bbox.append(int(max(0, box[1])))
            bbox.append(int(min(W, box[2])))
            bbox.append(int(min(H, box[3])))

            box_label(image, bbox, text)

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
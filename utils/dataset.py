import os
import cv2
import torch
import pickle
import random
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
    """
    Optimized custom dataset for action detection.
    Labels are only provided for the last (key) frame of each clip.
    """
    # Class constants
    VALID_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

    def __init__(self, config, phase, aug_config=None):
        if phase not in ['train', 'valid']:
            raise ValueError(f"Unknown phase: {phase}. Must be 'train' or 'valid'")
        self.phase = phase
        self.root_path = config['data_root']
        self.split_path = os.path.join(self.root_path, f"{self.phase}_list.txt")
        self.data_path = os.path.join(self.root_path, phase, "images")
        self.transform = VideoAugmentation(img_size=config['img_size'], aug_config=aug_config, mode=phase)
        self.clip_length = config['clip_length']
        self.sampling_rate = config['sampling_rate']
        self.num_classes = config['num_classes']
        self.img_size = (config['img_size'], config['img_size']) if isinstance(config['img_size'], int) else config['img_size']
        
        # Phase-specific settings
        self.is_train = (phase == 'train')
        self.normalize_boxes = self.is_train
        self.box_norm_factor = np.array([
            self.img_size[1], self.img_size[0], 
            self.img_size[1], self.img_size[0]
        ], dtype=np.float32) if self.normalize_boxes else None

        # Load file list
        with open(self.split_path, 'r') as file:
            self.lines = [line.strip() for line in sorted(file.readlines())]
        self.nSample = len(self.lines)

        # Set up and load label cache
        self.label_cache = self._load_cache()

    def _load_cache(self):
        cache_dir = os.path.join(self.root_path, self.phase, '_cache')
        cache_filename = f'{self.sampling_rate}x{self.clip_length}x{self.img_size[0]}x{self.img_size[1]}.pkl'
        cache_path = os.path.join(cache_dir, cache_filename)

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache file not found: {cache_path}\n"
                "Please build cache before creating Dataset."
            )

        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    def _get_clip_indices(self, video_path, key_frame_idx):
        """Get frame indices for the clip, with single directory scan."""
        # Single pass to get both file numbers and extension
        files = os.listdir(video_path)
        file_numbers = []
        img_ext = None
        
        for fname in files:
            if fname.lower().endswith(self.VALID_IMG_EXTENSIONS) and fname[:5].isdigit():
                if img_ext is None:
                    img_ext = os.path.splitext(fname)[1]
                file_numbers.append(int(fname[:5]))
        
        if not file_numbers:
            raise FileNotFoundError(f"{video_path} contains no valid image files")
        
        file_numbers.sort()
        
        # Get clip indices. Works position based sampling, not sequence based because works with positions, not actual frame numbers
        index_pos = file_numbers.index(key_frame_idx)
        clip_indices = sorted(file_numbers[:index_pos + 1][::-self.sampling_rate])[-self.clip_length:]
        
        # Pad with key frame if necessary
        if len(clip_indices) < self.clip_length:
            clip_indices = [key_frame_idx] * (self.clip_length - len(clip_indices)) + clip_indices
        
        return clip_indices, img_ext

    def _letterbox(self, video_path, clip_indices, img_ext):
        """
        Load video clip frames and letterbox in single operation (optimized).
        
        Returns:
            frames: List of letterboxed frames (target_size)
            shapes: ((h0, w0), ((r, r), (dw, dh))) - original shape and letterbox params for further rescaling later in validation
        """
        clip = []
        target_h, target_w = self.img_size
        
        # Get scale and padding (calculate once, same for all frames)
        first_path = os.path.join(video_path, f"{clip_indices[0]:05d}{img_ext}")
        first_frame = cv2.imread(first_path)
        h0, w0 = first_frame.shape[:2]  # Original shape
        
        # Scale factor (gain)
        r = min(target_h / h0, target_w / w0)
        if not self.is_train:  # validation: don't upscale
            r = min(r, 1.0)
        
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        
        # Compute padding
        dw = (target_w - new_w) / 2
        dh = (target_h - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Process all frames with same parameters
        for idx in clip_indices:
            path = os.path.join(video_path, f"{idx:05d}{img_ext}")
            frame = cv2.imread(path)
            
            # Single resize operation (better quality)
            if (new_h, new_w) != (h0, w0):
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Add padding
            frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            clip.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Return shapes for rescaling later
        shapes = ((h0, w0), ((r, r), (dw, dh)))
        return clip, shapes

    def _prepare_labels_for_augmentation(self, labels):
        """Convert labels to format expected by VideoAugmentation."""
        if self.is_train:
            # Convert to one-hot encoding
            onehot = np.zeros((len(labels), self.num_classes), dtype=np.float32)
            onehot[np.arange(len(labels)), labels] = 1
            return onehot.tolist()
        else:
            return labels.tolist()

    def _process_augmented_outputs(self, aug_boxes, aug_labels):
        """Convert augmented outputs to tensors."""
        if len(aug_boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            if self.is_train:
                labels_tensor = torch.zeros((0, self.num_classes), dtype=torch.float32)
            else:
                labels_tensor = torch.zeros((0,), dtype=torch.int64)
            return boxes_tensor, labels_tensor
        
        # Convert boxes
        boxes_array = np.array(aug_boxes, dtype=np.float32)
        if self.normalize_boxes:
            boxes_array /= self.box_norm_factor
        boxes_tensor = torch.from_numpy(boxes_array)
        
        # Convert labels
        if self.is_train:
            labels_array = np.array(aug_labels, dtype=np.float32)
            labels_tensor = torch.from_numpy(labels_array)
        else:
            labels_array = np.array(aug_labels, dtype=np.int64)
            labels_tensor = torch.from_numpy(labels_array).long()
        
        return boxes_tensor, labels_tensor
    
    @staticmethod
    def collate_fn(batch_data):
        clips  = []
        boxes  = []
        labels = []
        shapes = []
        for b in batch_data:
            clips.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            shapes.append(b[3])
        
        clips = torch.stack(clips, dim=0) # [batch_size, num_frame, C, H, W]
        return clips, boxes, labels, shapes

    def __len__(self):
        return self.nSample

    def __getitem__(self, index):
        """
        Returns:
            clip: Tensor (C, T, H, W)
            boxes: Tensor (N, 4) in normalized XYXY format (train) or XYXY pixels (valid)
            labels: Tensor (N, num_classes) for train or (N,) for valid
            shapes: tuple for rescaling (only in valid mode) or None (train mode)
        """
        # Parse key frame path
        key_frame_path = self.lines[index].rstrip()
        split_parts = key_frame_path.split('/')
        key_frame_idx = int(split_parts[-1].split('.')[-2])
        video_name = '/'.join(split_parts[2:-1])
        video_path = os.path.join(self.data_path, video_name)

        # Get clip indices and load frames (optimized single scan)
        clip_indices, img_ext = self._get_clip_indices(video_path, key_frame_idx)
        clip, shapes = self._letterbox(video_path, clip_indices, img_ext)

        # Get cached labels
        bboxes_list = []
        class_labels_list = []
        
        # Unpack shapes for bbox adjustment
        _, ((r, _), (dw, dh)) = shapes
        
        for frame in self.label_cache[index]:
            boxes = frame['boxes'].copy()
            labels = frame['labels'].copy()
            
            # Adjust boxes for letterbox transformation
            adjusted_boxes = []
            for bbox in boxes:
                x_min, y_min, x_max, y_max = bbox
                x_min = x_min * r + dw
                y_min = y_min * r + dh
                x_max = x_max * r + dw
                y_max = y_max * r + dh
                adjusted_boxes.append([x_min, y_min, x_max, y_max])
            
            bboxes_list.append(adjusted_boxes)
            class_labels_list.append(self._prepare_labels_for_augmentation(labels))

        # Apply augmentation (skip passing "shapes" since it is not used during augmentation and our images are relevant to "self.img_size")
        clip_tensor, aug_bboxes_list, aug_labels_list = self.transform(
            clip, bboxes_list, class_labels_list
        )

        # Extract and process last frame annotations
        last_frame_boxes = aug_bboxes_list[-1] # input to a model: bboxes of the last frame 
        last_frame_labels = aug_labels_list[-1] # input to a model: cls_ids of the last frame 
        
        boxes_tensor, labels_tensor = self._process_augmented_outputs(
            last_frame_boxes, last_frame_labels
        )

        return clip_tensor, boxes_tensor, labels_tensor, shapes
    
class VideoAugmentation:
    """
    Video augmentation using pure OpenCV (cv2).
    Maintains temporal consistency across frames.
    
    """ 
    def __init__(self, img_size, aug_config=None, mode='train'):
        """
        Args:
            img_size: Target image size for model input (height, width)
            mode: 'train' or 'valid' - determines augmentation strength
        """
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size  # (img_height, img_width)
        self.mode = mode
        self.hyp = aug_config # None for valid mode
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __call__(self, video_clip, bboxes_list, class_labels_list):
        """
        Apply augmentations to video clip with bounding boxes.
        
        Args:
            video_clip: List of PIL Images or numpy arrays (N, H, W, C)
            bboxes_list: List of bboxes per frame [[box1, box2], [box3], ...]
                        Format: [x_min, y_min, x_max, y_max]
            class_labels_list: List of class labels per frame [[cls1, cls2], [cls3], ...]
        
        Returns:
            augmented_video: Tensor (C, N, H, W)
            augmented_bboxes_list: List of augmented bboxes per frame
            augmented_labels_list: List of augmented labels per frame
            shapes: (orig_shape, (gain, pad)) for later rescaling in valid mode
        """
        # ============= Handling clip =======================
        # Convert to numpy if needed
        if isinstance(video_clip, np.ndarray):
            # Already stacked as (N, H, W, C)
            frames = [video_clip[i].copy() for i in range(len(video_clip))]
        elif isinstance(video_clip, list):
            # List of frames (could be PIL Images or numpy arrays)
            frames = []
            for img in video_clip:
                if hasattr(img, 'mode'):  # PIL Image
                    frames.append(np.array(img).copy())
                else:  # Already numpy array
                    frames.append(img.copy())
        else:
            raise ValueError(f"Unsupported video_clip type: {type(video_clip)}")
        
        # ============= Handling bboxes =======================
        bboxes_list = [
            [list(bbox) for bbox in bboxes] if len(bboxes) > 0 else []
            for bboxes in bboxes_list
        ]
        # ============= Handling classes =======================
        class_labels_list = [
            list(labels) if len(labels) > 0 else []
            for labels in class_labels_list
        ]
        
        if self.mode == 'train':
            frames, bboxes_list, class_labels_list = self._apply_train_transforms(
                frames, bboxes_list, class_labels_list
            )
        else:
            frames, bboxes_list, class_labels_list = self._apply_val_transforms(
                frames, bboxes_list, class_labels_list
            )
        
        # Normalize and convert to tensor (In-place Normalization operations for efficiency)
        augmented_frames = []
        for frame in frames:
            frame_float = frame.astype(np.float32)
            frame_float /= 255.0
            frame_float -= self.mean
            frame_float /= self.std
            
            augmented_frames.append(torch.from_numpy(frame_float).float())
        
        # Stack: (N, H, W, C) -> (C, N, H, W)
        augmented_video = torch.stack(augmented_frames, dim=0)
        augmented_video = augmented_video.permute(3, 0, 1, 2)  # Direct: (N, H, W, C) -> (C, N, H, W)
        
        return augmented_video, bboxes_list, class_labels_list
    
    def _apply_train_transforms(self, frames, bboxes_list, class_labels_list):
        """Apply training augmentations with temporal consistency."""

        # 1. GEOMETRIC TRANSFORMS
        geometric = self.hyp['geometric']
        
        # Random resized crop (with probability)
        if random.random() < geometric['random_resized_crop']:
            frames, bboxes_list, class_labels_list = self._random_resized_crop(
                frames, bboxes_list, class_labels_list,
                scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
        
        # Horizontal flip
        if random.random() < geometric['horizontal_flip']:
            frames, bboxes_list = self._horizontal_flip(frames, bboxes_list)
        
        # Affine transformation (scale, translate, shear)
        if random.random() < geometric['affine']:
            frames, bboxes_list, class_labels_list = self._affine_transform(
                frames, bboxes_list, class_labels_list,
                scale_range=(0.7, 1.1),
                translate_range=(-0.1, 0.1),
                shear_range=(-5, 5)
            )
        
        # Perspective transformation
        if random.random() < geometric['perspective']:
            frames, bboxes_list, class_labels_list = self._perspective_transform(
                frames, bboxes_list, class_labels_list,
                scale=(0.02, 0.05)
            )
        
        # save orig bboxes after all geometric augmentations beacuse they remove some unrealistic bboxes after augmentation
        original_bboxes_list = [
            [list(bbox) for bbox in bboxes] if len(bboxes) > 0 else []
            for bboxes in bboxes_list
        ]
        
        # 2. COLOR SPACE TRANSFORMS (with temporal consistency)
        # Sample augmentation parameters ONCE for entire video
        
        # HSV augmentation
        color_space = self.hyp['color']
        if random.random() < color_space['hsv']:
            h_shift = random.uniform(-8, 8)
            s_shift = random.uniform(-30, 30)
            v_shift = random.uniform(-25, 25)
            frames = self._hsv_augmentation(frames, h_shift, s_shift, v_shift)
        
        # Brightness and contrast
        if random.random() < color_space['brightness_contrast']:
            alpha = 1.0 + random.uniform(-0.2, 0.2)
            beta = random.uniform(-0.2, 0.2) * 255
            frames = self._brightness_contrast(frames, alpha, beta)
        
        # Color jitter
        if random.random() < color_space['color_jitter']:
            frames = self._color_jitter(
                frames,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        
        # RGB shift
        if random.random() < color_space['rgb_shift']:
            r_shift = random.uniform(-15, 15)
            g_shift = random.uniform(-15, 15)
            b_shift = random.uniform(-15, 15)
            frames = self._rgb_shift(frames, r_shift, g_shift, b_shift)
        
        # 3. NOISE AND BLUR (with temporal consistency)
        noise_blur = self.hyp['noise_blur']
        if random.random() < noise_blur['blur']:
            blur_type = random.choice(['gaussian', 'motion', 'median'])
            if blur_type == 'gaussian':
                kernel_size = random.choice(range(3, 8, 2))
                frames = self._gaussian_blur(frames, kernel_size)
            elif blur_type == 'motion':
                kernel_size = random.choice(range(3, 8, 2))
                angle = random.uniform(0, 360)
                frames = self._motion_blur(frames, kernel_size, angle)
            else:
                frames = self._median_blur(frames, kernel_size=5)
        
        # Gaussian noise (sample variance once)
        if random.random() < noise_blur['gaussian_noise']:
            variance = random.uniform(10.0, 50.0)
            frames = self._gaussian_noise(frames, variance)
        
        # 4. WEATHER EFFECTS (with temporal consistency)
        if random.random() < self.hyp['weather']:
            weather_type = random.choice(['rain', 'fog'])
            if weather_type == 'rain':
                frames = self._add_rain(frames)
            else:
                fog_coef = random.uniform(0.1, 0.3)
                frames = self._add_fog(frames, fog_coef)
        
        # 5. QUALITY DEGRADATION (with temporal consistency)
        quality_degr = self.hyp['quality']
        if random.random() < quality_degr['jpeg_compression']:
            quality = random.randint(75, 100)
            frames = self._jpeg_compression(frames, quality)
        
        if random.random() < quality_degr['downscale_upscale']:
            scale = random.uniform(0.5, 0.75)
            frames = self._downscale_upscale(frames, scale)
        
        # 6. COARSE DROPOUT (with temporal consistency)
        if random.random() < self.hyp['pixel']['coarse_dropout']:
            frames = self._coarse_dropout(
                frames,
                max_holes=8,
                hole_size_range=(8, 32)
            )
        
        filter_blank = self.hyp['filtering']    
        # 7. Filter and blank at the end
        frames, bboxes_list, class_labels_list = self._filter_and_blank_small_boxes(
            frames, original_bboxes_list, bboxes_list, class_labels_list, filter_blank['wh_thr'], filter_blank['ar_thr'], filter_blank['area_thr']
        )
        
        return frames, bboxes_list, class_labels_list
    
    def _apply_val_transforms(self, frames, bboxes_list, class_labels_list):
        """Apply validation augmentations (minimal, deterministic)."""

        return frames, bboxes_list, class_labels_list
    
    # ============ GEOMETRIC TRANSFORMS ============
    
    def _horizontal_flip(self, frames, bboxes_list):
        """Horizontal flip with bbox adjustment."""
        w = frames[0].shape[1]
        
        flipped_frames = [cv2.flip(frame, 1) for frame in frames]
        
        flipped_bboxes_list = []
        for bboxes in bboxes_list:
            flipped_bboxes = []
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                x_min_new = w - x_max
                x_max_new = w - x_min
                flipped_bboxes.append([x_min_new, y_min, x_max_new, y_max])
            flipped_bboxes_list.append(flipped_bboxes)
        
        return flipped_frames, flipped_bboxes_list
    
    def _random_resized_crop(self, frames, bboxes_list, class_labels_list, 
                            scale=(0.8, 1.0), ratio=(0.9, 1.1)):
        """Random resized crop with bbox filtering."""
        h, w = frames[0].shape[:2]
        target_h, target_w = self.img_size
        
        # Random crop parameters (sample once for temporal consistency)
        area = h * w
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])
        
        crop_w = int(np.sqrt(target_area * aspect_ratio))
        crop_h = int(crop_w / aspect_ratio)
        
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        
        # Crop frames
        cropped_frames = []
        for frame in frames:
            cropped = frame[y:y+crop_h, x:x+crop_w]
            resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            cropped_frames.append(resized)
        
        # Update and filter bboxes
        scale_x = target_w / crop_w
        scale_y = target_h / crop_h
        
        updated_bboxes_list = []
        updated_labels_list = []
        
        for bboxes, labels in zip(bboxes_list, class_labels_list):
            updated_bboxes = []
            updated_labels = []
            
            for bbox, label in zip(bboxes, labels):
                x_min, y_min, x_max, y_max = bbox
                
                # Translate and clip to crop region
                x_min = max(0, x_min - x)
                y_min = max(0, y_min - y)
                x_max = min(crop_w, x_max - x)
                y_max = min(crop_h, y_max - y)
                
                if x_max > x_min and y_max > y_min:
                    
                    # Scale to resized dimensions
                    x_min *= scale_x
                    y_min *= scale_y
                    x_max *= scale_x
                    y_max *= scale_y
                    
                    updated_bboxes.append([x_min, y_min, x_max, y_max])
                    updated_labels.append(label)
            
            updated_bboxes_list.append(updated_bboxes)
            updated_labels_list.append(updated_labels)
        
        return cropped_frames, updated_bboxes_list, updated_labels_list
    
    def _affine_transform(self, frames, bboxes_list, class_labels_list,
                         scale_range=(0.7, 1.1), translate_range=(-0.1, 0.1),
                         shear_range=(-5, 5)):
        """Affine transformation (scale, translate, shear)."""
        h, w = frames[0].shape[:2]
        
        # Random parameters (sample once for temporal consistency)
        scale = random.uniform(*scale_range)
        tx = random.uniform(*translate_range) * w
        ty = random.uniform(*translate_range) * h
        shear = random.uniform(*shear_range)
        
        # Affine matrix
        M = np.array([
            [scale, shear * np.pi / 180 * scale, tx],
            [0, scale, ty]
        ], dtype=np.float32)
        
        # Apply to frames
        transformed_frames = []
        for frame in frames:
            transformed = cv2.warpAffine(
                frame, M, (w, h), 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=(114, 114, 114)
            )
            transformed_frames.append(transformed)
        
        # Transform bboxes
        updated_bboxes_list = []
        updated_labels_list = []
        
        for bboxes, labels in zip(bboxes_list, class_labels_list):
            updated_bboxes = []
            updated_labels = []
            
            for bbox, label in zip(bboxes, labels):
                x_min, y_min, x_max, y_max = bbox
                
                # Transform corners
                corners = np.array([
                    [x_min, y_min, 1],
                    [x_max, y_min, 1],
                    [x_max, y_max, 1],
                    [x_min, y_max, 1]
                ], dtype=np.float32)
                
                transformed_corners = corners @ M.T
                
                # Get new bbox
                x_min_new = np.clip(transformed_corners[:, 0].min(), 0, w)
                y_min_new = np.clip(transformed_corners[:, 1].min(), 0, h)
                x_max_new = np.clip(transformed_corners[:, 0].max(), 0, w)
                y_max_new = np.clip(transformed_corners[:, 1].max(), 0, h)
                
                # Check validity
                if x_max_new > x_min_new and y_max_new > y_min_new:
                    updated_bboxes.append([x_min_new, y_min_new, x_max_new, y_max_new])
                    updated_labels.append(label)
        
            updated_bboxes_list.append(updated_bboxes)
            updated_labels_list.append(updated_labels)
        
        return transformed_frames, updated_bboxes_list, updated_labels_list
    
    def _perspective_transform(self, frames, bboxes_list, class_labels_list, scale=(0.02, 0.05)):
        """Perspective transformation."""
        h, w = frames[0].shape[:2]
        
        # Random perspective strength (sample once for temporal consistency)
        strength = random.uniform(*scale)
        
        # Source points (corners)
        src_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # Destination points (with random distortion)
        dst_points = np.float32([
            [random.uniform(0, w * strength), random.uniform(0, h * strength)],
            [w - random.uniform(0, w * strength), random.uniform(0, h * strength)],
            [w - random.uniform(0, w * strength), h - random.uniform(0, h * strength)],
            [random.uniform(0, w * strength), h - random.uniform(0, h * strength)]
        ])
        
        # Perspective matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply to frames
        transformed_frames = []
        for frame in frames:
            transformed = cv2.warpPerspective(
                frame, M, (w, h), 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=(114, 114, 114)
            )
            transformed_frames.append(transformed)
        
        # Transform bboxes
        updated_bboxes_list = []
        updated_labels_list = []
        
        for bboxes, labels in zip(bboxes_list, class_labels_list):
            updated_bboxes = []
            updated_labels = []
            
            for bbox, label in zip(bboxes, labels):
                x_min, y_min, x_max, y_max = bbox
                
                # Transform corners
                corners = np.array([
                    [[x_min, y_min]],
                    [[x_max, y_min]],
                    [[x_max, y_max]],
                    [[x_min, y_max]]
                ], dtype=np.float32)
                
                transformed_corners = cv2.perspectiveTransform(corners, M)
                transformed_corners = transformed_corners.reshape(-1, 2)
                
                # Get new bbox
                x_min_new = np.clip(transformed_corners[:, 0].min(), 0, w)
                y_min_new = np.clip(transformed_corners[:, 1].min(), 0, h)
                x_max_new = np.clip(transformed_corners[:, 0].max(), 0, w)
                y_max_new = np.clip(transformed_corners[:, 1].max(), 0, h)
                
                # Check validity
                if x_max_new > x_min_new and y_max_new > y_min_new:
                    updated_bboxes.append([x_min_new, y_min_new, x_max_new, y_max_new])
                    updated_labels.append(label)
            
            updated_bboxes_list.append(updated_bboxes)
            updated_labels_list.append(updated_labels)
        
        return transformed_frames, updated_bboxes_list, updated_labels_list
    
    # ============ COLOR TRANSFORMS ============
    
    def _hsv_augmentation(self, frames, h_shift, s_shift, v_shift):
        """HSV color augmentation with pre-sampled shifts."""
        augmented_frames = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Hue (0-179 in OpenCV)
            hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
            
            # Saturation (0-255)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_shift, 0, 255)
            
            # Value (0-255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_shift, 0, 255)
            
            hsv = hsv.astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            augmented_frames.append(rgb)
        
        return augmented_frames
    
    def _brightness_contrast(self, frames, alpha, beta):
        """Brightness and contrast adjustment with pre-sampled values."""
        augmented_frames = []
        for frame in frames:
            adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            augmented_frames.append(adjusted)
        
        return augmented_frames
    
    def _color_jitter(self, frames, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """Color jitter (sample once for temporal consistency)."""
        # Sample factors once
        b_factor = 1.0 + random.uniform(-brightness, brightness)
        c_factor = 1.0 + random.uniform(-contrast, contrast)
        s_factor = 1.0 + random.uniform(-saturation, saturation)
        h_shift = random.uniform(-hue, hue) * 179
        
        augmented_frames = []
        for frame in frames:
            # Brightness and contrast in RGB
            adjusted = cv2.convertScaleAbs(frame, alpha=c_factor, beta=(b_factor - 1) * 128)
            
            # Saturation and hue in HSV
            hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
            hsv = hsv.astype(np.uint8)
            
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            augmented_frames.append(rgb)
        
        return augmented_frames
    
    def _rgb_shift(self, frames, r_shift, g_shift, b_shift):
        """RGB channel shift with pre-sampled shifts."""
        augmented_frames = []
        for frame in frames:
            shifted = frame.astype(np.float32)
            shifted[:, :, 0] = np.clip(shifted[:, :, 0] + r_shift, 0, 255)
            shifted[:, :, 1] = np.clip(shifted[:, :, 1] + g_shift, 0, 255)
            shifted[:, :, 2] = np.clip(shifted[:, :, 2] + b_shift, 0, 255)
            augmented_frames.append(shifted.astype(np.uint8))
        
        return augmented_frames
    
    # ============ NOISE AND BLUR ============
    
    def _gaussian_blur(self, frames, kernel_size):
        """Gaussian blur with pre-sampled kernel size."""
        augmented_frames = []
        for frame in frames:
            blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            augmented_frames.append(blurred)
        
        return augmented_frames
    
    def _motion_blur(self, frames, kernel_size, angle):
        """Motion blur with pre-sampled parameters."""
        # Create motion blur kernel (sample once)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        augmented_frames = []
        for frame in frames:
            blurred = cv2.filter2D(frame, -1, kernel)
            augmented_frames.append(blurred)
        
        return augmented_frames
    
    def _median_blur(self, frames, kernel_size=5):
        """Median blur."""
        augmented_frames = []
        for frame in frames:
            blurred = cv2.medianBlur(frame, kernel_size)
            augmented_frames.append(blurred)
        
        return augmented_frames
    
    def _gaussian_noise(self, frames, variance):
        """Gaussian noise with pre-sampled variance."""
        sigma = variance ** 0.5
        
        augmented_frames = []
        for frame in frames:
            # Generate different noise for each frame (temporal variation is natural)
            noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
            noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            augmented_frames.append(noisy)
        
        return augmented_frames
    
    # ============ WEATHER EFFECTS ============
    
    def _add_rain(self, frames, brightness_coef=0.9, drop_width=1, blur_value=5):
        """Add rain effect with temporal consistency."""
        h, w = frames[0].shape[:2]
        
        # Sample rain pattern once for temporal consistency
        num_drops = random.randint(100, 300)
        rain_drops = []
        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(10, 30)
            rain_drops.append((x, y, length))
        
        augmented_frames = []
        for frame in frames:
            rain_layer = np.zeros_like(frame)
            
            # Draw same rain pattern (could add slight offset for motion)
            for x, y, length in rain_drops:
                cv2.line(rain_layer, (x, y), (x + drop_width, y + length), 
                        (200, 200, 200), 1)
            
            # Blur rain
            rain_layer = cv2.GaussianBlur(rain_layer, (blur_value, blur_value), 0)
            
            # Blend
            frame_adjusted = frame.astype(np.float32) * brightness_coef
            frame_adjusted = np.clip(frame_adjusted + rain_layer.astype(np.float32) * 0.3, 
                                    0, 255).astype(np.uint8)
            
            augmented_frames.append(frame_adjusted)
        
        return augmented_frames
    
    def _add_fog(self, frames, fog_coef, alpha_coef=0.1):
        """Add fog effect with pre-sampled fog coefficient."""
        augmented_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            
            # Create fog layer (white/gray)
            fog_color = np.ones_like(frame) * 200
            
            # Blend
            alpha = fog_coef * (1 + alpha_coef * np.random.randn(h, w, 1))
            alpha = np.clip(alpha, 0, 1)
            
            foggy = frame.astype(np.float32) * (1 - alpha) + fog_color * alpha
            foggy = np.clip(foggy, 0, 255).astype(np.uint8)
            
            augmented_frames.append(foggy)
        
        return augmented_frames
    
    def _add_sunflare(self, frames, cx, cy, radius):
        """Add sun flare effect with pre-sampled position."""
        augmented_frames = []
        
        for frame in frames:
            h, w = frame.shape[:2]
            
            # Create flare layer (same for all frames)
            flare_layer = np.zeros((h, w), dtype=np.float32)
            
            # Draw gradient circles
            num_circles = random.randint(1, 2)
            for i in range(num_circles):
                r = radius // (i + 1)
                cv2.circle(flare_layer, (cx, cy), r, 1.0, -1)
            
            # Apply Gaussian blur
            flare_layer = cv2.GaussianBlur(flare_layer, (51, 51), 0)
            
            # Normalize and expand to 3 channels
            if flare_layer.max() > 0:
                flare_layer = flare_layer / flare_layer.max()
            flare_layer = np.stack([flare_layer] * 3, axis=-1)
            
            # Blend (additive)
            flared = frame.astype(np.float32) + flare_layer * 100
            flared = np.clip(flared, 0, 255).astype(np.uint8)
            
            augmented_frames.append(flared)
        
        return augmented_frames
    
    # ============ QUALITY DEGRADATION ============
    
    def _jpeg_compression(self, frames, quality):
        """JPEG compression artifacts with pre-sampled quality."""
        augmented_frames = []
        for frame in frames:
            # Encode and decode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 
                                     encode_param)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            augmented_frames.append(decoded)
        
        return augmented_frames
    
    def _downscale_upscale(self, frames, scale):
        """Downscale and upscale with pre-sampled scale."""
        augmented_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Downscale
            downscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Upscale back
            upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)
            
            augmented_frames.append(upscaled)
        
        return augmented_frames
    
    # ============ PIXEL-LEVEL TRANSFORMS ============
    
    def _coarse_dropout(self, frames, max_holes=8, hole_size_range=(8, 32)):
        """Coarse dropout with temporal consistency."""
        h, w = frames[0].shape[:2]
        
        # Sample hole positions once for temporal consistency
        num_holes = random.randint(1, max_holes)
        holes = []
        for _ in range(num_holes):
            hole_h = random.randint(*hole_size_range)
            hole_w = random.randint(*hole_size_range)
            y = random.randint(0, max(1, h - hole_h))
            x = random.randint(0, max(1, w - hole_w))
            holes.append((x, y, hole_w, hole_h))
        
        augmented_frames = []
        for frame in frames:
            result = frame.copy()
            
            # Apply same holes to all frames
            for x, y, hole_w, hole_h in holes:
                result[y:y+hole_h, x:x+hole_w] = 0
            
            augmented_frames.append(result)
        
        return augmented_frames
    
    def _filter_and_blank_small_boxes(self, frames, original_bboxes_list, 
                                   augmented_bboxes_list, class_labels_list,
                                   wh_thr, ar_thr, area_thr):
        """
        Filter smaller boxes/labels than threshold and blank rejected regions.
        CRITICAL: Maintains 1:1 correspondence between boxes and labels.
        
        Args:
            frames: List of augmented frames
            original_bboxes_list: Boxes BEFORE augmentation (per frame)
            augmented_bboxes_list: Boxes AFTER augmentation (per frame)
            class_labels_list: Class labels per frame
            wh_thr: Minimum width/height in pixels (default: 10)
            ar_thr: Maximum aspect ratio (default: 20)
            area_thr: Minimum area retention ratio (default: 0.1)
        
        Returns:
            frames: Frames with rejected boxes blanked
            filtered_bboxes_list: Filtered boxes
            filtered_labels_list: Filtered labels
        """
        eps = 1e-16
        filtered_bboxes_list = []
        filtered_labels_list = []
        h, w = frames[0].shape[:2]

        for frame_idx, (orig_boxes, aug_boxes, labels) in enumerate(
            zip(original_bboxes_list, augmented_bboxes_list, class_labels_list)
        ):
            # Handle empty frames
            if len(aug_boxes) == 0 or not aug_boxes:
                filtered_bboxes_list.append([])
                filtered_labels_list.append([])
                continue
            
            aug_boxes = [box for box in aug_boxes if len(box) == 4]
            
            if len(aug_boxes) == 0:
                filtered_bboxes_list.append([])
                filtered_labels_list.append([])
                continue
            
            orig_boxes = np.array(orig_boxes)
            aug_boxes = np.array(aug_boxes)
            labels = np.array(labels)
            
            # Compute box candidates
            w1 = orig_boxes[:, 2] - orig_boxes[:, 0]
            h1 = orig_boxes[:, 3] - orig_boxes[:, 1]
            w2 = aug_boxes[:, 2] - aug_boxes[:, 0]
            h2 = aug_boxes[:, 3] - aug_boxes[:, 1]
            
            ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
            area_ratio = (w2 * h2) / (w1 * h1 + eps)
            
            # Filter criteria
            keep_mask = (w2 > wh_thr) & (h2 > wh_thr) & (area_ratio > area_thr) & (ar < ar_thr)
            
            # Blank rejected boxes - ONLY IN CURRENT FRAME
            rejected_boxes = aug_boxes[~keep_mask]
            for box in rejected_boxes:
                x1, y1, x2, y2 = box.astype(int)
                x1 = np.clip(x1, 0, w - 1)
                y1 = np.clip(y1, 0, h - 1)
                x2 = np.clip(x2, 0, w)
                y2 = np.clip(y2, 0, h)
                
                if x2 > x1 and y2 > y1:
                    # Only blank in the CURRENT frame
                    frames[frame_idx][y1:y2, x1:x2] = 114
            
            # Filter BOTH boxes AND labels
            kept_boxes = aug_boxes[keep_mask]
            kept_labels = labels[keep_mask]
            
            filtered_bboxes_list.append(kept_boxes.tolist() if len(kept_boxes) > 0 else [])
            filtered_labels_list.append(kept_labels.tolist() if len(kept_labels) > 0 else [])

        return frames, filtered_bboxes_list, filtered_labels_list
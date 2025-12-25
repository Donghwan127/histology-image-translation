import os
import random
import numpy as np
import torch
from PIL import Image
from diffusion_ffpe.my_utils import make_dataset, build_transform
import torch.utils.data as data
import torchvision.transforms.functional as F


def load_seg_feature(img_path, seg_dir):
    """
    Load segmentation feature corresponding to image path
    
    Args:
        img_path: e.g., "trainA/image_001.png"
        seg_dir: e.g., "train_seg_A"
    Returns:
        feature: (10,) numpy array
    """
    if seg_dir is None:
        return np.zeros(10, dtype=np.float32)
    
    basename = os.path.splitext(os.path.basename(img_path))[0]
    feat_path = os.path.join(seg_dir, f"{basename}.npy")
    
    if os.path.exists(feat_path):
        feature = np.load(feat_path).astype(np.float32)
    else:
        # Feature가 없으면 zero feature 반환
        print(f"⚠️ Feature not found: {feat_path}, using zeros")
        feature = np.zeros(10, dtype=np.float32)
    
    return feature


class UnpairedDataset(data.Dataset):
    def __init__(self, source_folder, target_folder, 
                 seg_source_folder=None, seg_target_folder=None,
                 image_prep=None):
        super().__init__()
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.seg_source_folder = seg_source_folder
        self.seg_target_folder = seg_target_folder

        self.l_imgs_src = make_dataset(self.source_folder, shuffle=True, seed=0)
        self.l_imgs_tgt = make_dataset(self.target_folder, shuffle=True, seed=0)

        self.T = build_transform(image_prep)

    def __len__(self):
        return len(self.l_imgs_src)

    def __getitem__(self, index):
        # Load images
        img_path_src = self.l_imgs_src[index]
        img_path_tgt = random.choice(self.l_imgs_tgt)
        
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])

        # ✅ Load segmentation features
        seg_feat_src = load_seg_feature(img_path_src, self.seg_source_folder)
        seg_feat_tgt = load_seg_feature(img_path_tgt, self.seg_target_folder)

        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "seg_features_src": torch.from_numpy(seg_feat_src),
            "seg_features_tgt": torch.from_numpy(seg_feat_tgt),
        }
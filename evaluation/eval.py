import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

import torch
import torch.nn.functional as F
import lpips

from cleanfid.fid import build_feature_extractor
from diffusion_ffpe.my_utils import (
    get_mu_sigma,
    get_features,
    calculate_fid,
    calculate_kid,
)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------
# DINO Structure Score (SSIM)
# ---------------------------
def calculate_dino_ssim(net_dino, fake_dir, real_dir):
    names = os.listdir(fake_dir)
    scores = []

    for name in tqdm(names, desc="[DINO-SSIM]"):
        fake_path = os.path.join(fake_dir, name)
        real_path = os.path.join(real_dir, name)

        fake = Image.open(fake_path).convert("RGB")
        real = Image.open(real_path).convert("RGB")

        a = net_dino.preprocess(fake).unsqueeze(0).cuda()
        b = net_dino.preprocess(real).unsqueeze(0).cuda()

        ssim_loss = net_dino.calculate_global_ssim_loss(a, b).item()
        scores.append(ssim_loss)

    return float(np.mean(scores))


# ---------------------------
# LPIPS (VGG)
# ---------------------------
def calculate_lpips(fake_dir, real_dir):
    loss_fn = lpips.LPIPS(net='vgg').cuda()
    names = os.listdir(fake_dir)
    scores = []

    for name in tqdm(names, desc="[LPIPS]"):
        fake = Image.open(os.path.join(fake_dir, name)).convert("RGB")
        real = Image.open(os.path.join(real_dir, name)).convert("RGB")

        fake_t = F.interpolate(
            transforms.ToTensor()(fake).unsqueeze(0).cuda(),
            size=(256, 256)
        )
        real_t = F.interpolate(
            transforms.ToTensor()(real).unsqueeze(0).cuda(),
            size=(256, 256)
        )

        lp = loss_fn(fake_t, real_t).item()
        scores.append(lp)

    return float(np.mean(scores))


# ---------------------------
# Main evaluation script
# ---------------------------
def main(args):
    print("Reference stats:", args.ref_path)
    print("Generated path:", args.data_path)
    print("Real image path:", args.real_path)
    print("=" * 60)

    # Clean-FID model
    feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

    # ---------------------------
    # FID
    # ---------------------------
    print("\n🔥 Calculating FID...")
    fid = calculate_fid(args.ref_path, args.data_path, feat_model)
    print("➡️ FID:", fid)

    # ---------------------------
    # KID
    # ---------------------------
    print("\n🔥 Calculating KID...")
    kid = calculate_kid(args.ref_path, args.data_path, feat_model)
    print("➡️ KID:", kid)

    # ---------------------------
    # LPIPS
    # ---------------------------
    print("\n🔥 Calculating LPIPS...")
    lp = calculate_lpips(args.data_path, args.real_path)
    print("➡️ LPIPS:", lp)

    # ---------------------------
    # DINO SSIM
    # ---------------------------
    print("\n🔥 Calculating DINO-SSIM...")
    from diffusion_ffpe.dino_struct import DinoStructureLoss
    net_dino = DinoStructureLoss()
    dino_ssim = calculate_dino_ssim(net_dino, args.data_path, args.real_path)
    print("➡️ DINO-SSIM:", dino_ssim)

    print("\n🚀 DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Generated image folder (A2B or B2A)")
    parser.add_argument("--real_path", type=str, required=True,
                        help="Real target images matching the generated ones")
    parser.add_argument("--ref_path", type=str, required=True,
                        help="Precomputed FID statistics (.npz)")
    args = parser.parse_args()

    main(args)

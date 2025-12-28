import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from DiFFPE.diffusion_ffpe.my_utils import calculate_fid, calculate_kid
from cleanfid.fid import build_feature_extractor

def parse_args():
    parser = argparse.ArgumentParser(description="Style Transfer Evaluation: FID, KID, and SSIM")
    # 경로 설정
    parser.add_argument("--data_path", type=str, required=True, help="Path to generated (Fake) FFPE patches")
    parser.add_argument("--ref_path", type=str, required=True, help="Path to real FFPE reference (folder or .npz)")
    parser.add_argument("--src_path", type=str, default=None, help="Path to original (Source) FF patches for SSIM calculation")
    
    # 평가 지표 활성화 여부
    parser.add_argument("--fid", action='store_true', default=True)
    parser.add_argument("--kid", action='store_true', default=False)
    parser.add_argument("--ssim", action='store_true', default=True)
    
    args = parser.parse_args()
    return args

def calculate_dataset_ssim(fake_path, src_path):
    """
    원본 FF(src)와 생성된 FFPE(fake) 사이의 구조적 유사도(SSIM)를 계산합니다.
    """
    fake_files = sorted([f for f in os.listdir(fake_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    ssim_values = []

    print(f"Calculating SSIM between {fake_path} and {src_path}...")
    for fname in tqdm(fake_files):
        src_file = os.path.join(src_path, fname)
        fake_file = os.path.join(fake_path, fname)
        
        if not os.path.exists(src_file):
            continue
            
        # 이미지 로드 및 그레이스케일 변환 (SSIM은 보통 밝기/구조 채널에서 계산)
        img_src = np.array(Image.open(src_file).convert('L'))
        img_fake = np.array(Image.open(fake_file).convert('L'))
        
        # 크기가 다를 경우 리사이즈
        if img_src.shape != img_fake.shape:
            img_fake = np.array(Image.fromarray(img_fake).resize((img_src.shape[1], img_src.shape[0])))

        score = ssim(img_src, img_fake, data_range=img_fake.max() - img_fake.min())
        ssim_values.append(score)
    
    return np.mean(ssim_values) if ssim_values else 0

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Evaluation Started ---")
    print(f"Target Data: {args.data_path}")

    # 1. 스타일 유사도 평가 (FID / KID)
    if args.fid or args.kid:
        # clean-fid 라이브러리의 추출기 사용
        feat_model = build_feature_extractor("clean", device)
        
        if args.fid:
            fid_score = calculate_fid(args.ref_path, args.data_path, feat_model)
            print(f"✅ [Result] FID Score: {fid_score:.4f}")
            
        if args.kid:
            kid_score = calculate_kid(args.ref_path, args.data_path, feat_model)
            print(f"✅ [Result] KID Score: {kid_score:.4f}")

    # 2. 구조 보존 평가 (SSIM)
    if args.ssim:
        if args.src_path is None:
            print("⚠️ [Warning] SSIM requires --src_path (Original FF images). Skipping SSIM.")
        else:
            avg_ssim = calculate_dataset_ssim(args.data_path, args.src_path)
            print(f"✅ [Result] Mean SSIM Score: {avg_ssim:.4f}")

    print(f"--- Evaluation Finished ---")

if __name__ == '__main__':
    main(parse_args())
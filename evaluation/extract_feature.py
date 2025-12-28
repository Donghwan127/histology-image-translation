import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import glob
from tqdm import tqdm
from pathlib import Path
import re

# ========================
# 설정
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 설정 - 세 가지 데이터 소스
REAL_FF_IMAGE_DIR = "files/patches/real_ff"
FAKE_FFPE_IMAGE_DIR = "files/patches/fake_ver4"
REAL_FFPE_IMAGE_DIR = "files/patches/real_ffpe"

REAL_FF_FEATURE_OUTPUT = "features/real_ff"
FAKE_FFPE_FEATURE_OUTPUT = "features/fake_ffpe"
REAL_FFPE_FEATURE_OUTPUT = "features/real_ffpe"

# ========================
# Feature Extractor 정의
# ========================

class FeatureExtractor(nn.Module):
    """ResNet50 기반 특징 추출기"""
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.squeeze(-1).squeeze(-1)
        return features

# ========================
# 파일명 파싱 함수 (새 파일명 형식에 맞게 수정)
# ========================

def parse_patch_filename(filename):
    """
    파일명에서 정보 추출
    
    예시 파일명들:
    - real_ffpe: TCGA-4B-A93V_TCGA-4B-A93V-01Z-00-DX1.C263DC1C-298D-47ED-AAFB-128043828530_thumbnail_TCGA-4B-A93V-01Z-00-DX1.C263DC1C-298D-47ED-AAFB-128043828530_thumbnail_patch_y128_x768.jpg
    - real_ff: TCGA-4B-A93V_TCGA-4B-A93V-01A-01-TSA.236A392B-4563-4891-BEE6-E93143865AA4_thumbnail_TCGA-4B-A93V-01A-01-TSA.236A392B-4563-4891-BEE6-E93143865AA4_thumbnail_patch_y128_x256.jpg
    - fake_ver4: 비슷한 패턴
    
    Returns:
        patient_id: TCGA-4B-A93V
        slide_id: C263DC1C-298D-47ED-AAFB-128043828530 또는 236A392B-4563-4891-BEE6-E93143865AA4
        patch_info: patch_y128_x768
    """
    # 확장자 제거
    name_without_ext = os.path.splitext(filename)[0]
    
    # Patient ID 추출: 첫 번째 '_' 전까지 (TCGA-XX-XXXX)
    first_underscore = name_without_ext.find('_')
    if first_underscore == -1:
        return None, None, None
    
    patient_id = name_without_ext[:first_underscore]
    
    # Patient ID 형식 검증 (TCGA-XX-XXXX)
    if not re.match(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$', patient_id):
        return None, None, None
    
    # Slide ID 추출: 첫 번째 '.' 뒤의 UUID 부분
    # UUID 패턴: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
    uuid_pattern = r'([A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12})'
    uuid_matches = re.findall(uuid_pattern, name_without_ext, re.IGNORECASE)
    
    if uuid_matches:
        slide_id = uuid_matches[0]  # 첫 번째 UUID를 slide_id로 사용
    else:
        slide_id = 'unknown'
    
    # Patch info 추출: patch_yXXX_xXXX 부분
    patch_pattern = r'(patch_y\d+_x\d+)'
    patch_match = re.search(patch_pattern, name_without_ext)
    
    if patch_match:
        patch_info = patch_match.group(1)
    else:
        patch_info = 'unknown'
    
    return patient_id, slide_id, patch_info

# ========================
# Feature 추출 메인 함수
# ========================

def extract_features_from_patches(image_dir, output_dir, image_type='FS'):
    """
    패치 이미지들에서 feature 추출
    """
    
    # Feature Extractor 로드
    print(f"Loading Feature Extractor...")
    feature_extractor = FeatureExtractor(pretrained=True).to(device)
    feature_extractor.eval()
    
    # Transform 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 이미지 파일 찾기 (jpg, png 모두)
    image_files = glob.glob(os.path.join(image_dir, "*.png")) + \
                  glob.glob(os.path.join(image_dir, "*.jpg"))
    
    print(f"Found {len(image_files)} {image_type} patch images")
    
    if len(image_files) == 0:
        print(f"WARNING: No images found in {image_dir}")
        return
    
    # 파싱 테스트 (처음 몇 개 파일)
    print(f"\n--- Parsing test for {image_type} ---")
    for i, img_path in enumerate(image_files[:3]):
        filename = os.path.basename(img_path)
        patient_id, slide_id, patch_info = parse_patch_filename(filename)
        print(f"  {filename[:80]}...")
        print(f"    -> patient_id: {patient_id}, slide_id: {slide_id[:20] if slide_id else None}..., patch_info: {patch_info}")
    print()
    
    # 환자별로 그룹화
    patient_patches = {}
    skipped = 0
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        patient_id, slide_id, patch_info = parse_patch_filename(filename)
        
        if patient_id is None:
            skipped += 1
            continue
        
        # 그룹 키: patient_id + slide_id
        group_key = f"{patient_id}/{slide_id}"
        
        if group_key not in patient_patches:
            patient_patches[group_key] = []
        
        patient_patches[group_key].append({
            'path': img_path,
            'filename': filename,
            'patch_info': patch_info
        })
    
    print(f"Grouped into {len(patient_patches)} patient-slide combinations")
    print(f"Skipped {skipped} files (parsing failed)")
    
    # 유니크한 환자 수 확인
    unique_patients = set([k.split('/')[0] for k in patient_patches.keys()])
    print(f"Unique patients: {len(unique_patients)}")
    
    # 각 그룹별로 처리
    for group_key, patches in tqdm(patient_patches.items(), desc=f"Processing {image_type}"):
        patient_id, slide_id = group_key.split('/', 1)
        
        # 출력 디렉토리 생성 (slide_id를 짧게 해시화하거나 그대로 사용)
        # slide_id가 너무 길 수 있으므로 앞 8자리만 사용
        slide_id_short = slide_id[:8] if slide_id != 'unknown' else 'unknown'
        output_patient_dir = os.path.join(output_dir, patient_id, f"slide_{slide_id_short}")
        os.makedirs(output_patient_dir, exist_ok=True)
        
        # 이미 처리된 경우 스킵
        existing_npys = glob.glob(os.path.join(output_patient_dir, "*.npy"))
        if len(existing_npys) >= len(patches):
            continue
        
        # 배치 단위로 특징 추출
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i+batch_size]
                
                batch_images = []
                valid_patches = []
                
                for patch_info in batch_patches:
                    try:
                        img = Image.open(patch_info['path']).convert('RGB')
                        img_tensor = transform(img)
                        batch_images.append(img_tensor)
                        valid_patches.append(patch_info)
                    except Exception as e:
                        print(f"Error loading {patch_info['filename']}: {e}")
                        continue
                
                if len(batch_images) == 0:
                    continue
                
                batch_tensors = torch.stack(batch_images).to(device)
                features = feature_extractor(batch_tensors)
                features_np = features.cpu().numpy()
                
                for feat, patch_info in zip(features_np, valid_patches):
                    npy_filename = f"{patch_info['patch_info']}.npy"
                    npy_path = os.path.join(output_patient_dir, npy_filename)
                    np.save(npy_path, feat)

# ========================
# 실행
# ========================

def main():
    print("=" * 70)
    print("Patch Feature Extraction Pipeline (3-way comparison)")
    print("=" * 70)
    
    # Real FF 패치 처리
    print("\n--- Processing Real FF Patches ---")
    os.makedirs(REAL_FF_FEATURE_OUTPUT, exist_ok=True)
    extract_features_from_patches(REAL_FF_IMAGE_DIR, REAL_FF_FEATURE_OUTPUT, image_type='Real_FF')
    
    # Fake FFPE (fake_ver4) 패치 처리
    print("\n--- Processing Fake FFPE (ver4) Patches ---")
    os.makedirs(FAKE_FFPE_FEATURE_OUTPUT, exist_ok=True)
    extract_features_from_patches(FAKE_FFPE_IMAGE_DIR, FAKE_FFPE_FEATURE_OUTPUT, image_type='Fake_FFPE')
    
    # Real FFPE 패치 처리
    print("\n--- Processing Real FFPE Patches ---")
    os.makedirs(REAL_FFPE_FEATURE_OUTPUT, exist_ok=True)
    extract_features_from_patches(REAL_FFPE_IMAGE_DIR, REAL_FFPE_FEATURE_OUTPUT, image_type='Real_FFPE')
    
    print("\n" + "=" * 70)
    print("Feature Extraction Complete!")
    print(f"Real FF features saved in: {REAL_FF_FEATURE_OUTPUT}")
    print(f"Fake FFPE features saved in: {FAKE_FFPE_FEATURE_OUTPUT}")
    print(f"Real FFPE features saved in: {REAL_FFPE_FEATURE_OUTPUT}")
    print("=" * 70)

if __name__ == "__main__":
    main()
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import glob
from tqdm import tqdm
from pathlib import Path

# ========================
# 설정
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 설정
FS_IMAGE_DIR = "./sur_a"
FFPE_IMAGE_DIR = "output_a2b_classifier"
FS_FEATURE_OUTPUT = "features/FS"
FFPE_FEATURE_OUTPUT = "features/FFPE"

# ========================
# Feature Extractor 정의
# ========================

class FeatureExtractor(nn.Module):
    """ResNet50 기반 특징 추출기"""
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # FC 레이어 제거, 2048차원 feature만 추출
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        features = self.feature_extractor(x)  # (batch, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch, 2048)
        return features

# ========================
# 파일명 파싱 함수
# ========================

def parse_patch_filename(filename):
    """
    파일명에서 정보 추출
    
    예: TCGA-4B-A93V-01A-01-TSA.236A392B-4563-4891-BEE6-E93143865AA4_thumbnail_fs_3.png
    
    Returns:
        patient_id: TCGA-4B-A93V
        slide_id: 236A392B-4563-4891-BEE6-E93143865AA4
        patch_info: fs_3
    """
    # 확장자 제거
    name_without_ext = os.path.splitext(filename)[0]
    
    # '-'로 split
    parts = name_without_ext.split('-')
    
    # Patient ID 추출 (TCGA-XX-XXXX)
    if len(parts) >= 3:
        patient_id = f"{parts[0]}-{parts[1]}-{parts[2]}"
    else:
        return None, None, None
    
    # Slide ID와 patch info 추출
    # 예: TCGA-4B-A93V-01A-01-TSA.236A392B-4563-4891-BEE6-E93143865AA4_thumbnail_fs_3
    if '.' in name_without_ext:
        # '.' 뒤의 부분이 slide ID
        after_dot = name_without_ext.split('.')[1]
        # '_thumbnail' 기준으로 split
        if '_thumbnail' in after_dot:
            slide_parts = after_dot.split('_thumbnail')
            slide_id = slide_parts[0]
            patch_info = '_thumbnail' + slide_parts[1] if len(slide_parts) > 1 else ''
        else:
            slide_id = after_dot
            patch_info = ''
    else:
        slide_id = 'unknown'
        patch_info = ''
    
    return patient_id, slide_id, patch_info

# ========================
# Feature 추출 메인 함수
# ========================

def extract_features_from_patches(image_dir, output_dir, image_type='FS'):
    """
    패치 이미지들에서 feature 추출
    
    입력 구조:
        image_dir/TCGA-4B-A93V-..._fs_3.png
        image_dir/TCGA-4B-A93V-..._fs_4.png
        image_dir/TCGA-50-5072-..._fs_1.png
    
    출력 구조:
        output_dir/TCGA-4B-A93V/slide_236A392B.../patch_fs_3.npy
                                                /patch_fs_4.npy
        output_dir/TCGA-50-5072/slide_XXX.../patch_fs_1.npy
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
    
    # 이미지 파일 찾기
    image_files = glob.glob(os.path.join(image_dir, "*.png")) + \
                  glob.glob(os.path.join(image_dir, "*.jpg"))
    
    print(f"Found {len(image_files)} {image_type} patch images")
    
    # 환자별로 그룹화
    patient_patches = {}
    for img_path in image_files:
        filename = os.path.basename(img_path)
        patient_id, slide_id, patch_info = parse_patch_filename(filename)
        
        if patient_id is None:
            print(f"Cannot parse {filename}, skipping")
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
    
    # 각 그룹별로 처리
    for group_key, patches in tqdm(patient_patches.items(), desc=f"Processing {image_type}"):
        patient_id, slide_id = group_key.split('/')
        
        # 출력 디렉토리 생성
        output_patient_dir = os.path.join(output_dir, patient_id, f"slide_{slide_id}")
        os.makedirs(output_patient_dir, exist_ok=True)
        
        # 이미 처리된 경우 스킵
        existing_npys = glob.glob(os.path.join(output_patient_dir, "*.npy"))
        if len(existing_npys) >= len(patches):
            print(f"Already processed {group_key}, skipping")
            continue
        
        # 배치 단위로 특징 추출
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i+batch_size]
                
                # 이미지 로드 및 transform
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
                
                # 배치 텐서 생성
                batch_tensors = torch.stack(batch_images).to(device)
                
                # 특징 추출
                features = feature_extractor(batch_tensors)  # (batch, 2048)
                features_np = features.cpu().numpy()
                
                # 각 패치별로 .npy 파일 저장
                for feat, patch_info in zip(features_np, valid_patches):
                    # 파일명 생성
                    patch_name = patch_info['patch_info'].replace('_thumbnail_', '')
                    npy_filename = f"patch_{patch_name}.npy"
                    npy_path = os.path.join(output_patient_dir, npy_filename)
                    
                    np.save(npy_path, feat)
        
        print(f"Saved {len(patches)} features for {group_key}")

# ========================
# 실행
# ========================

def main():
    print("=" * 70)
    print("Patch Feature Extraction Pipeline")
    print("=" * 70)
    
    # FS 패치 처리
    print("\n--- Processing FS Patches ---")
    os.makedirs(FS_FEATURE_OUTPUT, exist_ok=True)
    extract_features_from_patches(FS_IMAGE_DIR, FS_FEATURE_OUTPUT, image_type='FS')
    
    # FFPE 패치 처리
    print("\n--- Processing FFPE Patches ---")
    os.makedirs(FFPE_FEATURE_OUTPUT, exist_ok=True)
    extract_features_from_patches(FFPE_IMAGE_DIR, FFPE_FEATURE_OUTPUT, image_type='FFPE')
    
    print("\n=" * 70)
    print("Feature Extraction Complete!")
    print(f"FS features saved in: {FS_FEATURE_OUTPUT}")
    print(f"FFPE features saved in: {FFPE_FEATURE_OUTPUT}")
    print("=" * 70)

if __name__ == "__main__":
    main()
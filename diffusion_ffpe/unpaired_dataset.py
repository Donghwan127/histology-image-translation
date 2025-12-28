import random
import os  # 파일 존재 확인을 위해 추가
from PIL import Image
from diffusion_ffpe.my_utils import make_dataset, build_transform
import torch.utils.data as data
import torchvision.transforms.functional as F

class UnpairedDataset(data.Dataset):
    def __init__(self, source_folder, target_folder, image_prep=None):
        super().__init__()
        self.source_folder = source_folder
        self.target_folder = target_folder

        # 1. 초기 리스트 생성
        raw_imgs_src = make_dataset(self.source_folder, shuffle=True, seed=0)
        raw_imgs_tgt = make_dataset(self.target_folder, shuffle=True, seed=0)

        # 2. ✅ 핵심 보강: 실제 디스크에 존재하는 파일만 필터링
        print(f"🔍 Validating source files in {source_folder}...")
        self.l_imgs_src = [p for p in raw_imgs_src if os.path.exists(p)]
        
        print(f"🔍 Validating target files in {target_folder}...")
        self.l_imgs_tgt = [p for p in raw_imgs_tgt if os.path.exists(p)]

        print(f"✅ Dataset Summary: Source {len(self.l_imgs_src)} files, Target {len(self.l_imgs_tgt)} files valid.")

        self.T = build_transform(image_prep)

    def __len__(self):
        # 필터링된 리스트의 길이를 반환합니다.
        return len(self.l_imgs_src)

    def __getitem__(self, index):
        # 필터링된 리스트에서 안전하게 경로를 가져옵니다.
        img_path_src = self.l_imgs_src[index]
        img_path_tgt = random.choice(self.l_imgs_tgt)
        
        # 이제 Image.open 시 FileNotFoundError가 발생할 확률이 거의 없습니다.
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])

        return {"pixel_values_src": img_t_src, "pixel_values_tgt": img_t_tgt}
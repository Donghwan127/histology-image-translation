import os
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from cleanfid.fid import get_files_features, frechet_distance, kernel_distance


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path, seed=0, shuffle=False, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(path) or os.path.islink(path), '%s is not a valid directory' % path

    for root, _, fnames in sorted(os.walk(path, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    image_num = min(max_dataset_size, len(images))

    random.seed(seed)
    if shuffle:
        images = random.sample(images, int(image_num))
    else:
        images = images[:image_num]

    return images


def build_transform(image_prep="no_resize"):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS)
        ])
    elif image_prep == "no_resize":
        # 명시적으로 이미지를 그대로 반환하도록 설정
        return transforms.Compose([]) 
    else:
        # 이 부분이 실행되지 않도록 입력되는 img_prep 문자열을 확인해야 합니다.
        # train.py에서 "no_resize"를 보낸다면 위 elif에서 걸러집니다.
        raise NotImplementedError(f"transform is not Implemented for {image_prep}")

    return T


def get_mu_sigma(path, feat_model, transform=None):
    files = make_dataset(path, shuffle=True)
    features = get_files_features(files, model=feat_model, num_workers=0, batch_size=64, device='cuda', mode="clean",
                                  custom_fn_resize=None, description="", fdir=None, verbose=True,
                                  custom_image_tranform=transform)
    mu, sigma = np.mean(features, axis=0), np.cov(features, rowvar=False)
    return features, mu, sigma


def get_features(path, feat_model, transform=None):
    files = make_dataset(path, shuffle=True)
    features = get_files_features(files, model=feat_model, num_workers=0, batch_size=512, device='cuda', mode="clean",
                                  custom_fn_resize=None, description="", fdir=None, verbose=True,
                                  custom_image_tranform=transform)
    return features


def calculate_fid(ref_path, test_path, feat_model):
    if ref_path.endswith(".npz"):
        loaded_arrays = np.load(ref_path)
        ref_mu, ref_sigma = loaded_arrays['mu'], loaded_arrays['sigma']
    else:
        _, ref_mu, ref_sigma = get_mu_sigma(ref_path, feat_model)

    _, ed_mu, ed_sigma = get_mu_sigma(test_path, feat_model)
    fid_score = frechet_distance(ref_mu, ref_sigma, ed_mu, ed_sigma)
    return fid_score


def calculate_kid(ref_path, test_path, feat_model):
    if ref_path.endswith(".npz"):
        loaded_arrays = np.load(ref_path)
        ref_features = loaded_arrays['features']
    else:
        ref_features = get_features(ref_path, feat_model)

    ed_features = get_features(test_path, feat_model)
    kid_score = kernel_distance(ref_features, ed_features)
    return kid_score


def calculate_dino(data_path, test_path, net_dino):
    img_names = os.listdir(data_path)
    l_dino_scores = []
    for name in tqdm(img_names):
        fake = Image.open(os.path.join(data_path, name)).convert("RGB")
        real = Image.open(os.path.join(test_path, name)).convert("RGB")
        a = net_dino.preprocess(fake).unsqueeze(0).cuda()
        b = net_dino.preprocess(real).unsqueeze(0).cuda()
        dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
        l_dino_scores.append(dino_ssim)

    return np.mean(l_dino_scores)


def evaluate(model, net_dino, img_paths, fixed_emb, direction, fid_output_dir, img_prep, num_images, accelerator):
    l_dino_scores = []
    T_val = build_transform(img_prep)
    
    if num_images > 0:
        img_paths = img_paths[:num_images]

    my_img_paths = img_paths[accelerator.process_index::accelerator.num_processes]

    model.eval()
    for input_img_path in tqdm(my_img_paths, desc=f"GPU {accelerator.process_index} Evaluating"):
        # 파일 저장 경로 설정
        file_name = os.path.join(fid_output_dir, os.path.basename(input_img_path).replace(".png", ".jpg").replace(".tif", ".jpg"))
        
        with torch.no_grad():
            # 1. PIL 이미지 오픈
            raw_img = Image.open(input_img_path).convert("RGB")
            
            # 2. build_transform 적용 (리사이즈 등)
            input_img = T_val(raw_img)
            
            # 3. Tensor 변환 및 정규화 (모델 입력용)
            img_a = transforms.ToTensor()(input_img).unsqueeze(0).to(accelerator.device)
            img_a = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_a) # 3채널 정규화 확인
            
            # 4. 모델 호출
            # 기존 코드에서 fixed_emb[0:1]를 사용하는데, 배치 사이즈에 맞춰 repeat이 필요할 수 있습니다.
            eval_fake_b = model(img_a, direction, fixed_emb[0:1])
            
            # 5. 결과 저장 (Tensor -> PIL)
            output_tensor = eval_fake_b[0].float().cpu() * 0.5 + 0.5
            eval_fake_b_pil = transforms.ToPILImage()(output_tensor.clamp(0, 1))
            eval_fake_b_pil.save(file_name)
            
            # 6. DINO Score 계산
            dino_ssim = net_dino.calculate_global_ssim_loss(eval_fake_b, img_a).item()
            l_dino_scores.append(dino_ssim)
    
    accelerator.wait_for_everyone()
    return l_dino_scores

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
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    else:
        raise NotImplementedError("transform is not Implemented.")

    return T


def get_mu_sigma(path, feat_model, transform=None):
    files = make_dataset(path, shuffle=True)
    features = get_files_features(files, model=feat_model, num_workers=0, batch_size=256, device='cuda', mode="clean",
                                  custom_fn_resize=None, description="", fdir=None, verbose=True,
                                  custom_image_tranform=transform)
    mu, sigma = np.mean(features, axis=0), np.cov(features, rowvar=False)
    return features, mu, sigma


def get_features(path, feat_model, transform=None):
    files = make_dataset(path, shuffle=True)
    features = get_files_features(files, model=feat_model, num_workers=0, batch_size=128, device='cuda', mode="clean",
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


def load_seg_feature(img_path, seg_dir):
    """
    Load segmentation feature corresponding to image path
    
    Args:
        img_path: e.g., "validA/image_001.png"
        seg_dir: e.g., "valid_seg_A"
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
        print(f"⚠️ Seg feature not found: {feat_path}, using zeros")
        feature = np.zeros(10, dtype=np.float32)
    
    return feature


def evaluate(model, net_dino, img_path, fixed_emb, direction, fid_output_dir, img_prep, num_images, seg_folder=None):
    """
    Evaluate model on validation set
    
    Args:
        model: Diffusion_FFPE model
        net_dino: DINO structure loss network
        img_path: List of image paths
        fixed_emb: Text embeddings (B, 77, 1024)
        direction: "a2b" or "b2a"
        fid_output_dir: Output directory for generated images
        img_prep: Image preprocessing method
        num_images: Number of images to evaluate (-1 for all)
        seg_folder: ✅ Segmentation feature folder (e.g., "valid_seg_A")
    
    Returns:
        l_dino_scores: List of DINO-Struct scores
    """
    l_dino_scores = []
    T_val = build_transform(img_prep)

    for idx, input_img_path in enumerate(tqdm(img_path, desc="Evaluating")):
        if idx > num_images > 0:
            break
        
        file_name = os.path.join(fid_output_dir, os.path.basename(input_img_path).replace(".png", ".jpg"))
        
        with torch.no_grad():
            # Load and preprocess image
            input_img = T_val(Image.open(input_img_path).convert("RGB"))
            img_a = transforms.ToTensor()(input_img)
            img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
            
            # ✅ Load segmentation feature
            if seg_folder is not None:
                seg_feat = load_seg_feature(input_img_path, seg_folder)
                seg_feat = torch.from_numpy(seg_feat).unsqueeze(0).cuda()
            else:
                seg_feat = None
            
            # Generate fake image
            eval_fake_b = model(img_a, direction, fixed_emb[0:1], seg_features=seg_feat)  # ✅
            eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
            eval_fake_b_pil.save(file_name)
            
            # Compute DINO score
            a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
            b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
            dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
            l_dino_scores.append(dino_ssim)

    return l_dino_scores
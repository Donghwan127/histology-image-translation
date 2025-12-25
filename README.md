# Histology Image Translation with Diffusion

Frozen Section(FS) 조직 이미지를 FFPE 스타일로 변환하고,  
생성된 FFPE 이미지가 실제 임상 예측(생존율 등) 성능 향상에 기여하는지를 검증하는
Diffusion 기반 조직 이미지 변환 프레임워크입니다.


## 👥 Team

21기 고윤경   |  22기 서동환


## 🧬 Motivation

Frozen Section(FS)은 수술 중 신속 진단이 가능하지만  
염색 품질이 낮고 구조적 노이즈가 많아 예후 예측이나 정량 분석에 부적합합니다.  
반면 FFPE 조직 이미지는 고해상도·고정밀 분석이 가능하지만
제작에 수십 시간이 소요되어 실시간 임상 활용이 어렵습니다.

본 프로젝트는 **FS 이미지를 FFPE 스타일로 변환하여**
임상적 정보 손실을 최소화하면서,
**실제 생존율(survival rate) 예측 성능 향상 여부를 정량적으로 검증**하는 것을 목표로 합니다.


## 💡 Core Contributions

- **Unpaired Diffusion 기반 FS→FFPE 이미지 변환**
  - 병리 데이터 특성을 반영한 구조 보존 변환
  - DINO 기반 structural consistency loss 활용

- **Multi-scale Feature Fusion (MFF) + LoRA Adapter 적용**
  - Global / Local VAE encoder 분리
  - 조직 미세구조 보존 성능 향상

- **임상 예측 관점의 정량 검증**
  - 단순 시각적 변환이 아닌
  - *FS 대비 FFPE 변환 이미지가 survival prediction 성능을 얼마나 향상시키는지* 평가

---

## 🔍 Model Architecture

<img src="assets/Overview.png" width="900"/>

- Global / Local VAE encoder
- LoRA 기반 파라미터 효율적 미세조정
- Transformer 기반 FFPE generator
- Adversarial discriminator + structural loss

---

## 📂 Directory Structure

```

histology-image-translation/
├── assets/
│   ├── FF/                 # 원본 FS 이미지
│   ├── Generated FFPE/     # 생성된 FFPE 이미지
│   ├── Overview.png
│   └── visual.png
│
├── diffusion_ffpe/         # Diffusion-FFPE 모델 구현
├── evaluation/             # 생존율 / 분류 성능 평가
├── open_clip_custom/       # OpenCLIP 커스텀 모듈
├── train.py
├── train_bridge.py
└── inference.py

````


---

## 🧪 Experiments

본 프로젝트의 실험은 다음 3단계로 구성됩니다.

1. FS → FFPE Diffusion 기반 이미지 변환  
2. 변환된 FFPE 이미지로부터 특징(feature) 추출  
3. FS / FFPE 기반 survival prediction 성능 비교

### 1. Diffusion 모델 학습



```bash
python train.py
````

Bridge 구조를 사용할 경우:

```bash
python train_bridge.py
```


### 2. FFPE 이미지 생성


```bash
python inference.py \
  --input_dir assets/FF \
  --output_dir assets/Generated\ FFPE
```


### 3. Survival Prediction 평가


```bash
python evaluation/survival_predict.py
```

FS 원본 이미지 기반 모델과
Diffusion을 통해 생성한 FFPE 이미지 기반 모델의
생존율 예측 성능을 비교하여 임상적 유효성을 검증합니다.


---

## 📚 Reference

Qilai Zhang et al.,
**Diffusion-FFPE: A Diffusion Model for Unpaired Frozen-to-FFPE Histopathology Image Translation**
[https://github.com/QilaiZhang/Diffusion-FFPE](https://github.com/QilaiZhang/Diffusion-FFPE)

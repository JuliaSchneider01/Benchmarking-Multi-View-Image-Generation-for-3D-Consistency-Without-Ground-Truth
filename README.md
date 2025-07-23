
# Benchmarking Multi-View Image Generation for 3D Consistency Without Ground Truth 
Philipp Bauer, Julia Schneider


## 🔍 Method Overview 
<div align="center">
  <img src="assets/Pipeline.png" width="600"/>
</div>



## Abstract
Evaluating the 3D consistency of multi-view image generation systems remains a significant challenge, especially without access to ground truth 3D data. Traditional metrics often fall short in generative settings where multiple plausible outputs can exist. In this project, we propose a novel benchmarking methodology that separately assesses geometric and texture consistency across synthesized views, without relying on ground truth. Building upon recent advances such as MEt3R and leveraging self-supervised features (e.g., DINO), our pipeline employs feature-based comparisons and view-alignment techniques to robustly quantify multi-view coherence in both geometry and appearance. We validate the method across several generative models, demonstrating its effectiveness in identifying perceptual and structural inconsistencies. This approach offers a scalable, interpretable alternative for evaluating 3D-aware image generation and paves the way for standardized benchmarking in this field.



## 📌 Dependencies

    - Python 3.10.18
    - PyTorch: 2.7.1+cu118
    - CUDA: 11.8
    - PyTorch3D: 0.7.8
    - FeatUp: 0.1.2

NOTE: Pytorch3D and FeatUp are automatically installed alongside **MEt3R**.

Tested with *CUDA 11.8*, *PyTorch 2.7.1*, *Python 3.10*


## 💡 Example Usage

```python
    import torch
from met3r import MEt3R
from torchvision import transforms
from PIL import Image

# === Load and preprocess your images ===
def load_and_preprocess(image_path, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Now in [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# Replace with your own image paths
img1 = load_and_preprocess("/home/schneiderju/3DCV/met3r/FML_hard/photoconsistent-nvs/samples/000c3ab189999a83/samples/00000000/images/0004_hue.png")
img2 = load_and_preprocess("/home/schneiderju/3DCV/met3r/FML_hard/photoconsistent-nvs/samples/000c3ab189999a83/samples/00000000/images/0004.png")


# Stack images: (views=2, channels, H, W)
pair = torch.stack([img1, img2], dim=0)

# Add batch dimension: (batch=1, views=2, channels, H, W)
inputs = pair.unsqueeze(0).cuda()

# === Initialize MEt3R ===
metric = MEt3R(
    img_size=256,
    use_norm=True,
    backbone="dust3r",
    feature_backbone="dino16",
    feature_backbone_weights="mhamilton723/FeatUp",
    upsampler="featup",
    distance="cosine",
    freeze=True,
).cuda()

import matplotlib.pyplot as plt

# === Evaluate ===
with torch.no_grad():
    score, *rest = metric(
        images=inputs,
        return_overlap_mask=False,
        return_score_map=False,
        return_score_map_rgb=True,
        return_projections=True,
        return_rgb_projections=True,
        return_predictions=True
    )


# Should be between 0.30 - 0.35
print(f'Geometric score: {score.mean().item()}')
rgb_score = rest[0]
print(f'Texture score: {rgb_score.mean().item()}')

    
```


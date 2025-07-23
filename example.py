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

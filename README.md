
# Benchmarking Multi-View Image Generation for 3D Consistency Without Ground Truth 
<a href="https://mohammadasim98.github.io">Mohammad Asim</a><sup>1</sup>, <a href="https://geometric-rl.mpi-inf.mpg.de/people/Wewer.html">Christopher Wewer</a><sup>1</sup>, <a href="https://wimmerth.github.io">Thomas Wimmer</a><sup>1, 2</sup>, <a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele/">Bernt Schiele</a><sup>1</sup>,  <a href="https://geometric-rl.mpi-inf.mpg.de/people/lenssen.html">Jan Eric Lenssen</a><sup>1</sup>

*<sup>1</sup>Max Planck Institute for Informatics, Saarland Informatics Campus, <sup>2</sup>ETH Zurich*

<h4 align="left">
<a href="https://geometric-rl.mpi-inf.mpg.de/met3r/">Project Page</a>
</h4>

### `TL;DR: A differentiable metric to measure multi-view consistency between an image pair`. 

### 📣 News

- **15.04.2025** - Updates:
  - Added optical flow-based warping backbone using [`RAFT`](https://arxiv.org/abs/2003.12039).
  - Added `psnr`, `ssim`, `lpips`, `rmse`, and `mse` metrics on warped RGB images instead of feature maps.
  - Added `nearest`, `bilinear` and `bicubic` upsampling methods.
  - Refactored codebase structure.  
- **26.02.2025** - Accepted to [`CVPR 2025`](https://cvpr.thecvf.com/) 🎉!
- **10.01.2025** - Initial code releases.

## 🔍 Method Overview 
<div align="center">
  <img src="assets/method_overview.jpg" width="800"/>
</div>



## 📓 Abstract
Evaluating the 3D consistency of multi-view image generation systems remains a significant challenge, especially without access to ground truth 3D data. Traditional metrics often fall short in generative settings where multiple plausible outputs can exist. In this project, we propose a novel benchmarking methodology that separately assesses geometric and texture consistency across synthesized views, without relying on ground truth. Building upon recent advances such as MEt3R~\cite{asim24met3r} and leveraging self-supervised features (e.g., DINO~\cite{oquab2024dinov2learningrobustvisual}), our pipeline employs feature-based comparisons and view-alignment techniques to robustly quantify multi-view coherence in both geometry and appearance. We validate the method across several generative models, demonstrating its effectiveness in identifying perceptual and structural inconsistencies. This approach offers a scalable, interpretable alternative for evaluating 3D-aware image generation and paves the way for standardized benchmarking in this field.



## 📌 Dependencies

- Python 3.10.18
- PyTorch: 2.7.1+cu118
- CUDA: 11.8
- PyTorch3D: 0.7.8
- FeatUp: 0.1.2

NOTE: Pytorch3D and FeatUp are automatically installed alongside **MEt3R**.

Tested with *CUDA 11.8*, *PyTorch 2.4.1*, *Python 3.10*

## 🛠️ Quick Setup
Simply install **MEt3R** using the following command inside a bash terminal assuming prequisites are aleady installed and working.
```bash
pip install git+https://github.com/mohammadasim98/met3r
```


## 💡 Example Usage

Simply import and use **MEt3R** in your codebase as follows.

```python
import torch
from met3r import MEt3R

IMG_SIZE = 256

# Initialize MEt3R
metric = MEt3R(
    img_size=IMG_SIZE, # Default to 256, set to `None` to use the input resolution on the fly!
    use_norm=True, # Default to True 
    backbone="mast3r", # Default to MASt3R, select from ["mast3r", "dust3r", "raft"]
    feature_backbone="dino16", # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
    feature_backbone_weights="mhamilton723/FeatUp", # Default
    upsampler="featup", # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
    distance="cosine", # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
    freeze=True, # Default to True
).cuda()

# Prepare inputs of shape (batch, views, channels, height, width): views must be 2
# RGB range must be in [-1, 1]
# Reduce the batch size in case of CUDA OOM
inputs = torch.randn((10, 2, 3, IMG_SIZE, IMG_SIZE)).cuda()
inputs = inputs.clip(-1, 1)

# Evaluate MEt3R
score, *_ = metric(
    images=inputs, 
    return_overlap_mask=False, # Default 
    return_score_map=False, # Default 
    return_projections=False # Default 
)

# Should be between 0.25 - 0.35
print(score.mean().item())

# Clear up GPU memory
torch.cuda.empty_cache()
```

Checkout ```example.ipynb``` for more demo examples!

## 👷 Manual Install

Additionally **MEt3R** can also be installed manually in a local development environment. 
#### Install Prerequisites
```bash
pip install -r requirements.txt
```
#### Installing **FeatUp**
**MEt3R** relies on **FeatUp** to generate high resolution feature maps for the input images. Install **FeatUp** using the following command. 

```bash
pip install git+https://github.com/mhamilton723/FeatUp
```
Refer to [FeatUp](https://github.com/mhamilton723/FeatUp) for more details.

#### Installing **Pytorch3D**
**MEt3R** requires Pytorch3D to perform point projection and rasterization. Install it via the following command.  
```bash 
pip install git+https://github.com/facebookresearch/pytorch3d.git
```
In case of issues related to installing and building Pytorch3D, refer to [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more details. 

#### Installing **DUSt3R**
At the core of **MEt3R** lies [DUSt3R](https://github.com/naver/dust3r) which is used to generate the 3D point maps for feature unprojection and rasterization. We adopt **DUSt3R** as a submodule which can be downloaded as follows:
```bash
git submodule update --init --recursive
```


## 📘 Citation
When using **MEt3R** in your project, consider citing our work as follows.
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@inproceedings{asim24met3r,
    title = {MEt3R: Measuring Multi-View Consistency in Generated Images},
    author = {Asim, Mohammad and Wewer, Christopher and Wimmer, Thomas and Schiele, Bernt and Lenssen, Jan Eric},
    booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
    year = {2024},
}</code></pre>
  </div>
</section>

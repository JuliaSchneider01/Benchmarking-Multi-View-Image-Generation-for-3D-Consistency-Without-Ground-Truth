

import sys
import os
import os.path as path

from typing import Literal, Optional, Union

import torch
import numpy as np

from torch import Tensor
from torch.nn import Identity, functional as F
from pathlib import Path
from torch.nn import Module
from jaxtyping import Float, Bool
from typing import Union, Tuple
from einops import rearrange, repeat
from torchvision.models.optical_flow import raft_large
from torchmetrics.functional.image import structural_similarity_index_measure

from chrislib.data_util import load_image
from intrinsic.pipeline import load_models, run_pipeline

# Load Pytorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)


from lpips import LPIPS

#HERE_PATH = os.getcwd()
HERE_PATH = path.normpath(path.dirname(__file__))
MASt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../mast3r'))
DUSt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../mast3r/dust3r'))
MASt3R_LIB_PATH = path.join(MASt3R_REPO_PATH, 'mast3r')
DUSt3R_LIB_PATH = path.join(DUSt3R_REPO_PATH, 'dust3r')
# check the presence of models directory in repo to be sure its cloned
if path.isdir(MASt3R_LIB_PATH) and path.isdir(DUSt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, MASt3R_REPO_PATH)
    sys.path.insert(0, DUSt3R_REPO_PATH)
else:
    raise ImportError(f"mast3r and dust3r is not initialized, could not find: {MASt3R_LIB_PATH}.\n "
                    "Did you forget to run 'git submodule update --init --recursive' ?")
from dust3r.utils.geometry import xy_grid

def freeze_model(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()

def convert_to_buffer(module: torch.nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)

backbone_to_weights = {
    "mast3r": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    "dust3r": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
}

class MEt3R(Module):

    def __init__(
        self, 
        img_size: Optional[int] = 256, 
        use_norm: Optional[bool]=True,
        backbone: Literal["mast3r", "dust3r", "raft"] = "mast3r",
        feature_backbone: Optional[Literal["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]] = "dino16",
        feature_backbone_weights: Optional[Union[str, Path]] = "mhamilton723/FeatUp",
        upsampler: Optional[Literal["featup", "nearest", "bilinear", "bicubic"]] = "featup",
        distance: Literal["cosine", "lpips", "rmse", "psnr", "mse", "ssim"] = "cosine",
        freeze: bool=True,
        rasterizer_kwargs: dict = {}
    ) -> None:
        """Initialize MET3R

        Args:
            img_size (int, optional): Image size for rasterization. Set to None to allow for rasterization with the input resolution on the fly. Defaults to 224.
            use_norm (bool, optional): Whether to use norm layers in FeatUp. Refer to https://github.com/mhamilton723/FeatUp?tab=readme-ov-file#using-pretrained-upsamplers. Defaults to True.
            feature_backbone (str, optional): Feature backbone for FeatUp. Select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]. Defaults to "dino16".
            feature_backbone_weights (str | Path, optional): Weight path for FeatUp upsampler. Defaults to "mhamilton723/FeatUp".
            upsampler (str, optional): Set upsampling types. Defaults to "featup".
            distance (str): Select which distance to compute. Default to "cosine" for computing feature dissimilarity.
            freeze (bool, optional): Set whether to freeze the model. Defaults to True.
            rasterizer_kwargs (dict): Additional argument for point cloud render from PyTorch3D. Default to an empty dict. 
        """
        super().__init__()
        self.img_size = img_size
        self.upsampler = upsampler
        self.backbone = backbone
        self.distance = distance
        if upsampler == "featup" and "FeatUp" not in feature_backbone_weights:
            raise ValueError("Need to specify the correct weight path on huggingface for using `upsampler=\"featup\"`. Set `feature_backbone_weights=\"mhamilton723/FeatUp\"`")
            
        if distance == "cosine":
            if "FeatUp" in feature_backbone_weights:
                # Load featup
                from featup.util import norm, unnorm
                self.norm = norm
                if feature_backbone not in ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]:
                    raise ValueError("Provide `feature_backone` is not implemented for `FeatUp`. Please select from [\"dino16\", \"dinov2\", \"maskclip\", \"vit\", \"clip\", \"resnet50\"] in conjunction with `feature_backbone_weights=\"mhamilton723/FeatUp\"`")
                if use_norm is None:
                    raise ValueError("When using `FeatUp`, specify `use_norm` as either `True` or `False`. Currently it is set to `None`")
                
                featup = torch.hub.load(feature_backbone_weights, feature_backbone, use_norm=use_norm)
                self.feature_model = featup.model
                if upsampler == "featup":
                    self.upsampler_model = featup.upsampler
                    if freeze:
                        freeze_model(self.upsampler_model)
                        convert_to_buffer(self.upsampler_model, persistent=False)
                
            else:
                self.norm = Identity()
                self.feature_model = torch.hub.load(feature_backbone_weights, feature_backbone)
            
            if freeze:
                freeze_model(self.feature_model) 
                convert_to_buffer(self.feature_model, persistent=False)
            

        
        
        if backbone == "mast3r":
            from mast3r.model import AsymmetricMASt3R 
            self.backbone_model = AsymmetricMASt3R.from_pretrained(backbone_to_weights[backbone])
        elif backbone == "dust3r":
            from dust3r.model import AsymmetricCroCo3DStereo 
            self.backbone_model = AsymmetricCroCo3DStereo.from_pretrained(backbone_to_weights[backbone])
        elif backbone == "raft":
            self.backbone_model = raft_large(pretrained=True, progress=False)
        else:
            raise NotImplementedError("Specificed backbone for warping is not available. Please select from ['mast3r', 'dust3r', 'raft']")  

        if freeze:
            freeze_model(self.backbone_model) 
            convert_to_buffer(self.backbone_model, persistent=False)

        if backbone in ["mast3r", "dust3r"]:

            if self.img_size is not None:
                self.set_rasterizer(
                    image_size=img_size, 
                    points_per_pixel=10,
                    bin_size=0,
                    **rasterizer_kwargs
                )
            
            self.compositor = AlphaCompositor()
        
        if distance == "lpips":
            self.lpips = LPIPS(spatial=True)

    def _distance(self, inp1: Tensor, inp2: Tensor, mask: Optional[Tensor]=None, eps: float=1e-5):

        if self.distance == "cosine":
            # Get feature dissimilarity score map
            score_map = 1 - (inp1 * inp2).sum(1) / (torch.linalg.norm(inp1, dim=1) * torch.linalg.norm(inp2, dim=1) + eps) 
            score_map = score_map[:, None]
        elif self.distance == "mse":
            score_map = ((inp1 - inp2)**2).mean(1, keepdim=True)
        elif self.distance == "psnr":
            score_map = 20 * torch.log10(255.0 / (torch.sqrt(((inp1 - inp2)**2)).mean(1, keepdim=True) + eps))
        elif self.distance == "rmse":
            score_map = ((inp1 - inp2)**2).mean(1, keepdim=True)**0.5
        elif self.distance == "lpips":
            score_map = self.lpips(2 * inp1 - 1, 2 * inp2 - 1)
            score_map = score_map[:, None]
        elif self.distance == "ssim":
            _, score_map = structural_similarity_index_measure(inp1, inp2, return_full_image=True)

        result = [score_map[:, 0]]
        if mask is not None: 
            # Weighted average of score map with computed mask
            weighted = (score_map * mask[:, None]).sum(-1).sum(-1)  / (mask[:, None].sum(-1).sum(-1) + eps)
            result.append(weighted.mean(1))

        return tuple(result)
    
    def _interpolate(self, inp1: Tensor, inp2: Tensor):

        if self.upsampler == "featup":
            feat = self.upsampler_model(inp1, inp2)
            # Important for specific backbone which may not return with correct dimensions
            feat = F.interpolate(feat, (inp2.shape[-2:]), mode="bilinear")
        else:

            feat = F.interpolate(inp1, (inp2.shape[-2:]), mode=self.upsampler)

        return feat
    
    def _get_features(self, images):
        
        return self.feature_model(self.norm(images))

    def set_rasterizer(
        self,
        image_size, 
        points_per_pixel=10,
        bin_size=0,
        **kwargs
    ) -> None:
        raster_settings = PointsRasterizationSettings(
            image_size=image_size, 
            points_per_pixel=points_per_pixel,
            bin_size=bin_size,
            **kwargs
        )

        self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

    def render(
        self, 
        point_clouds: Pointclouds, 
        **kwargs
    ) -> Tuple[
            Float[Tensor, "b h w c"], 
            Float[Tensor, "b 2 h w n"]
        ]:
        """Adoped from Pytorch3D https://pytorch3d.readthedocs.io/en/latest/modules/renderer/points/renderer.html

        Args:
            point_clouds (pytorch3d.structures.PointCloud): Point cloud object to render 

        Returns:
            images (Float[Tensor, "b h w c"]): Rendered images
            zbuf (Float[Tensor, "b k h w n"]): Z-buffers for points per pixel
        """
        with torch.autocast("cuda", enabled=False):
            fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf

    def preprocess_iid_images(self, images, k=2):
        """
        Normalize input to [0, 1], split from (b, k, ...) to two arrays of shape (b, ...),
        and return two separate normalized NumPy arrays.

        Args:
            images (np.ndarray or torch.Tensor): Input image array or tensor.
            k (int): Number of images per group (fixed to 2 for this version).

        Returns:
            tuple: Two normalized numpy arrays in [0, 1], each with shape (b, ...)
        """
        assert k == 2, "This version only supports k=2"
        # Convert to NumPy if necessary
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()

        # images shape: (b, k, C, H, W)
        assert images.shape[1] == k, f"Expected second dimension to be {k}, got {images.shape[1]}"

        # Normalize each image independently to [0,1]
        b = images.shape[0]
        img1 = images[:, 0, ...].astype(np.float32)
        img2 = images[:, 1, ...].astype(np.float32)

        # Normalize per image in batch
        for i in range(b):
            img1_min, img1_max = img1[i].min(), img1[i].max()
            img2_min, img2_max = img2[i].min(), img2[i].max()

            img1[i] = (img1[i] - img1_min) / max(img1_max - img1_min, 1e-8)
            img2[i] = (img2[i] - img2_min) / max(img2_max - img2_min, 1e-8)

        print(f'img1.shape: {img1.shape}')

        return img1, img2

    def process_image_pair(self, img1, img2):
        """
        Takes two images as PyTorch tensors (CHW or HWC), stacks and normalizes them for input.

        Assumes input images are tensors with values in [-1, 1] or [0, 1], shape [C, H, W] or [H, W, C].
        Converts to shape (1, 2, C, H, W) and values in [0, 1].
        """
        # Ensure both images are tensors
        if not isinstance(img1, torch.Tensor):
            img1 = torch.from_numpy(img1)
        if not isinstance(img2, torch.Tensor):
            img2 = torch.from_numpy(img2)

        # Convert HWC -> CHW if needed
        if img1.ndim == 3 and img1.shape[-1] in [1, 3]:
            img1 = img1.permute(2, 0, 1)
        if img2.ndim == 3 and img2.shape[-1] in [1, 3]:
            img2 = img2.permute(2, 0, 1)

        # Stack the images: (k=2, C, H, W)
        images = torch.stack([img1.float(), img2.float()], dim=0)

        # Normalize from [-1, 1] → [0, 1]
        images = (images + 1) / 2

        # Add batch dimension: (1, 2, C, H, W)
        images_rgb = images.unsqueeze(0)

        return images_rgb

    def warp_image(self, image: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Warp an input image using an optical flow field and compute a mask for gaps.

        Args:
            image (torch.Tensor): The input image of shape (B, C, H, W), where
                                B is the batch size,
                                C is the number of channels,
                                H is the height,
                                W is the width.
            flow (torch.Tensor): The optical flow of shape (B, 2, H, W), where the 2 channels
                                correspond to the horizontal and vertical flow components.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The warped image of shape (B, C, H, W).
                - A mask of shape (B, 1, H, W) indicating gaps due to warping (1 for valid pixels, 0 for gaps).
        """
        B, C, H, W = image.shape

        # Generate a grid of coordinates for the image
        y, x = torch.meshgrid(
            torch.arange(H, device=image.device, dtype=torch.float32),
            torch.arange(W, device=image.device, dtype=torch.float32),
            indexing="ij"
        )

        # Normalize the grid coordinates to the range [-1, 1]
        x = x / (W - 1) * 2 - 1
        y = y / (H - 1) * 2 - 1

        grid = torch.stack((x, y), dim=2).unsqueeze(0)  # Shape: (1, H, W, 2)
        grid = grid.repeat(B, 1, 1, 1)  # Repeat for batch size

        # Normalize flow from pixel space to normalized coordinates
        flow = flow.clone()
        flow[:, 0, :, :] = flow[:, 0, :, :] / (W - 1) * 2  # Normalize horizontal flow
        flow[:, 1, :, :] = flow[:, 1, :, :] / (H - 1) * 2  # Normalize vertical flow

        # Add the flow to the grid
        flow = flow.permute(0, 2, 3, 1)  # Shape: (B, H, W, 2)
        warped_grid = grid + flow

        # Clip grid values to ensure they are within bounds
        warped_grid[..., 0] = torch.clamp(warped_grid[..., 0], -1, 1)
        warped_grid[..., 1] = torch.clamp(warped_grid[..., 1], -1, 1)

        # Use grid_sample to warp the image
        warped_image = F.grid_sample(image, warped_grid, mode="bilinear", padding_mode="border", align_corners=True)

        # Compute a mask for valid pixels
        mask = F.grid_sample(
            torch.ones((B, 1, H, W), device=image.device, dtype=image.dtype),
            warped_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        mask = (mask > 0.999).float()  # Threshold to create a binary mask

        return warped_image, mask

    def forward(
        self, 
        images: Float[Tensor, "b 2 c h w"], 
        return_overlap_mask: bool=False, 
        return_score_map: bool=False, 
        return_score_map_rgb: bool=False, 
        return_projections: bool=False,
        return_rgb_projections: bool=True,
        return_predictions: bool=True
    ) -> Tuple[
            float, 
            Bool[Tensor, "b h w"] | None, 
            Float[Tensor, "b h w"] | None, 
            Float[Tensor, "b 2 c h w"] | None
        ]:
        
        """Forward function to compute MET3R
        Args:
            images (Float[Tensor, "b 2 c h w"]): Normalized input image pairs with values ranging in [-1, 1],
            return_overlap_mask (bool, False): Return 2D map overlapping mask
            return_score_map (bool, False): Return 2D map of feature dissimlarity (Unweighted) 
            return_projections (bool, False): Return projected feature maps

        Return:
            score (Float[Tensor, "b"]): MET3R score which consists of weighted mean of feature dissimlarity
            mask (bool[Tensor, "b c h w"], optional): Overlapping mask
            feat_dissim_maps (bool[Tensor, "b h w"], optional): Feature dissimilarity score map
            proj_feats (bool[Tensor, "b h w c"], optional): Projected and rendered features
        """
        
        *_, h, w = images.shape
        
        # Set rasterization settings on the fly based on input resolution
        if self.img_size is None:
            raster_settings = PointsRasterizationSettings(
                    image_size=(h, w), 
                    radius = 0.01,
                    points_per_pixel = 10,
                    bin_size=0
                )
            self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

        
        b, k, *_ = images.shape
        images = rearrange(images, "b k c h w -> (b k) c h w")
        images = (images + 1) / 2

        if self.distance == "cosine":
            # NOTE: Compute features
            lr_feat = self._get_features(images)
            # NOTE: Transform feature to higher resolution either using `interpolate` or `FeatUp`
            hr_feat = self._interpolate(lr_feat, images)
            # K=2 since we only compare an image pairs
            hr_feat = rearrange(hr_feat, "(b k) ... -> b k ...", k=2)
        images = rearrange(images, "(b k) ... -> b k ...", k=2)
        images = 2 * images - 1

        # NOTE: Apply Backbone MASt3R/DUSt3R/RAFT to warp one view to the other and compute overlap masks
        if self.backbone == "raft":
            flow = self.backbone_model(images[:, 0, ...], images[:, 1, ...])[-1]

            if self.distance == "cosine":
                view1 = hr_feat[:, 0, ...]
                view2 = hr_feat[:, 1, ...]
            else:
                view1 = images[:, 0, ...]
                view2 = images[:, 1, ...]

            warped_view, mask = self.warp_image(view2, flow)
            rendering = torch.stack([view1, warped_view], dim=1)

        else:
            view1 = {"img": images[:, 0, ...], "instance": [""]}
            view2 = {"img": images[:, 1, ...], "instance": [""]}
            pred1, pred2 = self.backbone_model(view1, view2)

            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            # Assume hr_feat has shape [1, 2, 384, 256, 256]
            batch, k, d, h, w = hr_feat.shape

            # Step 1: Flatten spatial dims and normalize features
            features = rearrange(hr_feat, "b k d h w -> (b k h w) d")  # [1*2*256*256, 384]
            features_norm = torch.nn.functional.normalize(features, dim=-1)  # L2 normalize

            # Step 2: Project to RGB using PCA
            features_np = features_norm.cpu().numpy()
            pca = PCA(n_components=3)
            rgb_flat = pca.fit_transform(features_np)  # [2*256*256, 3]

            # Step 3: Reshape back and normalize to [0, 1]
            rgb = rgb_flat.reshape(batch, k, h, w, 3)  # [1, 2, 256, 256, 3]
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize to [0, 1]

            # Step 4: Plot and save images
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            output_dir = "./dino_feature_visualizations"
            os.makedirs(output_dir, exist_ok=True)

            for i in range(2):
                axes[i].imshow(rgb[0, i])
                axes[i].set_title(f"Image {i + 1}")
                axes[i].axis("off")

                # Save each image separately
                out_path = os.path.join(output_dir, f"dino_features_image_{i + 1}.png")
                plt.imsave(out_path, rgb[0, i])

            plt.suptitle("DINO Features (PCA to RGB)", fontsize=16)
            plt.tight_layout()
            plt.show()

            ptmps = torch.stack([pred1["pts3d"], pred2["pts3d_in_other_view"]], dim=1).detach()
            conf = torch.stack([pred1["conf"], pred2["conf"]], dim=1).detach()
 
            # NOTE: Get canonical point map using the confidences
            confs11 = conf.unsqueeze(-1) - 0.999
            canon = (confs11 * ptmps).sum(1) / confs11.sum(1)
            
            # Define principal point
            pp = torch.tensor([w /2 , h / 2], device=canon.device)
            
            
            # NOTE: Estimating fx and fy for a given canonical point map
            B, H, W, THREE = canon.shape
            assert THREE == 3

            # centered pixel grid
            pixels = xy_grid(W, H, device=canon.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
            canon = canon.flatten(1, 2)  # (B, HW, 3)

            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = canon.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.stack((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-2)[0]
            
            # Normalized focal length
            focal[..., 0] = 1 + focal[..., 0]/w
            focal[..., 1] = 1 + focal[..., 1]/h
            focal = repeat(focal, "b c -> (b k) c", k=2)
            # NOTE: Unproject feature on the point cloud
            ptmps = rearrange(ptmps, "b k h w c -> (b k) (h w) c", b=b, k=2)
            if self.distance == "cosine":
                # RGB TEST
                features = rearrange(hr_feat, "b k c h w -> (b k) (h w) c", k=2)

            else:
                images = (images + 1) / 2
                features = rearrange(images, "b k c h w-> (b k) (h w) c", k=2)

            images_rgb = (images + 1) / 2

            # load the models from the given paths
            iid_models = load_models('v2')

            # load an image (np float array in [0-1])
            iid_img_orig_1, iid_img_orig_2 = self.preprocess_iid_images(images)
            iid_img_orig_1 = np.squeeze(iid_img_orig_1, axis=0)  # (3, 256, 256)
            iid_img_orig_1 = np.transpose(iid_img_orig_1, (1, 2, 0))  # (256, 256, 3)
            iid_img_orig_2 = np.squeeze(iid_img_orig_2, axis=0)  # (3, 256, 256)
            iid_img_orig_2 = np.transpose(iid_img_orig_2, (1, 2, 0))  # (256, 256, 3)
            iid_1 = run_pipeline(iid_models, iid_img_orig_1, base_size=256)
            iid_2 = run_pipeline(iid_models, iid_img_orig_2, base_size=256)

            albedo_1 = iid_1['hr_alb']
            albedo_2 = iid_2['hr_alb']
            import cv2

            def resize_to_256(image):
                return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            import matplotlib.pyplot as plt
            # Plot the albedo image with axes showing resolution
            plt.imshow(resize_to_256(albedo_1))
            plt.title("Albedo")
            plt.xlabel("Width (pixels)")
            plt.ylabel("Height (pixels)")
            plt.grid(False)  # Optional: avoid grid lines
            plt.show()

            # Resize using cv2
            albedo_1_resized = torch.from_numpy(resize_to_256(albedo_1)).permute(2, 0, 1).float()  # HWC → CHW
            albedo_2_resized = torch.from_numpy(resize_to_256(albedo_2)).permute(2, 0, 1).float()

            images_albedo = self.process_image_pair(albedo_1_resized, albedo_2_resized)
            rgb_features = rearrange(images, "b k c h w -> (b k) (h w) c", k=2)

            albedo_features = rearrange(images_albedo, "b k c h w -> (b k) (h w) c", k=2)
            device = ptmps.device
            albedo_features = albedo_features.to(device)
            point_cloud = Pointclouds(points=ptmps, features=features)
            point_cloud_albedo = Pointclouds(points=ptmps, features=albedo_features)
            
            # NOTE: Project and Render
            R = torch.eye(3)
            R[0, 0] *= -1
            R[1, 1] *= -1
            R = repeat(R, "... -> (b k) ...", b=b, k=2)
            T = torch.zeros((3, ))
            T = repeat(T, "... -> (b k) ...", b=b, k=2)

            # Define Pytorch3D camera for projection
            cameras = PerspectiveCameras(device=ptmps.device, R=R, T=T, focal_length=focal)
            # Render via point rasterizer to get projected features
            with torch.autocast("cuda", enabled=False):
                rendering, zbuf = self.render(point_cloud, cameras=cameras, background_color=[-10000] * features.shape[-1])
                rendering_rgb, zbuf_rgb = self.render(point_cloud_albedo, cameras=cameras,
                                              background_color=[-10000] * rgb_features.shape[-1])
            rendering = rearrange(rendering, "(b k) h w c -> b k c h w",  b=b, k=2)
            rendering_rgb = rearrange(rendering_rgb, "(b k) h w c -> b k c h w", b=b, k=2)
            
            # Compute overlapping mask
            non_overlap_mask = (rendering == -10000)
            overlap_mask = (1 - non_overlap_mask.float()).prod(2).prod(1)
            non_overlap_mask_rgb = (rendering_rgb == -10000)
            overlap_mask_rgb = (1 - non_overlap_mask_rgb.float()).prod(2).prod(1)
            # Zero out regions which do not overlap
            rendering[non_overlap_mask] = 0.0
            rendering_rgb[non_overlap_mask_rgb] = 0.0

            # Mask for weighted sum
            mask = overlap_mask
            mask_rgb = overlap_mask_rgb

        # NOTE: Uncomment for incorporating occlusion masks along with overlap mask
        # zbuf = rearrange(zbuf, "(b k) ... -> b k ...",  b=b, k=2)
        # closest_z = zbuf[..., 0]
        # diff = (closest_z[:, 0, ...] - closest_z[:, 1, ...]).abs()
        # mask = (~(diff > 0.5) * (closest_z != -1).prod(1)) * mask
        
        # NOTE: Compute scores as either feature dissimilarity, RMSE, LPIPS, SSIM, MSE, or PSNR 
        score_map, weighted = self._distance(rendering[:, 0, ...], rendering[:, 1, ...], mask=mask)
        self.distance = "mse"
        score_map_rgb, weighted_rgb = self._distance(rendering_rgb[:, 0, ...], rendering_rgb[:, 1, ...], mask=mask_rgb)

        outputs = [weighted]
        outputs.append(weighted_rgb)
        if return_overlap_mask:
            outputs.append(mask)
            
        if return_score_map:
            outputs.append(score_map)
            
        if return_score_map_rgb:
            outputs.append(score_map_rgb)
        
        if return_projections:
            outputs.append(rendering)

        if return_rgb_projections:
            outputs.append(rendering_rgb)
            print(f'rendering_rgb.shape: {rendering_rgb.shape}')
        if return_predictions:
            outputs.append(pred1["conf"])
            outputs.append(pred2["conf"])
            print(f'rendering_rgb.shape: {rendering_rgb.shape}')

        return (*outputs, )


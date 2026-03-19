# Copyright 2025 Adobe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
3D Face Reconstruction Inference Pipeline

This module provides a complete pipeline for generating 3D face reconstructions
from single input images using multi-view diffusion and Gaussian splatting.
"""

import os
import yaml
import json
import importlib
import warnings
from typing import Tuple, Optional

import torch
import numpy as np
from PIL import Image
from einops import rearrange
from easydict import EasyDict as edict
from rich import print
from rembg import remove
from facenet_pytorch import MTCNN
from huggingface_hub import snapshot_download

from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import XFormersAttnProcessor, AttnProcessor2_0
from mvdiffusion.models.transformer_mv2d_image import (
    MVAttnProcessor,
    XFormersMVAttnProcessor,
    JointAttnProcessor,
    XFormersJointAttnProcessor,
)
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from gslrm.model.gaussians_renderer import render_turntable, imageseq2video
from utils_folder.face_utils import preprocess_image, preprocess_image_without_cropping

# Suppress FutureWarning from facenet_pytorch
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

# Configuration constants
DEFAULT_IMG_SIZE = 512
DEFAULT_TURNTABLE_VIEWS = 150
DEFAULT_TURNTABLE_FPS = 30
HF_REPO_ID = "wlyu/OpenFaceLift"

def download_weights_from_hf() -> str:
    """Download model weights from HuggingFace if not already present.
    
    Returns:
        Path to the downloaded repository
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Check if weights already exist locally
    mvdiffusion_path = os.path.join(script_directory, "checkpoints/mvdiffusion/pipeckpts")
    gslrm_path = os.path.join(script_directory, "checkpoints/gslrm/ckpt_0000000000021125.pt")
    prompt_embeds_path = os.path.join(script_directory, "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt")
    
    if os.path.exists(mvdiffusion_path) and os.path.exists(gslrm_path) and os.path.exists(prompt_embeds_path):
        print("Using local model weights")
        return script_directory
    
    print(f"Downloading model weights from HuggingFace: {HF_REPO_ID}")
    print("This may take a few minutes on first run...")
    
    # Download to local directory
    cache_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=os.path.join(script_directory, "checkpoints"),
        local_dir_use_symlinks=False,
    )
    
    print("Model weights downloaded successfully!")
    return script_directory

def get_model_paths() -> Tuple[str, str, str]:
    """Get paths to model checkpoints and config files."""
    script_directory = download_weights_from_hf()
    mvdiffusion_checkpoint_path = os.path.join(script_directory, "checkpoints/mvdiffusion/pipeckpts")
    gslrm_checkpoint_path = os.path.join(script_directory, "checkpoints/gslrm/ckpt_0000000000021125.pt")
    gslrm_config_path = os.path.join(script_directory, "configs/gslrm.yaml")
    return mvdiffusion_checkpoint_path, gslrm_checkpoint_path, gslrm_config_path



def initialize_face_detector(device: torch.device) -> MTCNN:
    """Initialize face detector."""
    return MTCNN(
        image_size=512, 
        margin=0, 
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], 
        factor=0.709,
        post_process=True, 
        device=device
    )


def _select_pipeline_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _select_gslrm_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _enable_xformers_attention(pipeline) -> None:
    if not is_xformers_available():
        print("xformers not available; using default attention")
        return
    try:
        pipeline.unet.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        print(f"xformers attention disabled: {exc}")


def _force_xformers_attn_processors(pipeline) -> None:
    if not is_xformers_available():
        return
    processors = {}
    for name, proc in pipeline.unet.attn_processors.items():
        if isinstance(proc, MVAttnProcessor):
            processors[name] = XFormersMVAttnProcessor()
        elif isinstance(proc, JointAttnProcessor):
            processors[name] = XFormersJointAttnProcessor()
        elif isinstance(proc, AttnProcessor2_0):
            processors[name] = XFormersAttnProcessor()
        else:
            processors[name] = proc
    pipeline.unet.set_attn_processor(processors)


def initialize_mvdiffusion_pipeline(mvdiffusion_checkpoint_path: str, device: torch.device):
    """Initialize MV Diffusion pipeline."""
    script_directory = download_weights_from_hf()
    
    torch_dtype = _select_pipeline_dtype(device)
    diffusion_pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        mvdiffusion_checkpoint_path,
        torch_dtype=torch_dtype,
    )
    diffusion_pipeline.enable_vae_slicing()
    _enable_xformers_attention(diffusion_pipeline)
    _force_xformers_attn_processors(diffusion_pipeline)
    
    color_prompt_embeddings = torch.load(
        os.path.join(script_directory, "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"),
        map_location="cpu",
    )
    
    return diffusion_pipeline, color_prompt_embeddings


def initialize_gslrm_model(gslrm_checkpoint_path: str, gslrm_config_path: str, device: torch.device):
    """Initialize GSLRM model."""
    model_config = edict(yaml.safe_load(open(gslrm_config_path, "r")))
    module_name, class_name = model_config.model.class_name.rsplit(".", 1)
    print(f"Loading model from {module_name} -> {class_name}")
    
    ModelClass = importlib.import_module(module_name).__dict__[class_name]
    gslrm_model = ModelClass(model_config)
    model_checkpoint = torch.load(gslrm_checkpoint_path, map_location="cpu")
    gslrm_model.load_state_dict(model_checkpoint["model"])
    return gslrm_model


def setup_camera_parameters(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Setup camera parameters for 6 views using local opencv_cameras.json."""
    script_directory = download_weights_from_hf()
    camera_file = os.path.join(script_directory, "utils_folder/opencv_cameras.json")
    
    with open(camera_file, 'r') as f:
        camera_data = json.load(f)["frames"]
    
    # Always use 6 views with indices [2, 1, 0, 5, 4, 3] like in gradio_app.py
    camera_indices = [2, 1, 0, 5, 4, 3]
    selected_cameras = [camera_data[i] for i in camera_indices]

    camera_intrinsics_list, camera_extrinsics_list = [], []
    for camera_frame in selected_cameras:
        camera_intrinsics_list.append(np.array([camera_frame["fx"], camera_frame["fy"], camera_frame["cx"], camera_frame["cy"]]))
        camera_extrinsics_list.append(np.linalg.inv(np.array(camera_frame["w2c"])))
    
    camera_intrinsics_array = np.stack(camera_intrinsics_list, axis=0).astype(np.float32)
    camera_extrinsics_array = np.stack(camera_extrinsics_list, axis=0).astype(np.float32)

    camera_intrinsics_tensor = torch.from_numpy(camera_intrinsics_array).float()[None]
    camera_extrinsics_tensor = torch.from_numpy(camera_extrinsics_array).float()[None]
    
    return camera_intrinsics_tensor, camera_extrinsics_tensor


def process_single_image(
    image_file: str,
    input_dir: str,
    output_dir: str,
    auto_crop: bool,
    unclip_pipeline,
    device: torch.device,
    seed: int,
    color_prompt_embedding: torch.Tensor,
    gs_lrm_model,
    demo_fxfycxcy: torch.Tensor,
    demo_c2w: torch.Tensor,
    guidance_scale_2D: float,
    step_2D: int,
    face_detector: Optional[MTCNN] = None
) -> None:
    """Process a single image through the 3D reconstruction pipeline."""
    print(f"Processing {image_file}")
    image_name = image_file.split(".")[0]

    input_image = Image.open(os.path.join(input_dir, image_file))
    input_image_np = np.array(input_image)

    demo_output_local_dir = os.path.join(output_dir, image_name)
    os.makedirs(demo_output_local_dir, exist_ok=True)
    # Preprocess image
    try:
        if auto_crop:
            input_image = preprocess_image(input_image_np)
        else:
            input_image = preprocess_image_without_cropping(input_image_np)
    except Exception as e:
        print(f"Failed to process {image_file}: {e}, applying fallback processing")
        try:
            input_image = remove(input_image)
            input_image = input_image.resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), Image.LANCZOS)
        except Exception as e2:
            print(f"Background removal also failed: {e2}, using original image")
            input_image = input_image.resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), Image.LANCZOS)

    input_image.save(os.path.join(demo_output_local_dir, "input.png"))

    # Generate multi-view images
    unclip_pipeline.to(device)
    execution_device = getattr(unclip_pipeline, "_execution_device", device)
    if hasattr(unclip_pipeline, "image_normalizer"):
        normalizer = unclip_pipeline.image_normalizer
        if hasattr(normalizer, "mean"):
            if isinstance(normalizer.mean, torch.nn.Parameter):
                normalizer.mean.data = normalizer.mean.data.to(execution_device)
            else:
                normalizer.mean = normalizer.mean.to(execution_device)
        if hasattr(normalizer, "std"):
            if isinstance(normalizer.std, torch.nn.Parameter):
                normalizer.std.data = normalizer.std.data.to(execution_device)
            else:
                normalizer.std = normalizer.std.to(execution_device)
    prompt_embeds = color_prompt_embedding.to(device=execution_device, dtype=unclip_pipeline.unet.dtype)
    generator = torch.Generator(device=execution_device)
    generator.manual_seed(seed)
    with torch.inference_mode():
        mv_imgs = unclip_pipeline(
            input_image, 
            None,
            prompt_embeds=prompt_embeds,
            guidance_scale=guidance_scale_2D,
            num_images_per_prompt=1, 
            num_inference_steps=step_2D,
            generator=generator,
            eta=1.0,
        ).images
    del prompt_embeds

    # Always use 6 views
    if len(mv_imgs) == 7:
        views = [mv_imgs[i] for i in [1, 2, 3, 4, 5, 6]]
    elif len(mv_imgs) == 6:
        views = [mv_imgs[i] for i in [0, 1, 2, 3, 4, 5]]
    else:
        raise ValueError(f"Unexpected number of views: {len(mv_imgs)}")

    # Save multi-view image
    lrm_input_save = Image.new("RGB", (DEFAULT_IMG_SIZE * len(mv_imgs), DEFAULT_IMG_SIZE))
    for i, view in enumerate(mv_imgs):
        lrm_input_save.paste(view, (DEFAULT_IMG_SIZE * i, 0))
    lrm_input_save.save(os.path.join(demo_output_local_dir, "multiview.png"))

    # Prepare input for 3D reconstruction
    lrm_input = np.stack([np.array(view) for view in views], axis=0)
    lrm_input = torch.from_numpy(lrm_input).float()[None].to(device) / 255
    lrm_input = rearrange(lrm_input, "b v h w c -> b v c h w")

    index = torch.stack([
        torch.zeros(lrm_input.size(1)).long(),
        torch.arange(lrm_input.size(1)).long(),
    ], dim=-1)
    demo_index = index[None].to(device)

    gs_dtype = _select_gslrm_dtype(device)
    fxfycxcy = demo_fxfycxcy.to(device, dtype=gs_dtype)
    c2w = demo_c2w.to(device, dtype=gs_dtype)

    gs_lrm_model.to(device, dtype=gs_dtype)
    lrm_input = lrm_input.to(dtype=gs_dtype)

    # Create batch
    batch = edict({
        "image": lrm_input,
        "c2w": c2w,
        "fxfycxcy": fxfycxcy,
        "index": demo_index,
    })

    # 3D reconstruction inference
    with torch.inference_mode(), torch.autocast(enabled=True, device_type="cuda", dtype=gs_dtype):
        result = gs_lrm_model.forward(batch, create_visual=False, split_data=True)

    # Save Gaussian splatting result
    result.gaussians[0].apply_all_filters(
        opacity_thres=0.04,
        scaling_thres=0.1,
        floater_thres=0.6,
        crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
        cam_origins=None,
        nearfar_percent=(0.0001, 1.0),
    ).save_ply(os.path.join(demo_output_local_dir, "gaussians.ply"))

    # Save rendered output
    comp_image = result.render[0].unsqueeze(0).detach()
    v = comp_image.size(1)
    if v > 10:
        comp_image = comp_image[:, :: v // 10, :, :, :]
    comp_image = rearrange(comp_image, "x v c h w -> (x h) (v w) c")
    comp_image = (comp_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(comp_image).save(os.path.join(demo_output_local_dir, "output.png"))
    
    # Generate turntable video
    vis_image = render_turntable(
        result.gaussians[0],
        rendering_resolution=DEFAULT_IMG_SIZE,
        num_views=DEFAULT_TURNTABLE_VIEWS,
    )
    vis_image = rearrange(vis_image, "h (v w) c -> v h w c", v=DEFAULT_TURNTABLE_VIEWS)
    vis_image = np.ascontiguousarray(vis_image)
    imageseq2video(
        vis_image, 
        os.path.join(demo_output_local_dir, "turntable.mp4"), 
        fps=DEFAULT_TURNTABLE_FPS
    )

    del fxfycxcy, c2w, lrm_input, demo_index, batch, result, vis_image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_images(
    input_dir: str,
    output_dir: str,
    auto_crop: bool,
    unclip_pipeline,
    device: torch.device,
    seed: int,
    color_prompt_embedding: torch.Tensor,
    gs_lrm_model,
    demo_fxfycxcy: torch.Tensor,
    demo_c2w: torch.Tensor,
    guidance_scale_2D: float,
    step_2D: int,
    face_detector: Optional[MTCNN] = None
) -> None:
    """Process all images in the input directory."""
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    image_files = sorted(os.listdir(input_dir))
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    for image_file in image_files:
        if not image_file.lower().endswith(valid_extensions):
            continue
            
        process_single_image(
            image_file, input_dir, output_dir, auto_crop,
            unclip_pipeline, device, seed, color_prompt_embedding,
            gs_lrm_model, demo_fxfycxcy, demo_c2w,
            guidance_scale_2D, step_2D, face_detector
        )


def main(
    input_dir: str = None,
    output_dir: str = None,
    auto_crop: bool = True,
    seed: int = 4,
    guidance_scale_2D: float = 3.0,
    step_2D: int = 50
) -> None:
    """Main function for 3D face reconstruction inference.
    
    Args:
        input_dir: Input directory containing images (default: examples/)
        output_dir: Output directory for results (default: outputs/)
        auto_crop: Auto crop the face (default: True)
        seed: Random seed for generating multi-view images (default: 4)
        guidance_scale_2D: Guidance scale for generating multi-view images (default: 3.0)
        step_2D: Number of steps for generating multi-view images (default: 50)
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Set default paths if not provided
    if input_dir is None:
        input_dir = os.path.join(script_directory, "examples")
    if output_dir is None:
        output_dir = os.path.join(script_directory, "outputs")

    # Setup device and paths
    computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mvdiffusion_checkpoint_path, gslrm_checkpoint_path, gslrm_config_path = get_model_paths()
    
    os.makedirs(output_dir, exist_ok=True)

    face_detector = None
    if auto_crop:
        face_detector = initialize_face_detector(computation_device)

    # Initialize models
    diffusion_pipeline, color_prompt_embeddings = initialize_mvdiffusion_pipeline(
        mvdiffusion_checkpoint_path, computation_device
    )
    gslrm_model = initialize_gslrm_model(gslrm_checkpoint_path, gslrm_config_path, computation_device)

    # Setup camera parameters (always 6 views)
    camera_intrinsics_tensor, camera_extrinsics_tensor = setup_camera_parameters(computation_device)
    
    # Process images
    process_images(
        input_dir, 
        output_dir, 
        auto_crop,
        diffusion_pipeline,
        computation_device,
        seed,
        color_prompt_embeddings,
        gslrm_model,
        camera_intrinsics_tensor,
        camera_extrinsics_tensor,
        guidance_scale_2D,
        step_2D,
        face_detector
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Face Reconstruction Inference Pipeline")
    parser.add_argument("--input_dir", "-i", type=str, help="Input directory containing images")
    parser.add_argument("--output_dir", "-o", type=str, help="Output directory for results")
    parser.add_argument("--auto_crop", action="store_true", default=True, help="Auto crop the face")
    parser.add_argument("--seed", type=int, default=4, help="Random seed")
    parser.add_argument("--guidance_scale_2D", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--step_2D", type=int, default=50, help="Number of diffusion steps")
    
    args = parser.parse_args()
    
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        auto_crop=args.auto_crop,
        seed=args.seed,
        guidance_scale_2D=args.guidance_scale_2D,
        step_2D=args.step_2D
    )

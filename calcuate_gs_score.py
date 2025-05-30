#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from utils.system_utils import autoChooseCudaDevice
autoChooseCudaDevice()
import torch
import os
from random import randint
import sys
from scene import Scene, GaussianModel
import math
from utils.general_utils import safe_state
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.sh_utils import eval_sh

def score_render(error_image, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    mesh_img = torch.zeros((3, viewpoint_camera.image_height, viewpoint_camera.image_width), device="cuda")
    mesh_depth = torch.full((1, viewpoint_camera.image_height, viewpoint_camera.image_width), torch.finfo(torch.float32).max, device="cuda", dtype=torch.float32)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=mesh_img,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color
    
    gs_score = torch.zeros_like(opacity)
    rendered_image, radii, rendered_depth, depth_image = rasterizer(
        error_image = error_image,
        gs_score = gs_score,
        depth_tolerance = pc.depth_tolerance,
        mesh_depth = mesh_depth,
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = rendered_image.clamp(0, 1)
    
    return {
        "gs_score": gs_score,
        "render": rendered_image,
        "rendered_depth": rendered_depth,  # depth
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,  # invdepth
        }
    
    
def record_gs_score(args, dataset, opt, pipe):
    first_iter = 0
    os.makedirs(dataset.model_path, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, dataset.depth_tolerance, mesh_path=dataset.mesh_path)
    scene = Scene(dataset, gaussians, ply_path=dataset.ply_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()

    progress_bar = tqdm(range(first_iter, len(viewpoint_stack)), desc="Record score progress")
    first_iter += 1
    
    # init per gs score
    gs_score = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
    for iteration in range(first_iter, len(viewpoint_stack) + 1):

        if iteration % 100 == 0:
            progress_bar.update(100)

        viewpoint_cam = viewpoint_stack[iteration-1]

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        with torch.no_grad():
            mesh_img, mesh_depth_img = gaussians.mesh.render(viewpoint_cam)
            
            # if the mesh render result is all balck, which means mesh did not show in this view, we skip this iter
            if (mesh_depth_img > 0).sum() < 2_636_088 / 8:
                continue
            gt_image = viewpoint_cam.original_image.cuda()
            error_image = torch.amax(torch.abs(gt_image - mesh_img), dim=0, keepdim=True)
            
            render_pkg = score_render(error_image, viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=False)
            render_image = render_pkg['render']
            gs_score_temp = render_pkg['gs_score']
            gs_score = torch.max(gs_score, gs_score_temp)
            
    print("Complete, save gs_score.")
    os.makedirs(os.path.dirname(args.output_path_gs_score), exist_ok=True)
    torch.save(gs_score, args.output_path_gs_score)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path_gs_score", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    record_gs_score(args, lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nAll complete.")

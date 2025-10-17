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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import time
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def test_FPS(args, model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    """
    input: Keep the same input parameters as render_set(...)
    output: the output is a more accurate FPS.
    """
    t_list_len = 200
    warmup_times = 5
    test_times = 10
    t_list = np.array([1.0] * t_list_len)
    step = 0
    fps_list = []
    while True:
        for view in views:
            step += 1
            torch.cuda.synchronize();
            t0 = time.time()
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, render_mode=args.render_mode)["render"]
            torch.cuda.synchronize();
            t1 = time.time()
            t_list[step % t_list_len] = t1 - t0

            if step % t_list_len == 0 and step > t_list_len * warmup_times:
                fps = 1.0 / t_list.mean()
                print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')
                fps_list.append(fps)
            if step > t_list_len * (test_times + warmup_times):
                # write fps info to a txt file
                with open(os.path.join(model_path, "point_cloud", "iteration_{}".format(iteration), "FPS.txt"), 'w') as f:
                    f.write("Average FPS: {:.5f}\n".format(np.mean(fps_list)))
                    f.write("FPS std: {:.5f}\n".format(np.std(fps_list)))
                print("Average FPS: {:.5f}, FPS std: {:.5f}".format(np.mean(fps_list), np.std(fps_list)))
                return
    
def render_set(args, model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, render_mode=args.render_mode)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_video(args, model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    n_fames = args.n_frames
    fps = 30
    height = views[0].image_height
    width = views[0].image_width
    traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(iteration))
    os.makedirs(traj_dir, exist_ok=True)
    print(f"rendering video to {traj_dir}, n_frames={n_fames}, fps={fps}, height={height}, width={width}")
    
    from utils.render_utils import generate_path
    import cv2
    cam_traj = generate_path(views, n_frames=n_fames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(traj_dir, "render_traj_color.mp4"), fourcc, fps, (width, height))
    
    for view in tqdm(cam_traj, desc="Rendering video"):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, render_mode=args.render_mode)["render"]
        frame = rendering.cpu().permute(1, 2, 0).numpy()
        frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    video.release()
    
    
    
def render_sets(args, dataset : ModelParams, pipeline : PipelineParams, separate_sh: bool):
    iteration = args.iteration
    skip_train = args.skip_train
    skip_test = args.skip_test
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.depth_tolerance, mesh_path=dataset.mesh_path)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, ply_path=dataset.ply_path)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.video:
            render_video(args, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_train:
            render_set(args, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
            render_set(args, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
        test_FPS(args, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mode", default="hybrid", type=str)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--n_frames", default=300, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(args, model.extract(args), pipeline.extract(args), SPARSE_ADAM_AVAILABLE)
    
    # Metrics
    print("Metrics " + args.model_path)
    from metrics import *
    evaluate([args.model_path])
    
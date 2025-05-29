from plyfile import PlyData, PlyElement
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import torch

def sparse_by_score(pcd, gs_score_path, threshold):
    origin = PlyData.read(pcd)
    vertex_data = origin["vertex"].data
    gs_score = torch.load(gs_score_path)
    mask = (gs_score > threshold).view(-1)
    mask = mask.cpu().numpy()
    print('num GS points:', gs_score.shape[0])
    print('choose GS points:', mask.sum())
    filtered_data = vertex_data[mask]
    return filtered_data


def main():
    parser = argparse.ArgumentParser(description='Choose GS according to score')
    parser.add_argument('--ply_path', type=str, required=True, 
                        help='Path to the GS point cloud.')
    parser.add_argument('--gs_score_path', type=str, required=True, 
                        help='Path to the gs score .pt path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the Output Path')
    parser.add_argument('--threshold', type=float, default=0.20,
                        help='Select those gs whose score is greater than this value')
    
    args = parser.parse_args()
    
    
    merged_data = sparse_by_score(
        pcd=args.ply_path,
        gs_score_path=args.gs_score_path,
        threshold=args.threshold
        )
    
    # save ply
    merged_ply = PlyElement.describe(merged_data, "vertex")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    print(f"Writing merged model to {args.output}...")
    PlyData([merged_ply]).write(args.output)
    print("Done!")


if __name__ == '__main__':
    main()
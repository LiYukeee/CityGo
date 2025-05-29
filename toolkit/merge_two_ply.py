"""
Merge two PLY files into one.
"""

import numpy as np
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

def read_PLY(pcd):
    origin = PlyData.read(pcd)
    vertex_data = origin["vertex"].data
    return vertex_data

def combine_two_ply(path_ply_A, path_ply_B, path_ply_output):
    """
    Combine two PLY files into one.
    """
    blocks = []
    ply_A = read_PLY(path_ply_A)
    ply_B = read_PLY(path_ply_B)
    blocks.append(ply_A)
    blocks.append(ply_B)
    
    merged_ply = np.concatenate(blocks, axis=0)
    merged_ply = PlyElement.describe(merged_ply, "vertex")
    PlyData([merged_ply]).write(path_ply_output)


if __name__ == "__main__":
    parser = ArgumentParser(description="Split ply file to building and non-building with shapes.")
    parser.add_argument("--PLY_PATH_A", default=None, type=str)
    parser.add_argument("--PLY_PATH_B", default=None, type=str)
    parser.add_argument("--OUTPUT_PLY_PATH", default=None, type=str)
    args = parser.parse_args()
    
    combine_two_ply(args.PLY_PATH_A, args.PLY_PATH_B, args.OUTPUT_PLY_PATH)
import numpy as np
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

def read_PLY(pcd, ratio=0.1):
    origin = PlyData.read(pcd)
    vertex_data = origin["vertex"].data
    mask = np.random.random(vertex_data.shape[0]) < ratio
    return vertex_data[mask]

if __name__ == "__main__":
    parser = ArgumentParser(description="Randomly sample points from a PLY file")
    parser.add_argument("--PLY_PATH", default=None, type=str)
    parser.add_argument("--OUTPUT_PLY_PATH", default=None, type=str)
    parser.add_argument("--RATIO", default=0.1, type=float)
    args = parser.parse_args()
    
    blocks = []
    ply_A = read_PLY(args.PLY_PATH, args.RATIO)
    blocks.append(ply_A)

    
    merged_ply = np.concatenate(blocks, axis=0)
    merged_ply = PlyElement.describe(merged_ply, "vertex")
    PlyData([merged_ply]).write(args.OUTPUT_PLY_PATH)
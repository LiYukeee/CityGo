from plyfile import PlyData, PlyElement
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm


def crop_block(pcd, bbox):
    origin = PlyData.read(pcd)
    vertex_data = origin["vertex"].data
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    
    if not isinstance(bbox, np.ndarray):
        bbox = np.load(bbox)
    # bbox = np.array([
    #     [110.733307, -760.016907, -5.],
    #     [289.238281, -620.095947, 300.]
    # ])
    mask = np.all((points >= bbox[0]) & (points <= bbox[1]), axis=1)
    filtered_data = vertex_data[mask]
    return filtered_data


def main():
    parser = argparse.ArgumentParser(description='Merge a Gaussian Splatting Model into a single bbox')
    parser.add_argument('--ply_path', type=str, required=True, 
                        help='Path to the GS point cloud.')
    parser.add_argument('--bbox_path', type=str, required=True, 
                        help='Path to the bounding box files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the merged model')
    
    args = parser.parse_args()
    
    
    merged_data = crop_block(
        pcd=args.ply_path,
        bbox=args.bbox_path
        )
    
    # save ply
    merged_ply = PlyElement.describe(merged_data, "vertex")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    print(f"Writing merged model to {args.output}...")
    PlyData([merged_ply]).write(args.output)
    print("Done!")


if __name__ == '__main__':
    main()
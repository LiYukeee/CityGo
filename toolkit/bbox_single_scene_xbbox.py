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
    
    mask = np.all((points >= bbox[0]) & (points <= bbox[1]), axis=1)
    filtered_data = vertex_data[mask]
    return filtered_data

def read_bbox(bbox):
    if not isinstance(bbox, np.ndarray):
        bbox = np.load(bbox)
    return bbox

def combine_bbox(bbox_list):
    x_min, y_min, z_min = np.inf, np.inf, np.inf
    x_max, y_max, z_max = -np.inf, -np.inf, -np.inf
    for bbox in bbox_list:
        x_min = min(x_min, bbox[0, 0])
        y_min = min(y_min, bbox[0, 1])
        z_min = min(z_min, bbox[0, 2])
        x_max = max(x_max, bbox[1, 0])
        y_max = max(y_max, bbox[1, 1])
        z_max = max(z_max, bbox[1, 2])
    merged_bbox = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])
    return merged_bbox
        
    
    

def crop_all(ply_path, bbox_path, n_blocks):
    bbox_list = []
    print(f"Processing {len(n_blocks)} blocks...")
    
    for block in tqdm(n_blocks):
        bbox_file = os.path.join(bbox_path, f'block_{block}', 'bbox.npy')
        if not os.path.exists(ply_path):
            print(f"Warning: PLY file for block {block} not found at {ply_path}")
            continue
        if not os.path.exists(bbox_file):
            print(f"Warning: Bbox file for block {block} not found at {bbox_file}")
            continue
        bbox = read_bbox(bbox_file)
        print(bbox)
        bbox_list.append(bbox)
    res_bbox = combine_bbox(bbox_list.copy())
    cropped = crop_block(ply_path, res_bbox)
    return cropped
    

def main():
    parser = argparse.ArgumentParser(description='Merge Gaussian Splatting Points to a bbox scene')
    parser.add_argument('--ply_path', type=str, required=True, 
                        help='Path to the GS point cloud.')
    parser.add_argument('--blocks_path', type=str, required=True, 
                        help='Path to the directory containing bounding box files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the merged model')
    parser.add_argument('--blocks', type=str, default='all',
                        help='Comma separated block IDs to merge, "all" for all blocks')
    
    args = parser.parse_args()
    
    if args.blocks.lower() == 'all':
        block_dirs = [d for d in os.listdir(args.blocks_path) if d.startswith('block_')]
        block_ids = [int(d.split('_')[1]) for d in block_dirs]
        n_blocks = sorted(block_ids)
    else:
        try:
            n_blocks = [int(b.strip()) for b in args.blocks.split(',')]
        except ValueError:
            print("Error: Block IDs must be integers")
            return 1
    
    print(f"Will process blocks: {n_blocks}")
    
    merged_data = crop_all(
        ply_path=args.ply_path,
        bbox_path=args.blocks_path,
        n_blocks=n_blocks
        )
    
    # save ply
    merged_ply = PlyElement.describe(merged_data, "vertex")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    print(f"Writing merged model to {args.output}...")
    PlyData([merged_ply]).write(args.output)
    print("Done!")


if __name__ == '__main__':
    main()
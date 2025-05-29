"""
Created by Chen Xi.

Restrict the chunk-trained GS within the BBOX and merge them.
"""
from plyfile import PlyData, PlyElement
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

def convert_sh0(ply_element):
    if isinstance(ply_element, str):
        ply_data = PlyData.read(ply_element)
        vertex_data = ply_data["vertex"]
    else:
        vertex_data = ply_element
    
    sh0s = [prop.name for prop in vertex_data.properties if not prop.name.startswith('f_rest')]
    npd = np.dtype([(prop.name, prop.val_dtype) for prop in vertex_data.properties if not prop.name.startswith('f_rest')])
    
    tsh0 = vertex_data[sh0s]
    tsh0 = PlyElement.describe(np.array(tsh0, dtype=npd), "vertex")
    return tsh0

def crop_block(pcd, bbox):
    origin = PlyData.read(pcd)
    vertex_data = origin["vertex"].data
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    
    if not isinstance(bbox, np.ndarray):
        bbox = np.load(bbox)
    
    mask = np.all((points >= bbox[0]) & (points <= bbox[1]), axis=1)
    filtered_data = vertex_data[mask]
    return filtered_data

def crop_all(gs_path, bbox_path, n_blocks, iterations):
    blocks = []
    print(f"Processing {len(n_blocks)} blocks...")
    
    for block in tqdm(n_blocks):
        block_dir = os.path.join(gs_path, f'block_{block}')
        if not os.path.exists(block_dir):
            print(f"Warning: Block {block} not found, skipping...")
            continue
        
        ply_path = os.path.join(block_dir, 'point_cloud', f'iteration_{iterations}', 'point_cloud.ply')
        bbox_file = os.path.join(bbox_path, f'block_{block}', 'bbox.npy')
        
        if not os.path.exists(ply_path):
            print(f"Warning: PLY file for block {block} not found at {ply_path}")
            continue
        
        if not os.path.exists(bbox_file):
            print(f"Warning: Bbox file for block {block} not found at {bbox_file}")
            continue
            
        try:
            cropped = crop_block(ply_path, bbox_file)
            print(len(cropped))
            blocks.append(cropped)
        except Exception as e:
            print(f"Error processing block {block}: {e}")
    
    if not blocks:
        raise ValueError("No valid blocks found to merge!")
    
    concat = np.concatenate(blocks, axis=0)
    return concat

def main():
    parser = argparse.ArgumentParser(description='Merge multiple Gaussian Splatting blocks into a single model')
    parser.add_argument('--gs_path', type=str, required=True, 
                        help='Path to the directory containing gaussian splatting model blocks')
    parser.add_argument('--blocks_path', type=str, required=True, 
                        help='Path to the directory containing bounding box files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the merged model')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='Iteration number to use for the models')
    parser.add_argument('--blocks', type=str, default='all',
                        help='Comma separated block IDs to merge, "all" for all blocks')
    parser.add_argument('--sh0', action='store_true',
                        help='Convert final merged model to SH0 (remove higher order spherical harmonics)')
    
    args = parser.parse_args()
    
    if args.blocks.lower() == 'all':
        block_dirs = [d for d in os.listdir(args.gs_path) if d.startswith('block_')]
        block_ids = [int(d.split('_')[1]) for d in block_dirs]
        n_blocks = sorted(block_ids)
    else:
        try:
            n_blocks = [int(b.strip()) for b in args.blocks.split(',')]
        except ValueError:
            print("Error: Block IDs must be integers")
            return 1
    
    print(f"Will process blocks: {n_blocks}")
    
    try:
        merged_data = crop_all(
            gs_path=args.gs_path,
            bbox_path=args.blocks_path,
            n_blocks=n_blocks,
            iterations=args.iterations
        )
        
        merged_ply = PlyElement.describe(merged_data, "vertex")
        
        if args.sh0:
            print("Converting merged model to SH0...")
            merged_ply = convert_sh0(merged_ply)
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        print(f"Writing merged model to {args.output}...")
        PlyData([merged_ply]).write(args.output)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
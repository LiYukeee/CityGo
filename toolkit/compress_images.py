import os
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def find_jpg_files(directory):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

def compress_img(input, output, scale_factor):
    original_image = Image.open(input)
    new_width = int(original_image.width * scale_factor)
    new_height = int(original_image.height * scale_factor)
    compressed_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    compressed_image.save(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge multiple Gaussian Splatting blocks into a single model')
    parser.add_argument('--path', type=str, required=True, 
                        help='Path to the directory containing images need to be compressed')
    parser.add_argument('--scale_factor', type=float, default=0.25,
                        help='Scale factor for image compression (default: 0.25)')
    args = parser.parse_args()
    
    jpg_files = find_jpg_files(args.path)
    for jpg_file in jpg_files:
        print(jpg_file)
        compress_img(jpg_file, jpg_file, args.scale_factor)
        



import os
import matplotlib.pyplot as plt
import torchvision

def depth2img(img, pth="./temp.png"):
    """
    _summary_ (x, y) tensor -> image
    """
    color = img[0].detach().to('cpu').numpy()
    height, width = color.shape
    dpi = 100
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)

    depth_map = plt.imshow(color, cmap='viridis', vmin=color.min(), vmax=color.max())
    plt.colorbar(depth_map, label='Depth')

    plt.title('Depth Map')
    os.makedirs(os.path.dirname(pth), exist_ok=True)
    plt.savefig(pth)
    

def saveimage(img, path="./temp.png"):
    torchvision.utils.save_image(img, path)
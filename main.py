import tifffile as tiff
from dotenv import load_dotenv
import os, glob
import numpy as np
import scipy.ndimage.morphology as snm
load_dotenv('.env')

import torch
import numpy as np
from PIL import Image
from skimage import data, io
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as snm


def print_num_of_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))


def norm_01(x):
    """
    normalize to 0 - 1
    """
    x = x - x.min()
    x = x / x.max()
    return x


def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x0, show=True, save=None):
    # switch
    x = 1 * x0
    if (len(x.shape) == 3) & (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    x = x - x.min()
    x = Image.fromarray(to_8bit(x))

    if show:
        io.imshow(np.array(x))
        plt.show()
    if save:
        x.save(save)


def open_tiff_to_npy(source):
    list_of_images = sorted(glob.glob(source + '*.tif'))
    npys = []
    for i in list_of_images:
        x = np.expand_dims(tiff.imread(i), 2)
        npys.append(x)
    npys = np.concatenate(npys, 2)
    return npys


def get_patch(x, destination, dx, filename):
# def get_patch(x, destination, dx=256):
    os.makedirs(destination, exist_ok=True)
    for i in range(x.shape[0] // dx):
        for j in range(x.shape[1] // dx):
            patch = (x[i*dx:(i+1)*dx, j*dx:(j+1)*dx])
            tiff.imsave(destination + filename + '_' + str(i).zfill(3) + '_' + str(j).zfill(3) + '.tif', patch)

def get_holes(x, threshold, distance):
    mask = (x <= threshold)
    mask0 = 1 * mask
    holes = (snm.distance_transform_edt(mask0) <= distance)
    return holes


if __name__ == '__main__':
    # Parameter setting
    dx = 512
    threshold = 150
    distance = 5
    slice_selected = [i for i in range(5)] #total : [i for i in range(200)]

    # read original
    source = os.environ.get('DATASET') + 'organoid_roi_3/'
    npys = open_tiff_to_npy(source)
    all_tif = sorted(os.listdir(source))
    # create folder
    os.makedirs('outputs/holes/', exist_ok=True)

    for z in slice_selected:
        filename = all_tif[z].split('.')[0] # 200.tif
    # save patches of original
        get_patch(npys[:, :, z], destination=os.environ.get('DATASET') + 'outputs/patches/', dx=dx, filename=filename)
    # get and save holes
        x = npys[:, :, z]
        hole = get_holes(x, threshold=threshold, distance=distance)
        tiff.imsave(os.environ.get('DATASET') + 'outputs/holes/' + filename + '.tif', hole)

    for z in slice_selected:
        filename = all_tif[z].split('.')[0]  # 200.tif
        # get holes
        holes = open_tiff_to_npy(os.environ.get('DATASET') + 'outputs/holes/')
        get_patch(holes[:, :, z], destination=os.environ.get('DATASET') + 'outputs/hole_patches/', dx=dx, filename=filename)

#CUDA_VISBIBLE_DEVICES=0 python main.py
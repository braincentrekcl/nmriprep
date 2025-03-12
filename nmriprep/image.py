import numpy as np
import rawpy
from PIL import Image

from .utils import rgb_to_grey, symmetrical_crop


def read_nef(path_to_file: str):
    with rawpy.imread(path_to_file) as raw:
        return raw.postprocess(
            use_camera_wb=True, no_auto_scale=True, no_auto_bright=True, output_bps=16
        )


def read_tiff(path_to_file):
    with Image.open(path_to_file) as img:
        return np.array(img)


def save_slice(array, out_name):
    im = Image.fromarray(array).save(out_name)
    return im


def convert_nef_to_grey(
    nef_file,
    crop_row=None,
    crop_col=None,
    flatfield_correction=None,
    invert=False,
):
    # Load the NEF file using rawpy
    print(f'Reading {nef_file.name}')
    rgb = read_nef(str(nef_file))
    grey = rgb_to_grey(rgb, flatfield_corr=flatfield_correction, invert=invert)

    if crop_row:
        row_lim = symmetrical_crop(grey.shape[0], crop_row)
        grey = grey[row_lim:-row_lim]
    if crop_col:
        col_lim = symmetrical_crop(grey.shape[1], crop_col)
        grey = grey[:, col_lim:-col_lim]
    return grey

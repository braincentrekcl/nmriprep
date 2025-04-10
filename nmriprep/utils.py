from re import findall

import numpy as np


def find_files(search_map):
    return sorted([fname for fname in search_map])


def parse_kv(fname: str) -> dict[str, str]:
    return {
        k: v
        for part in fname.split('_')
        if '-' in part
        for k, v in findall(r'(\w+)-([\w\d]+)', part)
    }


def do_flatfield_correction(im, flatfield, darkfield):
    return (im - darkfield) * np.mean(flatfield - darkfield) / (flatfield - darkfield)


def rgb_to_grey(rgb: np.array, flatfield_corr=None, invert=False):
    # Luminosity method to convert rgb to greyscale:
    # Grey = 0.2989*R + 0.5870*G + 0.1140*B
    grey = np.dot(rgb, [0.2989, 0.5870, 0.1140])
    if flatfield_corr:
        grey = do_flatfield_correction(
            grey, flatfield_corr['flat'], flatfield_corr['dark']
        )

    #  NB: the data inversion is to ensure that the darkest pixels
    # (i.e. the ones with the lowest transparency) have the highest
    # grey values and the lightest pixels (the most transparent)
    # have the lowest values. This is not the case in photographic
    # images where bright pixels often have the highest values
    return np.invert(grey.astype(np.uint16)) if invert else grey


def symmetrical_crop(range_array, quantile):
    return np.quantile(np.arange(range_array), quantile).astype(int)


def greyval_to_relative_OD(data: np.uint, bit=16):
    """
    When a 256-gray level digitizer is used, the grayness is
    expressed as a gray value on the linear scale between
    0 (brightest) and 255 (darkest). The GV can be transformed
    into an OD-like value, the relative optical density
    (ROD; Baskin and Stahl 1993)
    https://doi.org/10.1177/002215549804601006

    Extended by EM to handle other digitiser scales (e.g. 16 bit)
    """

    scale = (2**bit) - 1
    return np.log10(scale / (scale - data))


def rodbard(x, min_, slope, ed50, max_):
    return max_ + ((min_ - max_) / (1.0 + (x / ed50) ** slope))


def inverse_rodbard(y, min_, slope, ed50, max_):
    # inverse rodbard
    # https://www.myassays.com/four-parameter-logistic-regression.html

    return ed50 * (((min_ - max_) / (y - max_)) - 1.0) ** (1.0 / slope)


def normalise_by_region(df, region):
    region_df = df.query((f'region == "{region}"'))
    out = df.merge(
        region_df.assign(
            **{f'median_{region}_values': region_df['values'].apply(np.median)}
        ),
        on=['subj', 'slide', 'section'],
        suffixes=("", "_r")
    )
    return out['values'] / out[f'median_{region}_values']

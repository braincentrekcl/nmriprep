import importlib.resources
import json

import numpy as np
import pandas as pd

from ..image import convert_nef_to_grey
from ..plotting import plot_roi
from ..utils import parse_kv, rodbard


def get_image_patch(center_coord, square_apothem: int = 100):
    return slice(center_coord - square_apothem, center_coord + square_apothem)


def get_standard_value(array, medfilt_radius=40, square_size=900, roi_fig_name=None):
    import statistics

    from scipy.ndimage import binary_fill_holes
    from scipy.signal import find_peaks
    from scipy.spatial import distance
    from skimage import measure, morphology
    from skimage.exposure import histogram
    from skimage.filters import rank
    from skimage.segmentation import clear_border
    from skimage.util import img_as_ubyte

    print(f'extracting ROI for {roi_fig_name.stem}')

    center_coord = np.round(np.array(array.shape) / 2).astype(int)

    strel_med = morphology.disk(medfilt_radius)
    gray_medfilt = rank.median(img_as_ubyte(array / array.max()), strel_med)

    hist, bin_centers = histogram(gray_medfilt)
    peaks, _ = find_peaks(hist, distance=5, height=gray_medfilt.size / 100)
    if peaks.size < 2:  # background only in image
        gray_thresh = np.zeros_like(array).astype(bool)
        foreground = np.zeros_like(gray_thresh)
        label_image = measure.label(gray_thresh)
    else:
        troughs, _ = find_peaks(
            hist.max() - hist, distance=5, prominence=gray_medfilt.size / 100
        )
        peak_loc = bin_centers[peaks]
        trough_loc = bin_centers[troughs]
        thresholds = []
        for peak1, peak2 in zip(peak_loc[:-1], peak_loc[1:]):
            troughs = trough_loc[np.logical_and(trough_loc > peak1, trough_loc < peak2)]
            thresholds.append(
                min(troughs) if troughs else statistics.mean([peak1, peak2])
            )
            regions = np.digitize(gray_medfilt, bins=thresholds)

        square_apothem = 100
        mode = statistics.mode(
            regions[
                get_image_patch(center_coord[0]), get_image_patch(center_coord[1])
            ].ravel()
        )

        if (peaks.size == 2) & (mode == 1):  # background image
            gray_thresh = np.zeros_like(regions).astype(bool)
            foreground = np.zeros_like(gray_thresh)
            label_image = measure.label(gray_thresh)
        else:
            gray_thresh = regions == mode
            foreground = morphology.area_opening(
                morphology.binary_erosion(
                    binary_fill_holes(gray_thresh), morphology.disk(100)
                ),
                area_threshold=200000,
            )
            cleaned = clear_border(foreground)
            label_image = measure.label(cleaned)

    if label_image.max() == 1:
        roi = label_image == 1
    elif label_image.max() == 0:
        roi = np.zeros_like(gray_thresh)
        square_apothem = np.round(square_size / 2).astype(int)
        roi[
            get_image_patch(center_coord[0], square_apothem),
            get_image_patch(center_coord[1], square_apothem),
        ] = True
        if np.any(foreground):
            roi = roi & foreground
    else:
        dist = [
            distance.euclidean(region.centroid, center_coord)
            for region in measure.regionprops(label_image)
        ]
        roi = label_image == (np.argmin(dist) + 1)

    gray = np.invert(array.astype(np.uint16))
    if roi_fig_name:
        plot_roi(gray, roi, out_name=roi_fig_name)

    return np.median(gray[roi])


def calibrate_standard(
    standard_files, standard_type, flatfield_correction=None, out_dir=None
):
    from scipy.optimize import curve_fit

    from .. import data

    # load standard information
    with importlib.resources.open_text(data, 'standards.json') as f:
        standard_vals = pd.read_json(f)[standard_type].dropna()

    assert len(standard_files) == len(standard_vals)
    out_stem = '_'.join(
        f'{k}-{v}' for k, v
        in parse_kv(standard_files[0].stem).items()
        if 'standard' not in k
    )

    standards_df = pd.DataFrame(
        {
            std.stem: [
                convert_nef_to_grey(
                    std,
                    crop_row=0.2,
                    crop_col=0.2,
                    flatfield_correction=flatfield_correction,
                )
            ]
            for std in standard_files
        }
    ).melt(var_name='fname', value_name='gv_array')

    standards_df['median grey'] = standards_df.apply(
        lambda row: get_standard_value(
            row['gv_array'],
            roi_fig_name=(out_dir / f'{row["fname"]}-roi.png' if out_dir else None),
        ),
        axis=1,
    )

    # reverse standards as image order is from darkest activity to lightest
    standards_df['radioactivity (uCi/g)'] = standard_vals.iloc[::-1].to_list()

    # Fit the model
    X = standards_df['radioactivity (uCi/g)']
    y = standards_df['median grey']
    # https://www.myassays.com/four-parameter-logistic-regression.html
    print(f'fitting Rodbard curve for {out_stem}')
    popt, _ = curve_fit(
        rodbard,
        X,
        y,
        p0=[y.min(), 1, X.mean(), y.max()],
        bounds=([0.0, -np.inf, 0.0, 0.0], [2**16, np.inf, np.inf, 2**16]),
        maxfev=5000,
    )

    # save for calibrating slice images
    if out_dir:
        standards_df.to_json(out_dir / f'{out_stem}_standards.json')
        with (out_dir / f'{out_stem}_calibration.json').open(mode='w') as f:
            json.dump(dict(zip(['min', 'slope', 'ED50', 'max'], popt)), f)
    return popt, X, y, out_stem

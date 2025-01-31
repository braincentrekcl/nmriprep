import importlib.resources
import numpy as np
import pandas as pd
import json

from ..image import convert_nef_to_grey
from ..plotting import plot_roi
from ..utils import find_files, rodbard


def get_standard_value(
        array,
        row_start=900,
        col_start=1700,
        square_size=900,
        roi_fig_name=None
):
    if roi_fig_name:
        plot_roi(
            array,
            out_name=roi_fig_name,
            xy=(col_start, row_start),
            size=square_size
        )
    return array[
        row_start:row_start + square_size,
        col_start:col_start + square_size
    ].mean()


def calibrate_standard(
        sub_id,
        src_dir,
        standard_type,
        flatfield_correction=None,
        out_dir=None
):
    from .. import data
    from scipy.optimize import curve_fit

    # load standard information
    with importlib.resources.open_text(data, "standards.json") as f:
        standard_vals = pd.read_json(f)[standard_type].dropna()

    standard_files = find_files(src_dir.glob(f"*subj-{sub_id}*standard*.nef"))
    assert len(standard_files) == len(standard_vals)

    standards_df = pd.DataFrame(
        {std.stem: [convert_nef_to_grey(
                std,
                flatfield_correction=flatfield_correction
            )] for std in standard_files}
        ).melt(
            var_name="fname",
            value_name="gv_array"
        )
    standards_df['mean grey'] = standards_df.apply(
        lambda row: get_standard_value(
            row['gv_array'],
            roi_fig_name=(
                f"{out_dir / row['fname']}-roi.png" if out_dir else None
            )
        ),
        axis=1
    )

    # reverse standards as image order is from darkest activity to lightest
    standards_df['radioactivity (uCi/g)'] = standard_vals.iloc[::-1].to_list()

    # Fit the model
    X = standards_df['radioactivity (uCi/g)']
    y = standards_df['mean grey']
    # https://www.myassays.com/four-parameter-logistic-regression.html
    print(f"fitting Rodbard curve for {standard_files[0].stem[:-3]}")
    popt, _ = curve_fit(
        rodbard,
        X,
        y,
        p0=[y.min(), 1, X.mean(), y.max()],
        bounds=(
            [0.0, -np.inf, 0.0, 0.0],
            [2**16, np.inf, np.inf, 2**16]),
        maxfev=5000
    )

    # save for calibrating slice images
    out_stem = standard_files[0].stem[:-3]
    if out_dir:
        standards_df.to_json(f"{out_dir / out_stem}_standards.json")
        with open(out_dir / f"{out_stem}_calibration.json", 'w') as f:
            json.dump(
                dict(zip(["min", "slope", "ED50", "max"], popt)),
                f
            )
    return popt, X, y, out_stem

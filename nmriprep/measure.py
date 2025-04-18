import numpy as np
import pandas as pd
from skimage.measure import grid_points_in_poly

from .image import read_tiff
from .parser import get_roiextract_parser
from .utils import normalise_by_region, parse_kv


def summarise_vals(
    df, funcs=[np.median, np.mean, np.min, np.max, np.std, len], col='values'
):
    """
    Create a summary table of values from within a region
    """
    result_df = df.copy()
    return result_df.assign(
        **{f'{func.__name__}_{col}': df[col].apply(func) for func in funcs}
    ).drop(columns=col)


def roi_extract():
    """
    Extract data from napari `ROI manager` plugin files
    and their associated preprocessed tif files.
    """
    args = get_roiextract_parser().parse_args()
    input_dir = args.source_directory.absolute()
    roi_suffix = args.roi_suffix
    img_suffix = args.image_suffix
    output_name = args.output

    # find the ROI files
    roi_files = input_dir.rglob(f'*{roi_suffix}.json')
    if not roi_files:
        print('No ROI files found')
    else:
        roi_values = []
        for roi_file in roi_files:
            print(f'Processing {roi_file.name}')
            # read the corresponding image file and confirm it exists
            img_file = list(
                input_dir.rglob(roi_file.stem.replace(roi_suffix, f'{img_suffix}.tif*'))
            )[0]
            if any(['exclu' in str(path) for path in [img_file, roi_file]]):
                print(f'Excluding {roi_file}')
                continue
            img_data = read_tiff(img_file)
            img_info = parse_kv(img_file.stem)

            roi_df = pd.read_json(roi_file)
            out_df = roi_df['names'].apply(parse_kv).apply(pd.Series)
            out_df['values'] = roi_df['data'].apply(
                lambda x: img_data[grid_points_in_poly(img_data.shape, x)]
            )
            out_df = out_df.assign(**img_info)
            roi_values.append(out_df)

        main_df = pd.concat(roi_values, ignore_index=True)
        summary_df = summarise_vals(main_df)
        if args.norm_regions:
            merge_keys = [col for col in summary_df.columns if 'values' not in col]
            for region in args.norm_regions:
                array_col = f'{region}_values'
                main_df[array_col] = normalise_by_region(main_df, region)
                summary_df = summary_df.merge(
                    summarise_vals(
                        main_df[merge_keys + [array_col]],
                        funcs=[np.median, np.mean, np.min, np.max, np.std],
                        col=array_col,
                    ),
                    how='left',
                    on=merge_keys,
                    validate='1:1',
                )
        summary_df.to_csv(input_dir / f'{output_name}_summary.csv', index=False)

        if args.grouping_vars:
            main_df.groupby(args.grouping_vars, as_index=False, dropna=False).agg(
                {
                    col: lambda x: np.median(
                        np.concatenate([np.atleast_1d(val) for val in x])
                    )
                    for col in main_df.columns
                    if 'value' in col
                }
                # before taking the median, create an aggregate array from all values
            ).to_csv(input_dir / f'{output_name}_grouped_median.csv', index=False)
        else:
            main_df.to_json(input_dir / f'{output_name}.json')
    return

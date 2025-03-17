import pandas as pd
from skimage.measure import grid_points_in_poly

from .image import read_tiff
from .parser import get_roiextract_parser


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
            img_name = roi_file.stem.replace(roi_suffix, f'{img_suffix}.tif*')
            img_file = list(roi_file.parent.glob(img_name))[0]
            assert img_file.exists()
            img_data = read_tiff(img_file)

            roi_df = pd.read_json(roi_file)
            roi_values.append(
                roi_df.apply(
                    lambda x: pd.Series(
                        [
                            img_file.stem,
                            x['names'],
                            img_data[grid_points_in_poly(img_data.shape, x['data'])],
                        ],
                        index=['image', 'roi', 'values'],
                    ),
                    axis=1,
                )
            )

        pd.concat(roi_values, ignore_index=True).to_json(
            input_dir / f'{output_name}.json'
        )
    return

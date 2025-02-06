import napari
import json
import pandas as pd

from pathlib import Path
from .image import read_tiff


def extract_roi_data(input_dir):
    """
    Extract ROI data from napari ROI files found in subdirectories
    of `input_dir`, i.e. the `preproc` directory containing
    preprocessed tif files.

    Can (and should) be called from the napari iPython console
    """
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    # find all the ROI files
    roi_files = input_dir.rglob("*rois.json")
    if not roi_files:
        print("No ROI files found")
    else:
        viewer = napari.Viewer(show=False)
        img_names = []
        roi_names = []
        roi_values = []
        for roi_file in roi_files:
            print(f"Processing {roi_file}")
            # read the corresponding image file and confirm it exists
            img_file = (
                roi_file.parent /
                roi_file.stem.replace("rois", "ARG.tif")
            )
            assert img_file.exists()
            img_data = read_tiff(img_file)

            with open(roi_file, 'r') as f:
                roi_dict = json.load(f)
            for idx, roi in enumerate(roi_dict['data']):
                roi_name = roi_dict['names'][idx]
                roi_layer = viewer.add_labels(
                    viewer.add_shapes(
                        roi,
                        shape_type=roi_dict['shape_type'][idx],
                        name=roi_name
                    ).to_labels(
                        labels_shape=img_data.shape
                    )
                )
                # remove "shapes" layer
                viewer.layers.pop(-2)
                img_names.append(img_file.stem)
                roi_names.append(roi_name)
                roi_values.append(img_data[roi_layer.data == 1])
                # remove labels layer
                viewer.layers.pop()
        pd.DataFrame({
            "image": img_names,
            "roi": roi_names,
            "values": roi_values
        }).to_json(input_dir / "roi_values.json")
    return

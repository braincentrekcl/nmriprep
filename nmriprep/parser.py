from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path


def get_parser():
    """Build parser object."""

    parser = ArgumentParser(
        description='Convert .nef image to .nii with uCi/g voxels',
        formatter_class=ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        "standard_type",
        help="radioisotope of standards",
        choices=["H3", "C14"]
    )
    parser.add_argument(
        "source_directory",
        help="raw data directory",
        type=Path
        )
    parser.add_argument(
        "--flat-field",
        help="Path to flat field image",
        type=Path
    )
    parser.add_argument(
        "--dark-field",
        help="Path to dark field image",
        type=Path
    )
    parser.add_argument(
        "--output",
        help="derivative data directory",
        type=Path
        )
    parser.add_argument(
        "--crop-width",
        help="fraction of image width to remove from each end",
        type=float
        )
    parser.add_argument(
        "--crop-height",
        help="fraction of image height to remove from each end",
        type=float
        )
    parser.add_argument(
        "--rotate",
        help="number of 90º CW rotations",
        type=int,
        default=0
    )
    parser.add_argument(
        "--save-intermediate",
        help="generate content for assessment",
        action="store_true"
        )
    parser.add_argument(
        "--save-nii",
        help="generate nifti output",
        action="store_true"
        )
    parser.add_argument(
        "--save-tif",
        help="generate tif output",
        action="store_true"
        )
    parser.add_argument(
        "--mosaic-slices",
        help="slices indices to include " +
        "in mosaic plot (-1 indicates all slices)",
        type=int,
        nargs='*'
        )
    parser.add_argument(
        "--subject-id",
        help="optional list of subject IDs to process",
        nargs="*"
    )
    return parser

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path


def get_argprep_parser():
    """Build parser object."""

    parser = ArgumentParser(
        description='Convert .nef image to .nii with uCi/g voxels',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'standard_type', help='radioisotope of standards', choices=['H3', 'C14']
    )
    parser.add_argument('source_directory', help='raw data directory', type=Path)
    parser.add_argument('--flat-field', help='Path to flat field image', type=Path)
    parser.add_argument('--dark-field', help='Path to dark field image', type=Path)
    parser.add_argument('--output', help='derivative data directory', type=Path)
    parser.add_argument(
        '--crop-width',
        help='fraction of image width to remove from each end',
        type=float,
    )
    parser.add_argument(
        '--crop-height',
        help='fraction of image height to remove from each end',
        type=float,
    )
    parser.add_argument(
        '--rotate', help='number of 90º CW rotations', type=int, default=0
    )
    parser.add_argument(
        '--save-intermediate',
        help='generate content for assessment',
        action='store_true',
    )
    parser.add_argument('--save-nii', help='generate nifti output', action='store_true')
    parser.add_argument('--save-tif', help='generate tif output', action='store_true')
    parser.add_argument(
        '--mosaic-slices',
        help='slices indices to include ' + 'in mosaic plot (-1 indicates all slices)',
        type=int,
        nargs='*',
    )
    parser.add_argument(
        '--subject-id', help='optional list of subject IDs to process', nargs='*'
    )
    return parser


def get_fieldprep_parser():
    """Build parser object."""

    parser = ArgumentParser(
        description='Convert dark/flatfield .nef images for processing with argprep',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--flat-field', help='Path to directory containing flat field image', type=Path, nargs='*')
    parser.add_argument('--dark-field', help='Path to directory containing dark field image', type=Path, nargs='*')
    parser.add_argument('--output', help='Optionally specify output directory', type=Path, nargs='*')
    return parser


def get_roiextract_parser():
    """Build parser object."""

    parser = ArgumentParser(
        description='Extract data from ROIs defined'
        + ' by napari ROI manager .json files',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('source_directory', help='raw data directory', type=Path)
    parser.add_argument(
        '--image-suffix', help='suffix replacing `rois` in .tif files', default='ARG'
    )
    parser.add_argument(
        '--roi-suffix', help='suffix of ROI .json files', default='rois'
    )
    parser.add_argument(
        '--grouping-vars',
        help='filename keys to group on',
        nargs='*',
        type=str,
    )
    parser.add_argument(
        '--norm-regions',
        help='reference regions to normalise values',
        nargs='*',
        type=str,
    )
    parser.add_argument(
        '--output', help='name of output .json file', default='roi_values', type=str
    )
    return parser

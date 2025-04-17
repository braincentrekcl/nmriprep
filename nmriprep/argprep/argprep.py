from collections import defaultdict
import nibabel as nb
import numpy as np

from ..image import convert_nef_to_grey, save_slice
from ..parser import get_argprep_parser
from ..plotting import plot_curve, plot_mosaic, plot_single_slice
from ..utils import find_files, inverse_rodbard, rodbard
from .calibration import calibrate_standard
from .fieldprep import find_fields

def subject_workflow(sub_files, out_dir, args, verbose):
    sub_id = sub_files[0].parent.stem
    sub_dir = out_dir / sub_id
    sub_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = sub_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # attempt to find flat field info
    flatfield_correction = {}
    flatfield_correction['dark'] = find_fields(
        args.dark_field, sub_dir.glob('*darkfield.tif*')
    )
    flatfield_correction['flat'] = find_fields(
        args.flat_field, sub_dir.glob('*flatfield.tif*')
    )
    if any(v is None for v in flatfield_correction.values()):
        print('Skipping flat field correction...')
        flatfield_correction = None

    # calibrate standards to transform GV to radioactivity
    std_files = [fname for fname in sub_files if "standard" in fname.stem]
    if len(std_files) < 1:
        raise FileNotFoundError(f'No standard files found for {sub_id}')
    popt, std_rad, std_gv, std_stem = calibrate_standard(
        sub_id,
        std_files,
        args.standard_type,
        flatfield_correction=flatfield_correction,
        out_dir=sub_dir if verbose else None,
    )

    if verbose:
        plot_curve(std_rad, std_gv, rodbard(std_rad, *popt), sub_dir, std_stem)

    print(f'success! fitting data with parameters: {popt}')

    # find subject files
    slide_files = [fname for fname in sub_files if "slide" in fname.stem]
    if len(slide_files) < 1:
        raise FileNotFoundError(f'No slide files found for {sub_id}')

    # convert nefs to grey value
    data_gv = np.stack(
        [
            convert_nef_to_grey(
                fname,
                flatfield_correction=flatfield_correction,
                crop_row=args.crop_height,
                crop_col=args.crop_width,
                invert=True,
            )
            for fname in slide_files
        ],
        axis=2,
    )

    if args.rotate > 0:
        data_gv = np.rot90(data_gv, k=args.rotate, axes=(0, 1))

    # calibrate slice data
    print('Converting slide data to radioactivity')
    data_radioactivity = inverse_rodbard(data_gv, *popt)
    # clip "negative" values
    data_radioactivity[np.isnan(data_radioactivity)] = 0
    print('Success! Generating output...')

    if args.save_nii:
        # write out nii image (e.g. for Jim)
        # find information on number of slides from last file name
        last_section_parts = slide_files[-1].stem.split('_')
        max_slide = int(
            [
                last_section_parts[idx].split('-')[-1]
                for idx, val in enumerate(last_section_parts)
                if 'slide' in val
            ][0]
        )

        # iterate through slides and collect sections to concatenate
        for slide_no in range(max_slide):
            slide_val = str(slide_no + 1).zfill(2)
            slide_sections = {
                fname: idx
                for idx, fname in enumerate(slide_files)
                if f'slide-{str(slide_val).zfill(2)}' in str(fname.stem)
            }

            nb.Nifti1Image(
                data_radioactivity[..., sorted(slide_sections.values())],
                affine=None,
            ).to_filename(
                sub_dir / f'{std_stem}_slide-{slide_val}_desc-preproc_ARG.nii.gz'
            )

    if args.save_tif:
        print('Saving individual files...')
        for idx, fname in enumerate(slide_files):
            print(f'{fname.stem}')
            save_slice(
                data_radioactivity[..., idx],
                sub_dir / f'{fname.stem}_desc-preproc_ARG.tif',
            )

            if verbose:
                # plot image
                plot_single_slice(
                    data_radioactivity[..., idx],
                    fig_dir / f'{fname.stem}_desc-preproc_ARG.png',
                )

                # plot fits
                plot_curve(
                    std_rad=std_rad,
                    std_gv=std_gv,
                    fitted_gv=rodbard(std_rad, *popt),
                    out_dir=fig_dir,
                    out_stem=fname.stem,
                    data_rad=data_radioactivity[..., idx].ravel(),
                    data_gv=data_gv[..., idx].ravel(),
                )
    if args.mosaic_slices:
        sliced = (
            data_radioactivity
            if -1 in args.mosaic_slices
            else data_radioactivity[..., args.mosaic_slices]
        )
        plot_mosaic(sliced, out_dir / f'{std_stem}_desc-preproc_ARG.png')


def main():
    args = get_argprep_parser().parse_args()
    src_dir = args.source_directory.absolute()
    out_dir = src_dir.parent / 'preproc' if not args.output else args.output
    verbose = args.save_intermediate

    # identify subjects for pipeline
    fnames = find_files(src_dir.rglob('*.nef'))
    subdirs = defaultdict(list)
    [subdirs[p.parent].append(p) for p in fnames]
    all_subject_directories = [subdir for subdir in subdirs.keys()]

    if args.subject_id:
        print(f'Processing subjects: {args.subject_id}')
        subjects_to_process = [
            subj for subj in all_subject_directories
            if subj.stem.replace("sub-","") in args.subject_id
        ]
    else:
        print('Processing all subjects')
        subjects_to_process = all_subject_directories

    for sub_id in subjects_to_process:
        sub_files = subdirs[sub_id]
        subject_workflow(sub_files, out_dir, args, verbose)


if __name__ == '__main__':
    main()

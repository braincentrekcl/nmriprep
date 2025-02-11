import nibabel as nb
import numpy as np

from .calibration import calibrate_standard
from ..parser import get_argprep_parser
from ..image import convert_nef_to_grey, read_tiff, save_slice
from ..plotting import plot_curve, plot_single_slice, plot_mosaic
from ..utils import find_files, rodbard, inverse_rodbard


def main():
    args = get_argprep_parser().parse_args()
    src_dir = args.source_directory.absolute()
    out_dir = src_dir.parent / "preproc" if not args.output else args.output
    verbose = args.save_intermediate

    # attempt to find flat field info
    flatfield_correction = {}
    if args.dark_field and args.flat_field:
        flatfield_correction['dark'] = read_tiff(args.dark_field)
        flatfield_correction['flat'] = read_tiff(args.flat_field)
    else:
        # try searching for a preprocessed darkfield
        latest_ff = find_files(out_dir.glob("*flatfield.tiff"))
        latest_df = find_files(out_dir.glob("*darkfield.tiff"))
        if latest_ff and latest_df:
            flatfield_correction['dark'] = read_tiff(latest_df[-1])
            flatfield_correction['flat'] = read_tiff(latest_ff[-1])
        else:
            print("Skipping flat field correction...")
            flatfield_correction = None

    # identify subjects for pipeline
    all_subject_ids = set([
        part.split('-')[1]
        for fpath in src_dir.glob('*.nef')
        for part in fpath.stem.split('_')
        if "subj" in part
    ])

    if args.subject_id:
        print(f"Processing subjects: {args.subject_id}")
        subjects_to_process = [
            subj for subj in all_subject_ids if subj in args.subject_id
        ]
    else:
        print("Processing all subjects")
        subjects_to_process = all_subject_ids

    for sub_id in subjects_to_process:
        sub_dir = out_dir / sub_id
        sub_dir.mkdir(exist_ok=True, parents=True)
        fig_dir = sub_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # calibrate standards to transform GV to radioactivity
        popt, std_rad, std_gv, std_stem = calibrate_standard(
            sub_id,
            src_dir,
            args.standard_type,
            flatfield_correction=flatfield_correction,
            out_dir=sub_dir if verbose else None,
        )

        if verbose:
            plot_curve(
                std_rad,
                std_gv,
                rodbard(std_rad, *popt),
                sub_dir,
                std_stem
            )

        print(f"success! fitting data with parameters: {popt}")

        # find subject files
        slide_files = find_files(src_dir.glob(f'*subj-{sub_id}*slide*.nef'))
        if len(slide_files) < 1:
            raise FileNotFoundError(
                f"No slide files found for {sub_id} in {src_dir}"
            )

        # convert nefs to grey value
        data_gv = np.stack([
            convert_nef_to_grey(
                fname,
                flatfield_correction=flatfield_correction,
                crop_row=args.crop_height,
                crop_col=args.crop_width,
            ) for fname in slide_files
            ],
            axis=2
        )

        # calibrate slice data
        print("Converting slide data to radioactivity")
        data_radioactivity = inverse_rodbard(data_gv, *popt)
        # clip "negative" values
        data_radioactivity[np.isnan(data_radioactivity)] = 0
        print("Success! Generating output...")

        if args.rotate > 0:
            data_radioactivity = np.rot90(
                data_radioactivity,
                k=args.rotate,
                axes=(0, 1)
            )

        out_stem = std_stem.split('_standard')[0]

        if args.save_nii:
            # write out nii image (e.g. for Jim)
            # find information on number of slides from last file name
            last_section_parts = slide_files[-1].stem.split('_')
            max_slide = int([
                last_section_parts[idx].split('-')[-1] for idx, val
                in enumerate(
                    last_section_parts
                ) if "slide" in val
            ][0])

            # iterate through slides and collect sections to concatenate
            for slide_no in range(max_slide):
                slide_val = str(slide_no + 1).zfill(2)
                slide_sections = {
                    fname: idx
                    for idx, fname in
                    enumerate(slide_files)
                    if f"slide-{str(slide_val).zfill(2)}" in str(fname.stem)
                }

                nb.Nifti1Image(
                    data_radioactivity[..., sorted(slide_sections.values())],
                    affine=None,
                ).to_filename(
                    sub_dir /
                    f"{out_stem}_slide-{slide_val}_desc-preproc_ARG.nii.gz"
                )

        if args.save_tif:
            print("Saving individual files...")
            for idx, fname in enumerate(slide_files):
                print(f"{fname.stem}")
                save_slice(
                    data_radioactivity[..., idx],
                    sub_dir / f"{fname.stem}_desc-preproc_ARG.tif"
                )

                if verbose:
                    # plot image
                    plot_single_slice(
                        data_radioactivity[..., idx],
                        fig_dir / f"{fname.stem}_desc-preproc_ARG.png"
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
                data_radioactivity if -1 in args.mosaic_slices
                else data_radioactivity[..., args.mosaic_slices]
            )
            plot_mosaic(sliced, out_dir / f"{out_stem}_desc-preproc_ARG.png")


if __name__ == "__main__":
    main()

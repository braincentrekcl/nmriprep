import numpy as np

from ..image import convert_nef_to_grey, save_slice
from ..parser import get_fieldprep_parser
from ..utils import find_files, parse_kv


def fieldprep():
    args = get_fieldprep_parser().parse_args()

    if args.dark_field:
        raise NotImplementedError("Dark field processing coming soon...")
    
    if args.flat_field:
        ff_dir = args.flat_field
        fnames = find_files(ff_dir.glob('*flatfield*.nef'))
        if len(fnames) < 1:
            raise FileNotFoundError(
                f'No flat field files found in {ff_dir.absolute()}'
            )
        out_dir = ff_dir.parent / 'preproc' if not args.output else args.output
        out_dir.mkdir(parents=True, exist_ok=True)

        data = np.stack(
            [ convert_nef_to_grey(fname) for fname in fnames] ,
            axis=2,
        ).mean(axis=2)
        out_stem = '_'.join(
            f'{k}-{v}' for k, v
            in parse_kv(fnames[0]).items()
            if "flatfield" not in k
        )
        save_slice(data, out_dir / f"{out_stem}_flatfield.tif" )
    return

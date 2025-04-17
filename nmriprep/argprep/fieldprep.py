import numpy as np
from collections import defaultdict
from pathlib import Path

from ..image import convert_nef_to_grey, save_slice
from ..parser import get_fieldprep_parser
from ..utils import find_files, parse_kv


def fieldprep():
    args = get_fieldprep_parser().parse_args()

    if args.dark_field:
        raise NotImplementedError('Dark field processing coming soon...')

    if args.flat_field:
        ff_dir = args.flat_field
        fnames = find_files(ff_dir.rglob('*flatfield*.nef'))
        if len(fnames) < 1:
            raise FileNotFoundError(f'No flat field files found in {ff_dir.absolute()}')
        
        subdirs = defaultdict(list)
        [subdirs[p.parent].append(p) for p in fnames]

        for subdir in subdirs.keys():
            sub_files = subdirs[subdir]
            fname_parts = parse_kv(sub_files[0].stem)
            out_stem = '_'.join(
                f'{k}-{v}' for k, v in fname_parts.items() if 'flatfield' not in k
            )
            out_dir = Path(
                subdir.str.lower().replace('sourcedata', 'preproc')
            ) if not args.output else args.output
            out_dir.mkdir(parents=True, exist_ok=True)

            data = np.median(
                np.stack(
                    [convert_nef_to_grey(fname) for fname in sub_files],
                    axis=2,
                ),
                axis=2,
            )
            save_slice(data, out_dir / f'{out_stem}_flatfield.tif')
    return

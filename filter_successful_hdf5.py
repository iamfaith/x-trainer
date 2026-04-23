#!/usr/bin/env python3
"""
Filter HDF5 dataset files to keep only episodes whose `success` attribute is True.

Usage:
  python scripts/tools/filter_successful_hdf5.py --input datasets/dataset.hdf5
  python scripts/tools/filter_successful_hdf5.py --input datasets --output_dir datasets/filtered

Options:
  --include-missing    Treat episodes with no `success` attribute as successful (default: False)
  --in-place           Overwrite original files (use with caution)
  --verbose            Print details
"""

from __future__ import annotations
import argparse
import os
import h5py
import glob
import shutil
from typing import Iterable


def filter_file(input_path: str, output_path: str, include_missing: bool = False, verbose: bool = False) -> int:
    """Copy only successful demos from input_path into output_path.

    Returns number of demos copied.
    """
    print(input_path, output_path)
    with h5py.File(input_path, 'r') as fin:
        if 'data' not in fin:
            raise RuntimeError(f"Input file {input_path} has no 'data' group")
        demo_names = list(fin['data'].keys())
        successful = []
        total_samples = 0
        for name, grp in fin['data'].items():
            succ = grp.attrs.get('success', None)
            print(name, succ)
            if succ:
                successful.append(name)
                total_samples += int(grp.attrs.get('num_samples', 0))

        # for name in demo_names:
        #     grp = fin['data'][name]
        #     succ = grp.attrs.get('success', None)
        #     if succ is True or (succ is None and include_missing):
        #         successful.append(name)
        #         total_samples += int(grp.attrs.get('num_samples', 0))

        # if verbose:
        print(f"{input_path}: found {len(demo_names)} demos, {len(successful)} successful:[{successful}]")

        # create output file
        # ensure output folder exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with h5py.File(output_path, 'w') as fout:
            # copy root-level items except 'data' (keep env args / metadata)
            for key in fin.keys():
                if key == 'data':
                    continue
                try:
                    fin.copy(key, fout)
                except Exception:
                    # fallback: skip non-copyable items
                    if verbose:
                        print(f"Warning: could not copy top-level group {key}")
            # copy root attrs
            for k, v in fin.attrs.items():
                try:
                    fout.attrs[k] = v
                except Exception:
                    if verbose:
                        print(f"Warning: could not copy root attr {k}")

            # create data group and populate
            data_grp = fout.create_group('data')
            data_grp.attrs['total'] = total_samples
            for name in successful:
                # preserve original demo name
                fin.copy(f'data/{name}', data_grp, name=name)

    return len(successful)


def iter_inputs(path: str) -> Iterable[str]:
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, '*.hdf5')) + glob.glob(os.path.join(path, '*.hf5'))):
            yield f
    elif os.path.isfile(path):
        yield path
    else:
        # try glob
        for f in sorted(glob.glob(path)):
            yield f


def main():
    parser = argparse.ArgumentParser(description="Filter HDF5 datasets keeping only successful demos.")
    parser.add_argument('--input', '-i', required=False, help='Input hdf5 file path or directory or glob', default='./datasets/test.hdf5')
    parser.add_argument('--output', '-o', required=False, help='Output file path or directory. If input is dir, output must be dir.')
    parser.add_argument('--include-missing', action='store_true', help='Treat missing success attr as successful')
    parser.add_argument('--in-place', action='store_true', help='Overwrite original files (will backup original with .bak)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    inputs = list(iter_inputs(args.input))
    if len(inputs) == 0:
        print('No input files found')
        return

    # determine output targets
    outputs = []
    if args.output:
        if len(inputs) > 1 and not os.path.isdir(args.output):
            raise RuntimeError('When filtering multiple inputs, --output must be a directory')
        if os.path.isdir(args.output):
            for inp in inputs:
                outputs.append(os.path.join(args.output, os.path.basename(inp)))
        else:
            # single file
            outputs.append(args.output)
    else:
        # default: if single input -> input_successful.hdf5; if multiple -> input_dir/filtered/
        if len(inputs) == 1:
            inp = inputs[0]
            base = os.path.splitext(inp)[0]
            outputs.append(base + '_successful.hdf5')
        else:
            out_dir = os.path.join(os.path.dirname(inputs[0]), 'filtered')
            os.makedirs(out_dir, exist_ok=True)
            for inp in inputs:
                outputs.append(os.path.join(out_dir, os.path.basename(inp)))

    for inp, outp in zip(inputs, outputs):
        if args.in_place:
            # backup
            bak = inp + '.bak'
            if not os.path.exists(bak):
                shutil.copy2(inp, bak)
        try:
            n = filter_file(inp, outp, include_missing=args.include_missing, verbose=args.verbose)
            print(f'Wrote {n} successful demos to {outp}')
            if args.in_place:
                # replace original
                shutil.move(outp, inp)
                print(f'Replaced original file {inp} (backup at {bak})')
        except Exception as e:
            print(f'Error processing {inp}: {e}')


if __name__ == '__main__':
    main()

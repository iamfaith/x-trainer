#!/usr/bin/env python3
"""
Merge successful demos from all HDF5 files in a directory into one HDF5 file.
Filtering uses the same logic as:

    succ = grp.attrs.get('success', None)
    if succ:
        # treat as successful

Options:
  --include-missing    Treat missing `success` attr as successful
  --verbose            Print progress

Example:
  python scripts/tools/merge_successful_hdf5.py --input datasets --output merged_successful.hdf5
"""

from __future__ import annotations
import argparse
import os
import glob
import h5py
from typing import Iterable, Tuple, List


def iter_hdf5_files(dirpath: str) -> Iterable[str]:
    for p in sorted(glob.glob(os.path.join(dirpath, "*.hdf5")) + glob.glob(os.path.join(dirpath, "*.hf5"))):
        yield p


def collect_successful_names(path: str, include_missing: bool = False, verbose: bool = False) -> Tuple[List[str], int]:
    """Return list of successful demo names in file and total samples for those demos."""
    successful: List[str] = []
    total_samples = 0
    with h5py.File(path, 'r') as fin:
        if 'data' not in fin:
            if verbose:
                print(f"Skipping {path}: no 'data' group")
            return successful, total_samples
        for name, grp in fin['data'].items():
            succ = grp.attrs.get('success', None)
            if verbose:
                print(path, name, succ)
            # follow the provided logic: treat as successful when `succ` is truthy
            if succ or (succ is None and include_missing):
                successful.append(name)
                total_samples += int(grp.attrs.get('num_samples', 0))
    return successful, total_samples


def merge_successful(input_dir: str, output_path: str, include_missing: bool = False, verbose: bool = False) -> Tuple[int, int]:
    """Merge all successful demos from HDF5 files in `input_dir` into `output_path`.
    Returns (num_demos_merged, total_samples).
    """
    files = list(iter_hdf5_files(input_dir))
    if not files:
        raise RuntimeError(f"No hdf5 files found in {input_dir}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    merged_count = 0
    merged_samples = 0

    with h5py.File(output_path, 'w') as fout:
        data_grp = fout.create_group('data')
        # iterate input files and copy successful demos
        for fpath in files:
            if verbose:
                print(f"Processing {fpath}...")
            with h5py.File(fpath, 'r') as fin:
                if 'data' not in fin:
                    if verbose:
                        print(f"  skipping {fpath}: no data group")
                    continue
                for name, grp in fin['data'].items():
                    succ = grp.attrs.get('success', None)
                    if succ or (succ is None and include_missing):
                        # make a unique name to avoid collisions: <filebase>__<original_name>
                        filebase = os.path.splitext(os.path.basename(fpath))[0]
                        new_name = f"{filebase}__{name}"
                        try:
                            fin.copy(f"data/{name}", data_grp, name=new_name)
                            merged_count += 1
                            merged_samples += int(grp.attrs.get('num_samples', 0))
                            if verbose:
                                print(f"  copied {name} -> {new_name}")
                        except Exception as e:
                            if verbose:
                                print(f"  warning: failed copying {name} from {fpath}: {e}")
        data_grp.attrs['total'] = merged_samples

    return merged_count, merged_samples


def main():
    parser = argparse.ArgumentParser(description="Merge successful demos from a directory of HDF5 files.")
    parser.add_argument('--input', '-i', required=True, help='Input directory containing .hdf5 files')
    parser.add_argument('--output', '-o', required=True, help='Output merged hdf5 file path')
    parser.add_argument('--include-missing', action='store_true', help='Treat missing success attr as successful')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    args.verbose = True
    num, samples = merge_successful(args.input, args.output, include_missing=args.include_missing, verbose=args.verbose)
    print(f"Wrote {num} successful demos to {args.output} (total samples: {samples})")


if __name__ == '__main__':
    main()

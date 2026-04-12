"""Command-line interface for magirrep."""

import argparse
import os
import sys

from magirrep.pipeline import run_analysis, run_displacive_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="magirrep",
        description="Determine the active magnetic or displacive irrep from an mCIF/CIF file",
    )
    parser.add_argument("mcif_file", help="Path to the mCIF (or CIF) file")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print debug details at each pipeline stage",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--displacive",
        action="store_true",
        help="Displacive/mechanical representation only (all atoms, no det factor)",
    )
    mode_group.add_argument(
        "--magnetic",
        action="store_true",
        help="Magnetic representation only (skip displacive pass; faster)",
    )
    parser.add_argument(
        "--kvector",
        default=None,
        metavar="KX,KY,KZ",
        help="Propagation vector for displacive mode from a plain CIF, e.g. '0,1/2,0' (default: 0,0,0)",
    )
    parser.add_argument(
        "--output", "-o",
        nargs="?", const="AUTO", default=None,
        metavar="FILE",
        help="Save output to FILE (omit filename to auto-generate {stem}_magirrep.txt)",
    )
    parser.add_argument(
        "--distort", type=float, metavar="AMP", nargs="?", const=0.1, default=None,
        help="Generate distorted CIF files per (irrep, Wyckoff site, BV). "
             "AMP is the displacement amplitude in Angstroms (default 0.1). "
             "Requires --displacive.",
    )
    parser.add_argument(
        "--keep-magnetic", action="store_true",
        help="With --distort on mCIF input: preserve magnetic moments (writes .mcif files).",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Output directory for distorted CIF files (default: current working directory).",
    )
    args = parser.parse_args(argv)

    out_file = args.output
    if out_file == "AUTO":
        stem = os.path.splitext(os.path.basename(args.mcif_file))[0]
        out_file = f"{stem}_magirrep.txt"

    if args.displacive:
        run_displacive_analysis(args.mcif_file, kvector_str=args.kvector,
                                verbose=args.verbose, output_file=out_file,
                                distort_amplitude=args.distort,
                                keep_magnetic=args.keep_magnetic,
                                out_dir=args.out_dir)
    else:
        # Default: combined magnetic+displacive; --magnetic suppresses the displacive pass
        run_analysis(args.mcif_file, verbose=args.verbose, output_file=out_file,
                     displacive_pass=(not args.magnetic))

"""Command-line interface for magirrep."""

import argparse
import sys

from magirrep.pipeline import run_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="magirrep",
        description="Determine the active magnetic irrep from an mCIF file",
    )
    parser.add_argument("mcif_file", help="Path to the mCIF file")
    args = parser.parse_args(argv)

    run_analysis(args.mcif_file)

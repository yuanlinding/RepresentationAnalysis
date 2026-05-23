# magirrep

> **Disclaimer:** This software is research code under active development and has not been
> fully tested across all space groups, k-points, or mCIF conventions. Use results with
> caution and verify against established tools (e.g. Bilbao REPRES, BasIreps) where possible.

Bertaut representational analysis for magnetic structures. Given an mCIF file from
[Bilbao MAGNDATA](https://www.cryst.ehu.es/magndata/) or a hand-crafted mCIF, determine
which magnetic irreducible representation(s) drive the paramagnetic-to-magnetically-ordered
phase transition. Also computes the phonon/mechanical representation for any CIF file.

Supports both MAGNDATA-style mCIF files and plain/hand-crafted mCIF files that use older
CIF conventions (`_symmetry_Int_Tables_number`, `_atom_site_moment_crystalaxis_x`).

## Installation

```bash
conda activate findmagsym          # spglib 2.5.0, spgrep 0.3.5, pymatgen, numpy 1.26.0
pip install -e ".[dev]"            # installs magirrep + gemmi + seekpath + pytest
```

## Usage

### Magnetic analysis (default)

```bash
magirrep file.mcif                   # combined magnetic + phonon report
magirrep file.mcif --magnetic        # magnetic representation only (faster)
magirrep file.mcif -v                # verbose debug output at each pipeline stage
magirrep file.mcif --output          # auto-save to {stem}_magirrep.txt (also prints to terminal)
magirrep file.mcif -o result.txt     # save to named file
```

### Phonon / mechanical representation

```bash
magirrep file.mcif  --phonon                        # phonon-only from mCIF
magirrep file.cif   --phonon --kvector 0,1/2,0      # phonon from plain CIF with explicit k
```

### Displacive representational analysis

```bash
magirrep file.mcif --displacive                     # displacive modes (all atoms)
magirrep file.mcif --displacive --kvector 0,1/2,0   # with explicit k-vector
```

### Generate distorted structures

After displacive analysis, write one CIF (or mCIF) per (irrep, Wyckoff site, basis vector):

```bash
magirrep file.mcif --displacive --distort           # default amplitude 0.1 Å
magirrep file.mcif --displacive --distort 0.05      # custom amplitude in Å
magirrep file.mcif --displacive --distort --keep-magnetic   # mCIF with original moments
magirrep file.mcif --displacive --distort --out-dir ./distorted/
```

### All options

```
positional arguments:
  mcif_file              Path to the mCIF (or CIF) file

optional arguments:
  -h, --help             show this help message and exit
  -v, --verbose          Print debug details at each pipeline stage
  --phonon               Phonon/mechanical representation only (all atoms, no det factor)
  --magnetic             Magnetic representation only (skip phonon pass; faster)
  --displacive           Displacive representational analysis (all atoms, phonon convention)
  --kvector KX,KY,KZ     Propagation vector (for --phonon or --displacive on plain CIF),
                         e.g. '0,1/2,0'  (default: 0,0,0)
  --distort [AMP]        Generate distorted CIF/mCIF files after --displacive analysis.
                         AMP is the displacement amplitude in Angstroms (default: 0.1).
  --keep-magnetic        With --distort on mCIF input: preserve magnetic moments (writes
                         .mcif files with the MSG of the distorted structure).
  --out-dir DIR          Output directory for distorted files (default: current directory).
  --output [FILE], -o [FILE]
                         Save output to FILE. Omit filename to auto-generate
                         {input_stem}_magirrep.txt. Output is also printed to the terminal.
```

`--phonon`, `--magnetic`, and `--displacive` are mutually exclusive.

## Output

The combined report contains 10 sections followed by a validation block:

| Section | Content |
|---------|---------|
| (1) | Parent space group — IT number, symbol, crystal system, point group |
| (2)+(3) | Propagation vector and little group G_k — all {R\|t} coset representatives |
| (4)+(5) | Magnetic atoms (Wyckoff sites) + permutation table (atom→atom, lattice vectors L) |
| (6) | Representation characters — χ_perm, det(R), Tr(R), χ_axial, χ_mag per G_k operation |
| (7) | Irrep character table of G_k (Bilbao-style labels, conjugacy classes, active irreps marked) |
| (8) | **Decomposition** — building-block reps and combined summary table |
| (9) | Basis vectors — symmetry-adapted modes for magnetic atoms and for all atoms (phonon) |
| (10) | Moment–irrep consistency — actual moments projected onto active irrep subspace |

`--phonon` mode omits the magnetic block; `--magnetic` mode omits the phonon block.

For the mathematical background see [`docs/theory.md`](docs/theory.md).

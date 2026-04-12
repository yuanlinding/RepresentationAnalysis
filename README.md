# magirrep

Bertaut representational analysis for magnetic structures. Given an mCIF file from
[Bilbao MAGNDATA](https://www.cryst.ehu.es/magndata/), determine which magnetic
irreducible representation(s) drive the paramagnetic-to-magnetically-ordered phase
transition. Also computes the phonon/mechanical representation for any CIF file.

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

Each output file is **symmetrized**: spglib detects the actual subgroup of the distorted
structure. The symmetry symbol is appended to the filename:

```
{input_stem}_{irrep}_{element}_psi{n}_{symbol}.cif      # plain CIF
{input_stem}_{irrep}_{element}_psi{n}_{symbol}.mcif     # mCIF with moments
```

Examples (CuMnAs, k=Γ, `--keep-magnetic`):
```
0.222_CuMnAs_GM5-_Mn_psi13_Pmn2_1.mcif     # type-1 MSG (no antiunitary ops)
0.222_CuMnAs_GM5-_Mn_psi14_Pm'n2_1'.mcif   # type-3 MSG (half antiunitary)
0.222_CuMnAs_GM1+_Mn_psi1_Pm'mn.mcif       # GM1+ preserves Pm'mn
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
| (8) | **Decomposition** — building-block reps and combined summary table (see below) |
| (9) | Basis vectors — symmetry-adapted modes for magnetic atoms and for all atoms (phonon) |
| (10) | Moment–irrep consistency — actual moments projected onto active irrep subspace |

**Section (8) — building-block decompositions:**

```
Γ_perm  [mag]   = ...    (permutation of magnetic atoms)
Γ_axial          = ...    (χ = det(R)·Tr(R))
Γ_mag   = Γ_perm ⊗ Γ_axial   ← primary magnetic result

Γ_perm  [all]   = ...    (permutation of all atoms)
Γ_polar          = ...    (χ = Tr(R))
Γ_mech  = Γ_perm ⊗ Γ_polar   [total]

  Per-Wyckoff-site contributions to Γ_mech:
    Γ_mech(Mn (c)) = ...
    Γ_mech(Cu (a)) = ...
    ...

  Irrep     dim   n_μ(mag)  n_μ(mech)     η
  ─────────────────────────────────────────────
  mGM5-       2         1          3   1.000  ← ACTIVE
```

`--phonon` mode omits the magnetic block; `--magnetic` mode omits the phonon block.

## Validation targets

| File | Active irrep | k | Parent SG | ‖M−M_rec‖/‖M‖ |
|------|-------------|---|-----------|----------------|
| `1.6_NiO.mcif` | **mL3+** (dim=2) | L=(½,½,½) | #225 Fm-3m (FCC) | 0.000 ✓ |
| `0.15_MnF2.mcif` | **mGM3+** (dim=1, AFM) | Γ | #136 P4₂/mnm | 0.000 ✓ |
| `1.708_CrPS4.mcif` | **mX2** (dim=1) | (0,0,½) | #5 C2 (C-centered) | 0.000 ✓ |
| `0.222_CuMnAs.mcif` | **mGM5−** (dim=2) | Γ | #129 P4/nmm | 0.000 ✓ |

> **Note on k-point labels**: For monoclinic C2 (CrPS4), seekpath labels the zone-boundary
> point (0,0,½) as **X** while Bilbao MAGNDATA uses **A**. The physics is correct; only the
> label in the validation line differs (`mX2` identified vs `mA2` stored).

Expected phonon decompositions:

- **CuMnAs** `Γ_mech = 2·mGM1+ ⊕ 3·mGM2- ⊕ 3·mGM5+ ⊕ 3·mGM5- ⊕ 1·mGM3+`
- **NiO** `Γ_mech = 3·mL1+ ⊕ 1·mL1- ⊕ 1·mL2+ ⊕ 3·mL2- ⊕ 4·mL3+ ⊕ 4·mL3-`

## Running tests

```bash
pytest -v       # 40 tests
```

## Architecture

```
mCIF / CIF
  → parse_mcif.py          parse fields, k-vector, transforms, structure
  → mag_rep.py             map atoms to parent cell; compute χ_mag, χ_disp, χ_perm
  → little_group.py        get SG operations (spglib); filter little group G_k
  → irrep_decompose.py     get irrep matrices (spgrep); reduce formula; basis vectors
  → irrep_label.py         Bilbao-style labels via dimension-sorted ordering + seekpath
  → bilbao_match.py        optional HTTP cross-check against Bilbao REPRES
  → pipeline.py            orchestration + formatted 10-section report
  → cli.py                 argparse entry point
```

For the mathematical background see [`docs/theory.md`](docs/theory.md).

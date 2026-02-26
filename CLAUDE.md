# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bertaut representational analysis tool: given an mCIF file (from Bilbao MAGNDATA), determine which
magnetic irreducible representation(s) drive the paramagnetic-to-magnetically-ordered phase transition.
Complements the existing `findmagsym` tool (which finds the magnetic space group) by identifying the
**active irrep** of the parent space group at propagation vector **k**.

## Environment Setup

The project uses the `findmagsym` conda environment:
```bash
conda activate findmagsym
# has: spglib 2.5.0, spgrep 0.3.5, pymatgen, numpy 1.26.0
pip install -e ".[dev]"   # installs magirrep + gemmi + pytest
```

Python path: `/Users/ldyuan/Apps/anaconda3/envs/findmagsym/bin/python`

## Running

```bash
# Any of these work:
magirrep tests/data/0.222_CuMnAs.mcif
python -m magirrep tests/data/0.222_CuMnAs.mcif
```

Validation targets:
- `magirrep tests/data/0.222_CuMnAs.mcif` — expect active irrep **mGM5-** (dim=2, k=Gamma, SG #129)
- `magirrep tests/data/1.6_NiO.mcif` — expect active irrep **mL3+** (dim=8, small dim=2, k=L, SG #225)

## Testing

```bash
pytest -v                              # run all tests
pytest tests/test_little_group.py -v   # run tests that don't need gemmi
```

## Architecture

The pipeline in `src/magirrep/pipeline.py` is: **parse → transform → symmetry ops → irreps → decompose → label**.

### Data Flow

```
mCIF file
  → parse_mcif.parse_mcif_fields()     # extract IT number, k-vector, transforms
  → parse_mcif.get_magnetic_structure() # pymatgen Structure with magnetic moments
  → mag_rep.map_atoms_to_parent_cell()  # child cell coords → standard parent cell coords
  → little_group.get_parent_sg_operations()  # (R, t) pairs from spglib
  → irrep_decompose.get_little_group_irreps()  # spgrep irrep matrices + mapping
  → mag_rep.compute_characters()        # χ_mag(g) for each little-group operation
  → irrep_decompose.decompose()         # n_μ = (1/|G_k|) Σ χ_μ* χ_mag
  → irrep_label.irrep_name()            # Bilbao-style label (stub)
```

### Key Modules

- **src/magirrep/cli.py**: Argparse entry point
- **src/magirrep/pipeline.py**: Orchestration (`run_analysis()`)
- **src/magirrep/parse_mcif.py**: Extracts mCIF fields via gemmi, k-vector via `fractions.Fraction`, transforms via pymatgen `SymmOp`
- **src/magirrep/little_group.py**: Maps IT→Hall number (spglib cache), retrieves SG operations, filters little group G_k via R^T·k ≡ k (mod Z³)
- **src/magirrep/mag_rep.py**: Transforms child-cell positions/moments to parent cell; computes magnetic representation characters χ_mag(g) = det(R)·Tr(R)·Σ_fixed exp(-2πi k·L)
- **src/magirrep/irrep_decompose.py**: Wraps `spgrep.get_spacegroup_irreps()` via `build_reference_crystal()`; applies reduction formula with `mapping_little_group`
- **src/magirrep/irrep_label.py**: Minimal stub mapping k-vectors to labels (GM, L, X, M)
- **scripts/magrep_bilbao_parentdetect.py**: Standalone 600-line alternative implementation with Bilbao HTTP querying and auto parent-cell detection. Has syntax error at line 334. Contains useful logic for D(g) matrices and projection operators not yet in the modular pipeline.

### Key spgrep API

```python
# get_spacegroup_irreps takes a crystal (lattice, positions, numbers) and handles
# primitive/conventional conversion internally.  Returns 4-tuple:
irreps, rotations, translations, mapping_little_group = spgrep.get_spacegroup_irreps(
    lattice, positions, numbers, kpoint)
# irreps[alpha][i] is matrix for the i-th little-group operation
# mapping_little_group[i] is the index of that op in rotations/translations
# chi_mag must be computed for ALL rotations, then indexed: chi_lg = chi_mag[mapping_little_group]
```

## Known Issues

**P5 (Irrep labeling) — OPEN**: spgrep's 0-based irrep indices ≠ Bilbao labels. No parity
suffix (±) is computed. `irrep_label.py` is a minimal stub that returns placeholder names
like "mGM5" instead of the correct "mGM5-". The correct Bilbao label requires computing the
character under the inversion/time-reversal element and cross-referencing the Bilbao database.

## Resolved Issues

**P0 (gemmi)**: gemmi is now installed via `pip install -e ".[dev]"`.

**P1 (Primitive vs conventional)**: `get_little_group_irreps()` now uses
`spgrep.get_spacegroup_irreps(lattice, positions, numbers, kpoint)` with a reference crystal
built by `build_reference_crystal()`. spgrep handles the primitive/conventional conversion
internally; the 192-op conventional Fm-3m cell no longer causes a crash.

**P2 (Transform direction) — was a false alarm**: `map_atoms_to_parent_cell()` correctly
implements `r_parent = P @ r_child + p`, which is the proper ITA/Bilbao convention for
`_parent_transform_Pp_abc`. The CLAUDE.md entry was incorrect.

**P3 (spgrep mapping ignored)**: `decompose()` now takes `mapping_little_group` explicitly and
uses `chi_lg = chi_mag[mapping_little_group]` with divisor `len(mapping_little_group)`.

**Bug 4 (Zero-moment atoms)**: `run_analysis()` filters out atoms with
`np.linalg.norm(m_vec) < 0.01` before computing chi_mag. Pymatgen assigns magmom=0 to
non-magnetic species (Cu, As, O); including them inflated chi_mag.

**Bug 5 (Supercell duplicates)**: `_deduplicate_positions()` collapses repeated parent-cell
sites that arise when a supercell mCIF is mapped back to the parent cell (e.g., 32 Ni in the
NiO 2×2×2 supercell → 4 unique conventional-cell positions).

**Bug 6 (Conventional vs primitive for chi_mag)**: `_select_primitive_atoms()` reduces the
conventional-cell atom list to one representative per primitive cell (e.g., 4 Ni → 1 for
Fm-3m FCC). spgrep's returned irreps are indexed by primitive-cell operations, so chi_mag
must also use primitive-cell atoms.

## Conventions

- Magnetic moments in mCIF are in crystal axis (`_atom_site_moment.crystalaxis_x/y/z`), stored in a separate loop from atom positions
- Bilbao `child_transform_Pp_abc` convention: `(P, p)` means `r_parent = P · r_child + p` for coordinates
- k-vectors are in fractional reciprocal coordinates of the parent cell
- Axial vector (pseudovector) transform: `m' = det(P) · P · m`

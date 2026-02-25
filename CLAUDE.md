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
- **src/magirrep/irrep_decompose.py**: Wraps `spgrep.get_spacegroup_irreps_from_primitive_symmetry()`; applies reduction formula
- **src/magirrep/irrep_label.py**: Minimal stub mapping k-vectors to labels (GM, L, X, M)
- **scripts/magrep_bilbao_parentdetect.py**: Standalone 600-line alternative implementation with Bilbao HTTP querying and auto parent-cell detection. Has syntax error at line 334. Contains useful logic for D(g) matrices and projection operators not yet in the modular pipeline.

### Key spgrep API

```python
irreps, mapping_little_group = spgrep.get_spacegroup_irreps_from_primitive_symmetry(
    rotations, translations, kpoint)
# irreps[alpha][i] is matrix for (rotations[mapping_little_group[i]], translations[mapping_little_group[i]])
# mapping_little_group: shape (little_group_order,) — indices into the input rotation/translation arrays

# Simpler alternative (handles primitive/conventional internally):
irreps, mapping = spgrep.get_spacegroup_irreps(hall_number, kpoint)
```

## Known Critical Issues (see docs/plan.md Section 5 for details)

**P0**: gemmi not installed — blocks execution.

**P1 (Primitive vs conventional)**: `spglib.get_symmetry_from_database()` returns conventional-cell operations but spgrep needs primitive-cell operations. SG #225 (Fm-3m): 192 conventional vs 48 primitive ops. Fix: use `spgrep.get_spacegroup_irreps(hall_number, kpoint)` which handles this internally.

**P2 (Transform inversion)**: `map_atoms_to_parent_cell()` applies the child transform *forward* (`M @ r + t`) but needs the *inverse* (`M⁻¹ @ (r - t)`). Hidden for CuMnAs (identity transform), wrong for NiO (diag(2,2,2) doubles instead of halving coordinates).

**P3 (spgrep mapping ignored)**: Code ignores `mapping_little_group` — chi_mag is computed for all SG ops while irreps are indexed by little-group ops. The decomposition formula sums mismatched arrays.

**P5 (Irrep labeling)**: spgrep indices ≠ Bilbao labels. No parity suffix computation. Stub only.

## Conventions

- Magnetic moments in mCIF are in crystal axis (`_atom_site_moment.crystalaxis_x/y/z`), stored in a separate loop from atom positions
- Bilbao `child_transform_Pp_abc` convention: `(P, p)` means `r_parent = P · r_child + p` for coordinates
- k-vectors are in fractional reciprocal coordinates of the parent cell
- Axial vector (pseudovector) transform: `m' = det(P) · P · m`

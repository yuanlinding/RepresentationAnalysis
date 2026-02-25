# Magnetic Irrep Analysis: Project Documentation

## 1. Goal

Build a Python tool that reads an mCIF file and determines — via Bertaut representational
analysis — which magnetic irreducible representation(s) drive the paramagnetic-to-magnetically-ordered
phase transition. This complements the existing `findmagsym` tool (which identifies the MSG) by
identifying the **active irrep** of the parent space group at the propagation vector **k**.

Validation targets (Bilbao MAGNDATA):
- `0.222_CuMnAs.mcif` --> active irrep **mGM5-** (dim=2, k=Gamma, SG #129 P4/nmm)
- `1.6_NiO.mcif` --> active irrep **mL3+** (dim=8, small dim=2, k=L=[1/2,1/2,1/2], SG #225 Fm-3m)

---

## 2. Algorithm (9 Steps)

### Inputs (from mCIF)

| mCIF field                                   | meaning                                        |
|----------------------------------------------|------------------------------------------------|
| `_parent_space_group.IT_number`              | Parent SG international number (e.g., 129, 225)|
| `_parent_propagation_vector.kxkykz`          | k-vector in parent reciprocal coords           |
| `_parent_space_group.child_transform_Pp_abc` | Supercell transform parent --> magnetic cell    |
| `_parent_space_group.transform_Pp_abc`       | Non-standard --> standard parent setting        |
| `_atom_site_fract_x/y/z`                     | Atom positions (in magnetic cell)               |
| `_atom_site_moment.crystalaxis_x/y/z`        | Magnetic moments in crystal axis                |
| `_irrep_id`, `_irrep_dimension`              | Reference values for validation                 |

### Steps

1. **Parse mCIF** -- extract all fields above using gemmi.cif; use pymatgen for full structure.
2. **Reconstruct parent cell** -- apply child_transform (e.g., 2a,2b,2c for NiO) to map from
   magnetic cell back to parent; then apply transform_Pp_abc to reach the standard parent setting.
   Identify all symmetry-inequivalent magnetic atoms.
3. **Get parent SG operations** -- use `spglib.get_symmetry_from_database(hall_number)` to retrieve
   all (R, t) pairs in the standard setting.
4. **Find little group G_k** -- filter operations satisfying R^T . k = k (mod Z^3).
5. **Build magnetic representation characters** -- for each g = (R, t) in G_k:
   - chi_axial = det(R) * Tr(R)  (axial-vector character)
   - Sum over fixed atoms: atom i contributes exp(-2*pi*i * k.L) where L = R*r_i + t - r_i
   - chi_mag(g) = chi_axial * sum_of_phases
6. **Get little-group irreps** -- call `spgrep.get_spacegroup_irreps_from_primitive_symmetry()`
   which returns irrep matrices for each little-group operation plus a mapping array.
7. **Decompose** -- reduction formula: n_mu = (1/|G_k|) * sum_g chi_mu(g)* chi_mag(g)
8. **Identify active irrep** -- projection operator test using full D(g) matrices.
9. **Label** -- map to Bilbao notation (m + k-label + index +/- suffix).

---

## 3. File Structure

```
magnetic irreps/
|-- plan.md                           # This document
|-- main.py                           # CLI entry point: python main.py <file.mcif>
|-- parse_mcif.py                     # gemmi-based mCIF field parser
|-- little_group.py                   # Parent SG operations + little group extraction
|-- mag_rep.py                        # Magnetic representation character computation
|-- irrep_decompose.py                # spgrep wrapper + reduction formula
|-- irrep_label.py                    # k-point labeling + Bilbao notation mapping
|-- magrep_bilbao_parentdetect.py     # Standalone advanced implementation (Bilbao HTTP + auto parent detection)
|-- 0.222_CuMnAs.mcif                # Test: CuMnAs (k=Gamma, expect mGM5-)
|-- 1.6_NiO.mcif                     # Test: NiO (k=L, expect mL3+)
```

---

## 4. Current Implementation Status

### Environment

The best-fit conda environment is **`findmagsym`** (`/Users/ldyuan/Apps/anaconda3/envs/findmagsym/bin/python`),
which has spglib 2.5.0, spgrep 0.3.5, pymatgen, numpy 1.26.0. **gemmi is not installed anywhere** --
it needs to be added, OR the mCIF parsing can be rewritten to use pymatgen/regex only.

Run with:
```bash
conda activate findmagsym
pip install gemmi          # NEEDED
python main.py 0.222_CuMnAs.mcif
```

### Module-by-module status

#### parse_mcif.py -- PARTIALLY WORKING, needs gemmi install

| Function              | Status  | Issues                                                   |
|-----------------------|---------|----------------------------------------------------------|
| `parse_mcif_fields()` | Draft   | Uses gemmi (not installed). k-vector regex is fragile.    |
| `parse_kvector()`     | Working | Fraction parsing is correct.                             |
| `parse_transform()`   | Working | Uses pymatgen SymmOp -- handles 'a,b,c;-1/4,1/4,0' etc. |
| `get_magnetic_structure()` | Draft | Uses pymatgen Structure.from_file(). Needs verification that magmom is correctly parsed from mCIF format. |

**Key concern**: pymatgen's `Structure.from_file()` for mCIF may not parse `_atom_site_moment.crystalaxis_*`
fields correctly. The CuMnAs file stores moments in a **separate loop** (`_atom_site_moment.label` ...),
not in the `_atom_site` loop. pymatgen may return zero moments. This needs testing.

#### little_group.py -- WORKING (logic complete)

| Function                    | Status  | Issues                                               |
|-----------------------------|---------|------------------------------------------------------|
| `get_hall_number()`         | Working | Cache lookup IT-->Hall via spglib. Correct.          |
| `get_parent_sg_operations()`| Working | Uses `spglib.get_symmetry_from_database()`. Correct. |
| `find_little_group()`       | Working | Uses R^T @ k test. Has dead code from earlier iteration (first loop is `pass`). |

**Key concern**: Returns operations in the **conventional** cell, but spgrep requires **primitive** cell
operations. For SG #225 (Fm-3m, FCC), the conventional cell has 192 operations while the primitive
cell has 48. This mismatch will cause spgrep to fail or give wrong results.

#### mag_rep.py -- PARTIALLY WORKING, has subtle issues

| Function                     | Status | Issues                                                    |
|------------------------------|--------|-----------------------------------------------------------|
| `map_atoms_to_parent_cell()` | Draft  | Transform convention is unclear. See detailed analysis below. |
| `compute_characters()`       | Draft  | Missing the phase factor phi(g) = exp(2*pi*i * k.t). Only has the lattice-translation phase, not the translation-operation phase. |

**Transform convention issue**: The Bilbao convention for child_transform_Pp_abc `(P, p)` means:
- `(a', b', c') = (a, b, c) * P` (basis transform)
- `r_parent = P * r_child + p` (coordinate transform)

The code applies `r_p = M @ r_child + t` which would be correct IF `M, t = P, p`. But `parse_transform()`
uses pymatgen's SymmOp which interprets `'2a,2b,2c;0,0,0'` as `r' = 2*r + 0`, giving M=diag(2,2,2).
This is the **forward** transform (parent-->child in coords). But we need the **inverse** (child-->parent):
`r_parent = M^{-1} @ (r_child - t)`. The current code applies M forward, which is **wrong for NiO**
(it would scale coordinates by 2 instead of 1/2).

**Character formula issue**: The full magnetic character is:
```
chi_mag(g) = sum_i [ det(R) * Tr(R) * delta(R*r_i + t, r_i mod 1) * exp(-2*pi*i * k . L_i) ]
```
The code implements this. However, some references include an additional overall phase
`exp(2*pi*i * k . t)` that is NOT in the current code. For k=0 (CuMnAs) this doesn't matter,
but for k != 0 (NiO) it could matter. Need to verify against Bertaut's original formula.

#### irrep_decompose.py -- PARTIALLY WORKING, has API issues

| Function                    | Status | Issues                                                  |
|-----------------------------|--------|---------------------------------------------------------|
| `get_little_group_irreps()` | Draft  | Calls `spgrep.get_spacegroup_irreps_from_primitive_symmetry()`. Returns `(irreps, mapping_little_group)`. |
| `decompose()`               | Draft  | Reduction formula is correct in principle, but indexing must match. |
| `find_active_irrep()`       | Working| Simple threshold check.                                 |

**Critical issue**: spgrep returns irrep matrices indexed by a `mapping_little_group` array.
`irreps[alpha][i]` is the matrix for operation `rotations[mapping_little_group[i]]`. The current code
ignores `mapping_little_group` entirely -- it assumes irrep matrices are in the same order as the
input rotations. This is **wrong**: only little-group operations have irrep matrices, and their
indices must be mapped back to the input operation list.

The `chi_mag` array in main.py is computed for ALL space group operations, but the irreps are
only for little-group operations. The decomposition sum must use matching indices.

#### irrep_label.py -- MINIMAL STUB

| Function       | Status | Issues                                                        |
|----------------|--------|---------------------------------------------------------------|
| `kpoint_label()`| Stub  | Only 4 k-points hardcoded (GM, L, X, M). No SG dependence.   |
| `irrep_name()` | Stub   | Returns `m{k-label}{idx}` with no parity suffix (+/-).       |

This cannot produce correct Bilbao labels like "mGM5-" because:
1. spgrep's irrep ordering != Bilbao's ordering
2. No parity (inversion character) computation
3. No mapping between spgrep index and Bilbao index

#### magrep_bilbao_parentdetect.py -- BROKEN (syntax error), most complete logic

This 606-line standalone file has the most complete implementation but:
- **Syntax error** at line 334 (multiline string in regex literal)
- Uses `requests` to query Bilbao REPRES web API (fragile, network-dependent)
- Has auto-detection of parent cell transform (brute-force diagonal search)
- Has full D(g) matrix construction and projection operators
- Has operation-ordering mapping between Bilbao and local G_k
- Unit test only uses a trivially-constructed test CIF, not the real mCIF files

---

## 5. Critical Issues to Resolve (Priority Order)

### P0: Environment setup
- Install gemmi in `findmagsym` env: `conda activate findmagsym && pip install gemmi`
- OR: rewrite parse_mcif.py to avoid gemmi (use regex + pymatgen only)

### P1: Primitive vs conventional cell mismatch
**This is the single biggest correctness issue.**

spgrep requires **primitive-cell** operations. `spglib.get_symmetry_from_database(hall_number)`
returns **conventional-cell** operations. For cubic SGs like #225 (Fm-3m), conventional has 192 ops
but primitive has only 48.

**Fix**: Use `spglib.get_symmetry_from_database()` and then reduce to primitive. OR construct
a primitive cell and call `spglib.get_symmetry()` on it. spglib's `standardize_cell()` or
`find_primitive()` can help. Alternatively, use spgrep's `get_spacegroup_irreps()` which takes
a Hall number directly and handles the primitive reduction internally.

### P2: Transform convention (child --> parent)
The code currently applies `M @ r + t` but this is the forward transform (parent --> child basis).
For going FROM child TO parent coordinates, we need `M^{-1} @ (r - t)` or equivalently
`M^{-1} @ r - M^{-1} @ t`.

For CuMnAs: child_transform is `a,b,c;0,0,0` (identity), so this bug is hidden.
For NiO: child_transform is `2a,2b,2c;0,0,0`, so M=diag(2,2,2). Applying M forward would double
coordinates, but we need to halve them. **This will produce wrong atom positions for NiO.**

### P3: spgrep mapping_little_group usage
The return value of `spgrep.get_spacegroup_irreps_from_primitive_symmetry()` is:
```python
(irreps, mapping_little_group)
```
where `mapping_little_group` is an array of shape `(little_group_order,)`. The i-th little-group
operation corresponds to `rotations[mapping_little_group[i]]`.

The code must:
1. Extract the little-group operations using this mapping
2. Compute chi_mag ONLY for these operations, in this exact order
3. Then apply the reduction formula

### P4: Character formula completeness
For k != 0, the character of the magnetic representation at operation g = {R|t} is:

    chi_mag(g) = det(R) * Tr(R) * sum_{i: R*r_i+t = r_i + L_i} exp(-2*pi*i * k . L_i)

Note: there is NO separate exp(2*pi*i * k . t) prefactor in the standard Bertaut formula when
using the "permutation + axial" decomposition. The phase comes entirely from the lattice
translation L_i = R*r_i + t - r_i. The current code in mag_rep.py is correct on this point.
However, the magrep_bilbao_parentdetect.py version includes an extra `phi = exp(2j*pi*k.t)`,
which is used in the full-matrix D(g) construction (not just characters). Need to verify which
convention matches spgrep's convention.

### P5: Irrep labeling (Bilbao notation)
spgrep's irrep indices have no relation to Bilbao's label scheme (mGM5-, mL3+, etc.).
Options:
- **Option A**: Query Bilbao REPRES web API (as in magrep_bilbao_parentdetect.py). Fragile.
- **Option B**: Use spgrep's character table + known rules (dimension, inversion character,
  k-point symmetry) to match against a precomputed table of Bilbao labels.
- **Option C**: Use the `irrep` Python package (if available) which directly outputs Bilbao labels.
- **Option D**: For validation, compare n_mu decomposition pattern against expected and report
  match/mismatch without trying to produce the exact Bilbao label.

---

## 6. Strategy to Make It Work

### Phase 1: Get the math right on CuMnAs (k=0, simplest case)

CuMnAs has k=0 (Gamma point) and identity child transform. This eliminates P2 and simplifies P4.

1. Install gemmi (or bypass it for field parsing)
2. Fix P1: ensure we pass primitive-cell operations to spgrep
   - For SG #129 (P4/nmm), primitive = conventional (tetragonal), so 16 ops
   - Call `spgrep.get_spacegroup_irreps_from_primitive_symmetry(rot, trans, k=[0,0,0])`
3. Fix P3: use `mapping_little_group` to select the right operations for chi_mag
4. Run and verify n_mu decomposition gives exactly one irrep with n_mu=1
5. Compare dimension with expected (dim=2)

### Phase 2: Handle NiO (k != 0, supercell)

1. Fix P2: invert the child transform correctly
   - For `2a,2b,2c;0,0,0`: M=diag(2,2,2), so r_parent = M^{-1} @ r_child = r_child / 2
   - NiO child cell has Ni at (0,0,0); parent cell Ni at (0,0,0). This happens to be the same,
     but O at (0.75,0.5,0) in child --> (0.375,0.25,0) in parent. Check against expected Fm-3m
     Wyckoff positions.
2. For SG #225 (Fm-3m), conventional cell has 192 ops, primitive has 48.
   Must use primitive ops. Use spglib to find primitive operations:
   ```python
   dataset = spglib.get_symmetry_from_database(hall_number)  # 192 ops (conventional)
   # Need to find primitive operations instead
   ```
   Alternative: construct a primitive FCC cell and call `spglib.get_symmetry()`.
3. Verify k=[1/2,1/2,1/2] little group has the expected order
4. Run decomposition and check for mL3+ (dim=8, small dim=2)

### Phase 3: Irrep labeling

Start with Option D (pattern matching without exact labels). If needed, try Option B or C.

### Phase 4: Generalization

- Handle more mCIF files from MAGNDATA
- Handle non-standard parent settings (transform_Pp_abc != identity)
- Handle multi-orbit structures
- Clean up magrep_bilbao_parentdetect.py syntax error and consider merging useful logic

---

## 7. Key spgrep API Reference

```python
import spgrep

# Main function we use:
irreps, mapping = spgrep.get_spacegroup_irreps_from_primitive_symmetry(
    rotations,      # (N_ops, 3, 3) int array -- MUST be primitive cell
    translations,   # (N_ops, 3) float array
    kpoint,         # (3,) float array
)
# Returns:
#   irreps: list of arrays, each shape (little_group_order, dim, dim)
#   mapping: array of shape (little_group_order,)
#     irreps[alpha][i] is the matrix for operation (rotations[mapping[i]], translations[mapping[i]])

# Alternative that takes Hall number directly:
irreps, mapping = spgrep.get_spacegroup_irreps(
    hall_number,    # int (1-530)
    kpoint,         # (3,) float
)
# This internally handles primitive-cell reduction
```

`get_spacegroup_irreps(hall_number, kpoint)` is simpler and handles the primitive/conventional
issue internally. This may be the better choice.

---

## 8. Dependencies

| Package  | Required | Available in findmagsym env | Notes                          |
|----------|----------|-----------------------------|--------------------------------|
| numpy    | yes      | 1.26.0                      |                                |
| spglib   | yes      | 2.5.0                       |                                |
| spgrep   | yes      | 0.3.5                       |                                |
| pymatgen | yes      | yes                         | For Structure I/O, SymmOp      |
| gemmi    | yes      | **NO -- needs install**     | For mCIF field parsing         |

```bash
conda activate findmagsym
pip install gemmi
```

---

## 9. Expected CLI Output

```
python main.py 0.222_CuMnAs.mcif

=== Magnetic Irrep Analysis ===
File: 0.222_CuMnAs.mcif
Parent SG: P4/nmm (IT #129)
Propagation vector k = [0, 0, 0]  -->  Gamma (GM) point

Little group G_k: |G_k| = 16 operations (full SG at Gamma)

Magnetic atoms (parent cell): Mn1 at (0.000, 0.500, 0.336)

Magnetic representation decomposition:
  Gamma_mag = 1 x [Irrep #5, dim=2, label: mGM5-]

Active irrep: mGM5-
  Dimension: 2  (small dim: 2)

Validation:
  Stored _irrep_id = mGM5-  matches computed result
```

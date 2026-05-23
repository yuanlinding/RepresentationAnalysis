# Theory: Bertaut Representational Analysis for Magnetic Structures

## 1. Physical Setting

A crystal undergoes a **paramagnetic-to-magnetically-ordered phase transition** on cooling
below the Néel temperature T_N. Above T_N the crystal has a high-symmetry space group G
(the **parent** space group). Below T_N magnetic order sets in and the symmetry is lowered.

Landau theory states that a continuous transition is driven by a single **order parameter**
that transforms as an **irreducible representation** (irrep) of G. Finding that irrep from
the observed magnetic structure is what `magirrep` does.

The result:

- classifies the phase transition and its symmetry-breaking pattern
- links to the magnetic space group found by `findmagsym`
- constrains symmetry-allowed physical effects (magnetoelectric coupling, phonon softening, …)

## 2. Space Groups and Their Operations

A space group G is an infinite group of Euclidean isometries. Each element is a pair
`{R | t}` where R is a point-group rotation (3×3 integer matrix in fractional coordinates)
and **t** is a fractional-coordinate translation vector. The action on a position **r** is:

```
{R | t} r  =  R r + t
```

Multiplication:

```
{R₂ | t₂} {R₁ | t₁}  =  {R₂R₁ | R₂t₁ + t₂}
```

The pure lattice translations `{E | n}` (integer vectors **n**) form the translation subgroup
T. The quotient G/T is the finite crystallographic point group; a conventional unit cell
contributes N operations `(R_i, t_i)`.

In code: `spglib.get_symmetry_from_database(hall_number)` returns the arrays `rotations[i]`
and `translations[i]` for the conventional cell. SG #129 (P4/nmm) has 16; SG #225 (Fm-3m)
has 192.

## 3. Bloch Functions and the Propagation Vector

Magnetic ordering is periodic with period set by the **propagation vector k** (fractional
reciprocal coordinates of the parent cell). Neutron diffraction shows magnetic Bragg peaks
at **Q = G + k**.

Bloch basis functions are:

```
φ_{k,ν,α}(r)  ∝  e^{ik·r}  ×  (spin on atom ν in direction α)
```

A pure lattice translation `{E | n}` multiplies such a function by `e^{ik·n}`.

Special cases:

- **k = (0,0,0)** (Γ): magnetic cell equals the nuclear cell. Example: CuMnAs.
- **k = (½,½,½)** (L): magnetic cell doubles in each direction. Example: NiO (2×2×2 supercell).

## 4. The Little Group G_k

The action of a rotation R on a reciprocal-space vector **k** is:

```
R ∘ k  =  R^T k      (mod Z³)
```

(Transpose because real-space rotations act contravariantly on reciprocal-space vectors.)

The **little group** of **k** is the subgroup of G that maps **k** to an equivalent vector:

```
G_k  =  { {R | t} ∈ G  :  R^T k ≡ k  (mod Z³) }
```

Only elements of G_k can mix Bloch functions at **k** with each other; remaining elements
send them to a different **star arm** of **k**.

In code (`little_group.py: find_little_group`):

```python
k_prime = R.T @ kpoint
diff = k_prime - kpoint
if np.allclose(diff - np.round(diff), 0, atol=tol):   # diff ∈ Z³ → element is in G_k
```

The little group at Γ is always the full point group. At a general **k** it is a proper
subgroup.

## 5. Irreducible Representations of G_k

For each irrep α of G_k there is a set of unitary matrices `D^α(g)`, one per element
g ∈ G_k, satisfying the homomorphism property:

```
D^α(g₂ g₁)  =  D^α(g₂) D^α(g₁)
```

The **character** is the trace: `χ^α(g) = Tr D^α(g)`. Characters depend only on the
conjugacy class. The **Great Orthogonality Theorem** for characters:

```
(1/|G_k|)  Σ_{g ∈ G_k}  χ^α*(g) χ^β(g)  =  δ_{αβ}
```

This identity is the mathematical engine behind Bertaut's decomposition.

In code: `spgrep.get_spacegroup_irreps(lattice, positions, numbers, kpoint)` returns:

- `irreps[α][i]` — matrix for the i-th little-group operation
- `rotations`, `translations` — all N conventional-cell operations
- `mapping_little_group[i]` — index into `rotations`/`translations` of the i-th little-group op

spgrep works with the **primitive** cell. The reference crystal in `build_reference_crystal`
places atoms at a general Wyckoff position (full-multiplicity orbit) to guarantee exactly the
target space group is detected, not an accidental supergroup.

## 6. The Magnetic Representation

### Basis

For N_mag magnetic atoms in the primitive cell, define 3N_mag basis functions — one per
(atom, spin direction) pair. These span the space on which the magnetic representation acts.

### Action of a symmetry operation

Let `g = {R | t} ∈ G_k`. It acts in two parts:

**Permutation of atoms**: Atom at **r_ν** maps to R **r_ν** + **t**. Atom ν is **fixed**
(maps to itself modulo a lattice vector) iff:

```
R r_ν + t - r_ν  ∈  Z³
```

When this holds the lattice vector is **L_ν** = round(R **r_ν** + **t** − **r_ν**).

**Transformation of spin direction**: Spins are **axial vectors** (pseudovectors). Under R:

```
m  →  det(R) · R · m
```

`det(R) = +1` for proper rotations, `−1` for improper ones (inversions, mirrors,
rotoreflections). This sign distinguishes the magnetic representation from an ordinary
vector representation.

### The character formula

```
χ_mag({R|t})  =  [ det(R) · Tr(R) ]  ×  [ Σ_{ν: fixed} exp(−2πi k · L_ν) ]
```

**Factor 1** — `det(R)·Tr(R)`: character of the axial-vector representation of R.

| Operation           | det(R) | Tr(R) | product |
| ------------------- | ------ | ----- | ------- |
| Identity E          | +1     | +3    | +3      |
| Inversion I         | −1    | −3   | +3      |
| C₂ (proper 2-fold) | +1     | −1   | −1     |
| σ (mirror)         | −1    | +1    | −1     |

**Factor 2** — phase sum over fixed atoms. At k = Γ all phases are 1, so this factor equals
the number of fixed atoms. At a general **k** each fixed atom contributes a complex phase
that depends on whether the atom shifted by a pure lattice translation and, if so, which one.

In code (`mag_rep.py: compute_characters`):

```python
chi_axial = np.linalg.det(R) * np.trace(R)
# for each fixed atom ν:
L = np.round(R @ r + t - r)
atom_phase = np.exp(-2j * np.pi * np.dot(kpoint, L))
trace_perm += atom_phase
chi_mag[i] = chi_axial * trace_perm
```

## 7. The Displacive/Mechanical Representation

The **mechanical** (displacement) representation Γ_mech describes all phonon modes at **k**
for ALL atoms in the primitive cell (magnetic and non-magnetic alike).

### Character formula

Atomic displacements are **polar vectors** (not axial), so the `det(R)` sign factor is absent:

```
χ_disp({R|t})  =  Tr(R)  ×  [ Σ_{ν: fixed} exp(−2πi k · L_ν) ]
```

In code (`mag_rep.py: compute_displacive_characters`):

```python
chi_disp[i] = np.trace(R) * trace_perm   # no det(R)
```

### Building-block decompositions

All representations can be expressed through three primitive building blocks:

| Name     | χ(g)                             | Role                    |
| -------- | --------------------------------- | ----------------------- |
| Γ_perm  | Σ_{fixed atoms} exp(−2πi k·L) | permutation of atoms    |
| Γ_polar | Tr(R)                             | single polar-vector DOF |
| Γ_axial | det(R)·Tr(R)                     | single axial-vector DOF |

The full reps are then outer products:

```
Γ_mag   = Γ_perm[mag] ⊗ Γ_axial    (magnetic atoms, axial moments)
Γ_mech  = Γ_perm[all] ⊗ Γ_polar    (all atoms, polar displacements)
```

Each is decomposed independently using Bertaut's reduction formula.
Section (8) of the output shows all six decompositions together, with a combined
summary table listing n_μ(mag) and n_μ(mech) side by side.

### Per-Wyckoff breakdown

The total Γ_mech is also decomposed per Wyckoff orbit (groups of atoms of the same
element at the same Wyckoff letter). This identifies which atoms contribute which phonon
branches and is printed in section (8) after the Γ_mech total line.

### Dimension check

```
Σ_α  n_α · d_α  =  3 · N_all_prim
```

where N_all_prim is the number of atoms in the primitive cell. This is printed as a
validation at the end of the report.

## 8. Bertaut's Decomposition Formula

Any representation Γ decomposes into irreps of G_k:

```
Γ  =  Σ_α  n_α · Γ^α
```

By the Great Orthogonality Theorem the multiplicity is:

```
n_α  =  (1/|G_k|)  Σ_{g ∈ G_k}  χ^α*(g) · χ(g)
```

For a physically valid magnetic structure all `n_α` must be **non-negative integers**.

For a continuous (Landau) transition exactly **one** irrep is active (`n_α = 1`; the physical
magnetic structure lives entirely in the basis functions of that one irrep).

In code (`irrep_decompose.py: decompose`):

```python
chi_lg = chi_rep[mapping_little_group]          # restrict to G_k operations
for irrep in irreps:
    chi_irrep = np.array([np.trace(mat) for mat in irrep])
    n = np.sum(chi_lg * np.conj(chi_irrep)) / len(mapping_little_group)
```

`mapping_little_group` is essential: `chi_rep` is computed for all N conventional
operations but the sum runs only over the `|G_k|` little-group operations.

The same `decompose()` function handles Γ_mag, Γ_mech, Γ_perm, Γ_axial, and Γ_polar —
only the input character array differs.

## 9. Coordinate Conventions and the mCIF File

### Two-step transform (child cell → standard parent cell)

Bilbao MAGNDATA mCIF files contain two transforms following the ITA/Bilbao convention
`r_new = P · r_old + p`:

**`child_transform_pp_abc`** = (P_child, p_child): child (magnetic) cell → parent cell

```
r_parent  =  P_child · r_child  +  p_child
```

**`transform_pp_abc`** = (P_parent, p_parent): parent cell → standard ITA setting

```
r_std  =  P_parent · r_parent  +  p_parent
```

Full chain from child to standard:

```
r_std  =  P_parent · (P_child · r_child + p_child) + p_parent
```

For NiO: P_child = diag(2,2,2) (2×2×2 supercell) — this halves the coordinates when mapping
32 Ni sites in the supercell back to 4 Ni sites in the conventional cell.

### Moment transform (axial vector)

Magnetic moments in crystal-axis units transform as:

```
m_new  =  sign(det(P)) · P · m_old
```

applied once for P_child and once for P_parent (`mag_rep.py: map_atoms_to_parent_cell`).

The `sign(det(P))` factor (not the full `det(P)`) is the correct handedness correction for
axial vectors. For a pure scaling like P = diag(2,2,2) (NiO 2×2×2 supercell), det(P) = 8,
but moments should not be amplified by 8. Using sign(det) = +1 leaves the magnitude unchanged
while still reversing sign under improper transforms (det = −1). For rotations and reflections
(det = ±1) both formulas are equivalent.

### Origin choice

Many centrosymmetric space groups have two standard settings:

- **OC1**: origin at a high-site-symmetry point
- **OC2**: origin at the inversion centre (Bilbao standard)

Using OC1 operations with OC2 atom positions causes a coordinate system mismatch that
produces non-integer `n_α`. The code always selects OC2 Hall numbers
(`little_group.py: get_hall_number`):

```python
oc2 = [h for h, c in candidates if c == '2']
_HALL_NUMBER_CACHE[it_no] = oc2[0] if oc2 else candidates[0][0]
```

## 10. Primitive vs. Conventional Cell

spgrep computes irreps for the **primitive** cell. The character formula must also use
atoms from **one primitive cell** for dimensional consistency:

```
Σ_α  n_α · d_α  =  3 · N_mag_prim
```

For Fm-3m (SG #225, F-centred cubic): the conventional cell has 4 formula units;
the primitive cell has 1. `_select_primitive_atoms` in `pipeline.py` reduces the atom list
to one representative per primitive cell by converting to primitive fractional coordinates
and keeping distinct sites.

## 11. Self-Consistency Checks

| Check                                         | Meaning                                                  |
| --------------------------------------------- | -------------------------------------------------------- |
| All `n_α ∈ Z≥0`                          | Magnetic structure is physically consistent              |
| `Σ_α n_α · d_α = 3 N_mag_prim`         | Total spin degrees of freedom conserved                  |
| `Σ_α n_α · d_α = 3 N_all_prim`         | Total phonon degrees of freedom conserved                |
| Exactly one `n_α ≠ 0`                     | Continuous (Landau) transition driven by a single irrep  |
| Active irrep label matches mCIF `_irrep_id` | Correct identification vs. Bilbao database               |
| `‖M − M_rec‖/‖M‖ ≈ 0`                 | Actual moments lie entirely in the active irrep subspace |

Validation cases:

**CuMnAs** (SG #129, k = Γ, 2 Mn per primitive cell → 6 spin DOF):

```
Γ_mag  = 1·mGM1- ⊕ 1·mGM2+ ⊕ 1·mGM5+ ⊕ 1·mGM5-   (1+1+2+2 = 6) ✓
Active irrep: mGM5-  (dim=2, η=1.000) ✓

Γ_mech = 2·mGM1+ ⊕ 3·mGM2- ⊕ 3·mGM5+ ⊕ 3·mGM5- ⊕ 1·mGM3+   (Σ n·d = 18 = 3×6) ✓
```

**NiO** (SG #225, k = L = (½,½,½), 1 Ni per primitive cell → 3 spin DOF):

```
Γ_mag  = 1·mL1+ ⊕ 1·mL3+   (1+2 = 3) ✓
Active irrep: mL3+  (dim=2, deg=4, η=1.000) ✓

Γ_mech = 3·mL1+ ⊕ 1·mL1- ⊕ 1·mL2+ ⊕ 3·mL2- ⊕ 4·mL3+ ⊕ 4·mL3-   (Σ n·d = 24 = 3×8) ✓
```

## 12. Irrep Labelling

The Bilbao labels such as "mGM5−" and "mL3+" encode:

- **Point label**: GM (Γ), L, X, M, … — the high-symmetry k-point
- **Number**: sequential index at that k-point
- **Parity suffix ±**: character under inversion (if present in G_k). "+" irreps are even, "−" are odd.

### Parity suffix (±)

Computed in `irrep_decompose.compute_parity_suffixes()`: if inversion {−1|0} is in G_k,
its little-group index is located, and the character `χ^α({−1|0})` determines the suffix —
positive character → "+", negative → "−". Operations without inversion get no suffix.

### k-point label

`irrep_label.kpoint_label()` uses **seekpath** (Setyawan–Curtarolo 2010) to look up the
Brillouin-zone path for the given space group and map the propagation vector to a label.
The k-vector is converted from conventional to primitive reciprocal coordinates before lookup.
Fallbacks handle zone-boundary points and SGs not in the seekpath database.

### Irrep numbering

spgrep's 0-based irrep indices are not the same as Bilbao's 1-based sequential numbers.
`irrep_label.bilbao_ordered_labels()` sorts irreps within each parity class by dimension
(ascending) and assigns consecutive integers 1, 2, 3, …. This matches the Bilbao ordering
for the common little groups tested (D₄h, D₃d, D₂h). The optional `bilbao_match.py` module
can cross-check by querying the Bilbao REPRES web API (currently blocked by Cloudflare CAPTCHA).

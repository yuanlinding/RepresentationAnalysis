"""Orchestration pipeline: parse -> transform -> irreps -> decompose -> label."""

import sys

import numpy as np
import spglib

from magirrep import parse_mcif, mag_rep, irrep_decompose, irrep_label
from magirrep.little_group import build_reference_crystal


def _deduplicate_positions(positions: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Keep only unique rows in *positions* (wrapped to [0,1)) within *tol*."""
    unique = []
    for r in positions:
        if not any(np.allclose(r, u, atol=tol) for u in unique):
            unique.append(r)
    return np.array(unique) if unique else np.zeros((0, 3))


def _select_primitive_atoms(conv_positions: np.ndarray, it_number: int,
                             tol: float = 1e-4) -> np.ndarray:
    """
    From a list of positions in conventional-cell fractional coordinates,
    return only those belonging to ONE primitive cell.

    For primitive Bravais lattices (P-type) this is a no-op. For centered
    lattices (F, I, A, B, C, R) it reduces e.g. 4 Ni atoms (Fm-3m
    conventional) to 1 Ni (FCC primitive cell), which is required because
    spgrep's returned irreps index over primitive-cell operations.

    The selection is done by converting positions to primitive fractional
    coordinates and keeping one representative per unique primitive position.
    Conventional-cell fractional coordinates are returned.
    """
    # Build a reference crystal to determine the primitive cell lattice
    lattice_conv, _, _ = build_reference_crystal(it_number)
    cell_ref = (lattice_conv, np.array([[0.0, 0.0, 0.0]]), np.array([1]))
    prim_cell = spglib.find_primitive(cell_ref, symprec=1e-3)

    if prim_cell is None:
        # Already primitive — nothing to do
        return conv_positions

    prim_lattice = prim_cell[0]

    # Transformation: r_prim = (P^T)^{-1} @ r_conv  where P = A_prim @ A_conv^{-1}
    P = prim_lattice @ np.linalg.inv(lattice_conv)
    P_inv_T = np.linalg.inv(P.T)

    seen_prim: list = []
    result: list = []
    for r_conv in conv_positions:
        r_prim = (P_inv_T @ r_conv) % 1.0
        if not any(np.allclose(r_prim, s, atol=tol) for s in seen_prim):
            seen_prim.append(r_prim)
            result.append(r_conv)

    return np.array(result) if result else np.zeros((0, 3))


def _print_summary(path, mcif_fields, kpoint, g_k_len, active_irreps, irrep_data):
    """Print analysis results to stdout."""
    print("=== Magnetic Irrep Analysis ===")
    print(f"File: {path}")
    print(f"Parent SG IT #: {mcif_fields.get('it_number')}")
    print(f"Propagation vector k = {mcif_fields.get('kvector_str')}  ->  {irrep_label.kpoint_label(kpoint)} point\n")

    print(f"Little group |G_k| = {g_k_len} operations\n")

    print("Magnetic representation decomposition:")
    for idx, n in active_irreps:
        irrep_dim = irrep_data[idx][0].shape[0] if len(irrep_data[idx]) > 0 else 0
        label = irrep_label.irrep_name(kpoint, mcif_fields.get('it_number'), idx, irrep_dim)
        print(f"  Gamma_mag += {n} x [Irrep #{idx+1}, small dim={irrep_dim}, label: {label}]")

    print("\nValidation:")
    expected_id = mcif_fields.get('expected_irrep_id')
    print(f"  Stored _irrep_id = {expected_id}")


def run_analysis(mcif_path: str):
    """Run the full magnetic irrep analysis pipeline on *mcif_path*."""

    # 1. Parse fields
    fields = parse_mcif.parse_mcif_fields(mcif_path)
    kpoint = parse_mcif.parse_kvector(fields['kvector_str'])

    child_M, child_t = parse_mcif.parse_transform(fields['child_transform_str'])
    if 'parent_transform_str' in fields:
        parent_M, parent_t = parse_mcif.parse_transform(fields['parent_transform_str'])
    else:
        parent_M, parent_t = np.eye(3), np.zeros(3)

    it_number = fields['it_number']

    # 2. Get magnetic structure and extract NONZERO-moment atoms only
    #    (pymatgen assigns magmom=0 to all atoms; non-magnetic species must be excluded)
    structure = parse_mcif.get_magnetic_structure(mcif_path)
    mag_positions = []
    magmoms = []
    for site in structure:
        raw_m = site.properties.get("magmom")
        if raw_m is None:
            raw_m = getattr(site, "magmom", None)

        if raw_m is not None:
            if hasattr(raw_m, 'global_moment'):
                m_vec = raw_m.global_moment
            elif hasattr(raw_m, 'moment'):
                m_vec = raw_m.moment
            else:
                m_vec = raw_m

            m_vec = np.array(m_vec)
            if m_vec.shape == () or m_vec.shape == (1,):
                m_vec = np.array([0.0, 0.0, float(m_vec)])

            # Bug 4 fix: skip atoms with zero (or negligible) magnetic moment
            if np.linalg.norm(m_vec) < 0.01:
                continue

            mag_positions.append(site.frac_coords)
            magmoms.append(m_vec)

    mag_positions = np.array(mag_positions)
    magmoms = np.array(magmoms)

    if len(mag_positions) == 0:
        print("No nonzero magnetic moments found in the structure.")
        sys.exit(1)

    # 3. Transform magnetic positions to parent cell
    parent_positions, parent_magmoms = mag_rep.map_atoms_to_parent_cell(
        mag_positions, magmoms, child_M, child_t, parent_M, parent_t
    )

    # Bug 5/6 fix: (a) deduplicate — supercell gives repeated parent-cell sites
    # (32 Ni in NiO 2×2×2 → 4 unique conventional-cell positions);
    # (b) reduce to primitive cell — spgrep irreps index primitive-cell ops,
    # so chi_mag must use primitive-cell atoms (4 Ni → 1 for Fm-3m FCC).
    parent_positions = _deduplicate_positions(parent_positions)
    parent_positions = _select_primitive_atoms(parent_positions, it_number)

    if len(parent_positions) == 0:
        print("No magnetic positions found in primitive cell after transformation.")
        sys.exit(1)

    # 4. Get irreps and the conventional-cell ops spgrep used internally
    #    (Bug 1 fix: uses spgrep.get_spacegroup_irreps, which handles primitive conversion;
    #     Bug 3 fix: uses preferred origin choice 2 via build_reference_crystal)
    irreps, rotations, translations, mapping_little_group = irrep_decompose.get_little_group_irreps(
        it_number, kpoint
    )

    if len(irreps) == 0:
        print("Spgrep returned no irreps.")
        sys.exit(1)

    # 5. Compute magnetic representation characters using the SAME ops spgrep returned
    chi_mag = mag_rep.compute_characters(rotations, translations, kpoint, parent_positions)

    # 6. Decompose (Bug 2/P3 fix: filter chi_mag via mapping_little_group)
    n_mu_array = irrep_decompose.decompose(irreps, chi_mag, mapping_little_group)
    active_irreps = irrep_decompose.find_active_irrep(n_mu_array)

    # 7. Print summary
    _print_summary(mcif_path, fields, kpoint, len(mapping_little_group), active_irreps, irreps)

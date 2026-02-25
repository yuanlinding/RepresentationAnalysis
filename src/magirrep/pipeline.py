"""Orchestration pipeline: parse -> transform -> irreps -> decompose -> label."""

import sys

import numpy as np

from magirrep import parse_mcif, little_group, mag_rep, irrep_decompose, irrep_label


def _print_summary(path, mcif_fields, kpoint, g_k_len, active_irreps, irrep_data):
    """
    irrep_data: list of irreps returned by spgrep
    """
    print("=== Magnetic Irrep Analysis ===")
    print(f"File: {path}")
    print(f"Parent SG IT #: {mcif_fields.get('it_number')}")
    print(f"Propagation vector k = {mcif_fields.get('kvector_str')}  ->  {irrep_label.kpoint_label(kpoint)} point\n")

    print(f"Little group |G_k| = {g_k_len} operations\n")

    print("Magnetic representation decomposition (full):")
    for idx, n in active_irreps:
        irrep_dim = len(irrep_data[idx][0]) if len(irrep_data[idx]) > 0 else 0
        label = irrep_label.irrep_name(kpoint, mcif_fields.get('it_number'), idx, irrep_dim)
        print(f"  Gamma_mag += {n} x [Irrep #{idx+1}, dim={irrep_dim}, label: {label}]")

    print("\nValidation:")
    expected_id = mcif_fields.get('expected_irrep_id')
    print(f"  Stored _irrep_id = {expected_id}")


def run_analysis(mcif_path: str):
    """Run the full magnetic irrep analysis pipeline on *mcif_path*."""

    # 1. Parse fields
    fields = parse_mcif.parse_mcif_fields(mcif_path)
    kpoint = parse_mcif.parse_kvector(fields['kvector_str'])

    # Let's get parent and child transforms
    child_M, child_t = parse_mcif.parse_transform(fields['child_transform_str'])
    if 'parent_transform_str' in fields:
        parent_M, parent_t = parse_mcif.parse_transform(fields['parent_transform_str'])
    else:
        # Default to identity
        parent_M, parent_t = np.eye(3), np.zeros(3)

    it_number = fields['it_number']

    # 2. Get magnetic structure
    structure = parse_mcif.get_magnetic_structure(mcif_path)
    # Extract magnetic atoms
    mag_positions = []
    magmoms = []
    for site in structure:
        raw_m = None
        if "magmom" in site.properties:
            raw_m = site.properties["magmom"]
        elif hasattr(site, "magmom") and site.magmom is not None:
            raw_m = site.magmom

        if raw_m is not None:
            # Check if it's a pymatgen Magmom object
            if hasattr(raw_m, 'global_moment'):
                m_vec = raw_m.global_moment
            elif hasattr(raw_m, 'moment'):
                m_vec = raw_m.moment
            else:
                m_vec = raw_m

            # Ensure it is a 3-vector
            m_vec = np.array(m_vec)
            if m_vec.shape == () or m_vec.shape == (1,):
                m_vec = np.array([0.0, 0.0, float(m_vec)])

            mag_positions.append(site.frac_coords)
            magmoms.append(m_vec)
    mag_positions = np.array(mag_positions)
    magmoms = np.array(magmoms)

    if len(mag_positions) == 0:
        print("No magnetic moments found in the structure.")
        sys.exit(1)

    # 3. Transform magnetic positions to parent cell
    parent_positions, parent_magmoms = mag_rep.map_atoms_to_parent_cell(
        mag_positions, magmoms, child_M, child_t, parent_M, parent_t
    )

    # 4. Little group symmetry
    rotations, translations = little_group.get_parent_sg_operations(it_number)

    irreps_data = irrep_decompose.get_little_group_irreps(rotations, translations, kpoint)
    irreps = irreps_data[0]  # list of irreps

    if len(irreps) == 0:
        print("Spgrep didn't return any irreps.")
        sys.exit(1)

    num_ops = len(irreps[0])
    if num_ops != len(rotations):
        print(f"Warning: spgrep returned {num_ops} operations, but spglib has {len(rotations)} operations.")

    chi_mag = mag_rep.compute_characters(rotations, translations, kpoint, parent_positions)

    # 5. Decompose
    n_mu_array = irrep_decompose.decompose(irreps, chi_mag)
    active_irreps = irrep_decompose.find_active_irrep(n_mu_array)

    # 6. Print Summary
    _print_summary(mcif_path, fields, kpoint, num_ops, active_irreps, irreps)

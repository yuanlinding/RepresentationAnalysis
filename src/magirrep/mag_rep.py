import numpy as np

def map_atoms_to_parent_cell(positions, magmoms, child_transform_M, child_transform_t, parent_transform_M, parent_transform_t):
    """
    Transforms magnetic-cell fractional coordinates and magnetic moments to the 
    standard parent setting fractional coordinates.
    
    positions: Nx3 array
    magmoms: Nx3 array
    child_transform_M: 3x3 array (parent to child basis transform)
    child_transform_t: 3x1 array (parent to child origin shift)
    parent_transform_M: 3x3 array (standard parent to parent basis transform)
    """
    parent_positions = []
    parent_magmoms = []
    wrap_offsets = []

    # Usually: r_child = P_child^-1 r_parent - p_child
    # r_parent = P_child (r_child + p_child)
    # P_child is child_transform_M.
    # Wait, pymatgen's SymmOp parsed from '2a,2b,2c;0,0,0' gives the forward matrix M where a'=Ma.
    # The transformation of fractional coordinates r' = M^-T r ? No.
    # If basis vectors are row vectors A: A_child = M @ A_parent
    # Then fractional coordinates x column vector: A_parent.T @ x_parent = A_child.T @ x_child
    # A_parent.T @ x_parent = (M @ A_parent).T @ x_child = A_parent.T @ M.T @ x_child
    # So x_parent = M.T @ x_child.
    # Wait, the transformations in CIF child_transform_Pp_abc are usually P, p where:
    # A_child = P @ A_parent
    # The Bilbao convention uses (P,p) as:
    # r_parent = P @ r_child + p
    # So we just use M @ r_child + t.
    # We apply this first for the child transform, then for the parent transform.

    for r, m in zip(positions, magmoms):
        # 1. From child supercell to parent cell
        r_p = child_transform_M @ r + child_transform_t

        # 2. From parent cell to standard parent setting (raw, before wrapping)
        r_std_raw = parent_transform_M @ r_p + parent_transform_t

        # Wrapping to [0, 1); track integer offset for canonical-representative selection
        r_std = r_std_raw % 1.0
        L_wrap = np.round(r_std_raw - r_std).astype(int)

        # The magnetic moment vector transforms as an axial vector.
        # Only the sign of det(P) matters (handedness correction); using the full det
        # magnifies moments by scale^3 for pure supercell scalings (e.g. det(2I)=8).
        P1 = child_transform_M
        m_p = np.sign(np.linalg.det(P1)) * P1 @ m

        P2 = parent_transform_M
        m_std = np.sign(np.linalg.det(P2)) * P2 @ m_p

        parent_positions.append(r_std)
        parent_magmoms.append(m_std)
        wrap_offsets.append(L_wrap)

    return np.array(parent_positions), np.array(parent_magmoms), np.array(wrap_offsets)

def build_mag_rep_matrices(rotations, translations, kpoint, parent_positions,
                            mapping_little_group, tol=1e-4):
    """
    Build the full 3N×3N D(g) matrices of the magnetic representation for each
    little-group operation.

    For operation g = {R | t}, atom src maps to atom dst with lattice shift L:
        R @ r_src + t ≈ r_dst + L   (r_dst in parent_positions, L in Z³)
    Block (dst, src):
        D(g)[3*dst:3*dst+3, 3*src:3*src+3] = det(R) * R * exp(-2πi k·L)

    Tr(D(g)) == chi_mag(g) for each little-group op (self-consistency check).

    Parameters
    ----------
    rotations : np.ndarray, shape (N_ops, 3, 3)
    translations : np.ndarray, shape (N_ops, 3)
    kpoint : array-like, shape (3,)
    parent_positions : np.ndarray, shape (N_prim, 3)
        Primitive-cell magnetic atom positions in fractional coordinates.
    mapping_little_group : array-like
        Indices into rotations/translations for the little-group ops.

    Returns
    -------
    D_matrices : list[np.ndarray]
        One 3N×3N complex matrix per little-group operation, in the same order
        as mapping_little_group.
    """
    N = len(parent_positions)
    kpoint = np.asarray(kpoint, dtype=float)
    D_matrices = []

    for idx in mapping_little_group:
        R = rotations[idx].astype(float)
        t = translations[idx]
        det_R = np.linalg.det(R)
        D = np.zeros((3 * N, 3 * N), dtype=complex)

        for src, r_src in enumerate(parent_positions):
            r_transformed = R @ r_src + t
            for dst, r_dst in enumerate(parent_positions):
                diff = r_transformed - r_dst
                if np.allclose(diff - np.round(diff), 0, atol=tol):
                    L = np.round(diff)
                    phase = np.exp(-2j * np.pi * np.dot(kpoint, L))
                    D[3*dst:3*dst+3, 3*src:3*src+3] = det_R * R * phase
                    break  # each src maps to exactly one dst

        D_matrices.append(D)

    return D_matrices


def compute_permutation_rep(rotations, translations, kpoint,
                             parent_positions, mapping_little_group, tol=1e-4):
    """
    For each little-group operation compute the permutation representation data.

    Parameters
    ----------
    rotations : np.ndarray, shape (N_ops, 3, 3)
    translations : np.ndarray, shape (N_ops, 3)
    kpoint : array-like, shape (3,)
    parent_positions : np.ndarray, shape (N, 3)
    mapping_little_group : array-like

    Returns
    -------
    atom_mappings : list[list[(int, np.ndarray)]]
        atom_mappings[i][src] = (dst_idx, L_vec) where R@r_src+t ≈ r_dst+L_vec.
        dst_idx is 0-based; L_vec is integer array of shape (3,).
    chi_perm : np.ndarray, shape (|G_k|,), complex
        chi_perm[i] = Σ_{j: R r_j+t ≡ r_j (mod Z³)} exp(−2πi k·L)
    chi_axial : np.ndarray, shape (|G_k|,), float
        chi_axial[i] = det(R) · Tr(R)
    """
    N = len(parent_positions)
    kpoint = np.asarray(kpoint, dtype=float)

    atom_mappings = []
    chi_perm  = np.zeros(len(mapping_little_group), dtype=complex)
    chi_axial = np.zeros(len(mapping_little_group), dtype=float)

    for i_lg, idx in enumerate(mapping_little_group):
        R = rotations[idx].astype(float)
        t = translations[idx]
        det_R = np.linalg.det(R)
        tr_R  = np.trace(R)
        chi_axial[i_lg] = det_R * tr_R

        mappings_this_op = []
        chi_p = 0.0 + 0.0j

        for src, r_src in enumerate(parent_positions):
            r_transformed = R @ r_src + t
            found = False
            for dst, r_dst in enumerate(parent_positions):
                diff = r_transformed - r_dst
                if np.allclose(diff - np.round(diff), 0, atol=tol):
                    L_vec = np.round(diff).astype(int)
                    mappings_this_op.append((dst, L_vec))
                    if src == dst:
                        chi_p += np.exp(-2j * np.pi * np.dot(kpoint, L_vec))
                    found = True
                    break
            if not found:
                mappings_this_op.append((-1, np.zeros(3, dtype=int)))

        atom_mappings.append(mappings_this_op)
        chi_perm[i_lg] = chi_p

    return atom_mappings, chi_perm, chi_axial


def compute_characters(R_lk, t_lk, kpoint, mag_positions, tol=1e-4):
    """
    Computes the character of the magnetic representation for each operation in the little group.
    
    R_lk: Array of 3x3 integer matrices (rotations in real space)
    t_lk: Array of 3x1 float vectors (translations in real space fractional)
    kpoint: 3x1 float vector
    mag_positions: Nx3 array of magnetic atom fractional positions
    
    Returns:
    chi_mag: Array of length |G_k| containing the character for each operation.
    """
    chi_mag = np.zeros(len(R_lk), dtype=complex)
    
    for i, (R, t) in enumerate(zip(R_lk, t_lk)):
        # We sum the phase for each invariant atom. 
        # The phase is ONLY exp(-2 pi i k . L).
        
        # 2. Axial vector character: det(R) * Tr(R)
        # Note: Tr(R) is the trace of the 3x3 rotation matrix in real space.
        chi_axial = np.linalg.det(R) * np.trace(R)
        
        # 3. Number of fixed atoms modulo lattice translations with phase shift
        # Fixed point condition: r_j = R r_i + t (mod lattice)
        # However, for an orbit, it's Tr(permutation matrix). 
        # A magnetic site r_i contributes if R r_i + t = r_i (mod 1).
        # Wait, if there are multiple magnetic atoms in the primitive cell (or conventional cell),
        # we sum over all of them.
        
        trace_perm = 0.0 + 0.0j
        for r in mag_positions:
            r_transformed = R @ r + t
            diff = r_transformed - r
            
            # Check if diff is integer
            if np.allclose(diff - np.round(diff), 0, atol=tol):
                # The atom maps to itself up to a lattice translation L = r_transformed - r
                L = np.round(diff)
                # The phase factor associated with this lattice translation is e^{-i k . L}
                # Wait, if r_j = R r_i + t - L, then L = R r_i + t - r_i.
                # In Bertaut's method, the full phase is e^{-i k . L}.
                # Actually, the base function transforms as T_L \phi_k = e^{-i k . L} \phi_k
                atom_phase = np.exp(-2j * np.pi * np.dot(kpoint, L))
                trace_perm += atom_phase
                
        chi_mag[i] = chi_axial * trace_perm
        
    return chi_mag

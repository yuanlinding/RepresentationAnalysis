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
        
        # 2. From parent cell to standard parent setting
        r_std = parent_transform_M @ r_p + parent_transform_t
        
        # Wrapping to [0, 1)
        r_std = r_std % 1.0
        
        # The magnetic moment vector transforms as an axial vector.
        # m_std = det(P) * P @ m_child
        # This requires the real-space Cartesian transformation or just fractional?
        # Typically magnetic moments in mCIF are given in Cartesian axis or Crystal axis?
        # `_atom_site_moment.crystalaxis_x` implies they are in the basis of the primitive/conventional cell corresponding to the coordinates.
        # Let's assume they transform covariantly with the lattice (since they are in crystal axis).
        # Actually, if m is in crystal axis, m = m_1 a + m_2 b + m_3 c.
        # Thus m transforms exactly like a coordinate displacement vector (a true vector).
        # m_parent = P @ m_child.
        # But wait, it's an axial vector! So m_parent = det(P) * P @ m_child.
        # det(P) is the parity of the transformation. Usually transformations are right-handed, so det(P) > 0.
        P1 = child_transform_M
        m_p = np.linalg.det(P1) * P1 @ m
        
        P2 = parent_transform_M
        m_std = np.linalg.det(P2) * P2 @ m_p
        
        parent_positions.append(r_std)
        parent_magmoms.append(m_std)
        
    return np.array(parent_positions), np.array(parent_magmoms)

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

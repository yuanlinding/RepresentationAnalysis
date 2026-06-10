import numpy as np
import spgrep

from magirrep.little_group import build_reference_crystal


def get_little_group_irreps(it_number: int, kpoint: np.ndarray):
    """
    Compute irreducible representations of the little group at *kpoint* for
    space group IT number *it_number*.

    Uses ``spgrep.get_spacegroup_irreps`` with a reference crystal built in the
    preferred origin choice (OC2 when available), so that the returned
    rotations/translations are in the same coordinate system as parent-cell
    atom positions produced by ``parse_mcif.parse_transform``.

    Returns
    -------
    irreps : list[list[np.ndarray]]
        ``irreps[alpha][i]`` is the matrix for the i-th little-group operation.
    rotations : np.ndarray, shape (N, 3, 3)
        Conventional-cell rotations (same ordering as spglib detected).
    translations : np.ndarray, shape (N, 3)
        Conventional-cell translations.
    mapping_little_group : np.ndarray, shape (|G_k|,)
        ``mapping_little_group[i]`` is the index of the i-th little-group op
        in the ``rotations`` / ``translations`` arrays.
    """
    lattice, positions, numbers = build_reference_crystal(it_number)
    irreps, rotations, translations, mapping_little_group = spgrep.get_spacegroup_irreps(
        lattice, positions, numbers, kpoint
    )
    return irreps, rotations, translations, mapping_little_group


def decompose(irreps, chi_mag: np.ndarray, mapping_little_group: np.ndarray,
              translations: np.ndarray = None, kpoint=None) -> np.ndarray:
    """
    Decompose the magnetic representation into irreducible representations.

    n_μ = (1/|G_k|) Σ_{g ∈ G_k} χ_μ*(g) χ_mag(g)

    The sum runs over ALL listed little-group operations.  For centered
    Bravais lattices the conventional-cell op list contains each coset of the
    primitive translation group n_c times (once per centering translation
    t_c), and at zone-boundary k both χ_μ and χ_mag pick up the same phase
    exp(−2πi k·t_c) under g → g·{E|t_c} — spgrep small representations
    satisfy this by construction, and the physical characters do too provided
    atom matching is done modulo the PRIMITIVE lattice (see
    mag_rep._match_with_centering).  The products χ_μ* χ_mag are therefore
    constant across centering copies and the full-sum average is exact.

    Parameters
    ----------
    irreps :
        List of irreps; ``irreps[alpha][i]`` is the matrix for the i-th
        little-group operation.
    chi_mag :
        Characters of the magnetic representation for ALL space group
        operations (indexed the same way as the ``rotations`` returned by
        ``get_little_group_irreps``).
    mapping_little_group :
        Indices into ``chi_mag`` selecting the little-group operations.
    translations, kpoint :
        Unused; retained for call-site compatibility.

    Returns
    -------
    n_mu : np.ndarray
        Multiplicities (non-negative integers for a valid representation;
        non-integer output indicates inconsistent input characters).
    """
    chi_lg = chi_mag[np.asarray(mapping_little_group)]
    little_group_order = len(mapping_little_group)

    n_mu = []
    for irrep in irreps:
        chi_irrep = np.array([np.trace(mat) for mat in irrep])
        n = np.sum(chi_lg * np.conj(chi_irrep)) / little_group_order
        n_mu.append(np.real(n))

    return np.array(n_mu)


def find_active_irrep(n_mu_array: np.ndarray):
    """
    Return (index, multiplicity) pairs for irreps with n_μ > 0.001.
    """
    active = []
    for i, n in enumerate(n_mu_array):
        if n > 1e-3:
            active.append((i, round(n, 3)))
    return active


def compute_parity_suffixes(irreps, rotations, mapping_little_group) -> list:
    """
    Compute the Bilbao ± parity suffix for each irrep.

    The suffix is determined by the character at the inversion element I = {−E | 0}
    in the little group:
        χ^α(I) > 0  →  '+' (gerade)
        χ^α(I) < 0  →  '−' (ungerade)
        I ∉ G_k     →  '' (no suffix)

    Parameters
    ----------
    irreps :
        List of irreps; irreps[alpha][i] is the matrix for the i-th little-group op.
    rotations : np.ndarray, shape (N_ops, 3, 3)
        All space-group rotations (same indexing as mapping_little_group).
    mapping_little_group : array-like
        Indices into rotations selecting the little-group operations.

    Returns
    -------
    suffixes : list[str]
        One string ('+', '-', or '') per irrep.
    """
    # Find inversion (R == -I) among little-group operations
    i_inv = None
    for i, idx in enumerate(mapping_little_group):
        if np.allclose(rotations[idx], -np.eye(3)):
            i_inv = i
            break

    suffixes = []
    for irrep in irreps:
        if i_inv is None:
            suffixes.append('')
        else:
            chi_inv = np.real(np.trace(irrep[i_inv]))
            suffixes.append('+' if chi_inv > 0 else '-')
    return suffixes


def compute_projection_operators(irreps, D_matrices_lg, mapping_little_group) -> list:
    """
    Compute the projection operator onto each irrep subspace.

    P^α = (d_α / |G_k|) Σ_{g ∈ G_k} χ^α*(g) D(g)

    where d_α is the irrep dimension.

    Parameters
    ----------
    irreps :
        List of irreps; irreps[alpha][i] is the d_α×d_α matrix for the i-th op.
    D_matrices_lg : list[np.ndarray]
        D matrices for each little-group op, in the same order as mapping_little_group.
    mapping_little_group : array-like

    Returns
    -------
    proj_ops : list[np.ndarray]
        One 3N×3N Hermitian projection matrix per irrep.
    """
    n_lg = len(mapping_little_group)
    N3 = D_matrices_lg[0].shape[0]
    proj_ops = []
    for irrep in irreps:
        d_alpha = irrep[0].shape[0]
        chi_irrep = np.array([np.trace(mat) for mat in irrep])
        P = np.zeros((N3, N3), dtype=complex)
        for i in range(n_lg):
            P += np.conj(chi_irrep[i]) * D_matrices_lg[i]
        P *= d_alpha / n_lg
        proj_ops.append(P)
    return proj_ops


def compute_basis_vectors(proj_ops, n_atoms, tol=1e-6):
    """
    Find symmetry-adapted basis vectors for each irrep via projection + Gram-Schmidt.

    For each irrep α, projects every standard basis vector e_{3i+j} through P^α
    and orthogonalises the resulting vectors.  The span of {P^α e_i} equals the
    irrep-α subspace of the magnetic representation.

    Parameters
    ----------
    proj_ops : list[np.ndarray]
        Projection operators P^α, each of shape (3*n_atoms, 3*n_atoms).
    n_atoms : int
        Number of magnetic atoms (primitive cell).
    tol : float
        Threshold below which a projected vector is considered zero.

    Returns
    -------
    all_basis : list[list[np.ndarray]]
        all_basis[alpha] = list of orthonormal complex basis vectors of shape
        (3*n_atoms,) spanning the irrep-α subspace.
        Length = d_α · n_μ for active irreps, 0 for inactive ones.
    """
    all_basis = []
    for P in proj_ops:
        basis = []
        for i in range(3 * n_atoms):
            e = np.zeros(3 * n_atoms, dtype=complex)
            e[i] = 1.0
            v = P @ e
            for b in basis:          # Gram-Schmidt
                v -= np.vdot(b, v) * b
            if np.linalg.norm(v) > tol:
                v /= np.linalg.norm(v)
                basis.append(v)
        all_basis.append(basis)
    return all_basis


def identify_active_irrep(active_irreps, projection_ops, moment_vector) -> list:
    """
    Identify which active irrep the actual moment vector belongs to.

    For each active irrep α, computes η_α = ‖P^α M‖ / ‖M‖ where M is the
    stacked moment vector. The irrep with η ≈ 1 is the physically active one.

    Parameters
    ----------
    active_irreps : list of (idx, multiplicity)
        From find_active_irrep().
    projection_ops : list[np.ndarray]
        All projection operators (indexed by irrep index).
    moment_vector : np.ndarray, shape (3*N_prim,)
        Flattened array of primitive-cell magnetic moments.

    Returns
    -------
    identified : list of (idx, multiplicity, eta, M_proj)
        Sorted by η descending.  M_proj is the raw projected vector.
    """
    M_norm = np.linalg.norm(moment_vector)
    results = []
    for idx, n in active_irreps:
        P = projection_ops[idx]
        M_proj = P @ moment_vector
        eta = np.linalg.norm(M_proj) / M_norm if M_norm > 1e-10 else 0.0
        eta = min(1.0, eta)
        results.append((idx, n, eta, M_proj))
    results.sort(key=lambda x: -x[2])
    return results

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


def decompose(irreps, chi_mag: np.ndarray, mapping_little_group: np.ndarray) -> np.ndarray:
    """
    Decompose the magnetic representation into irreducible representations.

    n_μ = (1/|G_k|) Σ_{g ∈ G_k} χ_μ*(g) χ_mag(g)

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

    Returns
    -------
    n_mu : np.ndarray
        Multiplicities (should be non-negative integers for a valid magnetic
        representation).
    """
    chi_lg = chi_mag[mapping_little_group]
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

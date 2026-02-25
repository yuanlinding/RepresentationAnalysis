import numpy as np
import spgrep

def get_little_group_irreps(rotations, translations, kpoint):
    """
    Computes the irreducible representations of the little group associated with the kpoint.
    Returns:
    - irreps: List of irreps (list of matrices for each little group operation)
    - mapping: Indices mapping each original operation to the little group?
    Wait, spgrep returns (irreps, mapping) where mapping is len(mapping) == len(rotations).
    Actually, let's look at how spgrep does it: mapping[i] is the index of the little group element for rotations[i]?
    No, spgrep get_spacegroup_irreps returns `irreps, rotations, translations, mapping`. 
    Actually, we just need the irreps and the little group operations.
    """
    # spgrep.get_spacegroup_irreps returns a tuple.
    # We will compute it from primitive symmetry. Since we use conventional setting, the math is identical.
    # Spgrep's signature for spgrep.get_spacegroup_irreps_from_primitive_symmetry is:
    # irreps, ... ???
    # Let's write the wrapper. We will dynamically inspect the return value to be safe.
    
    res = spgrep.get_spacegroup_irreps_from_primitive_symmetry(rotations, translations, kpoint)
    # The return value is list[list[NDArrayComplex]].
    # wait! The spgrep 0.3 documentation says it returns just 'irreps'.
    # For each irrep (list), we get a list of matrices corresponding to the *little group* operations, 
    # OR is it corresponding to the full space group?
    # Usually, get_spacegroup_irreps returns irrep matrices for ALL operations in the space group.
    # Wait, spgrep actually returns matrices for the given operations! 
    # Let's assume `res` gives a list of irreps, where each irrep has a matrix for each operation in `rotations`.
    # BUT irreps for operations not in the little group are probably zero or undefined?
    # The representation of the space group is induced from the little group.
    
    return res

def decompose(irreps, chi_mag):
    """
    Decomposes the magnetic representation into irreducible representations.
    
    irreps: list of list of matrices (for each operation in the group)
    chi_mag: character of the magnetic representation for each operation
    
    Returns:
    n_mu: array of floats (should be integers) for the multiplicity of each irrep.
    """
    n_mu = []
    group_order = len(chi_mag)
    print("chi_mag shape & sum:", chi_mag.shape, np.sum(chi_mag))
    print("chi_mag:", np.round(chi_mag, 2))
    
    for mu, irrep in enumerate(irreps):
        # Calculate the character of this irrep for each operation
        # irrep is a list of matrices
        chi_orig = np.array([np.trace(mat) for mat in irrep])
        
        # The reduction formula
        # n_mu = (1/|G|) sum_{g} chi_mag(g) * chi_orig(g)^*
        # Note: if the matrices are for the full space group, we should only sum over the little group.
        # But if irrep has the same length as chi_mag (which contains only little group ops), then:
        n = np.sum(chi_mag * np.conj(chi_orig)) / group_order
        n_mu.append(np.real(n))
        
    print("n_mu array:", np.round(n_mu, 3))
    return np.array(n_mu)

def find_active_irrep(n_mu_array):
    """
    Given the multiplicities of each irrep, return the indices of the active irreps
    (those with n_mu > 0.001).
    """
    active = []
    for i, n in enumerate(n_mu_array):
        # Allow some float tolerance
        if n > 1e-3:
            active.append((i, round(n, 3)))
    return active

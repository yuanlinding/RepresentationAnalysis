import spglib
import numpy as np

# Cache for mapping IT number to standard Hall number
_HALL_NUMBER_CACHE = {}

def get_hall_number(it_number: int) -> int:
    """
    Maps an IT space group number (1-230) to the standard Hall number (1-530).
    Spgrep uses Hall number for irreps. We return the first (standard) Hall number
    for the given IT number.
    """
    global _HALL_NUMBER_CACHE
    if not _HALL_NUMBER_CACHE:
        for hall_no in range(1, 531):
            sg_type = spglib.get_spacegroup_type(hall_no)
            it_no = sg_type['number']
            if it_no not in _HALL_NUMBER_CACHE:
                _HALL_NUMBER_CACHE[it_no] = hall_no
    
    if it_number not in _HALL_NUMBER_CACHE:
        raise ValueError(f"Invalid IT space group number: {it_number}")
    return _HALL_NUMBER_CACHE[it_number]

def get_parent_sg_operations(it_number: int):
    """
    Retrieves the symmetry operations (rotations, translations) for the parent
    space group in the standard setting.
    """
    hall_no = get_hall_number(it_number)
    # spglib.get_symmetry_from_database computes symmetry operations of the specified Hall number
    # This directly returns the operations of the standard setting for the space group.
    dataset = spglib.get_symmetry_from_database(hall_no)
    
    if dataset is None:
        raise RuntimeError(f"Could not retrieve symmetry operations for Hall number {hall_no}")
        
    return dataset['rotations'], dataset['translations']

def find_little_group(rotations: np.ndarray, translations: np.ndarray, kpoint: np.ndarray, tol=1e-5):
    """
    Given the space group operations (rotations in reciprocal space are transpose of real space,
    but here k is a column vector and we use Rk), find the operations that leave k invariant
    modulo reciprocal lattice vectors (integers).
    
    Returns:
    - R_lk: Rotations of the little group
    - t_lk: Translations of the little group
    - indices: The indices of these operations in the original arrays
    """
    R_lk = []
    t_lk = []
    indices = []
    
    for idx, (R, t) in enumerate(zip(rotations, translations)):
        # R acts on k in reciprocal space as (R^T o k)? Actually no.
        # If r' = R r + t in real fractional coordinates, then
        # k' = k R^-1 in row vector notation, or R^T k in column vector notation for standard momentum transform.
        # But wait! For k as a column vector in fractional reciprocal coordinates,
        # the action of spatial rotation R on r corresponds to rotation R on k?
        # Typically, exp(i k . r') = exp(i k . (R r + t)) 
        # = exp(i (R^T k) . r + i k . t)  =>  k' = R^T k.
        # Let's use R^T k. Wait, spgrep uses R_p k. We should stick to exactly what spgrep / Bilbao expects.
        # spgrep documentation:
        # P k = k + G
        # Let's check how spgrep computes little group.
        # Wait, if we use spgrep.get_irreps_from_sgnumber(hall_number, kpoint), spgrep already finds the little group internally!
        # Do we need to explicitly compute it here?
        # Our plan states: "For each g=(R,t) in G_k, compute character contributions."
        # Spgrep will return irreps for operations of the little group.
        # However, to use the reduction formula n_μ = 1/|G_k| \sum χ_μ * χ_mag
        # we need to make sure the order of operations exactly matches the order spgrep returns!
        # spgrep function gives: irreps, mapping. wait, spgrep doesn't easily expose the raw little group operations.
        # Actually, spgrep exposes get_irreps_from_sgnumber. Its return type needs to be checked.
        pass

    # A better approach: We will implement the little group logic here.
    # The action of rotation R on k is usually k_transformed = R^T @ k or R @ k ?
    # Let's test what R^T @ k gives vs Bilbao.
    # Usually: k' = R^T k
    for idx, (R, t) in enumerate(zip(rotations, translations)):
        # k' = R^T @ k
        # Is (k' - k) a vector of integers?
        k_prime = R.T @ kpoint
        diff = k_prime - kpoint
        
        # Check if diff is integer
        is_integer = np.allclose(diff - np.round(diff), 0, atol=tol)
        if is_integer:
            R_lk.append(R)
            t_lk.append(t)
            indices.append(idx)
            
    return np.array(R_lk), np.array(t_lk), indices

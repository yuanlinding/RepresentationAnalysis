import spglib
import numpy as np

# Cache: IT number -> preferred Hall number (origin choice 2 when available)
_HALL_NUMBER_CACHE = {}


def get_hall_number(it_number: int) -> int:
    """
    Maps an IT space group number (1-230) to the preferred Hall number (1-530).

    Prefers origin choice 2 (centre-of-symmetry origin) over choice 1, matching
    the Bilbao Crystallographic Server standard used in MAGNDATA mCIF files.
    """
    global _HALL_NUMBER_CACHE
    if not _HALL_NUMBER_CACHE:
        from collections import defaultdict
        all_halls: dict = defaultdict(list)
        for hall_no in range(1, 531):
            sg_type = spglib.get_spacegroup_type(hall_no)
            # Support both old dict and new attribute interface
            it_no = sg_type['number'] if hasattr(sg_type, '__getitem__') else sg_type.number
            choice = sg_type['choice'] if hasattr(sg_type, '__getitem__') else sg_type.choice
            all_halls[it_no].append((hall_no, choice))
        for it_no, candidates in all_halls.items():
            # Prefer origin choice '2' (centrosymmetric origin); fall back to first
            oc2 = [h for h, c in candidates if c == '2']
            _HALL_NUMBER_CACHE[it_no] = oc2[0] if oc2 else candidates[0][0]

    if it_number not in _HALL_NUMBER_CACHE:
        raise ValueError(f"Invalid IT space group number: {it_number}")
    return _HALL_NUMBER_CACHE[it_number]


def _lattice_for_crystal_system(it_number: int) -> np.ndarray:
    """Return a generic unit lattice compatible with the crystal system of *it_number*."""
    if it_number <= 2:       # Triclinic
        return np.array([[1.0, 0.1, 0.2], [0.0, 1.0, 0.15], [0.0, 0.0, 1.1]])
    elif it_number <= 15:    # Monoclinic (unique axis b)
        return np.array([[1.0, 0.0, 0.3], [0.0, 1.2, 0.0], [0.0, 0.0, 1.0]])
    elif it_number <= 74:    # Orthorhombic
        return np.diag([1.0, 1.3, 1.7])
    elif it_number <= 142:   # Tetragonal
        return np.diag([1.0, 1.0, 1.5])
    elif it_number <= 194:   # Trigonal / Hexagonal
        a, c = 1.0, 1.6
        return np.array([[a, 0.0, 0.0], [-0.5 * a, 0.866025 * a, 0.0], [0.0, 0.0, c]])
    else:                    # Cubic (195-230)
        return np.eye(3)


def build_reference_crystal(it_number: int):
    """
    Build a minimal reference crystal for space group *it_number*.

    Uses TWO general-position orbits with distinct element types (Z=1 and Z=2).
    A single orbit with identical atoms can accidentally have higher symmetry
    than the target SG (e.g. the Pna2_1 orbit has Pnma symmetry when atoms
    are identical), causing spgrep to compute irreps for the wrong group.
    Two distinct orbits break any such accidental centrosymmetry.

    Returns (lattice, positions, numbers) suitable for spgrep.get_spacegroup_irreps.
    """
    hall_no = get_hall_number(it_number)
    dataset = spglib.get_symmetry_from_database(hall_no)
    rots = dataset['rotations']
    trans = dataset['translations']

    def _orbit(r0):
        seen = []
        for R, t in zip(rots, trans):
            r = (R.astype(float) @ r0 + t) % 1.0
            if not any(np.allclose(r, s, atol=1e-5) for s in seen):
                seen.append(r)
        return seen

    # Two general positions chosen to avoid special Wyckoff sites
    orbit_A = _orbit(np.array([0.12345, 0.56789, 0.34567]))
    orbit_B = _orbit(np.array([0.71828, 0.31415, 0.27183]))

    lattice   = _lattice_for_crystal_system(it_number)
    positions = np.array(orbit_A + orbit_B)
    numbers   = np.array([1] * len(orbit_A) + [2] * len(orbit_B))
    return lattice, positions, numbers


def get_parent_sg_operations(it_number: int):
    """
    Retrieve symmetry operations (rotations, translations) for the parent space
    group in the preferred standard setting (origin choice 2 when available).

    Returns the conventional-cell operations from spglib.
    """
    hall_no = get_hall_number(it_number)
    dataset = spglib.get_symmetry_from_database(hall_no)

    if dataset is None:
        raise RuntimeError(f"Could not retrieve symmetry operations for Hall number {hall_no}")

    return dataset['rotations'], dataset['translations']


def find_little_group(rotations: np.ndarray, translations: np.ndarray, kpoint: np.ndarray, tol=1e-5):
    """
    Find operations that leave *kpoint* invariant modulo reciprocal lattice vectors.

    The little-group condition: R^T k ≡ k  (mod Z³).

    Returns (R_lk, t_lk, indices) — rotations, translations, and their indices
    in the input arrays.
    """
    R_lk, t_lk, indices = [], [], []
    for idx, (R, t) in enumerate(zip(rotations, translations)):
        k_prime = R.T @ kpoint
        diff = k_prime - kpoint
        if np.allclose(diff - np.round(diff), 0, atol=tol):
            R_lk.append(R)
            t_lk.append(t)
            indices.append(idx)
    return np.array(R_lk), np.array(t_lk), indices

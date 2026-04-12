"""Generate distorted CIF/mCIF files from symmetry-adapted basis vectors."""

import os
from fractions import Fraction

import numpy as np


def _build_parent_lattice(structure, child_M: np.ndarray) -> np.ndarray:
    """Return (3,3) parent lattice matrix (rows = a, b, c in Angstroms).

    Relationship: A_child = child_M @ A_parent  (row-vector convention,
    consistent with map_atoms_to_parent_cell where r_parent = child_M @ r_child).
    """
    A_child = structure.lattice.matrix   # (3,3), rows = lattice vectors
    return np.linalg.inv(child_M) @ A_child


def _normalize_bv_cartesian(bv_complex: np.ndarray, A_parent: np.ndarray,
                             N: int) -> tuple:
    """Convert a complex (3N,) BV to normalized real fractional displacements.

    Returns
    -------
    bv_frac_unit : (N,3) ndarray  — real fractional, unit Cartesian norm
    norm_cart : float             — original Cartesian norm (for zero-check)
    """
    bv_real = np.real(bv_complex).reshape(N, 3)   # (N,3) fractional
    bv_cart = bv_real @ A_parent                   # (N,3) Cartesian
    norm_cart = np.linalg.norm(bv_cart)
    if norm_cart < 1e-10:
        return bv_real, 0.0
    bv_frac_unit = bv_real / norm_cart   # scale factor in fractional space
    return bv_frac_unit, norm_cart


def _parse_rational_kpoint(kpoint: np.ndarray) -> tuple:
    """Convert a kpoint array to (p_arr, q_arr) of integers via Fraction.

    Returns denominators q_arr (length 3) for supercell construction.
    """
    q_arr = np.ones(3, dtype=int)
    for i, x in enumerate(kpoint):
        f = Fraction(float(x)).limit_denominator(100)
        q_arr[i] = abs(f.denominator) if f.denominator != 0 else 1
    return q_arr


def _atomic_number(label: str) -> int:
    """Atomic number from element or species string (e.g. 'Mn', 'Sr2+', 'O2-')."""
    from pymatgen.core.periodic_table import get_el_sp
    try:
        return int(get_el_sp(label).Z)
    except Exception:
        return 1


def _site_short(site_name: str) -> str:
    """Extract element symbol from Wyckoff group name, e.g. 'Mn (c)' → 'Mn', 'Sr2+ (a)' → 'Sr'."""
    from pymatgen.core.periodic_table import get_el_sp
    raw = site_name.split()[0]
    try:
        return get_el_sp(raw).element.symbol
    except Exception:
        return raw


def _analyze_symmetry(A_parent: np.ndarray, atom_labels: list,
                      positions: np.ndarray, magmoms: np.ndarray = None,
                      symprec: float = 0.01) -> tuple:
    """Run spglib symmetry analysis and return (label_str, dataset).

    The label_str is the crystallographic symbol suitable for file names:
      - CIF  (no magmoms): nuclear SG symbol, e.g. 'Pmn2_1' or 'P4-nmm'
        ('/' replaced by '-' to avoid path-separator issues)
      - mCIF (with magmoms): MSG BNS symbol, e.g. "Pm'mn" or 'Pmn2_1'

    The dataset is the raw spglib result object (SpglibMagneticDataset or
    SpglibDataset) so callers can re-use it without a second spglib call,
    guaranteeing that the filename label and the file content are consistent.
    Returns ('unknown', None) on failure.
    """
    import spglib

    pos_wrapped = positions % 1.0
    numbers = [_atomic_number(s) for s in atom_labels]

    if magmoms is not None:
        cell_mag = (A_parent, pos_wrapped, numbers, magmoms)
        try:
            ds = spglib.get_magnetic_symmetry_dataset(cell_mag, symprec=symprec)
            if ds is not None:
                mtype = spglib.get_magnetic_spacegroup_type(ds.uni_number)
                label = _lookup_bns_name(mtype.bns_number).replace('/', '-')
                return (label, ds)
        except Exception:
            pass
        return ('unknown', None)
    else:
        cell = (A_parent, pos_wrapped, numbers)
        try:
            ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
            if ds is not None:
                label = ds.international.replace('/', '-').replace(' ', '')
                return (label, ds)
        except Exception:
            pass
        return ('unknown', None)


def _rot_trans_to_xyz(R: np.ndarray, t: np.ndarray,
                       time_reversal: int = None) -> str:
    """Convert integer rotation R and translation t to CIF xyz notation.

    If *time_reversal* is 0 or 1, appends ',+1' or ',-1' for mCIF format.
    """
    labels = ['x', 'y', 'z']
    parts = []
    for i in range(3):
        s = ''
        for j in range(3):
            c = int(round(R[i, j]))
            if c == 1:
                s += f'+{labels[j]}'
            elif c == -1:
                s += f'-{labels[j]}'
            elif c != 0:
                s += f'{c:+d}{labels[j]}'
        ti = float(t[i]) % 1.0
        if abs(ti) > 1e-8 and abs(ti - 1.0) > 1e-8:
            frac = Fraction(ti).limit_denominator(24)
            s += f'+{frac.numerator}/{frac.denominator}'
        s = s.lstrip('+') or '0'
        parts.append(s)
    result = ','.join(parts)
    if time_reversal is not None:
        result += ',+1' if time_reversal == 0 else ',-1'
    return result


def _lookup_bns_name(bns_number_str: str) -> str:
    """Look up BNS MSG name (e.g. 'Pmn2_1') from pymatgen's SQLite database.

    *bns_number_str* is a string like '31.123' as returned by spglib.
    Returns the BNS label string, or the number itself if lookup fails.
    """
    try:
        import sqlite3, os
        import pymatgen.symmetry as _pm_sym
        db = os.path.join(os.path.dirname(_pm_sym.__file__),
                          'symm_data_magnetic.sqlite')
        parts = bns_number_str.split('.')
        bns1, bns2 = int(parts[0]), int(parts[1])
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute('SELECT BNS_label FROM space_groups WHERE BNS1=? AND BNS2=?',
                    (bns1, bns2))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else bns_number_str
    except Exception:
        return bns_number_str


def _cell_params_from_matrix(A: np.ndarray) -> tuple:
    """Return (a, b, c, alpha, beta, gamma) from a (3,3) row-lattice matrix."""
    a_vec, b_vec, c_vec = A[0], A[1], A[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    cos_al = np.dot(b_vec, c_vec) / (b * c)
    cos_be = np.dot(a_vec, c_vec) / (a * c)
    cos_ga = np.dot(a_vec, b_vec) / (a * b)
    alpha = np.degrees(np.arccos(np.clip(cos_al, -1, 1)))
    beta  = np.degrees(np.arccos(np.clip(cos_be, -1, 1)))
    gamma = np.degrees(np.arccos(np.clip(cos_ga, -1, 1)))
    return a, b, c, alpha, beta, gamma


def _write_mcif_spglib(fname: str, A_parent: np.ndarray, atom_labels: list,
                        positions: np.ndarray, magmoms: np.ndarray,
                        symprec: float = 0.01,
                        precomputed_ds=None):
    """Write a symmetrized mCIF using spglib magnetic symmetry analysis.

    Uses *precomputed_ds* (SpglibMagneticDataset) if supplied, so the same
    spglib result that determined the filename also determines the file
    content — guaranteeing they are consistent.  Otherwise calls spglib
    internally.

    Writes:
      - Cell parameters
      - BNS MSG number and symbol (e.g. "Pm'mn")
      - Full MSG symmetry operations with time-reversal ±1
      - Asymmetric unit atoms (from equivalent_atoms)
      - Magnetic moments at each asymmetric-unit site
    """
    import spglib

    pos_wrapped = positions % 1.0
    numbers = [_atomic_number(s) for s in atom_labels]
    N = len(atom_labels)

    # ── Magnetic symmetry analysis ──────────────────────────────────────────
    if precomputed_ds is not None:
        ds_mag = precomputed_ds
    else:
        cell_mag = (A_parent, pos_wrapped, numbers, magmoms)
        ds_mag = spglib.get_magnetic_symmetry_dataset(cell_mag, symprec=symprec)

    if ds_mag is None:
        # Fallback: write P1 mCIF with all atoms and magmoms
        _write_mcif_p1(fname, A_parent, atom_labels, pos_wrapped, magmoms)
        return

    mtype      = spglib.get_magnetic_spacegroup_type(ds_mag.uni_number)
    bns_number = mtype.bns_number            # e.g. "31.123"
    bns_name   = _lookup_bns_name(bns_number) # e.g. "Pm'mn"
    msg_type   = mtype.type                   # 1,2,3,4

    # ── Asymmetric unit ─────────────────────────────────────────────────────
    # equivalent_atoms[i] == i  ↔  atom i is the canonical representative
    asym_idx = [i for i in range(N) if ds_mag.equivalent_atoms[i] == i]

    # ── Cell parameters ─────────────────────────────────────────────────────
    a, b, c, alpha, beta, gamma = _cell_params_from_matrix(A_parent)

    # ── Build CIF text ───────────────────────────────────────────────────────
    lines = []
    lines.append("# generated by magirrep (spglib magnetic symmetry analysis)")
    data_name = os.path.splitext(os.path.basename(fname))[0].replace(' ', '_')
    lines.append(f"data_{data_name}")
    lines.append("")
    lines.append(f"_space_group_magn.number_BNS   {bns_number}")
    lines.append(f"_space_group_magn.name_BNS     \"{bns_name}\"")
    lines.append(f"_space_group_magn.type          {msg_type}")
    lines.append("")
    lines.append(f"_cell_length_a    {a:.6f}")
    lines.append(f"_cell_length_b    {b:.6f}")
    lines.append(f"_cell_length_c    {c:.6f}")
    lines.append(f"_cell_angle_alpha {alpha:.4f}")
    lines.append(f"_cell_angle_beta  {beta:.4f}")
    lines.append(f"_cell_angle_gamma {gamma:.4f}")
    lines.append("")

    # MSG symmetry operations
    lines.append("loop_")
    lines.append("  _space_group_symop_magn_operation.id")
    lines.append("  _space_group_symop_magn_operation.xyz")
    for op_i, (R, t, tr) in enumerate(zip(ds_mag.rotations, ds_mag.translations,
                                           ds_mag.time_reversals), start=1):
        xyz = _rot_trans_to_xyz(R, t, time_reversal=int(tr))
        lines.append(f"  {op_i} {xyz}")
    lines.append("")

    # Centering (trivial — one entry for primitive; real centering is in ops above)
    lines.append("loop_")
    lines.append("  _space_group_symop_magn_centering.id")
    lines.append("  _space_group_symop_magn_centering.xyz")
    lines.append("  1 x,y,z,+1")
    lines.append("")

    # Atom sites (asymmetric unit)
    lines.append("loop_")
    lines.append("  _atom_site_label")
    lines.append("  _atom_site_type_symbol")
    lines.append("  _atom_site_fract_x")
    lines.append("  _atom_site_fract_y")
    lines.append("  _atom_site_fract_z")

    # Label counter per element
    label_count: dict = {}
    site_labels = []
    for i in asym_idx:
        elem = atom_labels[i]
        label_count[elem] = label_count.get(elem, 0) + 1
        lbl = f"{elem}{label_count[elem]}"
        site_labels.append(lbl)
        r = pos_wrapped[i]
        lines.append(f"  {lbl} {elem} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}")
    lines.append("")

    # Magnetic moments (only for sites with |m| > 0.01 µB)
    mag_sites = [(lbl, i) for lbl, i in zip(site_labels, asym_idx)
                 if np.linalg.norm(magmoms[i]) > 0.01]
    if mag_sites:
        lines.append("loop_")
        lines.append("  _atom_site_moment.label")
        lines.append("  _atom_site_moment.crystalaxis_x")
        lines.append("  _atom_site_moment.crystalaxis_y")
        lines.append("  _atom_site_moment.crystalaxis_z")
        for lbl, i in mag_sites:
            m = magmoms[i]
            lines.append(f"  {lbl} {m[0]:.4f} {m[1]:.4f} {m[2]:.4f}")
        lines.append("")

    with open(fname, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _write_mcif_p1(fname: str, A_parent: np.ndarray, atom_labels: list,
                   positions: np.ndarray, magmoms: np.ndarray):
    """Fallback: write P1 mCIF with all atoms when MSG detection fails."""
    a, b, c, alpha, beta, gamma = _cell_params_from_matrix(A_parent)
    lines = ["# generated by magirrep (P1 fallback — MSG detection failed)"]
    data_name = os.path.splitext(os.path.basename(fname))[0].replace(' ', '_')
    lines.append(f"data_{data_name}")
    lines.append("")
    lines.append("_space_group_magn.number_BNS   1.1")
    lines.append("_space_group_magn.name_BNS     \"P1\"")
    lines.append(f"_cell_length_a    {a:.6f}")
    lines.append(f"_cell_length_b    {b:.6f}")
    lines.append(f"_cell_length_c    {c:.6f}")
    lines.append(f"_cell_angle_alpha {alpha:.4f}")
    lines.append(f"_cell_angle_beta  {beta:.4f}")
    lines.append(f"_cell_angle_gamma {gamma:.4f}")
    lines.append("")
    lines.append("loop_")
    lines.append("  _atom_site_label  _atom_site_type_symbol")
    lines.append("  _atom_site_fract_x  _atom_site_fract_y  _atom_site_fract_z")
    label_count: dict = {}
    site_labels = []
    for i, (elem, r) in enumerate(zip(atom_labels, positions % 1.0)):
        label_count[elem] = label_count.get(elem, 0) + 1
        lbl = f"{elem}{label_count[elem]}"
        site_labels.append(lbl)
        lines.append(f"  {lbl} {elem} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}")
    lines.append("")
    mag_sites = [(lbl, i) for i, lbl in enumerate(site_labels)
                 if np.linalg.norm(magmoms[i]) > 0.01]
    if mag_sites:
        lines.append("loop_")
        lines.append("  _atom_site_moment.label")
        lines.append("  _atom_site_moment.crystalaxis_x")
        lines.append("  _atom_site_moment.crystalaxis_y")
        lines.append("  _atom_site_moment.crystalaxis_z")
        for lbl, i in mag_sites:
            m = magmoms[i]
            lines.append(f"  {lbl} {m[0]:.4f} {m[1]:.4f} {m[2]:.4f}")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _write_cif(fname: str, A_parent: np.ndarray, atom_labels: list,
               positions: np.ndarray, magmoms: np.ndarray = None,
               symprec: float = 0.01, precomputed_ds=None):
    """Write a symmetrized CIF or mCIF.

    CIF (magmoms=None):
        Uses pymatgen's CifWriter with symprec, which calls spglib internally
        to detect the correct subgroup.

    mCIF (magmoms provided):
        Calls _write_mcif_spglib with the *precomputed_ds* dataset (if given)
        so that the filename label and the file content both derive from the
        same single spglib call — no inconsistency possible.
    """
    pos_wrapped = positions % 1.0

    if magmoms is not None:
        _write_mcif_spglib(fname, A_parent, atom_labels, pos_wrapped,
                           magmoms, symprec=symprec, precomputed_ds=precomputed_ds)
    else:
        from pymatgen.core import Lattice, Structure
        from pymatgen.io.cif import CifWriter
        struct = Structure(Lattice(A_parent), atom_labels, pos_wrapped)
        cw = CifWriter(struct, symprec=symprec)
        cw.write_file(fname)


def _distort_gamma(
    stem: str,
    parent_positions: np.ndarray,
    atom_labels: list,
    A_parent: np.ndarray,
    all_basis: list,
    active_irreps: list,
    bilbao_labels: dict,
    wyckoff_groups: list,
    amplitude: float,
    keep_magnetic: bool,
    parent_magmoms: np.ndarray,
    out_dir: str,
    suffix: str,
) -> list:
    """Generate distorted files for k=Γ (distort parent cell directly)."""
    N = len(parent_positions)
    generated = []

    # Global BV counter (psi_n), 1-based, consistent with printed section 9
    psi_global = 1

    # Iterate over irreps in label-alphabetical order (matches section 9 order)
    sorted_active = sorted(active_irreps,
                           key=lambda x: bilbao_labels.get(x[0], f"z{x[0]}"))

    for irrep_idx, _n_mu in sorted_active:
        label = bilbao_labels.get(irrep_idx, f"irrep{irrep_idx}")
        bvs = all_basis[irrep_idx]   # list of (3N,) complex arrays
        bv_start = psi_global        # first global psi index for this irrep
        psi_global += len(bvs)

        for site_name, site_indices in wyckoff_groups:
            elem = _site_short(site_name)

            # Find which BVs are non-zero at this site
            for local_n, bv_complex in enumerate(bvs):
                # Check if BV has any amplitude at atoms in this site
                max_amp = max(
                    np.linalg.norm(np.real(bv_complex[3*i:3*i+3]))
                    for i in site_indices
                )
                if max_amp < 1e-3:
                    continue

                bv_frac_unit, norm = _normalize_bv_cartesian(bv_complex, A_parent, N)
                if norm < 1e-10:
                    continue

                # Displaced positions
                new_positions = parent_positions + amplitude * bv_frac_unit

                # Magmoms (pass through if requested)
                moms = parent_magmoms if keep_magnetic else None

                psi_n = bv_start + local_n
                sg_label, sym_ds = _analyze_symmetry(A_parent, atom_labels,
                                                      new_positions, moms)
                fname = os.path.join(out_dir,
                                     f"{stem}_{label}_{elem}_psi{psi_n}_{sg_label}{suffix}")
                _write_cif(fname, A_parent, atom_labels, new_positions, moms,
                           precomputed_ds=sym_ds)
                generated.append(fname)

    return generated


def _distort_kpoint(
    stem: str,
    parent_positions: np.ndarray,
    atom_labels: list,
    A_parent: np.ndarray,
    kpoint: np.ndarray,
    all_basis: list,
    active_irreps: list,
    bilbao_labels: dict,
    wyckoff_groups: list,
    amplitude: float,
    keep_magnetic: bool,
    parent_magmoms: np.ndarray,
    out_dir: str,
    suffix: str,
) -> list:
    """Generate distorted supercell files for commensurate k ≠ Γ."""
    from pymatgen.core import Lattice, Structure

    N = len(parent_positions)
    q_arr = _parse_rational_kpoint(kpoint)   # supercell diagonal
    T = np.diag(q_arr)                       # e.g. diag(2,2,2) for L-point

    # Build undistorted parent Structure
    A_parent_lat = Lattice(A_parent)
    parent_struct = Structure(A_parent_lat, atom_labels, parent_positions % 1.0)

    # Supercell lattice and atom enumeration (pymatgen make_supercell)
    super_struct_undist = parent_struct.copy()
    super_struct_undist.make_supercell(T)
    # super_struct_undist now has N * q1*q2*q3 atoms, ordered by unit cell

    n_super = q_arr[0] * q_arr[1] * q_arr[2]
    N_super = N * n_super

    generated = []
    psi_global = 1
    sorted_active = sorted(active_irreps,
                           key=lambda x: bilbao_labels.get(x[0], f"z{x[0]}"))

    for irrep_idx, _n_mu in sorted_active:
        label = bilbao_labels.get(irrep_idx, f"irrep{irrep_idx}")
        bvs = all_basis[irrep_idx]
        bv_start = psi_global
        psi_global += len(bvs)

        for site_name, site_indices in wyckoff_groups:
            elem = _site_short(site_name)

            for local_n, bv_complex in enumerate(bvs):
                max_amp = max(
                    np.linalg.norm(np.real(bv_complex[3*i:3*i+3]))
                    for i in site_indices
                )
                if max_amp < 1e-3:
                    continue

                bv_frac_unit, norm = _normalize_bv_cartesian(bv_complex, A_parent, N)
                if norm < 1e-10:
                    continue

                # Build supercell fractional positions with phase modulation
                new_frac_super = []   # in supercell fractional coords
                new_labels_super = []
                new_moms_super = []   # optional

                for n3 in range(q_arr[2]):
                    for n2 in range(q_arr[1]):
                        for n1 in range(q_arr[0]):
                            n_vec = np.array([n1, n2, n3], dtype=float)
                            phase = np.exp(2j * np.pi * np.dot(kpoint, n_vec))

                            for atom_i in range(N):
                                u_frac_i = amplitude * np.real(
                                    bv_frac_unit[atom_i] * phase)
                                r_parent_i = parent_positions[atom_i]
                                r_super_i = (r_parent_i + n_vec + u_frac_i) / q_arr

                                new_frac_super.append(r_super_i)
                                new_labels_super.append(atom_labels[atom_i])
                                if keep_magnetic and parent_magmoms is not None:
                                    new_moms_super.append(parent_magmoms[atom_i])

                new_frac_super = np.array(new_frac_super)
                A_super = super_struct_undist.lattice.matrix   # q1*a × q2*b × q3*c

                moms = np.array(new_moms_super) if keep_magnetic and parent_magmoms is not None else None
                psi_n = bv_start + local_n
                sg_label, sym_ds = _analyze_symmetry(A_super, new_labels_super,
                                                      new_frac_super, moms)
                fname = os.path.join(out_dir,
                                     f"{stem}_{label}_{elem}_psi{psi_n}_{sg_label}{suffix}")
                _write_cif(fname, A_super, new_labels_super, new_frac_super, moms,
                           precomputed_ds=sym_ds)
                generated.append(fname)

    return generated


def generate_distorted_cifs(
    path: str,
    parent_positions: np.ndarray,
    atom_labels: list,
    it_number: int,
    kpoint: np.ndarray,
    child_M: np.ndarray,
    structure,
    all_basis: list,
    active_irreps: list,
    bilbao_labels: dict,
    wyckoff_groups: list,
    amplitude: float = 0.1,
    keep_magnetic: bool = False,
    parent_magmoms: np.ndarray = None,
    out_dir: str = None,
) -> list:
    """Generate one distorted CIF (or mCIF) per (irrep, Wyckoff site, BV).

    Parameters
    ----------
    path : str
        Input file path (used for stem).
    parent_positions : (N,3) ndarray
        Fractional coords in parent cell.
    atom_labels : list[str]
        Element symbols, length N.
    it_number : int
        IT space-group number (unused directly; reserved for future labelling).
    kpoint : (3,) ndarray
        Propagation vector (fractional, parent cell reciprocal).
    child_M : (3,3) ndarray
        Child→parent basis transform (r_parent = child_M @ r_child).
    structure : pymatgen Structure
        Child-cell structure (for lattice).
    all_basis : list
        all_basis[alpha] = list of (3N,) complex BVs for irrep alpha.
    active_irreps : list of (int, float)
        (irrep_idx, n_mu) pairs.
    bilbao_labels : dict
        {irrep_idx: label_str}
    wyckoff_groups : list of (str, list[int])
        [(site_name, atom_indices), ...]
    amplitude : float
        Displacement amplitude in Angstroms (Cartesian norm of full BV).
    keep_magnetic : bool
        If True and parent_magmoms is not None, write mCIF files with moments.
    parent_magmoms : (N,3) ndarray or None
        Magnetic moments in fractional coordinates.
    out_dir : str or None
        Output directory (default: current working directory).

    Returns
    -------
    list[str] : paths of generated files.
    """
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(path))[0]
    suffix = ".mcif" if (keep_magnetic and parent_magmoms is not None) else ".cif"

    A_parent = _build_parent_lattice(structure, child_M)

    # Determine if k ≈ Γ (all components zero)
    is_gamma = np.allclose(kpoint, 0.0, atol=1e-6)

    if is_gamma:
        return _distort_gamma(
            stem, parent_positions, atom_labels, A_parent,
            all_basis, active_irreps, bilbao_labels, wyckoff_groups,
            amplitude, keep_magnetic, parent_magmoms, out_dir, suffix)
    else:
        return _distort_kpoint(
            stem, parent_positions, atom_labels, A_parent, kpoint,
            all_basis, active_irreps, bilbao_labels, wyckoff_groups,
            amplitude, keep_magnetic, parent_magmoms, out_dir, suffix)

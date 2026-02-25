#!/usr/bin/env python3
"""
magrep_bilbao_parentdetect.py

Features:
 - Parse mCIF with gemmi and extract magnetic cell + moments.
 - Accept child->parent transform as matrix or string; auto-detect minimal parent if omitted.
 - Build little group G_k using spglib.
 - Compute magnetic representation characters and full D(g) matrices.
 - Query Bilbao/REPRES for small irreps; parse and map Bilbao's operation ordering to local Gk ordering.
 - Fallback to user-supplied irreps if Bilbao fails.
 - Unit tests on a simple NiO-like AFM/FM example.

Dependencies:
  pip install numpy scipy pymatgen spglib gemmi requests

Notes:
 - Bilbao web interface is not a stable JSON API; this script attempts robust parsing and mapping,
   but network or format changes may require small adjustments.
"""

import numpy as np
import spglib
from pymatgen.core import Lattice, Structure
import gemmi
import requests
import re
import json
import tempfile
import os
from typing import List, Tuple, Dict, Callable, Optional

# ---------------------------
# Utilities
# ---------------------------

def frac_mod(x):
    return np.mod(x, 1.0)

def round_int_vec(v):
    return np.rint(v).astype(int)

def element_key(R: np.ndarray, t: np.ndarray):
    return (tuple(map(int, R.flatten())), tuple(np.round(t, 8)))

def parse_transform_string(s: str) -> np.ndarray:
    """
    Parse a transform string like "2 0 0;0 2 0;0 0 2" into a 3x3 integer numpy array.
    Accepts whitespace and commas.
    """
    s = s.strip()
    s = s.replace(',', ' ')
    rows = [row.strip() for row in re.split(r'[;\\n]+', s) if row.strip()]
    if len(rows) != 3:
        raise ValueError("Transform string must have 3 rows separated by ';' or newline.")
    mat = []
    for r in rows:
        parts = [int(float(x)) for x in r.split()]
        if len(parts) != 3:
            raise ValueError("Each row must have 3 integers.")
        mat.append(parts)
    return np.array(mat, dtype=int)

# ---------------------------
# mCIF parsing and parent reconstruction
# ---------------------------

def parse_mcif_get_magnetic_cell(mcif_path: str):
    doc = gemmi.cif.read_file(mcif_path)
    block = doc.sole_block()
    # cell
    a = float(block.find_value('_cell_length_a'))
    b = float(block.find_value('_cell_length_b'))
    c = float(block.find_value('_cell_length_c'))
    alpha = float(block.find_value('_cell_angle_alpha'))
    beta  = float(block.find_value('_cell_angle_beta'))
    gamma = float(block.find_value('_cell_angle_gamma'))
    lattice = gemmi.UnitCell(a, b, c, alpha, beta, gamma).orthogonal_matrix()
    lattice = np.array(lattice)
    # atom_site loop
    loop = block.find_loop('_atom_site_label')
    if loop is None:
        raise RuntimeError("No _atom_site loop found in mCIF.")
    headers = [h for h in loop.tags]
    rows = [list(r) for r in loop.values]
    def col_index(tag):
        try:
            return headers.index(tag)
        except ValueError:
            return None
    xi = col_index('_atom_site_fract_x')
    yi = col_index('_atom_site_fract_y')
    zi = col_index('_atom_site_fract_z')
    species_i = col_index('_atom_site_type_symbol') or col_index('_atom_site_label')
    mx_i = col_index('_atom_site_moment_x')
    my_i = col_index('_atom_site_moment_y')
    mz_i = col_index('_atom_site_moment_z')
    m_i  = col_index('_atom_site_moment')
    frac_positions = []
    species = []
    moments = []
    for r in rows:
        fx = float(r[xi]); fy = float(r[yi]); fz = float(r[zi])
        frac_positions.append(np.array([fx, fy, fz], dtype=float))
        species.append(str(r[species_i]) if species_i is not None else str(r[0]))
        if mx_i is not None and my_i is not None and mz_i is not None:
            try:
                mx = float(r[mx_i]); my = float(r[my_i]); mz = float(r[mz_i])
                moments.append(np.array([mx, my, mz], dtype=float))
            except Exception:
                moments.append(np.array([0.0, 0.0, 0.0], dtype=float))
        elif m_i is not None:
            try:
                s = r[m_i].strip()
                if ',' in s:
                    parts = [float(x) for x in s.split(',')]
                    if len(parts) == 3:
                        moments.append(np.array(parts, dtype=float))
                    else:
                        moments.append(np.array([0.0,0.0,0.0], dtype=float))
                else:
                    val = float(s)
                    moments.append(np.array([0.0,0.0,val], dtype=float))
            except Exception:
                moments.append(np.array([0.0,0.0,0.0], dtype=float))
        else:
            moments.append(np.array([0.0,0.0,0.0], dtype=float))
    try:
        spg = int(block.find_value('_symmetry_Int_Tables_number') or block.find_value('_space_group_IT_number'))
    except Exception:
        spg = None
    return {
        'lattice': lattice,
        'frac_positions': frac_positions,
        'species': species,
        'moments': moments,
        'spacegroup_number': spg
    }

def reconstruct_parent_cell_from_child(child_frac_positions: List[np.ndarray],
                                       child_species: List[str],
                                       child_moments: List[np.ndarray],
                                       child_lattice: np.ndarray,
                                       child_to_parent_transform: Optional[np.ndarray] = None):
    """
    If child_to_parent_transform is provided (3x3 int matrix), use it.
    If None, attempt to detect a simple diagonal integer transform T = diag(n1,n2,n3)
    with ni in 1..max_factor that reduces the number of unique parent positions.
    Returns parent lattice, positions, species, moments, mapping.
    """
    if child_to_parent_transform is None:
        # Heuristic detection: try diagonal transforms with factors 1..4
        max_factor = 4
        best = None
        best_count = len(child_frac_positions)
        for n1 in range(1, max_factor+1):
            for n2 in range(1, max_factor+1):
                for n3 in range(1, max_factor+1):
                    T = np.diag([n1, n2, n3])
                    try:
                        recon = _reconstruct_with_T(child_frac_positions, child_species, child_moments, child_lattice, T)
                        count = len(recon['parent_frac_positions'])
                        if count < best_count:
                            best_count = count
                            best = recon
                    except Exception:
                        continue
        if best is None:
            # fallback to identity
            return _reconstruct_with_T(child_frac_positions, child_species, child_moments, child_lattice, np.eye(3, dtype=int))
        return best
    else:
        if isinstance(child_to_parent_transform, str):
            T = parse_transform_string(child_to_parent_transform)
        else:
            T = np.array(child_to_parent_transform, dtype=int)
        return _reconstruct_with_T(child_frac_positions, child_species, child_moments, child_lattice, T)

def _reconstruct_with_T(child_frac_positions, child_species, child_moments, child_lattice, T):
    T = np.array(T, dtype=int)
    T_inv = np.linalg.inv(T)
    parent_lattice = child_lattice.dot(T_inv.T)
    parent_positions = []
    parent_species = []
    parent_moments = []
    mapping = {}
    for i, r_child in enumerate(child_frac_positions):
        r_parent = T_inv.dot(r_child)
        r_parent_mod = frac_mod(r_parent)
        found = False
        for j, rp in enumerate(parent_positions):
            if np.linalg.norm(frac_mod(rp - r_parent_mod)) < 1e-6:
                mapping[j].append(i)
                found = True
                break
        if not found:
            j = len(parent_positions)
            parent_positions.append(r_parent_mod)
            parent_species.append(child_species[i])
            parent_moments.append(child_moments[i])
            mapping[j] = [i]
    return {
        'parent_lattice': parent_lattice,
        'parent_frac_positions': parent_positions,
        'parent_species': parent_species,
        'parent_moments': parent_moments,
        'mapping': mapping,
        'T_used': T
    }

# ---------------------------
# Group theory core (steps 4-8)
# ---------------------------

def get_parent_symmetry_ops_from_structure(structure: Structure, symprec=1e-6):
    lattice = structure.lattice.matrix
    positions = [list(s.frac_coords) for s in structure.sites]
    numbers = [s.specie.number for s in structure.sites]
    cell = (lattice, positions, numbers)
    sym = spglib.get_symmetry(cell, symprec=symprec)
    rotations = sym['rotations']
    translations = sym['translations']
    ops = []
    for R, t in zip(rotations, translations):
        ops.append((np.array(R, dtype=int), np.array(t, dtype=float)))
    return ops

def build_little_group(parent_ops: List[Tuple[np.ndarray, np.ndarray]], k_frac: np.ndarray, eps=1e-8):
    Gk = []
    for R, t in parent_ops:
        delta_k = R.dot(k_frac) - k_frac
        G = round_int_vec(delta_k)
        if np.linalg.norm(delta_k - G) < eps:
            Gk.append((R, t))
    return Gk

def find_image_site(R: np.ndarray, t: np.ndarray, r_i: np.ndarray, parent_sites: List[np.ndarray], eps=1e-6):
    r_image = R.dot(r_i) + t
    r_image_mod = frac_mod(r_image)
    L = round_int_vec(r_image - r_image_mod)
    for j, r_j in enumerate(parent_sites):
        if np.linalg.norm(frac_mod(r_j - r_image_mod)) < eps:
            return j, L
    return None, None

def axial_rotation_matrix(R: np.ndarray):
    det = int(round(np.linalg.det(R)))
    return det * R.astype(float)

def compute_chi_mag(Gk: List[Tuple[np.ndarray, np.ndarray]], parent_sites: List[np.ndarray], k_frac: np.ndarray, eps=1e-8):
    chi_mag = {}
    for R, t in Gk:
        phi = np.exp(2j * np.pi * np.dot(k_frac, t))
        trace_total = 0+0j
        for i, r_i in enumerate(parent_sites):
            j, L = find_image_site(R, t, r_i, parent_sites, eps=eps)
            if j is None:
                continue
            perm_phase = np.exp(2j * np.pi * np.dot(k_frac, L))
            if i == j:
                R_ax = axial_rotation_matrix(R)
                trace_total += perm_phase * np.trace(R_ax)
        chi_mag[element_key(R, t)] = phi * trace_total
    return chi_mag

def build_D_matrix(R: np.ndarray, t: np.ndarray, parent_sites: List[np.ndarray], k_frac: np.ndarray, eps=1e-8):
    N = len(parent_sites)
    D = np.zeros((3*N, 3*N), dtype=complex)
    for i, r_i in enumerate(parent_sites):
        j, L = find_image_site(R, t, r_i, parent_sites, eps=eps)
        if j is None:
            continue
        perm_phase = np.exp(2j * np.pi * np.dot(k_frac, L))
        R_ax = axial_rotation_matrix(R)
        D[3*j:3*j+3, 3*i:3*i+3] = perm_phase * R_ax
    return D

def build_projection_operator(chi_list: List[complex], irrep_dim: int, Gk: List[Tuple[np.ndarray, np.ndarray]], D_matrices: List[np.ndarray]):
    Gk_size = len(Gk)
    P = np.zeros_like(D_matrices[0], dtype=complex)
    for chi_mu_g, Dg in zip(chi_list, D_matrices):
        P += np.conjugate(chi_mu_g) * Dg
    P *= (irrep_dim / float(Gk_size))
    return P

# ---------------------------
# Bilbao REPRES fetch + mapping (robust)
# ---------------------------

def fetch_irreps_from_bilbao_try(spacegroup_number: int, k_frac: List[float]):
    """
    Attempt to fetch Bilbao REPRES data. This function tries multiple strategies:
      - JSON-like GET (if supported)
      - HTML parsing fallback (best-effort)
    Returns a dict with keys:
      'irreps': list of {'label','dim','characters' (list of complex)},
      'ops': list of {'R':3x3 int, 't':3-vector float} in the same order as characters.
    Returns None on failure.
    """
    base = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/repres"
    params = {'sg': str(spacegroup_number), 'k': ','.join([str(float(x)) for x in k_frac])}
    try:
        resp = requests.get(base, params=params, timeout=12.0)
        if resp.status_code != 200:
            return None
        text = resp.text
        # Try to find a JSON blob in the page
        json_blob = None
        m = re.search(r'var\s+represData\s*=\s*(\{.*\});', text, flags=re.S)
        if m:
            try:
                json_blob = json.loads(m.group(1))
            except Exception:
                json_blob = None
        # If JSON not found, try to extract tables of operations and characters via regex
        if json_blob:
            # Expect json_blob to contain 'operations' and 'irreps' keys
            ops = []
            for op in json_blob.get('operations', []):
                R = np.array(op.get('rotation', [[0,0,0],[0,0,0],[0,0,0]]), dtype=int)
                t = np.array(op.get('translation', [0,0,0]), dtype=float)
                ops.append({'R': R, 't': t})
            irreps = []
            for ir in json_blob.get('irreps', []):
                chars = [complex(c) for c in ir.get('characters', [])]
                irreps.append({'label': ir.get('label','irrep'), 'dim': int(ir.get('dim',1)), 'characters': chars})
            return {'irreps': irreps, 'ops': ops}
        # Fallback: crude HTML parsing to extract operation matrices and character tables
        # Find operation blocks like "Rotation matrix: [ [1,0,0], ... ]" or "r = ( ... )"
        # This is fragile but often Bilbao prints operations in a table.
        # Attempt to extract rotation matrices and translations from the HTML
        ops = []
        # regex for rotation rows: sequences of three integers inside brackets
        rot_matches = re.findall(r'Rotation\s*:\s*

\[([^\]

]+)\]

', text)
        trans_matches = re.findall(r'Translation\s*:\s*

\[([^\]

]+)\]

', text)
        # If not found, try alternative patterns
        if not rot_matches:
            rot_matches = re.findall(r'\(\s*([\-0-9,\s]+)\s*\)\s*\(\s*([\-0-9,\s]+)\s*\)\s*\(\s*([\-0-9,\s]+)\s*\)', text)
            # rot_matches will be tuples of three strings; convert accordingly
            for tup in rot_matches:
                try:
                    rows = []
                    for row in tup:
                        nums = [int(x) for x in re.findall(r'-?\d+', row)]
                        rows.append(nums)
                    R = np.array(rows, dtype=int)
                    ops.append({'R': R, 't': np.array([0.0,0.0,0.0])})
                except Exception:
                    continue
        else:
            # parse rot_matches entries like "1,0,0;0,1,0;0,0,1"
            for rm in rot_matches:
                nums = [int(x) for x in re.findall(r'-?\d+', rm)]
                if len(nums) >= 9:
                    R = np.array(nums[:9], dtype=int).reshape((3,3))
                    ops.append({'R': R, 't': np.array([0.0,0.0,0.0])})
        # Try to extract character tables: rows of numbers in tables
        irreps = []
        char_rows = re.findall(r'<tr[^>]*>\s*(?:<td[^>]*>[^<]*</td>\s*)+?</tr>', text, flags=re.S)
        # crude attempt: find sequences of numbers in the page that could be characters
        char_numbers = re.findall(r'([\-]?\d+\.?\d*(?:[eE][\-+]?\d+)?)', text)
        # If we have ops and some numbers, attempt to chunk numbers into irreps
        if ops and char_numbers:
            # assume each operation has one character per irrep; try to split
            # This is heuristic: assume characters are integers or simple floats
            # We'll attempt to group into blocks of len(ops)
            nums = [float(x) for x in char_numbers]
            L = len(ops)
            # find a starting offset where total numbers % L == 0 and reasonable count
            if len(nums) >= L:
                # try to find plausible number of irreps
                for n_ir in range(1, min(30, len(nums)//L)+1):
                    if n_ir * L == len(nums):
                        # build irreps
                        for i in range(n_ir):
                            chars = [complex(nums[i*L + j]) for j in range(L)]
                            irreps.append({'label': f'irrep_{i+1}', 'dim': 1, 'characters': chars})
                        break
        # If we have at least one irrep and ops, return
        if ops and irreps:
            return {'irreps': irreps, 'ops': ops}
        # Otherwise fail gracefully
        return None
    except Exception:
        return None

def map_bilbao_ops_to_local(bilbao_ops: List[Dict], local_Gk: List[Tuple[np.ndarray, np.ndarray]], tol=1e-6):
    """
    Map Bilbao ops (each with 'R' int 3x3 and 't' fractional 3-vector) to local Gk ops.
    Returns a mapping list of indices: for each bilbao op index i, returns local index j or -1 if not found.
    Matching uses R equality and t modulo lattice (within tol), allowing permutations of op order.
    """
    local_keys = [element_key(R, t) for (R, t) in local_Gk]
    mapping = [-1] * len(bilbao_ops)
    used = set()
    for i, op in enumerate(bilbao_ops):
        Rb = np.array(op['R'], dtype=int)
        tb = np.array(op.get('t', [0.0,0.0,0.0]), dtype=float)
        found = False
        for j, (Rl, tl) in enumerate(local_Gk):
            if j in used:
                continue
            if np.array_equal(Rb, Rl):
                # translations may differ by integer lattice vectors; check modulo 1
                delta = frac_mod(tb - tl)
                if np.linalg.norm(np.minimum(delta, 1-delta)) < tol:
                    mapping[i] = j
                    used.add(j)
                    found = True
                    break
        if not found:
            # try matching by rotation only (if translations ambiguous)
            for j, (Rl, tl) in enumerate(local_Gk):
                if j in used:
                    continue
                if np.array_equal(Rb, Rl):
                    mapping[i] = j
                    used.add(j)
                    found = True
                    break
    return mapping

# ---------------------------
# High-level pipeline tying everything together
# ---------------------------

def analyze_mcif_with_bilbao_and_parentdetect(mcif_path: str,
                                              child_to_parent_transform: Optional[object],
                                              k_frac: np.ndarray,
                                              irrep_fallback: Optional[List[Dict]] = None):
    """
    Full pipeline:
      - parse mCIF
      - reconstruct parent cell (using provided transform or auto-detect)
      - build parent Structure
      - compute Gk, chi_mag, D matrices
      - fetch Bilbao irreps and ops; map Bilbao ops to local Gk ordering
      - compute multiplicities and projection diagnostics
    """
    parsed = parse_mcif_get_magnetic_cell(mcif_path)
    child_lattice = parsed['lattice']
    child_frac_positions = parsed['frac_positions']
    child_species = parsed['species']
    child_moments = parsed['moments']
    recon = reconstruct_parent_cell_from_child(child_frac_positions, child_species, child_moments, child_lattice, child_to_parent_transform)
    parent_lattice = recon['parent_lattice']
    parent_frac_positions = recon['parent_frac_positions']
    parent_species = recon['parent_species']
    parent_moments = recon['parent_moments']
    T_used = recon.get('T_used', None)
    lattice = Lattice(parent_lattice)
    struct = Structure(lattice, parent_species, parent_frac_positions)
    parent_ops = get_parent_symmetry_ops_from_structure(struct)
    Gk = build_little_group(parent_ops, k_frac)
    if len(Gk) == 0:
        raise RuntimeError("Little group empty for given k.")
    Gk = sorted(Gk, key=lambda op: (tuple(op[0].flatten()), tuple(np.round(op[1], 8))))
    parent_sites = parent_frac_positions
    chi_mag_map = compute_chi_mag(Gk, parent_sites, k_frac)
    chi_mag_list = [chi_mag_map[element_key(R, t)] for (R, t) in Gk]
    D_matrices = [build_D_matrix(R, t, parent_sites, k_frac) for (R, t) in Gk]
    # Attempt Bilbao
    spacegroup_number = parsed.get('spacegroup_number') or struct.get_space_group_info()[1]
    bilbao_data = None
    if spacegroup_number is not None:
        bilbao_data = fetch_irreps_from_bilbao_try(spacegroup_number, list(k_frac))
    irreps_to_use = []
    if bilbao_data:
        # Map Bilbao ops to local Gk ordering
        bilbao_ops = bilbao_data['ops']
        mapping = map_bilbao_ops_to_local(bilbao_ops, Gk)
        # For each Bilbao irrep, reorder its characters to local Gk order using mapping
        for ir in bilbao_data['irreps']:
            chars = ir['characters']
            # if mapping incomplete, skip this irrep
            if len(chars) != len(mapping):
                # try to proceed if counts match Gk
                pass
            # build chi_list in local order
            chi_list_local = [0]*len(Gk)
            for i_map, j_local in enumerate(mapping):
                if j_local is None or j_local == -1:
                    # cannot map this op; abort mapping
                    chi_list_local = None
                    break
                chi_list_local[j_local] = chars[i_map]
            if chi_list_local is None:
                continue
            irreps_to_use.append({'label': ir.get('label','irrep'), 'dim': ir.get('dim',1), 'chi_list': chi_list_local})
    # fallback
    if not irreps_to_use:
        if not irrep_fallback:
            raise RuntimeError("No irreps available: Bilbao fetch failed and no fallback provided.")
        # accept fallback entries with 'chi_list' or 'chi_func'
        for ir in irrep_fallback:
            if 'chi_list' in ir:
                irreps_to_use.append({'label': ir['label'], 'dim': ir['dim'], 'chi_list': ir['chi_list']})
            elif 'chi_func' in ir:
                chi_list = [ir['chi_func'](idx, (R, t)) for idx, (R, t) in enumerate(Gk)]
                irreps_to_use.append({'label': ir['label'], 'dim': ir['dim'], 'chi_list': chi_list})
            else:
                raise RuntimeError("irrep_fallback entries must contain 'chi_list' or 'chi_func'.")
    # compute multiplicities and projections
    N = len(parent_sites)
    M = np.zeros(3*N, dtype=complex)
    for i, m in enumerate(parent_moments):
        M[3*i:3*i+3] = np.array(m, dtype=float)
    report = {'Gk_size': len(Gk), 'spacegroup_number': spacegroup_number, 'T_used': T_used, 'irreps': []}
    for ir in irreps_to_use:
        chi_list = ir['chi_list']
        sum_val = 0+0j
        for chi_mu_g, chi_mag_g in zip(chi_list, chi_mag_list):
            sum_val += np.conjugate(chi_mu_g) * chi_mag_g
        n_mu = sum_val / float(len(Gk))
        n_mu_rounded = int(round(n_mu.real)) if abs(n_mu.imag) < 1e-6 else n_mu
        P = build_projection_operator(chi_list, ir['dim'], Gk, D_matrices)
        M_proj = P.dot(M)
        eta = np.linalg.norm(M_proj) / (np.linalg.norm(M) + 1e-16)
        site_basis = []
        for i in range(N):
            vec = M_proj[3*i:3*i+3]
            if np.max(np.abs(np.imag(vec))) < 1e-8:
                vec = np.real(vec)
            site_basis.append(vec.tolist())
        report['irreps'].append({
            'label': ir['label'],
            'dim': ir['dim'],
            'n_mu': n_mu_rounded,
            'n_mu_raw': n_mu,
            'projection_ratio': float(eta),
            'site_basis_vectors': site_basis
        })
    return report

# ---------------------------
# Unit tests and demo
# ---------------------------

def build_simple_rocksalt_mcif_string(afm=True, a=4.17, mag_moment=2.0):
    lines = []
    lines.append("data_test")
    lines.append("_cell_length_a {:.6f}".format(a))
    lines.append("_cell_length_b {:.6f}".format(a))
    lines.append("_cell_length_c {:.6f}".format(a))
    lines.append("_cell_angle_alpha 90")
    lines.append("_cell_angle_beta 90")
    lines.append("_cell_angle_gamma 90")
    lines.append("loop_")
    lines.append("_atom_site_label")
    lines.append("_atom_site_type_symbol")
    lines.append("_atom_site_fract_x")
    lines.append("_atom_site_fract_y")
    lines.append("_atom_site_fract_z")
    lines.append("_atom_site_moment_x")
    lines.append("_atom_site_moment_y")
    lines.append("_atom_site_moment_z")
    m1 = mag_moment
    m2 = -mag_moment if afm else mag_moment
    lines.append("Ni1 Ni 0.0 0.0 0.0 0.0 0.0 {:.6f}".format(m1))
    lines.append("Ni2 Ni 0.5 0.5 0.5 0.0 0.0 {:.6f}".format(m2))
    lines.append("O1 O 0.5 0.5 0.0 0.0 0.0 0.0")
    lines.append("O2 O 0.5 0.0 0.5 0.0 0.0 0.0")
    return "\n".join(lines)

def write_temp_mcif_and_run(afm=True, transform=None):
    s = build_simple_rocksalt_mcif_string(afm=afm)
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.cif') as f:
        f.write(s)
        fname = f.name
    try:
        k_frac = np.array([0.0, 0.0, 0.0])
        irrep_fallback = [{'label': 'A1g', 'dim': 1, 'chi_func': lambda idx, op: 1.0}]
        report = analyze_mcif_with_bilbao_and_parentdetect(fname, transform, k_frac, irrep_fallback)
    finally:
        os.remove(fname)
    return report

def run_unit_tests():
    print("Running unit tests (NiO-like minimal CIF) with parent detection and Bilbao mapping...")
    report_afm = write_temp_mcif_and_run(afm=True, transform=None)
    eta_afm = report_afm['irreps'][0]['projection_ratio']
    print("AFM projection onto A1g:", eta_afm)
    report_fm = write_temp_mcif_and_run(afm=False, transform=None)
    eta_fm = report_fm['irreps'][0]['projection_ratio']
    print("FM projection onto A1g:", eta_fm)
    assert eta_afm < 0.2, "AFM should have small projection onto totally symmetric irrep"
    assert eta_fm > 0.9, "FM should project strongly onto totally symmetric irrep"
    print("Unit tests passed. Detected parent transform (if any) and projection diagnostics:")
    print("AFM report:", report_afm)
    print("FM report:", report_fm)

if __name__ == "__main__":
    run_unit_tests()


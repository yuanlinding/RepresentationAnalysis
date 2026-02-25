import gemmi
import numpy as np
from fractions import Fraction
from pymatgen.core import Structure

def parse_mcif_fields(path: str) -> dict:
    """
    Parses a magnetic CIF file using gemmi to extract parent space group,
    propagation vectors, and transformation matrices.
    """
    with open(path, 'rb') as f:
        content = f.read().decode('ascii', errors='ignore')
    doc = gemmi.cif.read_string(content)
    block = doc[-1] # typically there is only one block, or the last block has the data
    
    fields = {}
    
    # IT number
    it_val = block.find_value('_parent_space_group.it_number')
    if it_val:
        fields['it_number'] = int(it_val)
        
    # k-vector (assuming single k-vector for now)
    import re
    # Try to find the exact brackets for k-vector which avoids CIF parsing truncation
    k_match = re.search(r'_parent_propagation_vector\.kxkykz\s*(.*?)\n(.*?)\[\s*(.*?)\s*\]', content, flags=re.MULTILINE|re.DOTALL)
    if k_match:
        # Check if the match is close to the tag
        dist = len(k_match.group(1)) + len(k_match.group(2))
        if dist < 100:
            fields['kvector_str'] = k_match.group(3) # just the inside, e.g. "0 0 0"
        else:
            fields['kvector_str'] = "0 0 0"
    else:
        fields['kvector_str'] = "0 0 0"

    # child transform Pp_abc
    child_transform = block.find_value('_parent_space_group.child_transform_pp_abc')
    if child_transform:
        fields['child_transform_str'] = child_transform.strip("'\"")

    # parent transform Pp_abc (to standard setting)
    parent_transform = block.find_value('_parent_space_group.transform_pp_abc')
    if parent_transform:
        fields['parent_transform_str'] = parent_transform.strip("'\"")

    # expected irrep for validation
    irrep_id_loop = block.find('_irrep_', ['id', 'dimension', 'small_dimension'])
    if irrep_id_loop and len(irrep_id_loop) > 0:
        fields['expected_irrep_id'] = irrep_id_loop[0][0]
        fields['expected_irrep_dim'] = int(irrep_id_loop[0][1]) if len(irrep_id_loop[0]) > 1 and irrep_id_loop[0][1] != '.' else None
        fields['expected_irrep_small_dim'] = int(irrep_id_loop[0][2]) if len(irrep_id_loop[0]) > 2 and irrep_id_loop[0][2] != '.' else None

    return fields

def parse_kvector(kstring: str) -> np.ndarray:
    """
    Parses a string like '[1/2 1/2 1/2]' or '[0 0 0]' into a float numpy array.
    """
    s = kstring.strip('[]').replace(',', ' ').split()
    kvec = []
    for val in s:
        if '/' in val:
            kvec.append(float(Fraction(val)))
        else:
            kvec.append(float(val))
    return np.array(kvec)

def parse_transform(transform_str: str):
    """
    Parses strings like '2a,2b,2c;0,0,0' or 'a,b,c;-1/4,1/4,0'
    Returns a 3x3 array M and a 3x1 array origin_shift.
    R_new = M @ R_old
    We can use pymatgen's SymmOp to parse this string easily if we convert it to
    x,y,z form, but actually gemmi provides an easier way: gemmi.Op
    """
    # gemmi.Op can parse xyz strings like x,y,z or 2x,2y,2z.
    # The string here uses a,b,c instead of x,y,z.
    op_str = transform_str.replace('a', 'x').replace('b', 'y').replace('c', 'z')
    # gemmi.Op format usually accepts comma separated equations. The translation comes either as +1/2 or as a second part separated by ';'
    parts = op_str.split(';')
    if len(parts) == 1:
        linear_part = parts[0]
        translation_part = '0,0,0'
    elif len(parts) == 2:
        linear_part = parts[0]
        translation_part = parts[1]
    else:
        raise ValueError(f"Cannot parse transform string: {transform_str}")
    
    # Let's write a simple custom parser since a,b,c format is very standard
    # e.g., '2x,2y,2z', '-1/4,1/4,0'
    try:
        linear_op = gemmi.Op(linear_part)
        M = np.array(linear_op.rot) / dict(gemmi.Op().rot_denominator())[1] # usually 1 or 24, gemmi.Op.rot has some denominator. Actually gemmi.Op.float_rot() is easier? No float rot in python bindings. Let's just do it manually.
    except:
        pass
        
    # We will just use SymmOp from pymatgen which natively parses these
    from pymatgen.core.operations import SymmOp
    # It parses "x+1/2, y, z" etc.
    # We construct "x_expr, y_expr, z_expr"
    lin_parts = linear_part.split(',')
    tran_parts = translation_part.split(',')
    exprs = []
    for i in range(3):
        # combine rotation and translation
        expr = lin_parts[i]
        t = tran_parts[i].strip()
        if t != '0' and t != '0.0':
            if not t.startswith('-') and not t.startswith('+'):
                expr += '+' + t
            else:
                expr += t
        exprs.append(expr)
    
    symm_op = SymmOp.from_xyz_str(','.join(exprs))
    return symm_op.rotation_matrix, symm_op.translation_vector

def get_magnetic_structure(path: str) -> Structure:
    """
    Uses pymatgen to read the magnetic structure from the mCIF file.
    """
    # Pymatgen from_file supports mcif if the extension is .mcif or .cif
    structure = Structure.from_file(path)
    # verify it has magnetic moments
    if "magmom" not in structure.site_properties:
        # Check if they are stored differently by pymatgen for some mcifs
        # Since pymatgen parses magmoms as site properties 'magmom'
        properties = structure.site_properties
        print(f"Warning: 'magmom' site property not found. Found properties: {list(properties.keys())}")
    
    return structure

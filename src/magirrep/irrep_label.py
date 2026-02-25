import numpy as np

def kpoint_label(kpoint):
    """
    Tries to map a kpoint to a high symmetry letter.
    Currently a simple heuristic.
    """
    k = np.abs(np.array(kpoint))
    # Gamma point
    if np.allclose(k, [0, 0, 0], atol=1e-4):
        return "GM"
    # L point in FCC
    elif np.allclose(k, [0.5, 0.5, 0.5], atol=1e-4):
        return "L"
    # X point 
    elif np.allclose(k, [0.5, 0, 0], atol=1e-4) or \
         np.allclose(k, [0, 0.5, 0], atol=1e-4) or \
         np.allclose(k, [0, 0, 0.5], atol=1e-4):
        return "X"
    # M point
    elif np.allclose(k, [0.5, 0.5, 0], atol=1e-4):
        return "M"
    else:
        # Fallback formatting
        return f"[{kpoint[0]:.3f}_{kpoint[1]:.3f}_{kpoint[2]:.3f}]"

def irrep_name(kpoint, sg_number, irrep_idx, irrep_dim):
    """
    Tries to assemble a Bilbao-style label, e.g. mGM5-
    Since spgrep does not provide the exact parity suffix natively,
    we just return a placeholder that has the k-point, index and dim.
    """
    k_label = kpoint_label(kpoint)
    # 1-indexed for the irrep
    idx = irrep_idx + 1
    # m prefix for magnetic
    label = f"m{k_label}{idx}"
    return label

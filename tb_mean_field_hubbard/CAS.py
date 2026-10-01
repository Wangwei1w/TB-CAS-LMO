
# CAS.py
"""
Minimal Hubbard-CAS module for TB/LMO workflows.

Core design
-----------
- Active basis is a common spatial orbital basis C_spatial with shape
  (n_sites, n_active_orb).
- The many-body CAS basis uses spin orbitals generated from these spatial
  orbitals: p_alpha and p_beta.
- Only the total electron number N_elec is fixed by default.
- One-body term is h_pq for the active spatial basis.
- Interaction is a projected site-Hubbard interaction:
      sum_i U_i n_{i alpha} n_{i beta}
  with d_{i sigma} = sum_p C_spatial[i,p] a_{p sigma}.

This is not a full quantum-chemistry CASCI with arbitrary four-index ERI.
It is the Hubbard-CAS appropriate for TB/Hubbard models.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, Sequence[float]]

@dataclass
class HubbardCASResult:
    """Container for Hubbard-CAS calculation output."""

    energies: np.ndarray
    coeffs: np.ndarray
    basis: List[int]
    basis_index: Dict[int, int]
    n_orb: int
    n_elec: int
    h_spatial: np.ndarray
    C_spatial: np.ndarray
    U_site: np.ndarray
    H: np.ndarray

    def excitation_energies(self) -> np.ndarray:
        """Return E_k - E_0 for all computed roots."""
        return self.energies - self.energies[0]

# -----------------------------------------------------------------------------
# Bitstring / determinant utilities
# -----------------------------------------------------------------------------

def count_bits(x: int) -> int:
    """Return the population count of an integer bit string.

    Compatible with older Python versions that do not implement
    int.bit_count().
    """
    return bin(int(x)).count("1")

def spinorb_index(p: int, spin: int, n_orb: int) -> int:
    """
    Spin orbital ordering.

    spin = 0: alpha block, index p
    spin = 1: beta block, index n_orb + p
    """
    if spin not in (0, 1):
        raise ValueError("spin must be 0 for alpha or 1 for beta")
    return p if spin == 0 else n_orb + p

def generate_fock_basis(n_orb: int, n_elec: int) -> List[int]:
    """
    Generate all Slater determinants with fixed total electron number.

    Determinants are encoded as integers over 2*n_orb spin orbitals.
    """
    n_spinorb = 2 * int(n_orb)
    if n_elec < 0 or n_elec > n_spinorb:
        raise ValueError(f"n_elec={n_elec} is invalid for {n_spinorb} spin orbitals")

    basis = []
    for occ in combinations(range(n_spinorb), int(n_elec)):
        det = 0
        for k in occ:
            det |= 1 << k
        basis.append(det)
    return basis

def annihilate(det: int, k: int) -> Optional[Tuple[int, int]]:
    """Apply a_k to determinant. Return (new_det, sign), or None if empty."""
    if ((det >> k) & 1) == 0:
        return None
    lower = det & ((1 << k) - 1)
    sign = -1 if (count_bits(lower) % 2) else 1
    return det & ~(1 << k), sign

def create(det: int, k: int) -> Optional[Tuple[int, int]]:
    """Apply a_k^dagger to determinant. Return (new_det, sign), or None if occupied."""
    if ((det >> k) & 1) == 1:
        return None
    lower = det & ((1 << k) - 1)
    sign = -1 if (count_bits(lower) % 2) else 1
    return det | (1 << k), sign

def apply_one_body_op(det: int, p: int, q: int) -> Optional[Tuple[int, int]]:
    """Apply a_p^dagger a_q."""
    out = annihilate(det, q)
    if out is None:
        return None
    det1, s1 = out
    out = create(det1, p)
    if out is None:
        return None
    det2, s2 = out
    return det2, s1 * s2

def apply_two_body_op(det: int, p: int, q: int, r: int, s: int) -> Optional[Tuple[int, int]]:
    """
    Apply a_p^dagger a_q a_r^dagger a_s, right-to-left.

    This ordering matches the projected Hubbard term
        d_up^dagger d_up d_down^dagger d_down.
    """
    out = annihilate(det, s)
    if out is None:
        return None
    det1, sgn = out

    out = create(det1, r)
    if out is None:
        return None
    det2, s2 = out
    sgn *= s2

    out = annihilate(det2, q)
    if out is None:
        return None
    det3, s3 = out
    sgn *= s3

    out = create(det3, p)
    if out is None:
        return None
    det4, s4 = out
    sgn *= s4

    return det4, sgn

def determinant_occupations(det: int, n_orb: int) -> Tuple[str, str]:
    """Return alpha/beta occupation strings such as '1100', '1010'."""
    alpha = "".join("1" if ((det >> p) & 1) else "0" for p in range(n_orb))
    beta = "".join("1" if ((det >> (n_orb + p)) & 1) else "0" for p in range(n_orb))
    return alpha, beta

# -----------------------------------------------------------------------------
# Hamiltonian construction
# -----------------------------------------------------------------------------

def validate_hubbard_cas_inputs(
    h_spatial: ArrayLike,
    C_spatial: ArrayLike,
    U_site: Union[float, ArrayLike],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(h_spatial, dtype=complex)
    C = np.asarray(C_spatial, dtype=complex)

    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h_spatial must be a square matrix")
    if C.ndim != 2:
        raise ValueError("C_spatial must be a 2D array with shape (n_sites, n_orb)")
    if C.shape[1] != h.shape[0]:
        raise ValueError(
            f"C_spatial.shape[1]={C.shape[1]} must equal h_spatial.shape[0]={h.shape[0]}"
        )

    n_sites = C.shape[0]
    if np.isscalar(U_site):
        U = np.full(n_sites, float(U_site), dtype=float)
    else:
        U = np.asarray(U_site, dtype=float)
        if U.shape != (n_sites,):
            raise ValueError(f"U_site must have shape ({n_sites},), got {U.shape}")

    h = 0.5 * (h + h.conj().T)
    return h, C, U

def build_hubbard_cas_hamiltonian(
    h_spatial: ArrayLike,
    C_spatial: ArrayLike,
    U_site: Union[float, ArrayLike],
    n_elec: int,
    interaction_cutoff: float = 1e-12,
) -> Tuple[np.ndarray, List[int], Dict[int, int]]:
    """
    Build the Hubbard-CAS Hamiltonian in a fixed-N Fock basis.

    Parameters
    ----------
    h_spatial
        One-body Hamiltonian in the active common spatial orbital basis.
    C_spatial
        Active common spatial orbitals in the original site basis, shape
        (n_sites, n_active_orb). Columns must be orthonormal or near-orthonormal.
    U_site
        On-site Hubbard U per site, or scalar U for all sites.
    n_elec
        Total active electron number. N_alpha and N_beta are not fixed.
    interaction_cutoff
        Skip projected Hubbard matrix elements below this magnitude.

    Returns
    -------
    H, basis, basis_index
    """
    h, C, U = validate_hubbard_cas_inputs(h_spatial, C_spatial, U_site)
    n_orb = h.shape[0]

    basis = generate_fock_basis(n_orb=n_orb, n_elec=n_elec)
    basis_index = {det: i for i, det in enumerate(basis)}
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)

    # One-body part: sum_sigma sum_pq h_pq a_pσ^† a_qσ
    for j, det in enumerate(basis):
        for spin in (0, 1):
            for p in range(n_orb):
                pso = spinorb_index(p, spin, n_orb)
                for q in range(n_orb):
                    val = h[p, q]
                    if abs(val) < interaction_cutoff:
                        continue
                    qso = spinorb_index(q, spin, n_orb)
                    out = apply_one_body_op(det, pso, qso)
                    if out is None:
                        continue
                    det2, sign = out
                    i = basis_index.get(det2)
                    if i is not None:
                        H[i, j] += val * sign

    # Projected Hubbard interaction:
    # sum_i U_i sum_pqrs C_ip^* C_iq C_ir^* C_is
    #     a_pα^† a_qα a_rβ^† a_sβ
    Cc = C.conj()
    for site in range(C.shape[0]):
        Ui = U[site]
        if abs(Ui) < interaction_cutoff:
            continue
        # W[p,q,r,s] factorized by site. Loops are acceptable for small CAS.
        for p in range(n_orb):
            for q in range(n_orb):
                left = Ui * Cc[site, p] * C[site, q]
                if abs(left) < interaction_cutoff:
                    continue
                pso = spinorb_index(p, 0, n_orb)
                qso = spinorb_index(q, 0, n_orb)
                for r in range(n_orb):
                    for s in range(n_orb):
                        val = left * Cc[site, r] * C[site, s]
                        if abs(val) < interaction_cutoff:
                            continue
                        rso = spinorb_index(r, 1, n_orb)
                        sso = spinorb_index(s, 1, n_orb)
                        for j, det in enumerate(basis):
                            out = apply_two_body_op(det, pso, qso, rso, sso)
                            if out is None:
                                continue
                            det2, sign = out
                            i = basis_index.get(det2)
                            if i is not None:
                                H[i, j] += val * sign

    H = 0.5 * (H + H.conj().T)
    return H, basis, basis_index

def diagonalize_hubbard_cas(
    h_spatial: ArrayLike,
    C_spatial: ArrayLike,
    U_site: Union[float, ArrayLike],
    n_elec: int,
    n_roots: int = 3,
    interaction_cutoff: float = 1e-12,
) -> HubbardCASResult:
    """Build and diagonalize the Hubbard-CAS Hamiltonian."""
    h, C, U = validate_hubbard_cas_inputs(h_spatial, C_spatial, U_site)
    H, basis, basis_index = build_hubbard_cas_hamiltonian(
        h_spatial=h,
        C_spatial=C,
        U_site=U,
        n_elec=n_elec,
        interaction_cutoff=interaction_cutoff,
    )

    evals, evecs = np.linalg.eigh(H)
    n_roots = min(int(n_roots), len(evals))

    return HubbardCASResult(
        energies=np.real_if_close(evals[:n_roots]),
        coeffs=evecs[:, :n_roots],
        basis=basis,
        basis_index=basis_index,
        n_orb=h.shape[0],
        n_elec=int(n_elec),
        h_spatial=h,
        C_spatial=C,
        U_site=U,
        H=H,
    )

# -----------------------------------------------------------------------------
# RDM / natural orbitals
# -----------------------------------------------------------------------------

def compute_spin_resolved_1rdm(
    ci_vec: ArrayLike,
    basis: Sequence[int],
    basis_index: Dict[int, int],
    n_orb: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gamma_alpha[p,q] and gamma_beta[p,q] = <a_pσ^† a_qσ>.
    """
    c = np.asarray(ci_vec, dtype=complex)
    gamma = [np.zeros((n_orb, n_orb), dtype=complex), np.zeros((n_orb, n_orb), dtype=complex)]

    for j, det in enumerate(basis):
        cj = c[j]
        if abs(cj) < 1e-14:
            continue
        for spin in (0, 1):
            for p in range(n_orb):
                pso = spinorb_index(p, spin, n_orb)
                for q in range(n_orb):
                    qso = spinorb_index(q, spin, n_orb)
                    out = apply_one_body_op(det, pso, qso)
                    if out is None:
                        continue
                    det2, sign = out
                    i = basis_index.get(det2)
                    if i is None:
                        continue
                    gamma[spin][p, q] += np.conj(c[i]) * cj * sign

    return gamma[0], gamma[1]

def compute_spin_summed_1rdm(
    ci_vec: ArrayLike,
    basis: Sequence[int],
    basis_index: Dict[int, int],
    n_orb: int,
) -> np.ndarray:
    """Compute gamma[p,q] = gamma_alpha[p,q] + gamma_beta[p,q]."""
    ga, gb = compute_spin_resolved_1rdm(ci_vec, basis, basis_index, n_orb)
    return ga + gb

def compute_natural_orbitals_from_1rdm(gamma: ArrayLike, sort: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Natural orbitals from spin-summed spatial 1-RDM.

    Returns
    -------
    natorbs
        Columns are natural orbitals in the input active spatial basis.
    occs
        Natural occupations, typically between 0 and 2.
    """
    G = np.asarray(gamma, dtype=complex)
    G = 0.5 * (G + G.conj().T)
    occs, natorbs = np.linalg.eigh(G)
    if sort:
        idx = np.argsort(np.real(occs))[::-1]
        occs = occs[idx]
        natorbs = natorbs[:, idx]
    return natorbs, np.real_if_close(occs)

def compute_state_rdms_and_natorbs(result: HubbardCASResult, root: int = 0) -> Dict[str, np.ndarray]:
    """Compute spin-resolved 1-RDM, spin-summed 1-RDM, and natural orbitals."""
    ci = result.coeffs[:, root]
    gamma_a, gamma_b = compute_spin_resolved_1rdm(
        ci_vec=ci,
        basis=result.basis,
        basis_index=result.basis_index,
        n_orb=result.n_orb,
    )
    gamma = gamma_a + gamma_b
    natorbs_active, occs = compute_natural_orbitals_from_1rdm(gamma)
    natorbs_site = result.C_spatial @ natorbs_active
    return {
        "gamma_alpha": gamma_a,
        "gamma_beta": gamma_b,
        "gamma": gamma,
        "natural_orbitals_active": natorbs_active,
        "natural_orbitals_site": natorbs_site,
        "natural_occupations": occs,
    }

# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------

def ci_expansion_dataframe(
    result: HubbardCASResult,
    root: int = 0,
    coeff_cutoff: float = 1e-4,
    max_terms: Optional[int] = None,
) -> pd.DataFrame:
    """Return dominant CI coefficients for one root as a DataFrame."""
    coeff = result.coeffs[:, root]
    rows = []
    for det, amp in zip(result.basis, coeff):
        if abs(amp) < coeff_cutoff:
            continue
        alpha, beta = determinant_occupations(det, result.n_orb)
        rows.append({
            "alpha_occ": alpha,
            "beta_occ": beta,
            "coeff": amp,
            "weight": abs(amp) ** 2,
        })

    rows = sorted(rows, key=lambda x: x["weight"], reverse=True)
    if max_terms is not None:
        rows = rows[:max_terms]
    return pd.DataFrame(rows)

def print_cas_summary(
    result: HubbardCASResult,
    coeff_cutoff: float = 1e-3,
    max_terms: int = 20,
) -> None:
    """Print energies and dominant CI expansions for all computed roots."""
    print("=== Hubbard-CAS energies ===")
    for k, E in enumerate(result.energies):
        dE = E - result.energies[0]
        label = "ground" if k == 0 else f"excited {k}"
        print(f"root {k:2d} ({label:>9s}): E = {E:.12f},  dE = {dE:.12f}")

    for k in range(len(result.energies)):
        print(f"\n=== CI expansion: root {k} ===")
        df = ci_expansion_dataframe(
            result,
            root=k,
            coeff_cutoff=coeff_cutoff,
            max_terms=max_terms,
        )
        if df.empty:
            print(f"No coefficients above cutoff {coeff_cutoff}")
        else:
            print(df.to_string(index=False))

# -----------------------------------------------------------------------------
# Bridges to existing mfh_model / LMO.py workflow
# -----------------------------------------------------------------------------

def compute_h_eff_in_spatial_basis_from_mfh_model(
    mfh_model,
    C_spatial: ArrayLike,
    spin: int,
    target_state_indices: Sequence[int],
) -> np.ndarray:
    """
    Compute H_eff = C^dagger E C in a supplied spatial basis using mfh_model MOs.

    This mirrors the LMO.py logic without importing LMO.py.
    """
    C_spatial = np.asarray(C_spatial, dtype=complex)
    Psi = np.column_stack([
        np.asarray(mfh_model.evecs[spin][i], dtype=complex)
        for i in target_state_indices
    ])
    E = np.diag([mfh_model.evals[spin][i] for i in target_state_indices])
    C_proj = Psi.conj().T @ C_spatial
    H_eff = C_proj.conj().T @ E @ C_proj
    return 0.5 * (H_eff + H_eff.conj().T)

def compute_spin_averaged_h_spatial_from_mfh_model(
    mfh_model,
    C_spatial: ArrayLike,
    orb_indices_up: Sequence[int],
    orb_indices_dn: Sequence[int],
) -> np.ndarray:
    """Spin-averaged one-body Hamiltonian in the common spatial basis."""
    H_up = compute_h_eff_in_spatial_basis_from_mfh_model(
        mfh_model=mfh_model,
        C_spatial=C_spatial,
        spin=0,
        target_state_indices=orb_indices_up,
    )
    H_dn = compute_h_eff_in_spatial_basis_from_mfh_model(
        mfh_model=mfh_model,
        C_spatial=C_spatial,
        spin=1,
        target_state_indices=orb_indices_dn,
    )
    return 0.5 * (H_up + H_dn)

def run_hubbard_cas_from_common_spatial_basis(
    h_spatial: ArrayLike,
    C_spatial: ArrayLike,
    U_site: Union[float, ArrayLike],
    n_elec: int,
    n_roots: int = 3,
    print_summary: bool = True,
    coeff_cutoff: float = 1e-3,
) -> Dict[str, object]:
    """
    Convenience wrapper for the first-stage workflow.

    Returns result plus per-root RDM/natural-orbital dictionaries.
    """
    result = diagonalize_hubbard_cas(
        h_spatial=h_spatial,
        C_spatial=C_spatial,
        U_site=U_site,
        n_elec=n_elec,
        n_roots=n_roots,
    )

    if print_summary:
        print_cas_summary(result, coeff_cutoff=coeff_cutoff)

    state_data = [compute_state_rdms_and_natorbs(result, root=k) for k in range(len(result.energies))]

    return {
        "result": result,
        "state_data": state_data,
    }

def run_hubbard_cas_from_mfh_common_data(
    mfh_model,
    common_data: Dict[str, object],
    U_site: Union[float, ArrayLike],
    n_elec: int,
    n_roots: int = 3,
    C_key: str = "C_common",
    print_summary: bool = True,
    coeff_cutoff: float = 1e-3,
) -> Dict[str, object]:
    """
    Run Hubbard-CAS from common_data produced by LMO.py.

    Required common_data keys:
        - C_key, default "C_common"
        - paired_up
        - paired_dn
    """
    C_spatial = np.asarray(common_data[C_key], dtype=complex)
    orb_indices_up = common_data["paired_up"]
    orb_indices_dn = common_data["paired_dn"]

    h_spatial = compute_spin_averaged_h_spatial_from_mfh_model(
        mfh_model=mfh_model,
        C_spatial=C_spatial,
        orb_indices_up=orb_indices_up,
        orb_indices_dn=orb_indices_dn,
    )

    cas_data = run_hubbard_cas_from_common_spatial_basis(
        h_spatial=h_spatial,
        C_spatial=C_spatial,
        U_site=U_site,
        n_elec=n_elec,
        n_roots=n_roots,
        print_summary=print_summary,
        coeff_cutoff=coeff_cutoff,
    )
    cas_data["h_spatial"] = h_spatial
    cas_data["C_spatial"] = C_spatial
    return cas_data

def select_partially_occupied_natural_orbitals(
    natural_occupations: ArrayLike,
    occ_min: float = 1e-3,
    occ_max: float = 2.0 - 1e-3,
) -> np.ndarray:
    """Return indices of natural orbitals with nontrivial fractional occupation."""
    occ = np.asarray(natural_occupations, dtype=float)
    return np.where((occ > occ_min) & (occ < occ_max))[0]

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def diagonalize_tb_site_hamiltonian(
    h_site: ArrayLike,
    sort: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize a spin-independent U=0 TB Hamiltonian in the site basis.

    Parameters
    ----------
    h_site
        Site-basis one-body Hamiltonian, shape (n_sites, n_sites).
    sort
        If True, sort eigenvalues increasingly.

    Returns
    -------
    evals
        Canonical TB orbital energies.
    evecs
        Canonical TB orbitals in site basis. Columns are orbitals.
    """
    h = np.asarray(h_site, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h_site must be a square matrix")

    h = 0.5 * (h + h.conj().T)
    evals, evecs = np.linalg.eigh(h)

    if sort:
        idx = np.argsort(np.real(evals))
        evals = evals[idx]
        evecs = evecs[:, idx]

    return np.real_if_close(evals), evecs

def build_active_space_from_tb_canonical_orbitals(
    h_site: ArrayLike,
    CAS_indices: Sequence[int],
) -> Dict[str, np.ndarray]:
    """
    Build a CAS active spatial basis from U=0 TB canonical orbitals.

    Parameters
    ----------
    h_site
        Spin-independent U=0 TB Hamiltonian in site basis.
    CAS_indices
        Orbital indices selected from the ascending-energy canonical TB orbitals.

    Returns
    -------
    dict with:
        - evals_all
        - C_all
        - CAS_indices
        - active_energies
        - C_spatial
        - h_spatial

    Notes
    -----
    In this canonical basis, h_spatial is diagonal with selected orbital energies.
    The Hubbard interaction is still supplied later as U_site and projected into
    the active orbital basis by build_hubbard_cas_hamiltonian(...).
    """
    evals, C_all = diagonalize_tb_site_hamiltonian(h_site)
    CAS_indices = np.asarray(CAS_indices, dtype=int)

    if CAS_indices.ndim != 1:
        raise ValueError("CAS_indices must be a one-dimensional list/array")
    if len(CAS_indices) == 0:
        raise ValueError("CAS_indices cannot be empty")
    if np.min(CAS_indices) < 0 or np.max(CAS_indices) >= len(evals):
        raise ValueError(
            f"CAS_indices must lie between 0 and {len(evals)-1}; got {CAS_indices}"
        )

    active_energies = np.asarray(evals[CAS_indices], dtype=complex)
    C_spatial = np.asarray(C_all[:, CAS_indices], dtype=complex)
    h_spatial = np.diag(active_energies)

    return {
        "evals_all": evals,
        "C_all": C_all,
        "CAS_indices": CAS_indices,
        "active_energies": active_energies,
        "C_spatial": C_spatial,
        "h_spatial": h_spatial,
    }

def run_hubbard_cas_from_tb_canonical_orbitals(
    h_site: ArrayLike,
    U_site: Union[float, ArrayLike],
    CAS_indices: Sequence[int],
    n_elec: int,
    n_roots: int = 3,
    print_summary: bool = True,
    coeff_cutoff: float = 1e-3,
) -> Dict[str, object]:
    """
    Run Hubbard-CAS using U=0 TB canonical orbitals as the active basis.

    This is the benchmark-style route:
        1. diagonalize spin-independent U=0 TB Hamiltonian h_site
        2. select active canonical MOs by CAS_indices
        3. build h_spatial = diag(epsilon_active)
        4. project site Hubbard U into this active basis
        5. diagonalize the many-body CAS Hamiltonian

    The printed CI expansion is therefore in the selected U=0 TB canonical MO
    determinant basis.
    """
    active = build_active_space_from_tb_canonical_orbitals(
        h_site=h_site,
        CAS_indices=CAS_indices,
    )

    cas_data = run_hubbard_cas_from_common_spatial_basis(
        h_spatial=active["h_spatial"],
        C_spatial=active["C_spatial"],
        U_site=U_site,
        n_elec=n_elec,
        n_roots=n_roots,
        print_summary=print_summary,
        coeff_cutoff=coeff_cutoff,
    )

    cas_data.update(active)
    cas_data["basis_mode"] = "u0_tb_canonical"
    return cas_data

def get_h_site_from_object(obj, attr_candidates: Optional[Sequence[str]] = None) -> np.ndarray:
    """
    Best-effort helper to extract a site-basis TB Hamiltonian from an object.

    This is intentionally conservative because different TB model classes use
    different attribute names. Prefer passing h_site explicitly when possible.
    """
    if attr_candidates is None:
        attr_candidates = (
            "h_site", "H_site", "h_tb", "H_tb", "hamiltonian", "H",
            "h", "ham", "_hamiltonian",
        )

    for name in attr_candidates:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return arr

    raise AttributeError(
        "Could not find a square site-basis Hamiltonian on the supplied object. "
        "Pass h_site explicitly to run_hubbard_cas_from_tb_canonical_orbitals(...)."
    )

def run_hubbard_cas_from_mfh_u0_tb_canonical(
    mfh_model,
    U_site: Union[float, ArrayLike],
    CAS_indices: Sequence[int],
    n_elec: int,
    n_roots: int = 3,
    h_site: Optional[ArrayLike] = None,
    print_summary: bool = True,
    coeff_cutoff: float = 1e-3,
) -> Dict[str, object]:
    """
    Convenience wrapper for U=0 TB canonical-orbital CAS from an mfh_model-like object.

    Recommended usage is to pass h_site explicitly if the model's site-basis
    U=0 Hamiltonian has a known variable name. If h_site is None, this function
    tries to extract a square Hamiltonian from mfh_model or mfh_model.model.
    """
    if h_site is None:
        try:
            h_site = get_h_site_from_object(mfh_model)
        except AttributeError:
            if hasattr(mfh_model, "model"):
                h_site = get_h_site_from_object(mfh_model.model)
            elif hasattr(mfh_model, "pp"):
                h_site = get_h_site_from_object(mfh_model.pp)
            else:
                raise

    return run_hubbard_cas_from_tb_canonical_orbitals(
        h_site=h_site,
        U_site=U_site,
        CAS_indices=CAS_indices,
        n_elec=n_elec,
        n_roots=n_roots,
        print_summary=print_summary,
        coeff_cutoff=coeff_cutoff,
    )

def build_active_space_from_mfh_eigensystem(
    mfh_model,
    CAS_indices: Sequence[int],
    spin: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Build a CAS active basis directly from mfh_model.evals/evecs.

    Use this when you already have a spin-independent U=0 TB calculation stored
    as mfh_model.evals[spin] and mfh_model.evecs[spin]. In that case the active
    basis is exactly the selected canonical orbitals from that eigensystem.

    Parameters
    ----------
    mfh_model
        Object with mfh_model.evals[spin][i] and mfh_model.evecs[spin][i].
    CAS_indices
        Selected canonical orbital indices.
    spin
        Which eigensystem to read. For a true U=0 spin-independent calculation,
        spin=0 and spin=1 should be identical up to phase.
    """
    if not hasattr(mfh_model, "evals") or not hasattr(mfh_model, "evecs"):
        raise AttributeError("mfh_model must provide evals and evecs attributes")

    evals_spin = np.asarray(mfh_model.evals[spin])
    CAS_indices = np.asarray(CAS_indices, dtype=int)

    if CAS_indices.ndim != 1:
        raise ValueError("CAS_indices must be a one-dimensional list/array")
    if len(CAS_indices) == 0:
        raise ValueError("CAS_indices cannot be empty")
    if np.min(CAS_indices) < 0 or np.max(CAS_indices) >= len(evals_spin):
        raise ValueError(
            f"CAS_indices must lie between 0 and {len(evals_spin)-1}; got {CAS_indices}"
        )

    C_spatial = np.column_stack([
        np.asarray(mfh_model.evecs[spin][int(i)], dtype=complex)
        for i in CAS_indices
    ])
    active_energies = np.asarray(evals_spin[CAS_indices], dtype=complex)
    h_spatial = np.diag(active_energies)

    return {
        "CAS_indices": CAS_indices,
        "spin_used": spin,
        "active_energies": active_energies,
        "C_spatial": C_spatial,
        "h_spatial": h_spatial,
    }

def run_hubbard_cas_from_mfh_eigensystem_canonical(
    mfh_model,
    U_site: Union[float, ArrayLike],
    CAS_indices: Sequence[int],
    n_elec: int,
    n_roots: int = 3,
    spin: int = 0,
    print_summary: bool = True,
    coeff_cutoff: float = 1e-3,
) -> Dict[str, object]:
    """
    Run Hubbard-CAS directly from mfh_model.evals/evecs canonical orbitals.

    The canonical-orbital workflow is:
        1. prepare a U=0 / spin-independent model so evecs[0] == evecs[1]
        2. choose CAS_indices from mfh_model.evecs[spin]
        3. use h_spatial = diag(selected evals)
        4. project U_site into the selected orbital basis
        5. diagonalize Hubbard-CAS

    No external variable named h_site is required.
    """
    active = build_active_space_from_mfh_eigensystem(
        mfh_model=mfh_model,
        CAS_indices=CAS_indices,
        spin=spin,
    )

    cas_data = run_hubbard_cas_from_common_spatial_basis(
        h_spatial=active["h_spatial"],
        C_spatial=active["C_spatial"],
        U_site=U_site,
        n_elec=n_elec,
        n_roots=n_roots,
        print_summary=print_summary,
        coeff_cutoff=coeff_cutoff,
    )

    cas_data.update(active)
    cas_data["basis_mode"] = "mfh_eigensystem_canonical"
    return cas_data

def plot_cas_natural_orbitals(
    mfh_model,
    cas_data,
    root=0,
    save_dir="cas_natural_orbitals",
    prefix="CAS",
    show=True,
):
    """
    Plot and save CAS natural orbitals for a selected root.

    Uses the actual keys produced by compute_state_rdms_and_natorbs:
        natural_orbitals_active
        natural_orbitals_site
        natural_occupations

    Returns
    -------
    C_no_site : ndarray
        Natural orbitals in the original site basis, columns are NOs.
    occ : ndarray
        Natural occupations.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    state = cas_data["state_data"][root]
    occ = np.asarray(state["natural_occupations"])

    if "natural_orbitals_site" in state:
        C_no_site = np.asarray(state["natural_orbitals_site"])
    elif "natural_orbitals_active" in state:
        C_active = np.asarray(cas_data["C_spatial"])
        U_no = np.asarray(state["natural_orbitals_active"])
        C_no_site = C_active @ U_no
    else:
        raise KeyError(
            "state_data[root] does not contain natural_orbitals_site or "
            "natural_orbitals_active. Available keys: " + str(list(state.keys()))
        )

    n_no = C_no_site.shape[1]

    fig, axs = plt.subplots(1, n_no, figsize=(5 * n_no, 4))
    if n_no == 1:
        axs = [axs]

    for i in range(n_no):
        evec = C_no_site[:, i]
        title = f"Root {root} NO {i}, occ={occ[i]:.4f}"
        mfh_model.pp.plot_eigenvector(axs[i], evec, title=title)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    for i in range(n_no):
        fig, ax = plt.subplots(figsize=(5, 4))
        evec = C_no_site[:, i]
        mfh_model.pp.plot_eigenvector(
            ax,
            evec,
            title=f"NO {i}, occ={occ[i]:.4f}",
        )
        plt.tight_layout()
        filename = os.path.join(
            save_dir,
            f"{prefix}_root{root}_NO_{i}_occ_{occ[i]:.4f}.pdf",
        )
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {filename}")

    return C_no_site, occ

def print_cas_natural_orbital_expansion(cas_data, root=0, cutoff=1e-3):
    """
    Print natural orbitals in the imported active MO basis.
    NO_i = sum_p natural_orbitals_active[p, i] * active_MO_p
    """
    import numpy as np
    import pandas as pd

    state = cas_data["state_data"][root]
    U_no = np.asarray(state["natural_orbitals_active"])
    occ = np.asarray(state["natural_occupations"])

    active_indices = cas_data.get("CAS_indices", None)
    if active_indices is None:
        active_indices = list(range(U_no.shape[0]))

    rows = []
    for i in range(U_no.shape[1]):
        for p in range(U_no.shape[0]):
            c = U_no[p, i]
            if abs(c) >= cutoff:
                rows.append({
                    "NO": i,
                    "NO_occ": occ[i],
                    "active_MO": active_indices[p],
                    "coeff": c,
                    "abs_coeff": abs(c),
                })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df

def _cas_count_bits(x):
    return bin(int(x)).count("1")

def _cas_annihilate(det, k):
    if not ((det >> k) & 1):
        return None
    lower = det & ((1 << k) - 1)
    sign = -1 if (_cas_count_bits(lower) % 2) else 1
    return det & ~(1 << k), sign

def _cas_create(det, k):
    if (det >> k) & 1:
        return None
    lower = det & ((1 << k) - 1)
    sign = -1 if (_cas_count_bits(lower) % 2) else 1
    return det | (1 << k), sign

def _cas_apply_ops_right_to_left(det, ops):
    """
    ops: list of ("ann"/"cre", spinorb_index), applied from left to right
         as an operation sequence already ordered right-to-left.
    """
    phase = 1
    cur = det

    for op, k in ops:
        if op == "ann":
            out = _cas_annihilate(cur, k)
        elif op == "cre":
            out = _cas_create(cur, k)
        else:
            raise ValueError(f"Unknown op: {op}")

        if out is None:
            return None

        cur, s = out
        phase *= s

    return cur, phase

def _cas_spinorb(p, spin, n_orb):
    """
    spin=0: alpha
    spin=1: beta
    """
    return p if spin == 0 else n_orb + p

def _cas_get_ci_vector(result, root):
    coeffs = np.asarray(result.coeffs)
    n_basis = len(result.basis)

    if coeffs.shape[0] == n_basis:
        return coeffs[:, root]
    elif coeffs.shape[1] == n_basis:
        return coeffs[root, :]
    else:
        raise ValueError(
            f"Cannot infer CI coefficient layout: coeffs.shape={coeffs.shape}, "
            f"n_basis={n_basis}"
        )

def _cas_expect_diagonal(vec, basis, values):
    val = 0.0 + 0.0j
    for i, det in enumerate(basis):
        val += np.conj(vec[i]) * vec[i] * values(det)
    return val

def _cas_expect_operator(vec, basis, basis_index, ops_builder):
    """
    Generic expectation value <Psi|O|Psi>.
    ops_builder(det) should return:
        None
    or
        (new_det, phase)
    """
    val = 0.0 + 0.0j

    for j, det in enumerate(basis):
        out = ops_builder(det)
        if out is None:
            continue

        new_det, phase = out
        i = basis_index.get(new_det, None)
        if i is None:
            continue

        val += np.conj(vec[i]) * phase * vec[j]

    return val

def compute_spin_spin_correlation_matrix(cas_data, root=0):
    """
    Compute active-orbital spin-spin correlation matrix:

        C_ij = < S_i · S_j >

    where i,j are active spatial orbitals.

    Uses:
        S_i · S_j =
            S_i^z S_j^z
          + 1/2 (S_i^+ S_j^- + S_i^- S_j^+)

    Returns
    -------
    corr : ndarray, shape=(n_orb, n_orb)
        Active-orbital spin-spin correlation matrix.
    s2 : float
        Total <S^2> = sum_ij <S_i · S_j>.
    """

    result = cas_data["result"]
    basis = list(result.basis)
    basis_index = dict(result.basis_index)

    vec = _cas_get_ci_vector(result, root)
    vec = np.asarray(vec, dtype=complex)

    if "C_spatial" in cas_data:
        n_orb = np.asarray(cas_data["C_spatial"]).shape[1]
    elif "h_spatial" in cas_data:
        n_orb = np.asarray(cas_data["h_spatial"]).shape[0]
    else:
        raise KeyError("Cannot infer n_orb: cas_data needs C_spatial or h_spatial.")

    corr = np.zeros((n_orb, n_orb), dtype=complex)

    for i in range(n_orb):
        ia = _cas_spinorb(i, 0, n_orb)
        ib = _cas_spinorb(i, 1, n_orb)

        for j in range(n_orb):
            ja = _cas_spinorb(j, 0, n_orb)
            jb = _cas_spinorb(j, 1, n_orb)

            # Sz_i Sz_j = 1/4 (n_iα - n_iβ)(n_jα - n_jβ)
            def szsz_value(det, ia=ia, ib=ib, ja=ja, jb=jb):
                ni_a = (det >> ia) & 1
                ni_b = (det >> ib) & 1
                nj_a = (det >> ja) & 1
                nj_b = (det >> jb) & 1
                return 0.25 * (ni_a - ni_b) * (nj_a - nj_b)

            szsz = _cas_expect_diagonal(vec, basis, szsz_value)

            # S_i^+ S_j^- = a†_{iα} a_{iβ} a†_{jβ} a_{jα}
            def sp_sm(det, ia=ia, ib=ib, ja=ja, jb=jb):
                return _cas_apply_ops_right_to_left(
                    det,
                    [
                        ("ann", ja),
                        ("cre", jb),
                        ("ann", ib),
                        ("cre", ia),
                    ],
                )

            # S_i^- S_j^+ = a†_{iβ} a_{iα} a†_{jα} a_{jβ}
            def sm_sp(det, ia=ia, ib=ib, ja=ja, jb=jb):
                return _cas_apply_ops_right_to_left(
                    det,
                    [
                        ("ann", jb),
                        ("cre", ja),
                        ("ann", ia),
                        ("cre", ib),
                    ],
                )

            flip = 0.5 * (
                _cas_expect_operator(vec, basis, basis_index, sp_sm)
                + _cas_expect_operator(vec, basis, basis_index, sm_sp)
            )

            corr[i, j] = szsz + flip

    corr = np.real_if_close(corr)

# -------------------------------------------------
# connected spin-spin correlation
#

# -------------------------------------------------

    n_orb = corr.shape[0]

# <S_i^z>
    sz_expect = np.zeros(n_orb, dtype=float)

    for i in range(n_orb):

        ia = _cas_spinorb(i, 0, n_orb)
        ib = _cas_spinorb(i, 1, n_orb)

        def sz_value(det, ia=ia, ib=ib):
            ni_a = (det >> ia) & 1
            ni_b = (det >> ib) & 1
            return 0.5 * (ni_a - ni_b)

        sz_expect[i] = np.real_if_close(
            _cas_expect_diagonal(vec, basis, sz_value)
        )

# connected correlation
    corr_connected = corr - np.outer(sz_expect, sz_expect)

    corr_connected = np.real_if_close(corr_connected)

# total S^2 from raw correlation
    s2 = np.real_if_close(np.sum(corr))

    return (
        np.asarray(corr_connected, dtype=float),
        float(s2),
    )

def print_spin_spin_correlation(cas_data, root=0, labels=None, precision=6):
    corr, s2 = compute_spin_spin_correlation_matrix(cas_data, root=root)

    np.set_printoptions(precision=precision, suppress=True)

    print(f"=== Spin-spin correlation <S_i · S_j>-<S_i> · <S_j>, root {root} ===")
    print(corr)
    print("")
    print(f"<S^2> = sum_ij <S_i · S_j>-<S_i> · <S_j> = {s2:.12f}")

    if labels is not None:
        import pandas as pd
        df = pd.DataFrame(corr, index=labels, columns=labels)
        return df

    return corr, s2

def plot_spin_spin_correlation(
    cas_data,
    root=0,
    labels=None,
    title=None,
    save_path=None,
    show=True,
):
    import matplotlib.pyplot as plt

    corr, s2 = compute_spin_spin_correlation_matrix(cas_data, root=root)

    fig, ax = plt.subplots(figsize=(5, 4))
    corr = np.array(corr, dtype=float)
    corr[np.abs(corr) < 1e-12] = 0.0

    im = ax.imshow(corr, vmin=-0.75, vmax=0.75)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = corr.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    if title is None:
        title = f"Root {root}: <S_i·S_j>-<S_i> · <S_j>, <S²>={s2:.4f}"

    ax.set_title(title)
    ax.set_xlabel("active orbital j")
    ax.set_ylabel("active orbital i")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return corr, s2

def build_cas_lmo_observables(cas_data, C_lmo, root=0):
    """
    Build LMO-basis observables from CAS result.

    Returns an obs dict compatible with LMO.py functions:
        H_eff_avg
        B
        n_tot
        t_eff_abs
        J2_proxy = |t| |B|

    Notes:
        For spin-adapted CAS singlet, m = 0 by construction.
    """
    import numpy as np

    C_active = np.asarray(cas_data["C_spatial"], dtype=complex)
    h_active = np.asarray(cas_data["h_spatial"], dtype=complex)
    C_lmo = np.asarray(C_lmo, dtype=complex)

    # LMO in active basis
    U_active_to_lmo = C_active.conj().T @ C_lmo

    # one-body Hamiltonian in LMO basis
    H_lmo = U_active_to_lmo.conj().T @ h_active @ U_active_to_lmo

    state = cas_data["state_data"][root]

    # reconstruct spin-summed 1-RDM in active basis from natural orbitals
    U_no = np.asarray(state["natural_orbitals_active"], dtype=complex)
    occ = np.asarray(state["natural_occupations"], dtype=float)

    gamma_active = U_no @ np.diag(occ) @ U_no.conj().T

    # spin-summed 1-RDM in LMO basis
    gamma_lmo = U_active_to_lmo.conj().T @ gamma_active @ U_active_to_lmo

    n_tot = np.real(np.diag(gamma_lmo))

    # For spin-adapted singlet: n_up = n_dn = n_tot / 2
    n_up = 0.5 * n_tot
    n_dn = 0.5 * n_tot
    m = n_up - n_dn

    B = np.real(gamma_lmo)

    t_eff_abs = np.abs(H_lmo)
    B_abs = np.abs(B)

    J2_proxy = t_eff_abs * B_abs
    J1_proxy = np.zeros_like(J2_proxy)

    return {
        "H_eff_avg": H_lmo,
        "H_eff_charge": H_lmo,
        "H_eff_up": H_lmo,
        "H_eff_dn": H_lmo,
        "rho": gamma_lmo,
        "B": B,
        "B_charge": B,
        "n_tot": n_tot,
        "n_up": n_up,
        "n_dn": n_dn,
        "m": m,
        "t_eff_abs": t_eff_abs,
        "J1_proxy": J1_proxy,
        "J2_proxy": J2_proxy,
        "U_active_to_lmo": U_active_to_lmo,
    }

def rerun_cas_in_lmo_basis(cas_data, C_lmo, U_site, n_elec, n_roots=4, print_summary=True):
    """
    Re-run Hubbard-CAS using CAS-LMO as active spatial basis.

    This allows spin-spin correlation to be computed directly
    between localized CAS-LMOs.
    """

    C_active = np.asarray(cas_data["C_spatial"], dtype=complex)
    h_active = np.asarray(cas_data["h_spatial"], dtype=complex)
    C_lmo = np.asarray(C_lmo, dtype=complex)

    U_active_to_lmo = C_active.conj().T @ C_lmo
    h_lmo = U_active_to_lmo.conj().T @ h_active @ U_active_to_lmo

    cas_lmo_data = run_hubbard_cas_from_common_spatial_basis(
        h_spatial=h_lmo,
        C_spatial=C_lmo,
        U_site=U_site,
        n_elec=n_elec,
        n_roots=n_roots,
        print_summary=print_summary,
    )

    cas_lmo_data["C_spatial"] = C_lmo
    cas_lmo_data["h_spatial"] = h_lmo
    cas_lmo_data["U_active_to_lmo"] = U_active_to_lmo
    cas_lmo_data["h_lmo"] = h_lmo

    return cas_lmo_data

def _cas_expect_ops(vec, basis, basis_index, ops):
    """
    Expectation value of a fermionic operator string.

    ops should be ordered in the actual operator order from left to right:
        c† ... c ...

    This function applies them right-to-left to the ket.
    Each op is:
        ("cre", spinorb_index)
        ("ann", spinorb_index)
    """
    val = 0.0 + 0.0j

    for j, det in enumerate(basis):
        cur = det
        phase = 1

        for op, k in reversed(ops):
            if op == "ann":
                out = _cas_annihilate(cur, k)
            elif op == "cre":
                out = _cas_create(cur, k)
            else:
                raise ValueError(f"Unknown op: {op}")

            if out is None:
                cur = None
                break

            cur, s = out
            phase *= s

        if cur is None:
            continue

        i = basis_index.get(cur, None)
        if i is None:
            continue

        val += np.conj(vec[i]) * phase * vec[j]

    return val

def _cas_one_body_expect(vec, basis, basis_index, p, q, spin, n_orb):
    pso = _cas_spinorb(p, spin, n_orb)
    qso = _cas_spinorb(q, spin, n_orb)
    return _cas_expect_ops(
        vec, basis, basis_index,
        [("cre", pso), ("ann", qso)]
    )

def cache_active_spin_operator_expectations(cas_data, root=0):
    import numpy as np

    result = cas_data["result"]
    basis = list(result.basis)
    basis_index = dict(result.basis_index)
    vec = _cas_get_ci_vector(result, root)

    n_orb = np.asarray(cas_data["h_spatial"]).shape[0]

    E1 = np.zeros((2, 2, n_orb, n_orb), dtype=complex)
    E2 = np.zeros((2, 2, 2, 2, n_orb, n_orb, n_orb, n_orb), dtype=complex)

    for s1 in range(2):
        for s2 in range(2):
            for p in range(n_orb):
                for q in range(n_orb):
                    E1[s1, s2, p, q] = _cas_expect_ops(
                        vec, basis, basis_index,
                        [
                            ("cre", _cas_spinorb(p, s1, n_orb)),
                            ("ann", _cas_spinorb(q, s2, n_orb)),
                        ],
                    )

    for s1 in range(2):
        for s2 in range(2):
            for s3 in range(2):
                for s4 in range(2):
                    for p in range(n_orb):
                        for q in range(n_orb):
                            for r in range(n_orb):
                                for t in range(n_orb):
                                    E2[s1, s2, s3, s4, p, q, r, t] = _cas_expect_ops(
                                        vec, basis, basis_index,
                                        [
                                            ("cre", _cas_spinorb(p, s1, n_orb)),
                                            ("ann", _cas_spinorb(q, s2, n_orb)),
                                            ("cre", _cas_spinorb(r, s3, n_orb)),
                                            ("ann", _cas_spinorb(t, s4, n_orb)),
                                        ],
                                    )

    return {
        "root": root,
        "n_orb": n_orb,
        "E1": E1,
        "E2": E2,
    }

def spin_correlation_from_cached_expectations(cache, U_active_to_new, connected=True):
    import numpy as np

    E1 = cache["E1"]
    E2 = cache["E2"]

    U = np.asarray(U_active_to_new, dtype=complex)
    n_active, n_new = U.shape

    M = np.zeros((n_new, n_active, n_active), dtype=complex)
    for i in range(n_new):
        M[i] = np.outer(U[:, i], np.conj(U[:, i]))

    spin_vec = np.zeros((n_new, 3), dtype=complex)
    raw = np.zeros((n_new, n_new), dtype=complex)

    for i in range(n_new):
        Mi = M[i]

        sp = np.einsum("pq,pq->", Mi, E1[0, 1])
        sm = np.einsum("pq,pq->", Mi, E1[1, 0])
        sz = 0.5 * np.einsum("pq,pq->", Mi, E1[0, 0] - E1[1, 1])

        spin_vec[i, 0] = 0.5 * (sp + sm)
        spin_vec[i, 1] = (sp - sm) / (2.0j)
        spin_vec[i, 2] = sz

    for i in range(n_new):
        Mi = M[i]
        for j in range(n_new):
            Mj = M[j]

            szsz = 0.0 + 0.0j
            for s1, sign1 in [(0, +1), (1, -1)]:
                for s2, sign2 in [(0, +1), (1, -1)]:
                    szsz += (
                        0.25
                        * sign1
                        * sign2
                        * np.einsum("pq,rt,pqrt->", Mi, Mj, E2[s1, s1, s2, s2])
                    )

            sp_sm = np.einsum("pq,rt,pqrt->", Mi, Mj, E2[0, 1, 1, 0])
            sm_sp = np.einsum("pq,rt,pqrt->", Mi, Mj, E2[1, 0, 0, 1])

            raw[i, j] = szsz + 0.5 * (sp_sm + sm_sp)

    raw = np.real_if_close(raw)
    spin_vec = np.real_if_close(spin_vec)

    spin_dot = spin_vec @ spin_vec.T
    corr = raw - spin_dot if connected else raw

    corr = np.real_if_close(corr)
    s2 = float(np.real_if_close(np.sum(raw)))

    return (
        np.asarray(corr, dtype=float),
        np.asarray(raw, dtype=float),
        np.asarray(spin_vec, dtype=float),
        s2,
    )

def print_lmo_spin_correlation_cached(
    cas_data,
    U_active_to_lmo,
    root=0,
    labels=None,
    connected=True,
    precision=6,
):
    import numpy as np
    import pandas as pd

    cache = cache_active_spin_operator_expectations(cas_data, root=root)

    corr, raw, spin_vec, s2 = spin_correlation_from_cached_expectations(
        cache,
        U_active_to_lmo,
        connected=connected,
    )

    np.set_printoptions(precision=precision, suppress=True)

    print(f"=== root {root} spin correlation ===")
    print("connected =" if connected else "raw =")
    print(corr)
    print("")
    print("raw <S_i·S_j> =")
    print(raw)
    print("")
    print("<S_i> =")
    print(spin_vec)
    print("")
    print(f"<S^2> = {s2:.12f}")

    if labels is not None:
        return pd.DataFrame(corr, index=labels, columns=labels)

    return corr, raw, spin_vec, s2
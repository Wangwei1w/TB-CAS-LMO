# LMO.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def pipek_mezey_localization(C, model, max_iter=100, tol=1e-6, verbose=True):
    """
    Pipek–Mezey LMO: C.shape = (n_sites, n_orbs), real-valued
    """
    C_new = np.real(C.copy()).astype(np.float64)
    n_sites, n_orbs = C_new.shape

    for it in range(max_iter):
        max_angle = 0.0
        for i in range(n_orbs):
            for j in range(i + 1, n_orbs):
                c_i = C_new[:, i]
                c_j = C_new[:, j]

                q_i = c_i**2
                q_j = c_j**2
                delta_q = q_i - q_j

                pij = float(np.dot(delta_q, c_i * c_j))
                D = float(np.dot(delta_q, delta_q) - 4 * np.dot(c_i**2, c_j**2))

                theta = 0.5 * np.arctan2(2 * pij, D)

                if abs(theta) > 1e-12:  # Apply a nonzero rotation.
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    C_i_new = cos_t * c_i - sin_t * c_j
                    C_j_new = sin_t * c_i + cos_t * c_j
                    C_new[:, i] = C_i_new
                    C_new[:, j] = C_j_new
                    max_angle = max(max_angle, abs(theta))

        if verbose:
            print(f"[PM Iter {it}] max rotation angle: {np.degrees(max_angle):.6f} deg")
        if max_angle < tol:
            break

    return C_new

def pipek_mezey_localization_stable(
    C,
    site_groups=None,
    max_iter=1000,
    tol=1e-6,
    verbose=True,
    max_angle_deg=10.0,
    damping=0.2,
    min_improvement=1e-12,
):
    """Localize TB orbitals with damped pair rotations.
    
    Limit each rotation angle and reject steps that reduce the objective.
    Return localized coefficients, objective history and angle history."""

    C = np.asarray(C, dtype=float).copy()
    n_sites, n_orb = C.shape

    if site_groups is None:
        site_groups = [[i] for i in range(n_sites)]

    max_angle_rad = np.radians(max_angle_deg)

    f_history = []
    angle_history = []

    def objective(C_mat):
        f = 0.0
        for group in site_groups:
            q = np.sum(C_mat[group, :] ** 2, axis=0)
            f += np.sum(q ** 2)
        return float(f)

    for it in range(max_iter):
        f_before_iter = objective(C)
        max_angle = 0.0
        improved = False

        for i in range(n_orb - 1):
            for j in range(i + 1, n_orb):
                ci = C[:, i]
                cj = C[:, j]

                A = 0.0
                B = 0.0

                for group in site_groups:
                    xi = ci[group]
                    xj = cj[group]

                    q_i = np.sum(xi ** 2)
                    q_j = np.sum(xj ** 2)
                    q_ij = np.sum(xi * xj)

                    A += q_ij * (q_i - q_j)
                    B += q_ij ** 2 - 0.25 * (q_i - q_j) ** 2

                theta = 0.25 * np.arctan2(4.0 * A, -4.0 * B)

                theta *= damping
                theta = np.clip(theta, -max_angle_rad, max_angle_rad)

                if abs(theta) < 1e-14:
                    continue

                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                C_trial = C.copy()
                C_trial[:, i] = cos_t * ci - sin_t * cj
                C_trial[:, j] = sin_t * ci + cos_t * cj

                f_old = objective(C)
                f_new = objective(C_trial)

                if f_new > f_old + min_improvement:
                    C = C_trial
                    improved = True
                    max_angle = max(max_angle, abs(theta))

        f_after_iter = objective(C)
        f_history.append(f_after_iter)
        angle_history.append(np.degrees(max_angle))

        if verbose:
            print(
                f"[PM Stable Iter {it+1:4d}] "
                f"f = {f_after_iter:.12f}, "
                f"df = {f_after_iter - f_before_iter:+.3e}, "
                f"max angle = {np.degrees(max_angle):.6f} deg"
            )

        if max_angle < np.radians(tol):
            if verbose:
                print(
                    f"Converged: max angle "
                    f"{np.degrees(max_angle):.6f} deg < {tol} deg"
                )
            break

        if not improved:
            if verbose:
                print("No accepted rotation. Stop.")
            break

    return C, f_history, angle_history

def pipek_mezey_localization_linesearch(
    C,
    site_groups=None,
    max_iter=1000,
    tol=1e-6,
    verbose=True,
    max_angle_deg=45.0,
    n_grid=181,
    min_improvement=1e-12,
):
    """Localize real orbitals by maximizing the grouped population objective.
    
    For each orbital pair, search a uniform angular grid and accept only
    objective increases above min_improvement. site_groups=None uses one
    group per site. tol and the returned angle history are in degrees.
    Return C_lmo, f_history and angle_history; columns label orbitals."""

    C = np.asarray(C, dtype=float).copy()
    n_sites, n_orb = C.shape

    if site_groups is None:
        site_groups = [[i] for i in range(n_sites)]

    max_angle = np.radians(max_angle_deg)
    theta_grid = np.linspace(-max_angle, max_angle, n_grid)

    f_history = []
    angle_history = []

    def objective(C_mat):
        f = 0.0
        for group in site_groups:
            q = np.sum(C_mat[group, :] ** 2, axis=0)
            f += np.sum(q ** 2)
        return float(f)

    def rotate_pair(C_mat, i, j, theta):
        C_new = C_mat.copy()
        ci = C_mat[:, i].copy()
        cj = C_mat[:, j].copy()

        c = np.cos(theta)
        s = np.sin(theta)

        C_new[:, i] = c * ci - s * cj
        C_new[:, j] = s * ci + c * cj

        return C_new

    for it in range(max_iter):
        f_iter_start = objective(C)
        max_theta_accepted = 0.0
        improved = False

        for i in range(n_orb - 1):
            for j in range(i + 1, n_orb):
                f_old = objective(C)

                best_theta = 0.0
                best_f = f_old

                for theta in theta_grid:
                    if abs(theta) < 1e-15:
                        continue

                    C_trial = rotate_pair(C, i, j, theta)
                    f_trial = objective(C_trial)

                    if f_trial > best_f:
                        best_f = f_trial
                        best_theta = theta

                if best_f > f_old + min_improvement:
                    C = rotate_pair(C, i, j, best_theta)
                    improved = True
                    max_theta_accepted = max(max_theta_accepted, abs(best_theta))

        f_iter_end = objective(C)
        f_history.append(f_iter_end)
        angle_history.append(np.degrees(max_theta_accepted))

        if verbose:
            print(
                f"[PM LineSearch Iter {it+1:4d}] "
                f"f = {f_iter_end:.12f}, "
                f"df = {f_iter_end - f_iter_start:+.3e}, "
                f"max angle = {np.degrees(max_theta_accepted):.6f} deg"
            )

        if max_theta_accepted < np.radians(tol):
            if verbose:
                print(
                    f"Converged: max angle "
                    f"{np.degrees(max_theta_accepted):.6f} deg < {tol} deg"
                )
            break

        if not improved:
            if verbose:
                print("No accepted rotation. Stop.")
            break

    return C, f_history, angle_history

def damped_pipek_mezey_localization_tb(C, site_groups=None, max_iter=500, tol=1e-5, verbose=True, max_angle_deg=30, damping=0.5):
    """Localize TB orbitals using damped Pipek-Mezey pair rotations.
    
    C has shape (n_sites, n_orb). site_groups=None assigns one group per site.
    max_iter limits sweeps; tol and max_angle_deg are in degrees. damping
    scales each rotation. Return coefficients, objective and angle histories."""
    C = np.real(C.copy())
    n_sites, n_orb = C.shape
    max_angle_rad = np.radians(max_angle_deg)
    f_history = []
    angle_history = []

    if site_groups is None:
        site_groups = [[i] for i in range(n_sites)]

    def objective(C):
        f = 0.0
        for group in site_groups:
            proj = np.sum(C[group, :]**2, axis=0)
            f += np.sum(proj**2)
        return f

    for it in range(max_iter):
        max_angle = 0
        improved = False
        f_before = objective(C)

        for i in range(n_orb - 1):
            for j in range(i + 1, n_orb):
                ci, cj = C[:, i], C[:, j]
                delta_f = 0.0

                for group in site_groups:
                    pi = np.sum(ci[group]**2)
                    pj = np.sum(cj[group]**2)
                    pij = np.sum(ci[group] * cj[group])
                    delta_f += 4 * pij * (pi - pj)

                theta = 0.25 * np.arctan2(delta_f, 1e-10 + np.abs(delta_f))
                theta *= damping  # Damp the rotation to reduce oscillations.
                theta = np.clip(theta, -max_angle_rad, max_angle_rad)

                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
                C_new = C.copy()
                C_new[:, [i, j]] = C[:, [i, j]] @ R

                f_after = objective(C_new)
                if f_after > f_before:  # Accept only rotations that increase the objective.
                    C = C_new
                    f_before = f_after
                    max_angle = max(max_angle, np.abs(theta))
                    improved = True

        f_val = f_before
        f_history.append(f_val)
        angle_history.append(np.degrees(max_angle))

        if verbose:
            print(f"[PM Iter {it+1:3d}] f = {f_val:.6f}, max angle = {np.degrees(max_angle):.4f}°")

        if max_angle < np.radians(tol):
            if verbose:
                print(f"收敛达到角度阈值：{np.degrees(max_angle):.4f}° < {tol}°")
            break

        if not improved:
            if verbose:
                print("本轮无任何旋转提高目标函数，提前终止。")
            break

    return C, f_history, angle_history

def compute_natural_orbitals(density_matrix, sort=True):
    """Diagonalize a Hermitian one-particle density matrix.
    
    Return eigenvectors as columns and their occupations. sort=True orders
    occupations from largest to smallest; their normalization follows the input."""
    # Diagonalize the Hermitian density matrix.
    occs, natorbs = np.linalg.eigh(density_matrix)

    if sort:
        idx = np.argsort(occs)[::-1]
        occs = occs[idx]
        natorbs = natorbs[:, idx]

    return natorbs, occs

def boys_localization_tb(C, positions, max_iter=500, tol=1e-5, verbose=True,
                         max_angle_deg=30, damping=0.5):
    """Apply pair rotations to the TB orbital-centroid objective.
    
    This implementation maximizes the negative sum of squared pairwise
    centroid separations. Retained as implemented; it is not the conventional
    Boys spread minimization. tol is an angular threshold in degrees."""
    C = np.real(C.copy())
    n_sites, n_orb = C.shape
    max_angle_rad = np.radians(max_angle_deg)
    f_history = []
    angle_history = []

    def compute_centers(C):
        # Compute each orbital centroid.
        return np.array([np.sum((C[:, i][:, None]**2) * positions, axis=0) for i in range(n_orb)])

    def objective(R):
        # Sum squared pairwise centroid separations.
        f = 0.0
        for i in range(n_orb - 1):
            for j in range(i + 1, n_orb):
                f += np.sum((R[i] - R[j])**2)
        return -f  # Use the negative centroid-separation objective.

    for it in range(max_iter):
        max_angle = 0
        improved = False

        R_old = compute_centers(C)
        f_before = objective(R_old)

        for i in range(n_orb - 1):
            for j in range(i + 1, n_orb):
                ci, cj = C[:, i], C[:, j]

                Ri = np.sum((ci[:, None]**2) * positions, axis=0)
                Rj = np.sum((cj[:, None]**2) * positions, axis=0)
                Rij = np.sum(((ci * cj)[:, None]) * positions, axis=0)

                delta = 4 * np.dot(Rij, Ri - Rj)
                theta = 0.25 * np.arctan2(delta, 1e-10 + np.abs(delta))
                theta *= damping
                theta = np.clip(theta, -max_angle_rad, max_angle_rad)

                # Apply the pair rotation.
                Rmat = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta),  np.cos(theta)]])
                C_new = C.copy()
                C_new[:, [i, j]] = C[:, [i, j]] @ Rmat

                R_new = compute_centers(C_new)
                f_after = objective(R_new)

                if f_after > f_before:
                    C = C_new
                    f_before = f_after
                    max_angle = max(max_angle, np.abs(theta))
                    improved = True

        f_history.append(f_before)
        angle_history.append(np.degrees(max_angle))

        if verbose:
            print(f"[Boys Iter {it+1:3d}] f = {-f_before:.6f}, max angle = {np.degrees(max_angle):.4f}°")

        if max_angle < np.radians(tol):
            if verbose:
                print(f"收敛达到角度阈值：{np.degrees(max_angle):.4f}° < {tol}°")
            break
        if not improved:
            if verbose:
                print("无有效旋转，提前终止。")
            break

    return C, f_history, angle_history

def boys_localization_tb_fixed(C, positions, max_iter=500, tol=1e-6, verbose=True,
                               max_angle_deg=5, damping=1.0):
    """Apply damped pair rotations to the TB orbital-centroid objective.
    
    C has shape (n_sites, n_orb); positions has shape (n_sites, dimension).
    max_iter limits sweeps, tol and max_angle_deg are in degrees, and damping
    scales the angle. Return coefficients, objective and angle histories."""
    C = np.real(C.copy())
    n_sites, n_orb = C.shape
    max_angle_rad = np.radians(max_angle_deg)
    f_history = []
    angle_history = []

    def compute_centers(C):
        return np.array([np.sum((C[:, i]**2)[:, None] * positions, axis=0) for i in range(n_orb)])

    def objective(R):
        return -np.sum([np.sum((R[i] - R[j])**2)
                        for i in range(n_orb - 1)
                        for j in range(i + 1, n_orb)])

    for it in range(max_iter):
        max_angle = 0
        improved = False
        R_old = compute_centers(C)
        f_before = objective(R_old)

        for i in range(n_orb - 1):
            for j in range(i + 1, n_orb):
                ci, cj = C[:, i], C[:, j]
                Ri = np.sum((ci**2)[:, None] * positions, axis=0)
                Rj = np.sum((cj**2)[:, None] * positions, axis=0)
                Rij = np.sum((ci * cj)[:, None] * positions, axis=0)
                delta = 4 * np.dot(Rij, Ri - Rj)

                if np.abs(delta) < 1e-10:
                    continue

                theta = damping * 0.5 * max_angle_rad * np.sign(delta)

                Rmat = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                ])

                C_rot = C.copy()
                C_rot[:, [i, j]] = C[:, [i, j]] @ Rmat

                R_new = compute_centers(C_rot)
                f_after = objective(R_new)

                if f_after > f_before:
                    C = C_rot
                    f_before = f_after
                    improved = True
                    max_angle = max(max_angle, np.abs(theta))

        f_history.append(-f_before)
        angle_history.append(np.degrees(max_angle))

        if verbose:
            print(f"[Boys Iter {it+1:3d}] f = {-f_before:.6f}, max angle = {np.degrees(max_angle):.4f}°")

        if max_angle < np.radians(tol):
            print("已达到角度收敛阈值。")
            break

        if not improved:
            print("本轮无优化，提前终止。")
            break

    return C, f_history, angle_history

def build_tb_model_from_mo(C_mo_lmo, energies, tol=1e-10):
    """Construct H_LMO such that H_LMO @ C_mo_lmo = C_mo_lmo @ E.
    
    Columns of C_mo_lmo are MOs in the LMO basis. energies is a vector or
    diagonal matrix. Use the pseudoinverse if the coefficient matrix is singular."""
    C = np.array(C_mo_lmo)
    
    if energies.ndim == 1:
        E = np.diag(energies)
    else:
        E = np.array(energies)

    # Use an inverse, or a pseudoinverse for a singular matrix.
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C, rcond=tol)

    H_lmo = C @ E @ C_inv
    H_lmo = np.real_if_close(H_lmo)  # Discard only numerically negligible imaginary parts.

    return H_lmo

def flip_lmo_phases(C_lmo, indices_to_flip):
    """Return a copy of C_lmo with the selected orbital columns negated."""
    C_lmo_flipped = C_lmo.copy()
    for idx in indices_to_flip:
        if 0 <= idx < C_lmo.shape[1]:
            C_lmo_flipped[:, idx] *= -1
    return C_lmo_flipped

def analyze_lmo_phase_flips(C_mo, C_lmo, energies, flip_indices, plot_func=None, mo_indices=None):
    """Flip selected LMO phases and compute MO/LMO expansions and H_lmo.
    
    Optionally plot the transformed orbitals. Return both expansion tables
    and the reconstructed one-electron Hamiltonian."""
    C_lmo_flipped = flip_lmo_phases(C_lmo, flip_indices)

    # Optionally plot the phase-flipped LMOs.
    if plot_func is not None:
        fig, axs = plt.subplots(1, C_lmo.shape[1], figsize=(15, 4))
        for i in range(C_lmo.shape[1]):
            plot_func(axs[i], C_lmo_flipped[:, i], title=f"Flipped LMO {i}")
        plt.suptitle("Flipped LMO")
        plt.tight_layout()
        plt.show()

    # Compute MO/LMO expansion coefficients.
    from numpy.linalg import pinv
    proj_mo_lmo = pinv(C_lmo_flipped) @ C_mo
    proj_lmo_mo = pinv(C_mo) @ C_lmo_flipped

    print("MO 在 LMO 基下的展开（含相位翻转）：")
    print(np.round(proj_mo_lmo, 4))
    print("\nLMO 在 MO 基下的展开（含相位翻转）：")
    print(np.round(proj_lmo_mo, 4))

    # Construct the one-electron TB Hamiltonian.
    from numpy.linalg import LinAlgError
    try:
        C_inv = np.linalg.inv(proj_lmo_mo)
    except LinAlgError:
        C_inv = pinv(proj_lmo_mo)
    H_lmo = proj_lmo_mo @ np.diag(energies) @ C_inv
    H_lmo = np.real_if_close(H_lmo)
    
    print("\n构建的等效 TB 哈密顿量：")
    print(np.round(H_lmo, 4))

    return proj_mo_lmo, proj_lmo_mo, H_lmo

def compute_mo_lmo_expansion(C_mo, C_lmo, mo_indices=None, lmo_indices=None, tol=1e-3):
    """Return MO-in-LMO and LMO-in-MO coefficient tables.
    
    Input matrices have sites as rows and orbitals as columns. This legacy
    helper uses real parts. Coefficients below tol are zeroed for display;
    mo_indices and lmo_indices label the tables."""

    # Retain the legacy real-coefficient convention.
    C_mo = np.real(C_mo)
    C_lmo = np.real(C_lmo)

    # Project MOs onto the LMO basis.
    proj = C_lmo.T @ C_mo  

    # Construct the coefficient tables.
    proj_mo_to_lmo = pd.DataFrame(proj.T, columns=[f"LMO {j}" for j in range(proj.shape[0])])
    proj_lmo_to_mo = pd.DataFrame(proj, index=[f"LMO {j}" for j in range(proj.shape[0])],
                                         columns=[f"MO {i}" for i in range(proj.shape[1])])

    # Label rows and columns with the supplied orbital indices.
    if mo_indices:
        proj_mo_to_lmo.index = [f"MO {i}" for i in mo_indices]
    if lmo_indices:
        proj_lmo_to_mo.index = [f"LMO {i}" for i in lmo_indices]
        proj_mo_to_lmo.columns = [f"LMO {i}" for i in lmo_indices]

    # Suppress coefficients below the display threshold.
    proj_mo_to_lmo = proj_mo_to_lmo.where(np.abs(proj_mo_to_lmo) > tol, 0.0)
    proj_lmo_to_mo = proj_lmo_to_mo.where(np.abs(proj_lmo_to_mo) > tol, 0.0)

    return proj_mo_to_lmo, proj_lmo_to_mo

def plot_lmos(C_lmo, model, spin, orb_indices, cmap="viridis", vmin=None, vmax=None):
    """Plot localized-orbital site weights using model.positions (N by 2 or 3)."""
    coords = np.array(model.positions)
    n_orb = C_lmo.shape[1]
    fig, axs = plt.subplots(1, n_orb, figsize=(4*n_orb, 4), squeeze=False)
    for idx, ax in enumerate(axs[0]):
        weights = np.abs(C_lmo[:, idx])**2
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=weights, cmap=cmap, s=80, vmin=vmin, vmax=vmax)
        ax.set_title(f"{spin} LMO {orb_indices[idx]}")
        ax.axis("equal")
        ax.axis("off")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    
def get_target_occupations_from_indices(mfh_model, spin, target_state_indices):
    n_occ = int(mfh_model.num_spin_el[spin])
    return np.array(
        [1.0 if i < n_occ else 0.0 for i in target_state_indices],
        dtype=float
    )

def build_target_subspace_from_mos(mfh_model, spin, target_state_indices):
    """Return selected MO columns Psi and the diagonal energy matrix E.
    
    Psi has shape (n_sites, n_target); E has shape (n_target, n_target)."""
    Psi = np.column_stack([
        np.asarray(mfh_model.evecs[spin][i], dtype=complex)
        for i in target_state_indices
    ])

    E = np.diag([
        mfh_model.evals[spin][i]
        for i in target_state_indices
    ])

    return Psi, E

def compute_lmo_C_matrix(mfh_model, C_lmo, spin, target_state_indices):
    """Project LMO columns onto selected MO columns: C = Psi.conj().T @ Phi.
    
    Rows label selected MOs; columns label LMOs. Return C, Psi, E and Phi."""
    Psi, E = build_target_subspace_from_mos(
        mfh_model,
        spin=spin,
        target_state_indices=target_state_indices
    )

    Phi = np.asarray(C_lmo, dtype=complex)

    C = Psi.conj().T @ Phi

    return C, Psi, E, Phi

def compute_lmo_H_eff(mfh_model, C_lmo, spin, target_state_indices):
    """Transform selected one-electron energies into the LMO basis.
    
    H_eff = C.conj().T @ E @ C, where C contains MO-to-LMO overlaps."""
    C, Psi, E, Phi = compute_lmo_C_matrix(
        mfh_model,
        C_lmo=C_lmo,
        spin=spin,
        target_state_indices=target_state_indices
    )

    H_eff = C.conj().T @ E @ C

    return H_eff, C, Psi, E, Phi

def compute_lmo_density_matrix_from_C(C, occ):
    """
    rho = C^dagger diag(occ) C
    """
    occ = np.asarray(occ, dtype=float)

    if C.shape[0] != len(occ):
        raise ValueError(
            f"C.shape[0]={C.shape[0]} != len(occ)={len(occ)}"
        )

    rho = C.conj().T @ np.diag(occ) @ C

    return rho

def compute_pm_lmo_observables_from_C_lmo(
    mfh_model,
    C_lmo,
    target_state_indices,
):
    """Build spin densities, bond orders, hoppings and exchange proxies in an LMO basis.
    
    Return rho_up/dn, n_up/dn, n_tot, m, B, H_eff_up/dn/avg,
    t_eff_abs, J1_proxy and J2_proxy. These proxies are not fitted exchange
    constants. Orbital Hubbard U is computed separately."""

    # spin up
    H_eff_up, C_up, Psi_up, E_up, Phi_up = compute_lmo_H_eff(
        mfh_model,
        C_lmo=C_lmo,
        spin=0,
        target_state_indices=target_state_indices
    )

    occ_up = get_target_occupations_from_indices(
        mfh_model,
        spin=0,
        target_state_indices=target_state_indices
    )

    rho_up = compute_lmo_density_matrix_from_C(C_up, occ_up)

    # spin down
    H_eff_dn, C_dn, Psi_dn, E_dn, Phi_dn = compute_lmo_H_eff(
        mfh_model,
        C_lmo=C_lmo,
        spin=1,
        target_state_indices=target_state_indices
    )

    occ_dn = get_target_occupations_from_indices(
        mfh_model,
        spin=1,
        target_state_indices=target_state_indices
    )

    rho_dn = compute_lmo_density_matrix_from_C(C_dn, occ_dn)

    # local occupations
    n_up = np.real(np.diag(rho_up))
    n_dn = np.real(np.diag(rho_dn))
    n_tot = n_up + n_dn
    m = n_up - n_dn

    # bond order
    B = np.real(rho_up + rho_dn)

    # spin-averaged effective hopping
    H_eff_avg = 0.5 * (H_eff_up + H_eff_dn)
    t_eff_abs = np.abs(H_eff_avg)

    # proxy J
    mm_abs = np.abs(np.outer(m, m))
    B_abs = np.abs(B)

    J1_proxy = t_eff_abs**2 * mm_abs
    J2_proxy = t_eff_abs * B_abs

    return {
        "H_eff_up": H_eff_up,
        "H_eff_dn": H_eff_dn,
        "H_eff_avg": H_eff_avg,
        "C_up": C_up,
        "C_dn": C_dn,
        "rho_up": rho_up,
        "rho_dn": rho_dn,
        "n_up": n_up,
        "n_dn": n_dn,
        "n_tot": n_tot,
        "m": m,
        "B": B,
        "t_eff_abs": t_eff_abs,
        "J1_proxy": J1_proxy,
        "J2_proxy": J2_proxy,
    }

def print_pm_lmo_observables_summary(obs):
    np.set_printoptions(precision=6, suppress=True)

    print("=== n_up ===")
    print(obs["n_up"])

    print("\n=== n_dn ===")
    print(obs["n_dn"])

    print("\n=== n_tot ===")
    print(obs["n_tot"])

    print("\n=== m = n_up - n_dn ===")
    print(obs["m"])

    print("\n=== B = Re(rho_up + rho_dn) ===")
    print(obs["B"])

    print("\n=== H_eff_avg ===")
    print(obs["H_eff_avg"])

    print("\n=== |t_eff| ===")
    print(obs["t_eff_abs"])

    print("\n=== J1_proxy = |t_eff|^2 * |m_a m_b| ===")
    print(obs["J1_proxy"])

    print("\n=== J2_proxy = |t_eff| * |B_ab| ===")
    print(obs["J2_proxy"])

def extract_pm_lmo_pair_proxy(obs, alpha, beta, one_based=False):
    """Extract the density, hopping and exchange proxies for one LMO pair."""
    if one_based:
        alpha -= 1
        beta -= 1

    return {
        "alpha": alpha + 1 if one_based else alpha,
        "beta": beta + 1 if one_based else beta,
        "m_alpha": float(obs["m"][alpha]),
        "m_beta": float(obs["m"][beta]),
        "abs_malpha_mbeta": float(abs(obs["m"][alpha] * obs["m"][beta])),
        "B_ab": float(obs["B"][alpha, beta]),
        "abs_B_ab": float(abs(obs["B"][alpha, beta])),
        "t_eff_ab": complex(obs["H_eff_avg"][alpha, beta]),
        "abs_t_eff_ab": float(abs(obs["H_eff_avg"][alpha, beta])),
        "J1_proxy_ab": float(obs["J1_proxy"][alpha, beta]),
        "J2_proxy_ab": float(obs["J2_proxy"][alpha, beta]),
    }

def print_pm_lmo_pair_proxy(pair_info):
    print("=== P-M LMO Pair Proxy Summary ===")
    for k, v in pair_info.items():
        print(f"{k}: {v}")

def match_spin_orbitals_by_overlap(
    mfh_model,
    orb_indices_up,
    orb_indices_dn,
    use_abs=True,
    verbose=True,
):
    """Greedily pair up/down MOs using absolute spatial overlaps.
    
    Return paired index lists and overlap information, ordered by the supplied
    up-spin list. Both lists must contain the same number of orbitals."""

    C_up = np.column_stack([
        np.asarray(mfh_model.evecs[0][i], dtype=complex)
        for i in orb_indices_up
    ])

    C_dn = np.column_stack([
        np.asarray(mfh_model.evecs[1][j], dtype=complex)
        for j in orb_indices_dn
    ])

    O = C_up.conj().T @ C_dn

    if use_abs:
        score = np.abs(O)
    else:
        score = np.real(O)

    n_up = len(orb_indices_up)
    n_dn = len(orb_indices_dn)

    if n_up != n_dn:
        raise ValueError(
            f"orb_indices_up 和 orb_indices_dn 数量必须相同："
            f"len(up)={n_up}, len(dn)={n_dn}"
        )

    # Greedily match orbitals by overlap.
    remaining_up = set(range(n_up))
    remaining_dn = set(range(n_dn))

    pairs = []

    while remaining_up:
        best = None
        best_score = -np.inf

        for iu in remaining_up:
            for jd in remaining_dn:
                if score[iu, jd] > best_score:
                    best_score = score[iu, jd]
                    best = (iu, jd)

        iu, jd = best
        pairs.append((iu, jd, best_score, O[iu, jd]))

        remaining_up.remove(iu)
        remaining_dn.remove(jd)

    # Retain the input up-spin ordering.
    pairs = sorted(pairs, key=lambda x: x[0])

    paired_up = [orb_indices_up[iu] for iu, jd, s, o in pairs]
    paired_dn = [orb_indices_dn[jd] for iu, jd, s, o in pairs]

    pair_info = []
    for iu, jd, s, o in pairs:
        pair_info.append({
            "up_list_pos": iu,
            "dn_list_pos": jd,
            "up_mo": orb_indices_up[iu],
            "dn_mo": orb_indices_dn[jd],
            "overlap": o,
            "abs_overlap": abs(o),
        })

    if verbose:
        print("=== Spin orbital pairing by overlap ===")
        print("abs(<MO_up|MO_dn>) =")
        print(np.round(np.abs(O), 6))

        print("\nMatched pairs:")
        for item in pair_info:
            print(
                f"up MO {item['up_mo']}  <->  "
                f"dn MO {item['dn_mo']}  |overlap| = "
                f"{item['abs_overlap']:.6f}"
            )

    return paired_up, paired_dn, np.abs(O), pair_info

def build_common_spatial_subspace_from_spin_mos_paired(
    mfh_model,
    orb_indices_up,
    orb_indices_dn,
    weights=(0.5, 0.5),
    n_common=None,
    auto_pair=True,
    verbose=True,
):
    """Construct a common spatial subspace from selected up/down MOs.
    
    When auto_pair=True, first pair orbitals by absolute spatial overlap."""

    if len(orb_indices_up) != len(orb_indices_dn):
        raise ValueError(
            f"orb_indices_up 和 orb_indices_dn 数量必须相同："
            f"{len(orb_indices_up)} vs {len(orb_indices_dn)}"
        )

    if n_common is None:
        n_common = len(orb_indices_up)

    if auto_pair:
        paired_up, paired_dn, overlap_abs, pair_info = match_spin_orbitals_by_overlap(
            mfh_model=mfh_model,
            orb_indices_up=orb_indices_up,
            orb_indices_dn=orb_indices_dn,
            verbose=verbose,
        )
    else:
        paired_up = list(orb_indices_up)
        paired_dn = list(orb_indices_dn)
        overlap_abs = None
        pair_info = None

    w_up, w_dn = weights

    C_up = np.column_stack([
        np.asarray(mfh_model.evecs[0][i], dtype=complex)
        for i in paired_up
    ])

    C_dn = np.column_stack([
        np.asarray(mfh_model.evecs[1][j], dtype=complex)
        for j in paired_dn
    ])

    P_up = C_up @ C_up.conj().T
    P_dn = C_dn @ C_dn.conj().T

    P_common = w_up * P_up + w_dn * P_dn
    P_common = 0.5 * (P_common + P_common.conj().T)

    evals, evecs = np.linalg.eigh(P_common)
    idx = np.argsort(evals)[::-1]

    evals = evals[idx]
    evecs = evecs[:, idx]

    C_common = evecs[:, :n_common]

    return {
        "C_common": C_common,
        "common_evals": evals[:n_common],
        "P_common": P_common,
        "paired_up": paired_up,
        "paired_dn": paired_dn,
        "overlap_abs": overlap_abs,
        "pair_info": pair_info,
    }

def build_common_pm_lmo_basis_paired(
    mfh_model,
    orb_indices_up,
    orb_indices_dn,
    weights=(0.5, 0.5),
    n_common=None,
    auto_pair=True,
    site_groups=None,
    max_iter=1000,
    tol=1e-6,
    verbose=True,
    max_angle_deg=45.0,
    n_grid=181,
):
    """Build common spatial Pipek-Mezey LMOs from possibly different spin MO lists."""

    common_data = build_common_spatial_subspace_from_spin_mos_paired(
        mfh_model=mfh_model,
        orb_indices_up=orb_indices_up,
        orb_indices_dn=orb_indices_dn,
        weights=weights,
        n_common=n_common,
        auto_pair=auto_pair,
        verbose=verbose,
    )

    C_common = common_data["C_common"]

    imag_norm = np.max(np.abs(np.imag(C_common)))
    if imag_norm > 1e-10:
        raise ValueError(
            f"C_common has significant imaginary part: max imag = {imag_norm:.3e}. "
            "Current PM line-search localization assumes real orbitals."
        )

    C_common_real = np.real(C_common)

    C_lmo_common, f_history, angle_history = pipek_mezey_localization_linesearch(
        C_common_real,
        site_groups=site_groups,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        max_angle_deg=max_angle_deg,
        n_grid=n_grid,
    )

    common_data["C_common"] = C_common_real
    common_data["C_lmo_common"] = C_lmo_common
    common_data["f_history"] = f_history
    common_data["angle_history"] = angle_history

    return common_data

def compute_lmo_H_eff_for_spin_indices(
    mfh_model,
    C_lmo,
    spin,
    target_state_indices,
):
    C, Psi, E, Phi = compute_lmo_C_matrix(
        mfh_model=mfh_model,
        C_lmo=C_lmo,
        spin=spin,
        target_state_indices=target_state_indices,
    )

    H_eff = C.conj().T @ E @ C

    return H_eff, C, Psi, E, Phi

def compute_pm_lmo_observables_from_C_lmo_paired(
    mfh_model,
    C_lmo,
    orb_indices_up,
    orb_indices_dn,
):
    """Build MFH observables in a common LMO basis using separate up/down MO lists."""

    H_eff_up, C_up, Psi_up, E_up, Phi_up = compute_lmo_H_eff_for_spin_indices(
        mfh_model=mfh_model,
        C_lmo=C_lmo,
        spin=0,
        target_state_indices=orb_indices_up,
    )

    occ_up = get_target_occupations_from_indices(
        mfh_model,
        spin=0,
        target_state_indices=orb_indices_up,
    )

    rho_up = compute_lmo_density_matrix_from_C(C_up, occ_up)

    H_eff_dn, C_dn, Psi_dn, E_dn, Phi_dn = compute_lmo_H_eff_for_spin_indices(
        mfh_model=mfh_model,
        C_lmo=C_lmo,
        spin=1,
        target_state_indices=orb_indices_dn,
    )

    occ_dn = get_target_occupations_from_indices(
        mfh_model,
        spin=1,
        target_state_indices=orb_indices_dn,
    )

    rho_dn = compute_lmo_density_matrix_from_C(C_dn, occ_dn)

    n_up = np.real(np.diag(rho_up))
    n_dn = np.real(np.diag(rho_dn))
    n_tot = n_up + n_dn
    m = n_up - n_dn

    B_charge = np.real(rho_up + rho_dn)
    B_spin = np.real(rho_up - rho_dn)

    H_eff_charge = 0.5 * (H_eff_up + H_eff_dn)
    H_eff_spin = 0.5 * (H_eff_up - H_eff_dn)

    spin_splitting_ratio = np.abs(H_eff_spin) / (
        np.abs(H_eff_charge) + 1e-12
    )

    t_eff_abs = np.abs(H_eff_charge)

    mm_abs = np.abs(np.outer(m, m))
    B_abs = np.abs(B_charge)

    J1_proxy = t_eff_abs**2 * mm_abs
    J2_proxy = t_eff_abs * B_abs

    return {
        "H_eff_up": H_eff_up,
        "H_eff_dn": H_eff_dn,
        "H_eff_avg": H_eff_charge,
        "H_eff_charge": H_eff_charge,
        "H_eff_spin": H_eff_spin,
        "spin_splitting_ratio": spin_splitting_ratio,
        "C_up": C_up,
        "C_dn": C_dn,
        "rho_up": rho_up,
        "rho_dn": rho_dn,
        "n_up": n_up,
        "n_dn": n_dn,
        "n_tot": n_tot,
        "m": m,
        "B": B_charge,
        "B_charge": B_charge,
        "B_spin": B_spin,
        "t_eff_abs": t_eff_abs,
        "J1_proxy": J1_proxy,
        "J2_proxy": J2_proxy,
    }

def compute_pm_lmo_observables_common_basis_paired(
    mfh_model,
    orb_indices_up,
    orb_indices_dn,
    weights=(0.5, 0.5),
    n_common=None,
    auto_pair=True,
    site_groups=None,
    max_iter=1000,
    tol=1e-6,
    verbose=True,
    max_angle_deg=45.0,
    n_grid=181,
):
    """Pair spin MOs, construct a common spatial basis, localize and build MFH observables."""

    common_data = build_common_pm_lmo_basis_paired(
        mfh_model=mfh_model,
        orb_indices_up=orb_indices_up,
        orb_indices_dn=orb_indices_dn,
        weights=weights,
        n_common=n_common,
        auto_pair=auto_pair,
        site_groups=site_groups,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        max_angle_deg=max_angle_deg,
        n_grid=n_grid,
    )

    C_lmo_common = common_data["C_lmo_common"]

    obs = compute_pm_lmo_observables_from_C_lmo_paired(
        mfh_model=mfh_model,
        C_lmo=C_lmo_common,
        orb_indices_up=common_data["paired_up"],
        orb_indices_dn=common_data["paired_dn"],
    )

    common_data["obs"] = obs

    return common_data

def save_pm_lmo_observables_summary(
    obs,
    save_dir=None,
    prefix=None,
    print_out=True,
):
    """Write exchange-proxy observables to a text file and optionally print them."""

    import os
    import numpy as np

    np.set_printoptions(precision=6, suppress=True)

    lines = []

    def add_block(title, value):
        lines.append(f"=== {title} ===")
        lines.append(str(value))
        lines.append("")

    add_block("n_up", obs["n_up"])
    add_block("n_dn", obs["n_dn"])
    add_block("n_tot", obs["n_tot"])
    add_block("m = n_up - n_dn", obs["m"])
    add_block("B = Re(rho_up + rho_dn)", obs["B"])
    add_block("H_eff_avg", obs["H_eff_avg"])
    add_block("|t_eff|", obs["t_eff_abs"])
    add_block("J1_proxy = |t_eff|^2 * |m_a m_b|", obs["J1_proxy"])
    add_block("J2_proxy = |t_eff| * |B_ab|", obs["J2_proxy"])

    text = "\n".join(lines)

    # Print the summary.
    if print_out:
        print(text)

    # Save the results.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if prefix:
            filename = os.path.join(save_dir, f"{prefix}_pm_lmo_summary.txt")
        else:
            filename = os.path.join(save_dir, "pm_lmo_summary.txt")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"summary 已保存: {filename}")
        
def plot_and_save_common_lmos(
    mfh_model,
    C_lmo,
    u=None,
    save_dir="mo_lmo_plots",
    prefix=None,
    show=True,
):
    """Plot and save orbital columns in a common site basis.
    
    The model must expose pp.plot_eigenvector. If u is supplied, print
    u * sum(abs(phi)**4) for each orbital. save_dir and prefix control PDF
    filenames; show controls the combined display."""

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    C_lmo = np.asarray(C_lmo)
    n_lmo = C_lmo.shape[1]

    # Display the combined orbital figure.
    fig, axs = plt.subplots(1, n_lmo, figsize=(5 * n_lmo, 4))

    if n_lmo == 1:
        axs = [axs]

    for i in range(n_lmo):
        evec = C_lmo[:, i]

        if u is not None:
            U_orbital = np.sum(u * np.abs(evec) ** 4)
            print(f"Common LMO {i} U_orbital: {U_orbital}")

        mfh_model.pp.plot_eigenvector(
            axs[i],
            evec,
            title=f"Common LMO {i}"
        )

    plt.suptitle("Common spatial LMO", fontsize=14)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Save one PDF per LMO.
    for i in range(n_lmo):
        fig, ax = plt.subplots(figsize=(5, 4))
        evec = C_lmo[:, i]

        mfh_model.pp.plot_eigenvector(
            ax,
            evec,
            title=None
        )

        plt.tight_layout()

        if prefix:
            filename = os.path.join(save_dir, f"{prefix}_Common_LMO_{i}.pdf")
        else:
            filename = os.path.join(save_dir, f"Common_LMO_{i}.pdf")

        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close(fig)

        print(f"Common LMO {i} 图已保存: {filename}")
        
def plot_and_save_mos(
    mfh_model,
    spin,
    orb_indices,
    save_dir="mo_plots",
    prefix=None,
):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    n_orb = len(orb_indices)

    # Create the combined orbital figure.
    fig, axs = plt.subplots(1, n_orb, figsize=(5 * n_orb, 4))

    if n_orb == 1:
        axs = [axs]

    for i, idx in enumerate(orb_indices):
        evec = mfh_model.evecs[spin][idx]
        mfh_model.pp.plot_eigenvector(
            axs[i],
            evec,
            title=f"MO {idx} (spin {spin})"
        )

    plt.suptitle(f"MO spin {spin}", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Save individual orbital figures.
    for i, idx in enumerate(orb_indices):
        fig, ax = plt.subplots(figsize=(5, 4))
        evec = mfh_model.evecs[spin][idx]

        mfh_model.pp.plot_eigenvector(ax, evec, title=None)
        plt.tight_layout()

        if prefix:
            filename = os.path.join(save_dir, f"{prefix}_{idx}.pdf")
        else:
            filename = os.path.join(save_dir, f"MO_{idx}.pdf")

        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close(fig)

        print(f"MO {idx} 图已保存: {filename}")
        
def mix_two_lmos_plus_minus(C_lmo, idx_a, idx_b, mode="replace"):
    """Replace columns idx_a and idx_b by (a+b)/sqrt(2) and (a-b)/sqrt(2).
    
    Only mode="replace" is supported. Other columns are preserved."""

    import numpy as np

    C_new = np.array(C_lmo, dtype=complex).copy()

    phi_a = C_new[:, idx_a].copy()
    phi_b = C_new[:, idx_b].copy()

    phi_plus = (phi_a + phi_b) / np.sqrt(2.0)
    phi_minus = (phi_a - phi_b) / np.sqrt(2.0)

    if mode != "replace":
        raise ValueError('目前只支持 mode="replace"')

    C_new[:, idx_a] = phi_plus
    C_new[:, idx_b] = phi_minus

    return C_new

def analyze_plus_minus_lmo_pair(
    mfh_model,
    C_lmo,
    idx_a,
    idx_b,
    orb_indices_up,
    orb_indices_dn,
    u=None,
    save_dir=None,
    prefix=None,
    show=True,
):
    """Mix two LMO columns, plot them and rebuild MFH observables.
    
    Return mixed coefficients, MFH observables and the selected pair proxy.
    This function uses MFH occupations; it does not build CAS observables."""

    C_pm = mix_two_lmos_plus_minus(
        C_lmo=C_lmo,
        idx_a=idx_a,
        idx_b=idx_b,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if prefix is None:
        prefix = f"plus_minus_{idx_a}_{idx_b}"

    # Plot the orbital basis.
    plot_and_save_common_lmos(
        mfh_model=mfh_model,
        C_lmo=C_pm,
        u=u,
        save_dir=save_dir if save_dir is not None else "mo_lmo_plots",
        prefix=prefix,
        show=show,
    )

    # Rebuild MFH observables in the transformed basis.
    obs_pm = compute_pm_lmo_observables_from_C_lmo_paired(
        mfh_model=mfh_model,
        C_lmo=C_pm,
        orb_indices_up=orb_indices_up,
        orb_indices_dn=orb_indices_dn,
    )

    # Save the observable summary.
    save_pm_lmo_observables_summary(
        obs_pm,
        save_dir=save_dir,
        prefix=prefix,
        print_out=True,
    )

    # Report proxies for the selected pair.
    pair_info = extract_pm_lmo_pair_proxy(
        obs_pm,
        alpha=idx_a,
        beta=idx_b,
        one_based=False,
    )

    print("\n=== Plus/Minus selected pair proxy ===")
    print_pm_lmo_pair_proxy(pair_info)

    # Check orbital orthonormality.
    S = C_pm.conj().T @ C_pm
    offdiag = S - np.diag(np.diag(S))

    print("\n=== Transformed LMO orthogonality ===")
    print(np.real_if_close(S))
    print("max offdiag:", np.max(np.abs(offdiag)))
    print("max diag error:", np.max(np.abs(np.diag(S) - 1.0)))

    return C_pm, obs_pm, pair_info

def downfold_hopping_between_two_lmos(
    obs,
    lmo_a,
    lmo_b,
    bridge_indices=None,
    H_key="H_eff_avg",
    E_ref=None,
    C_lmo=None,
    hubbard_U=None,
    return_matrices=True,
):
    """Eliminate bridge orbitals from a selected one-electron LMO Hamiltonian.
    
    obs[H_key] supplies the Hamiltonian. lmo_a and lmo_b are target indices;
    bridge_indices=None selects all remaining orbitals. E_ref=None uses
    the mean target onsite energy. The resolvent (E_ref I - H_BB)^(-1)
    produces the bridge correction. Return direct, bridge and total hoppings,
    with optional matrices. When hubbard_U is supplied, C_lmo must describe
    the same basis; the legacy orbital-U helper expects a scalar interaction."""

    H = np.asarray(obs[H_key], dtype=complex)
    n = H.shape[0]

    S = [lmo_a, lmo_b]

    if bridge_indices is None:
        bridge_indices = [i for i in range(n) if i not in S]
    else:
        bridge_indices = list(bridge_indices)
#
    overlap = set(S).intersection(set(bridge_indices))
    if overlap:
        raise ValueError(
            f"bridge_indices 不能包含目标 LMO: {overlap}"
        )
#
    B = bridge_indices

    H_SS = H[np.ix_(S, S)]
    H_SB = H[np.ix_(S, B)]
    H_BS = H[np.ix_(B, S)]
    H_BB = H[np.ix_(B, B)]

    if E_ref is None:
        E_ref = 0.5 * (
            np.real(H[lmo_a, lmo_a]) + np.real(H[lmo_b, lmo_b])
        )

    if len(B) == 0:
        H_bridge = np.zeros_like(H_SS, dtype=complex)
        H_eff_2 = H_SS.copy()
        resolvent = None
    else:
        resolvent = np.linalg.inv(
            E_ref * np.eye(len(B), dtype=complex) - H_BB
        )

        H_bridge = H_SB @ resolvent @ H_BS
        H_eff_2 = H_SS + H_bridge

    t_direct = H_SS[0, 1]
    t_bridge = H_bridge[0, 1]
    t_total = H_eff_2[0, 1]

    result = {
        "H_key": H_key,
        "lmo_a": lmo_a,
        "lmo_b": lmo_b,
        "bridge_indices": B,
        "E_ref": E_ref,

        "t_direct": t_direct,
        "t_bridge": t_bridge,
        "t_total": t_total,

        "abs_t_direct": abs(t_direct),
        "abs_t_bridge": abs(t_bridge),
        "abs_t_total": abs(t_total),
    }

    if "m" in obs:
        m_a = obs["m"][lmo_a]
        m_b = obs["m"][lmo_b]

        result["m_a"] = m_a
        result["m_b"] = m_b
        result["abs_mamb"] = abs(m_a * m_b)

        result["J_direct_proxy"] = abs(t_direct) ** 2 * abs(m_a * m_b)
        result["J_bridge_proxy"] = abs(t_bridge) ** 2 * abs(m_a * m_b)
        result["J_total_proxy"] = abs(t_total) ** 2 * abs(m_a * m_b)

        result["signed_J_direct_texture_proxy"] = abs(t_direct) ** 2 * (m_a * m_b)
        result["signed_J_bridge_texture_proxy"] = abs(t_bridge) ** 2 * (m_a * m_b)
        result["signed_J_total_texture_proxy"] = abs(t_total) ** 2 * (m_a * m_b)
#-----------
    if "B" in obs:
        B_ab = obs["B"][lmo_a, lmo_b]
        abs_B_ab = abs(B_ab)

        result["B_ab"] = B_ab
        result["abs_B_ab"] = abs_B_ab

        result["J_tB_direct_proxy"] = abs(t_direct) * abs_B_ab
        result["J_tB_bridge_proxy"] = abs(t_bridge) * abs_B_ab
        result["J_tB_total_proxy"] = abs(t_total) * abs_B_ab
#-----------
    if hubbard_U is not None:
        if C_lmo is None:
            raise ValueError(
                "若提供 hubbard_U，必须同时提供 C_lmo，用于计算 orbital-resolved U。"
            )

        U_orb = compute_lmo_orbital_U(C_lmo, hubbard_U)
        U_eff = compute_pair_U_eff(U_orb, lmo_a, lmo_b)

        result["U_orb"] = U_orb
        result["U_a"] = U_orb[lmo_a]
        result["U_b"] = U_orb[lmo_b]
        result["U_eff"] = U_eff

        result["J_AFM_direct_4t2_over_U"] = compute_superexchange_J(
            t_direct, U_eff
        )
        result["J_AFM_bridge_4t2_over_U"] = compute_superexchange_J(
            t_bridge, U_eff
        )
        result["J_AFM_total_4t2_over_U"] = compute_superexchange_J(
            t_total, U_eff
        )
#----------
    if return_matrices:
        result["H_SS"] = H_SS
        result["H_SB"] = H_SB
        result["H_BS"] = H_BS
        result["H_BB"] = H_BB
        result["H_bridge"] = H_bridge
        result["H_eff_2"] = H_eff_2
        result["resolvent"] = resolvent

    return result

def downfold_hopping_between_two_lmos_spin_resolved_o(
    obs,
    lmo_a,
    lmo_b,
    bridge_indices=None,
    E_ref=None,
):
    """Calculate direct, bridge and total hoppings for the available spin channels."""

    import numpy as np

    keys = []

    if "H_eff_up" in obs:
        keys.append("H_eff_up")

    if "H_eff_dn" in obs:
        keys.append("H_eff_dn")

    if "H_eff_avg" in obs:
        keys.append("H_eff_avg")
    elif "H_eff_charge" in obs:
        keys.append("H_eff_charge")

    results = {}

    for key in keys:
        results[key] = downfold_hopping_between_two_lmos(
            obs=obs,
            lmo_a=lmo_a,
            lmo_b=lmo_b,
            bridge_indices=bridge_indices,
            H_key=key,
            E_ref=E_ref,
            return_matrices=True,
        )

    if "H_eff_up" in results and "H_eff_dn" in results:
        t_up = results["H_eff_up"]["t_total"]
        t_dn = results["H_eff_dn"]["t_total"]

        t_bridge_up = results["H_eff_up"]["t_bridge"]
        t_bridge_dn = results["H_eff_dn"]["t_bridge"]

        results["spin_decomposition"] = {
            "t_total_charge": 0.5 * (t_up + t_dn),
            "t_total_spin": 0.5 * (t_up - t_dn),
            "t_bridge_charge": 0.5 * (t_bridge_up + t_bridge_dn),
            "t_bridge_spin": 0.5 * (t_bridge_up - t_bridge_dn),
            "abs_t_total_charge": abs(0.5 * (t_up + t_dn)),
            "abs_t_total_spin": abs(0.5 * (t_up - t_dn)),
            "abs_t_bridge_charge": abs(0.5 * (t_bridge_up + t_bridge_dn)),
            "abs_t_bridge_spin": abs(0.5 * (t_bridge_up - t_bridge_dn)),
        }

    return results

def downfold_hopping_between_two_lmos_spin_resolved(
    obs,
    lmo_a,
    lmo_b,
    bridge_indices=None,
    E_ref=None,
    C_lmo=None,
    hubbard_U=None,
):
    keys = []

    if "H_eff_up" in obs:
        keys.append("H_eff_up")

    if "H_eff_dn" in obs:
        keys.append("H_eff_dn")

    if "H_eff_avg" in obs:
        keys.append("H_eff_avg")
    elif "H_eff_charge" in obs:
        keys.append("H_eff_charge")

    results = {}

    for key in keys:
        results[key] = downfold_hopping_between_two_lmos(
            obs=obs,
            lmo_a=lmo_a,
            lmo_b=lmo_b,
            bridge_indices=bridge_indices,
            H_key=key,
            E_ref=E_ref,
            C_lmo=C_lmo,
            hubbard_U=hubbard_U,
            return_matrices=True,
        )

    if "H_eff_up" in results and "H_eff_dn" in results:
        t_up = results["H_eff_up"]["t_total"]
        t_dn = results["H_eff_dn"]["t_total"]

        t_bridge_up = results["H_eff_up"]["t_bridge"]
        t_bridge_dn = results["H_eff_dn"]["t_bridge"]

        results["spin_decomposition"] = {
            "t_total_charge": 0.5 * (t_up + t_dn),
            "t_total_spin": 0.5 * (t_up - t_dn),
            "t_bridge_charge": 0.5 * (t_bridge_up + t_bridge_dn),
            "t_bridge_spin": 0.5 * (t_bridge_up - t_bridge_dn),
            "abs_t_total_charge": abs(0.5 * (t_up + t_dn)),
            "abs_t_total_spin": abs(0.5 * (t_up - t_dn)),
            "abs_t_bridge_charge": abs(0.5 * (t_bridge_up + t_bridge_dn)),
            "abs_t_bridge_spin": abs(0.5 * (t_bridge_up - t_bridge_dn)),
        }

    return results

def print_downfold_hopping_result(result):
    print("=== Downfolded hopping between two LMOs ===")
    print(f"H_key: {result['H_key']}")
    print(f"LMO pair: {result['lmo_a']} - {result['lmo_b']}")
    print(f"Bridge indices: {result['bridge_indices']}")
    print(f"E_ref: {result['E_ref']}")

    print("\n--- hopping ---")
    print(f"t_direct = {result['t_direct']}")
    print(f"t_bridge = {result['t_bridge']}")
    print(f"t_total  = {result['t_total']}")

    print("\n--- |hopping| ---")
    print(f"|t_direct| = {result['abs_t_direct']}")
    print(f"|t_bridge| = {result['abs_t_bridge']}")
    print(f"|t_total|  = {result['abs_t_total']}")

    # =========================
    # t^2 * |m_a m_b|
    # =========================
    if "J_total_proxy" in result:
        print("\n--- magnetic texture proxy: |t|^2 * |m_a m_b| ---")
        print(f"m_a = {result['m_a']}")
        print(f"m_b = {result['m_b']}")
        print(f"|m_a m_b| = {result['abs_mamb']}")

        print(f"J_t2m_direct = {result['J_direct_proxy']}")
        print(f"J_t2m_bridge = {result['J_bridge_proxy']}")
        print(f"J_t2m_total  = {result['J_total_proxy']}")

        print("\n--- signed texture proxy: |t|^2 * (m_a m_b) ---")
        print(f"signed_J_t2m_direct = {result['signed_J_direct_texture_proxy']}")
        print(f"signed_J_t2m_bridge = {result['signed_J_bridge_texture_proxy']}")
        print(f"signed_J_t2m_total  = {result['signed_J_total_texture_proxy']}")

    # =========================
    # |t| * |B_ab|
    # =========================
    if "B_ab" in result:
        print("\n--- covalency proxy: |t| * |B_ab| ---")
        print(f"B_ab = {result['B_ab']}")
        print(f"|B_ab| = {result['abs_B_ab']}")

        print(f"J_tB_direct = {result['J_tB_direct_proxy']}")
        print(f"J_tB_bridge = {result['J_tB_bridge_proxy']}")
        print(f"J_tB_total  = {result['J_tB_total_proxy']}")

    # =========================
    # 4t^2/U
    # =========================
    if "U_eff" in result:
        print("\n--- effective U ---")
        print(f"U_a   = {result['U_a']}")
        print(f"U_b   = {result['U_b']}")
        print(f"U_eff = {result['U_eff']}")

        print("\n--- AFM superexchange estimate: 4|t|^2/U_eff ---")
        print(f"J_AFM_direct = {result['J_AFM_direct_4t2_over_U']}")
        print(f"J_AFM_bridge = {result['J_AFM_bridge_4t2_over_U']}")
        print(f"J_AFM_total  = {result['J_AFM_total_4t2_over_U']}")
        
def compute_lmo_orbital_U(C_lmo, hubbard_U):
    """Compute U_alpha = hubbard_U * sum_i abs(phi_i_alpha)**4.
    
    This legacy helper expects a scalar site interaction."""

    C_lmo = np.asarray(C_lmo)

    U_orb = []

    for i in range(C_lmo.shape[1]):
        phi = C_lmo[:, i]
        U_eff = hubbard_U * np.sum(np.abs(phi)**4)
        U_orb.append(np.real_if_close(U_eff))

    return np.array(U_orb)

def compute_pair_U_eff(U_orb, alpha, beta):
    """
    pair effective U
    """

    return 0.5 * (U_orb[alpha] + U_orb[beta])

def compute_superexchange_J(
    t_eff,
    U_eff,
):
    """
    AFM superexchange estimate

    J = 4 t^2 / U
    """

    return 4.0 * np.abs(t_eff)**2 / U_eff

def decompose_bridge_eigenchannels(
    obs,
    lmo_a,
    lmo_b,
    bridge_indices=None,
    H_key="H_eff_avg",
    E_ref=None,
    sort_by_abs=True,
):
    import numpy as np
    import pandas as pd

    H = np.asarray(obs[H_key], dtype=complex)
    n = H.shape[0]

    targets = [lmo_a, lmo_b]

    if bridge_indices is None:
        bridge_indices = [i for i in range(n) if i not in targets]
    else:
        bridge_indices = list(bridge_indices)

    if len(bridge_indices) == 0:
        raise ValueError("bridge_indices is empty.")

    overlap = set(targets).intersection(bridge_indices)
    if overlap:
        raise ValueError(f"bridge_indices contains target LMO(s): {overlap}")

    if E_ref is None:
        E_ref = 0.5 * (np.real(H[lmo_a, lmo_a]) + np.real(H[lmo_b, lmo_b]))

    B = bridge_indices

    H_BB = H[np.ix_(B, B)]
    H_aB = H[np.ix_([lmo_a], B)].reshape(-1)
    H_Bb = H[np.ix_(B, [lmo_b])].reshape(-1)

    # bridge eigenchannels
    eps, W = np.linalg.eigh(H_BB)

    rows = []
    t_total = 0.0 + 0.0j

    for mu in range(len(B)):
        chi_mu = W[:, mu]

        V_a_mu = H_aB @ chi_mu
        V_mu_b = np.conj(chi_mu) @ H_Bb

        denom = E_ref - eps[mu]
        t_mu = V_a_mu * V_mu_b / denom

        t_total += t_mu

        # channel composition in original bridge LMO basis
        comp = {
            f"weight_LMO_{B[k]}": abs(chi_mu[k]) ** 2
            for k in range(len(B))
        }

        row = {
            "channel": mu,
            "epsilon_mu": eps[mu],
            "denominator": denom,
            "V_a_mu": V_a_mu,
            "V_mu_b": V_mu_b,
            "t_mu": t_mu,
            "abs_t_mu": abs(t_mu),
            "phase_t_mu": np.angle(t_mu),
        }
        row.update(comp)
        rows.append(row)

    df = pd.DataFrame(rows)

    if sort_by_abs:
        df = df.sort_values("abs_t_mu", ascending=False).reset_index(drop=True)

    # exact bridge contribution from full resolvent
    H_SB = H[np.ix_(targets, B)]
    H_BS = H[np.ix_(B, targets)]
    resolvent = np.linalg.inv(E_ref * np.eye(len(B), dtype=complex) - H_BB)
    H_bridge = H_SB @ resolvent @ H_BS
    t_bridge_exact = H_bridge[0, 1]

    info = {
        "H_key": H_key,
        "lmo_a": lmo_a,
        "lmo_b": lmo_b,
        "bridge_indices": B,
        "E_ref": E_ref,
        "t_bridge_sum_channels": t_total,
        "t_bridge_exact": t_bridge_exact,
        "abs_t_bridge_exact": abs(t_bridge_exact),
        "channel_table": df,
    }

    return info


import matplotlib.pyplot as plt
import numpy as np
import os

def build_density_matrix(mfh_model, spin=None):
    """Build the occupied-orbital density matrix for one spin or both spins.
    
    spin=0 selects alpha, spin=1 selects beta, and spin=None sums both.
    Return an (n_sites, n_sites) matrix."""
    n_sites = mfh_model.evecs[0].shape[0]
    rho = np.zeros((n_sites, n_sites), dtype=np.complex128)

    # Accumulate occupied-orbital projectors.
    def add_spin_density(spin_idx):
        evecs = mfh_model.evecs[spin_idx]
        n_occ = mfh_model.num_spin_el[spin_idx]
        local_rho = np.zeros_like(rho)
        for i in range(n_occ):
            psi = evecs[:, i]
            local_rho += np.outer(psi, psi.conj())
        return local_rho

    if spin is None:
        # Sum both spin channels.
        for s in [0, 1]:
            rho += add_spin_density(s)
    else:
        # Use the selected spin channel.
        rho = add_spin_density(spin)

    rho = np.real_if_close(rho)
    return rho

def get_homo_lumo_indices(mfh_model, spin=0, num_range=5):
    """Return MO indices from HOMO-num_range through LUMO+num_range."""
    evals = mfh_model.evals[spin]
    num_occ = mfh_model.num_spin_el[spin]
    homo = num_occ - 1
    lumo = num_occ

    indices = list(range(max(0, homo - num_range + 1), min(len(evals), lumo + num_range)))
    return indices

def plot_natural_orbitals(natorbs, occs, model, indices, spin=0, save_prefix=None):
    """Plot natural-orbital site weights for the selected indices."""
    import matplotlib.pyplot as plt

    positions = model.model_a.get_orb()
    for i in indices:
        fig, ax = plt.subplots(figsize=(5, 4))
        weights = np.abs(natorbs[:, i]) ** 2
        sc = ax.scatter(positions[:, 0], positions[:, 1], c=weights, cmap="viridis", s=80)
        ax.axis("equal")
        ax.axis("off")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Occupancy: {occs[i]:.3f}")

        if save_prefix:
            plt.savefig(f"{save_prefix}_NO_{i}_occ{occs[i]:.3f}.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_natural_orbitals1(mfh_model, natorbs, occs, spin=0, delta=3, save_dir=None, show_plot=True):
    """Plot natural-orbital columns around the selected spin HOMO/LUMO.
    
    Use delta to set the range, save_dir for optional PDF output and
    show_plot to control display. occs supplies the column occupations."""
    num_elec = mfh_model.num_spin_el[spin]
    num_orbs = natorbs.shape[1]
    
    homo = num_elec - 1
    lumo = homo + 1

    i_min = max(homo - delta, 0)
    i_max = min(lumo + delta + 1, num_orbs)

    print(f"显示自然轨道编号 [{i_min} - {i_max - 1}]，对应 MO 中约为 HOMO-{homo - i_min} 到 LUMO+{i_max - lumo - 1}")

    for i in range(i_min, i_max):
        vec = natorbs[:, i]
        occ = occs[i]

        title = f"Natural Orbital {i} (Occ = {occ:.2f})"

        # Display the figure.
        if show_plot:
            fig, ax = plt.subplots(figsize=(4, 4))
            mfh_model.pp.plot_eigenvector(ax, vec, title=title)
            plt.tight_layout()
            plt.show()

        # Save the orbital figure.
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(4, 4))
            mfh_model.pp.plot_eigenvector(ax, vec, title="")  # Omit the title.
            filepath = os.path.join(save_dir, f"NO_{i:03d}_occ_{occ:.2f}.pdf")
            fig.savefig(filepath, format="pdf")
            plt.close(fig)
            print(f"已保存: {filepath}")
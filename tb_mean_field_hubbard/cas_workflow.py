"""Orchestrate canonical Hubbard CAS and basis-consistent localized analyses.

Numerical kernels remain in CAS and LMO. All orbital and root indices are
zero-based. U_site_param is the many-body interaction, not the MFH input U.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import nullcontext
from math import comb
from typing import Optional, Sequence, Union
import numpy as np
from . import CAS, LMO
from .cas_output import result_log, save_results_txt

@dataclass
class PairAnalysis:
    """Targets and eliminated bridge orbitals in one explicitly selected basis.

    E_ref=None uses the mean target onsite energy in both analyses.
    channels=False permits an empty bridge for direct-hopping analysis.
    """
    lmo_a: int
    lmo_b: int
    bridge_indices: Optional[Sequence[int]] = None
    H_key: str = "H_eff_avg"
    E_ref: Optional[float] = None
    channels: bool = True

@dataclass
class CASWorkflowParameters:
    """Central configuration for run_cas_workflow.

    The input model must contain a complete spin-independent eigensystem.
    Prepare it with run_mfh(u=0); this API never reruns or modifies that model.
    Selected natural orbitals must be real within basis_tolerance because the
    existing Pipek-Mezey implementation uses real rotations.
    """
    active_indices: Sequence[int]
    U_site_param: Union[float, np.ndarray]
    n_elec_param: int
    n_roots_param: int = 6
    spin_param: int = 0
    root_param: int = 0
    report_roots: Sequence[int] = (0,)
    correlation_roots: Sequence[int] = (0,)
    selected_no_indices: Optional[Sequence[int]] = None
    localization: dict = field(default_factory=lambda: dict(
        site_groups=None, max_iter=1000, tol=1e-6, verbose=True,
        max_angle_deg=45.0, n_grid=181))
    lmo_pairs: Sequence[PairAnalysis] = ()
    mix_pair: Optional[Sequence[int]] = None
    mix_pairs: Sequence[PairAnalysis] = ()
    connected: bool = True
    coefficient_cutoff: float = 1e-3
    result_folder: Optional[Union[str, Path]] = None
    save_plots: bool = False
    show_plots: bool = False
    include_mfh_mix_observables: bool = True
    basis_tolerance: float = 1e-8

def _indices(values, size, name, nonempty=True):
    values = tuple(values)
    if (nonempty and not values) or any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
        or i < 0 or i >= size for i in values
    ) or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique integer indices in [0, {size}).")
    return values

def _check_basis(C, active, tolerance):
    """Reject norm loss or basis leakage rather than silently repairing it."""
    if not np.allclose(C.conj().T @ C, np.eye(C.shape[1]), atol=tolerance, rtol=0):
        raise ValueError("Orbital columns must be orthonormal.")
    if not np.allclose(active @ (active.conj().T @ C), C, atol=tolerance, rtol=0):
        raise ValueError("Localized orbitals must lie in the CAS active subspace.")

def _validate_model(model, p):
    """Compare reconstructed Hamiltonians, allowing phases and degeneracy rotations."""
    evals = np.asarray(model.evals)
    evecs = np.asarray(model.evecs)
    if evals.ndim != 2 or evals.shape[0] != 2 or evecs.shape != (2, evals.shape[1], evals.shape[1]):
        raise ValueError("Expected two complete eigensystems with eigenvectors stored as rows.")
    if not np.all(np.isfinite(evals)) or not np.all(np.isfinite(evecs)):
        raise ValueError("The input eigensystem must be finite.")
    h = []
    for spin in (0, 1):
        C = evecs[spin].T
        _check_basis(C, C, p.basis_tolerance)
        h.append((C * evals[spin]) @ C.conj().T)
    if not np.allclose(h[0], h[1], atol=p.basis_tolerance, rtol=0):
        raise ValueError("Use a U=0 / spin-independent one-electron model for canonical CAS.")
    # Equality alone cannot establish whether a spin-independent mean-field shift exists.
    if getattr(model, "one_electron_U", 0) != 0:
        raise ValueError("The recorded one-electron MFH U must be zero.")
    _indices((p.spin_param,), 2, "spin_param")
    active = _indices(p.active_indices, evals.shape[1], "active_indices")
    if not isinstance(p.n_elec_param, (int, np.integer)) or not 0 <= p.n_elec_param <= 2 * len(active):
        raise ValueError("n_elec_param is outside the active-space capacity.")
    if not isinstance(p.n_roots_param, (int, np.integer)) or not 1 <= p.n_roots_param <= comb(2 * len(active), p.n_elec_param):
        raise ValueError("n_roots_param exceeds the fixed-electron Hilbert-space dimension.")
    for name, roots in (("root_param", (p.root_param,)), ("report_roots", p.report_roots),
                        ("correlation_roots", p.correlation_roots)):
        _indices(roots, p.n_roots_param, name, nonempty=False)
    selected = _indices(range(len(active)) if p.selected_no_indices is None else p.selected_no_indices,
                        len(active), "selected_no_indices")
    U = np.asarray(p.U_site_param)
    if U.shape not in ((), (evals.shape[1],)) or not np.all(np.isfinite(U)) or np.iscomplexobj(U):
        raise ValueError("U_site_param must be a finite real scalar or one value per site.")
    if U.shape != () and (p.lmo_pairs or p.mix_pairs or p.save_plots):
        raise ValueError("Legacy pair/plot orbital-U helpers require scalar U_site_param.")
    if p.save_plots and p.result_folder is None:
        raise ValueError("save_plots requires result_folder.")
    if p.mix_pair is not None:
        if len(_indices(p.mix_pair, len(selected), "mix_pair")) != 2:
            raise ValueError("mix_pair must contain exactly two indices.")
    elif p.mix_pairs:
        raise ValueError("mix_pairs requires mix_pair.")
    for pair in (*p.lmo_pairs, *p.mix_pairs):
        targets = _indices((pair.lmo_a, pair.lmo_b), len(selected), "pair targets")
        bridges = tuple(i for i in range(len(selected)) if i not in targets) if pair.bridge_indices is None else pair.bridge_indices
        bridges = _indices(bridges, len(selected), "bridge_indices", nonempty=pair.channels)
        if set(targets).intersection(bridges):
            raise ValueError("Bridge indices must not include target orbitals.")
    return active, selected

def _pair_analyses(obs, C, pairs, U_site):
    """Use the same observables, coefficient matrix and reference energy for each pair."""
    results = []
    for pair in pairs:
        kwargs = dict(lmo_a=pair.lmo_a, lmo_b=pair.lmo_b,
                      bridge_indices=pair.bridge_indices, H_key=pair.H_key, E_ref=pair.E_ref)
        downfolding = LMO.downfold_hopping_between_two_lmos(
            obs=obs, C_lmo=C, hubbard_U=U_site, **kwargs)
        channels = LMO.decompose_bridge_eigenchannels(obs=obs, **kwargs) if pair.channels else None
        results.append(dict(parameters=asdict(pair), downfolding=downfolding, channels=channels))
    return results

def run_cas_workflow(mfh_model, parameters: CASWorkflowParameters) -> dict:
    """Run canonical CAS, NO analysis, localization, correlations and optional MIX.

    Returns separate ``canonical``, ``lmo`` and ``mix`` dictionaries, plus
    ``cas_data`` and the exact parameters. Basis coefficients accompany every
    observable block. This transforms observables; only pair analyses eliminate
    bridge orbitals. No CAS rediagonalization is performed after localization.
    Text exports use descriptive stage names and overwrite matching files.
    """
    p = parameters
    active_indices, selected = _validate_model(mfh_model, p)
    saved_parameters = asdict(p)
    if p.result_folder is not None:
        saved_parameters["result_folder"] = str(p.result_folder)
    output = None if p.result_folder is None else Path(p.result_folder) / f"CAS({p.n_elec_param},{len(active_indices)})"
    context = nullcontext() if output is None else result_log(output / "workflow_log.txt")
    with context:
        # Project site Hubbard U into canonical MOs and solve the fixed-N CAS.
        cas_data = CAS.run_hubbard_cas_from_mfh_eigensystem_canonical(
            mfh_model=mfh_model, U_site=p.U_site_param, CAS_indices=active_indices,
            n_elec=p.n_elec_param, n_roots=p.n_roots_param, spin=p.spin_param,
            coeff_cutoff=p.coefficient_cutoff)
        state = cas_data["state_data"][p.root_param]
        C_no_site = state["natural_orbitals_site"]
        canonical = dict(root=p.root_param, C_no_site=C_no_site,
                         natural_occupations=state["natural_occupations"],
                         reports={r: cas_data["state_data"][r] for r in p.report_roots},
                         expansion=CAS.print_cas_natural_orbital_expansion(
                             cas_data, root=p.root_param, cutoff=p.coefficient_cutoff),
                         spin_correlations={r: CAS.compute_spin_spin_correlation_matrix(cas_data, root=r)
                                            for r in p.correlation_roots})

        # Retain the notebook's real-orbital localization; reject complex norm loss.
        selected_no = C_no_site[:, selected]
        if np.max(np.abs(np.imag(selected_no))) > p.basis_tolerance:
            raise ValueError("The existing localization requires real natural orbitals.")
        C_no_part = np.real(selected_no)
        C_lmo, f_hist, angle_hist = LMO.pipek_mezey_localization_linesearch(C_no_part, **p.localization)
        _check_basis(C_lmo, cas_data["C_spatial"], p.basis_tolerance)
        obs_lmo = CAS.build_cas_lmo_observables(cas_data, C_lmo, root=p.root_param)
        lmo_data = dict(basis="CAS-LMO", root=p.root_param, C_lmo=C_lmo,
                        C_no_part=C_no_part, selected_no_indices=selected,
                        no_in_lmo=C_lmo.T @ C_no_part, f_hist=f_hist, angle_hist=angle_hist,
                        observables=obs_lmo, pairs=_pair_analyses(obs_lmo, C_lmo, p.lmo_pairs, p.U_site_param))

        # Rotate each root's spin operators into this one common localized basis.
        caches = {r: CAS.cache_active_spin_operator_expectations(cas_data, root=r) for r in p.correlation_roots}
        lmo_data["spin_correlations"] = {r: CAS.spin_correlation_from_cached_expectations(
            cache, obs_lmo["U_active_to_lmo"], connected=p.connected) for r, cache in caches.items()}

        mix_data = None
        if p.mix_pair is not None:
            # Replace two columns by their normalized sum/difference; retain all others.
            C_mix = LMO.mix_two_lmos_plus_minus(C_lmo, *p.mix_pair)
            _check_basis(C_mix, cas_data["C_spatial"], p.basis_tolerance)
            obs_mix = CAS.build_cas_lmo_observables(cas_data, C_mix, root=p.root_param)
            mix_data = dict(basis="CAS-LMO-MIX", root=p.root_param, C_lmo=C_mix,
                            mix_in_no=C_no_part.conj().T @ C_mix,
                            no_in_mix=C_mix.conj().T @ C_no_part,
                            weights_no_by_mix=np.abs(C_no_part.conj().T @ C_mix)**2,
                            observables=obs_mix,
                            pairs=_pair_analyses(obs_mix, C_mix, p.mix_pairs, p.U_site_param),
                            spin_correlations={r: CAS.spin_correlation_from_cached_expectations(
                                cache, obs_mix["U_active_to_lmo"], connected=p.connected) for r, cache in caches.items()})
            if p.include_mfh_mix_observables:
                # Preserve the original auxiliary MFH analysis under an explicit name.
                mfh_obs = LMO.compute_pm_lmo_observables_from_C_lmo_paired(
                    mfh_model, C_mix, active_indices, active_indices)
                mix_data["mfh_observables"] = mfh_obs
                mix_data["mfh_pair_proxy"] = LMO.extract_pm_lmo_pair_proxy(mfh_obs, *p.mix_pair)

        results = dict(parameters=saved_parameters, cas_data=cas_data, canonical=canonical,
                       lmo=lmo_data, mix=mix_data)
        if output is not None:
            # Separate stages prevent accidental reuse of CAS-LMO data in MIX analyses.
            save_results_txt(output / "parameters.txt", parameters=saved_parameters)
            save_results_txt(output / "CAS" / "canonical_data.txt", cas_data=cas_data, analysis=canonical)
            save_results_txt(output / "CAS-LMO" / "localized_analysis.txt", analysis=lmo_data)
            if mix_data is not None:
                save_results_txt(output / "CAS-LMO-MIX" / "mixed_analysis.txt", analysis=mix_data)
        if p.save_plots:
            CAS.plot_cas_natural_orbitals(mfh_model, cas_data, root=p.root_param,
                save_dir=output / "CAS", prefix=f"CAS_root{p.root_param}", show=p.show_plots)
            CAS.plot_spin_spin_correlation(cas_data, root=p.root_param,
                save_path=output / "CAS" / f"spin_correlation_root{p.root_param}.pdf", show=p.show_plots)
            for folder, data in (("CAS-LMO", lmo_data), ("CAS-LMO-MIX", mix_data)):
                if data is not None:
                    LMO.plot_and_save_common_lmos(mfh_model, data["C_lmo"], u=p.U_site_param,
                        save_dir=output / folder, prefix=f"{folder}_root{p.root_param}", show=p.show_plots)
        return results

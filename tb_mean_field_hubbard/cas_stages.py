"""Interactive CAS stages: choose each stage's parameters after inspecting its inputs."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from . import CAS, LMO
from .cas_output import result_log, save_results_txt
from .cas_workflow import CASWorkflowParameters, PairAnalysis, _validate_model, _indices, _check_basis


@dataclass
class OrbitalBasis:
    """Orbital columns and their originating CAS solution; no observables implied."""
    cas_data: dict
    name: str
    C_lmo: np.ndarray
    localization_root: int
    details: dict


@dataclass
class BasisObservables:
    """Keep a root's observables attached to the exact orbital basis used."""
    basis: OrbitalBasis
    root: int
    data: dict


class CASSession:
    """Inspect a solved CAS calculation without rerunning earlier stages.

    Methods return independent stage results. Changing the NO selection creates
    a new basis; old observable objects remain attached to their original basis.
    Numerical kernels remain in CAS.py and LMO.py.
    """

    def __init__(self, model, cas_data, output_dir=None, basis_tolerance=1e-8):
        self.model = model
        self.cas_data = cas_data
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.basis_tolerance = basis_tolerance
        self._spin_cache = {}

    @contextmanager
    def _stage(self, folder, name):
        path = None if self.output_dir is None else self.output_dir / folder
        with nullcontext() if path is None else result_log(path / f"{name}_log.txt"):
            yield path

    @staticmethod
    def _save(path, name, **data):
        if path is not None:
            save_results_txt(path / f"{name}_data.txt", **data)

    def _roots(self, roots):
        return _indices(roots, len(self.cas_data['result'].energies), 'roots')

    def _basis(self, basis):
        if basis.cas_data is not self.cas_data:
            raise ValueError('This basis belongs to a different CAS solution.')
        _check_basis(basis.C_lmo, self.cas_data['C_spatial'], self.basis_tolerance)

    def _plot_requirements(self, uniform_U=False):
        """Validate exports before starting an expensive numerical stage."""
        if self.output_dir is None:
            raise ValueError('Plot export requires result_folder.')
        if uniform_U:
            U = self.cas_data['result'].U_site
            if not np.allclose(U, U[0], atol=0, rtol=0):
                raise ValueError('The legacy orbital-U plot helper requires uniform site U.')

    def _plot_basis(self, basis, path, show):
        self._plot_requirements(uniform_U=True)
        U = self.cas_data['result'].U_site
        LMO.plot_and_save_common_lmos(self.model, basis.C_lmo, u=float(U[0]),
            save_dir=path, prefix=f'{basis.name}_root{basis.localization_root}', show=show)

    def report_roots(self, report_roots=(0,)):
        """Inspect energies and natural occupations before choosing an analysis root."""
        roots = self._roots(report_roots)
        with self._stage('CAS', 'root_report') as path:
            reports = {}
            energies = np.asarray(self.cas_data['result'].energies)
            energy_table = pd.DataFrame({'energy_eV': energies,
                                         'excitation_eV': energies - energies[0]})
            energy_table.index.name = 'root'
            print(energy_table.to_string())
            for root in roots:
                reports[root] = self.cas_data['state_data'][root]['natural_occupations']
                print(f'Root {root} natural occupations:', reports[root])
            self._save(path, 'root_report', report_roots=roots, occupations=reports,
                       energies=self.cas_data['result'].energies, energy_table=energy_table)
            return reports

    def natural_orbitals(self, root_param=0, cutoff_param=1e-3, save_plots=False, show=False):
        """Inspect one root's NO coefficients and occupations before selecting NOs."""
        self._roots((root_param,))
        if save_plots:
            self._plot_requirements()
        name = f'natural_orbitals_root{root_param}'
        with self._stage('CAS', name) as path:
            state = self.cas_data['state_data'][root_param]
            table = CAS.print_cas_natural_orbital_expansion(self.cas_data, root_param, cutoff_param)
            result = dict(root=root_param, C_no_site=state['natural_orbitals_site'],
                          occupations=state['natural_occupations'], expansion=table)
            if save_plots:
                if path is None:
                    raise ValueError('Plot export requires result_folder.')
                CAS.plot_cas_natural_orbitals(self.model, self.cas_data, root=root_param,
                    save_dir=path, prefix=f'CAS_root{root_param}', show=show)
            self._save(path, name, cutoff=cutoff_param, **result)
            return result

    def canonical_correlations(self, correlation_roots=(0,), save_plots=False, show=False):
        """Compute raw spin correlations in the active canonical-MO basis."""
        roots = self._roots(correlation_roots)
        if save_plots:
            self._plot_requirements()
        with self._stage('CAS', 'canonical_correlations') as path:
            result = {}
            for root in roots:
                result[root] = CAS.print_spin_spin_correlation(self.cas_data, root=root)
                if save_plots:
                    if path is None:
                        raise ValueError('Plot export requires result_folder.')
                    CAS.plot_spin_spin_correlation(self.cas_data, root=root,
                        save_path=path / f'spin_correlation_root{root}.pdf', show=show)
            self._save(path, 'canonical_correlations', correlation_roots=roots, correlations=result)
            return result

    def localize(self, root_param, selected_no_indices, *, site_groups=None,
                 max_iter=1000, tol=1e-6, verbose=True, max_angle_deg=45.0,
                 n_grid=181, save_plots=False, show=False):
        """Localize chosen NOs only; inspect the resulting LMOs before choosing pairs."""
        self._roots((root_param,))
        selected = _indices(selected_no_indices, self.cas_data['C_spatial'].shape[1], 'selected_no_indices')
        if save_plots:
            self._plot_requirements(uniform_U=True)
        name = f'localization_root{root_param}'
        with self._stage('CAS-LMO', name) as path:
            state = self.cas_data['state_data'][root_param]
            C_no = state['natural_orbitals_site'][:, selected]
            if np.max(np.abs(C_no.imag)) > self.basis_tolerance:
                raise ValueError('The existing localization requires real natural orbitals.')
            C_no = C_no.real
            options = dict(site_groups=site_groups, max_iter=max_iter, tol=tol,
                           verbose=verbose, max_angle_deg=max_angle_deg, n_grid=n_grid)
            C_lmo, f_hist, angle_hist = LMO.pipek_mezey_localization_linesearch(C_no, **options)
            _check_basis(C_lmo, self.cas_data['C_spatial'], self.basis_tolerance)
            details = dict(selected_no_indices=selected, C_no_part=C_no,
                           no_in_lmo=C_lmo.T @ C_no, f_hist=f_hist, angle_hist=angle_hist)
            basis = OrbitalBasis(self.cas_data, 'CAS-LMO', C_lmo, root_param, details)
            print('Selected NO occupations:', state['natural_occupations'][list(selected)])
            print('NO expansion: rows = selected NOs, columns = LMOs')
            print(pd.DataFrame(details['no_in_lmo'].T, index=selected))
            if save_plots:
                self._plot_basis(basis, path, show)
            self._save(path, name, root=root_param, C_lmo=C_lmo, options=options, **details)
            return basis

    def observables(self, basis, root_param=0):
        """Build CAS observables in the supplied basis; this is not downfolding."""
        self._basis(basis)
        self._roots((root_param,))
        name = f'observables_root{root_param}'
        with self._stage(basis.name, name) as path:
            data = CAS.build_cas_lmo_observables(self.cas_data, basis.C_lmo, root=root_param)
            LMO.print_pm_lmo_observables_summary(data)
            self._save(path, name, root=root_param, C_lmo=basis.C_lmo, observables=data)
            return BasisObservables(basis, root_param, data)

    def spin_correlations(self, basis, correlation_roots=(0,), connected=True):
        """Transform cached CI spin operators into a chosen LMO or MIX basis."""
        self._basis(basis)
        roots = self._roots(correlation_roots)
        transform = self.cas_data['C_spatial'].conj().T @ basis.C_lmo
        with self._stage(basis.name, 'spin_correlations') as path:
            result = {}
            for root in roots:
                if root not in self._spin_cache:
                    self._spin_cache[root] = CAS.cache_active_spin_operator_expectations(self.cas_data, root)
                result[root] = CAS.spin_correlation_from_cached_expectations(
                    self._spin_cache[root], transform, connected=connected)
                print(f'Root {root}: connected={connected}; S2={result[root][3]}')
                print(result[root][0])
            self._save(path, 'spin_correlations', correlation_roots=roots, connected=connected,
                       C_lmo=basis.C_lmo, correlations=result)
            return result

    def _pair(self, observations, pair, expected_basis=None):
        self._basis(observations.basis)
        if expected_basis is not None:
            self._basis(expected_basis)
            if observations.basis is not expected_basis:
                raise ValueError(
                    'Observables belong to an earlier or different basis. '
                    'Rerun the observables cell for the current basis before pair analysis.'
                )
        n = observations.basis.C_lmo.shape[1]
        targets = _indices((pair.lmo_a, pair.lmo_b), n, 'targets')
        bridges = tuple(i for i in range(n) if i not in targets) if pair.bridge_indices is None else pair.bridge_indices
        bridges = _indices(bridges, n, 'bridge_indices', nonempty=False)
        if set(targets).intersection(bridges):
            raise ValueError('Bridge indices must exclude target orbitals.')
        return dict(lmo_a=pair.lmo_a, lmo_b=pair.lmo_b, bridge_indices=bridges,
                    H_key=pair.H_key, E_ref=pair.E_ref)

    def downfold(self, observations, pair, *, expected_basis=None):
        """Eliminate bridges; expected_basis rejects stale notebook observables."""
        options = self._pair(observations, pair, expected_basis)
        U = self.cas_data['result'].U_site
        if not np.allclose(U, U[0], atol=0, rtol=0):
            raise ValueError('The legacy pair orbital-U helper requires uniform site U.')
        name = f'downfold_root{observations.root}_{pair.lmo_a}_{pair.lmo_b}'
        with self._stage(observations.basis.name, name) as path:
            result = LMO.downfold_hopping_between_two_lmos(observations.data,
                C_lmo=observations.basis.C_lmo, hubbard_U=float(U[0]), **options)
            LMO.print_downfold_hopping_result(result)
            self._save(path, name, parameters=options, C_lmo=observations.basis.C_lmo, result=result)
            return result

    def channels(self, observations, pair, *, expected_basis=None):
        """Decompose bridge hopping, optionally checking the current notebook basis."""
        options = self._pair(observations, pair, expected_basis)
        if not options['bridge_indices']:
            raise ValueError('Channel decomposition requires bridge orbitals.')
        name = f'channels_root{observations.root}_{pair.lmo_a}_{pair.lmo_b}'
        with self._stage(observations.basis.name, name) as path:
            result = LMO.decompose_bridge_eigenchannels(observations.data, **options)
            print(result['channel_table'])
            print('Channel sum:', result['t_bridge_sum_channels'])
            print('Exact bridge contribution:', result['t_bridge_exact'])
            self._save(path, name, parameters=options, C_lmo=observations.basis.C_lmo, result=result)
            return result

    def mix(self, basis, mix_pair, save_plots=False, show=False):
        """Form and inspect plus/minus orbitals before selecting MIX analysis pairs."""
        self._basis(basis)
        pair = _indices(mix_pair, basis.C_lmo.shape[1], 'mix_pair')
        if len(pair) != 2:
            raise ValueError('mix_pair must contain two distinct orbital indices.')
        if save_plots:
            self._plot_requirements(uniform_U=True)
        name = f'mixing_{pair[0]}_{pair[1]}'
        with self._stage('CAS-LMO-MIX', name) as path:
            C_mix = LMO.mix_two_lmos_plus_minus(basis.C_lmo, *pair)
            _check_basis(C_mix, self.cas_data['C_spatial'], self.basis_tolerance)
            C_no = basis.details['C_no_part']
            details = dict(mix_pair=pair, C_no_part=C_no,
                           mix_in_no=C_no.conj().T @ C_mix, no_in_mix=C_mix.conj().T @ C_no)
            result = OrbitalBasis(self.cas_data, 'CAS-LMO-MIX', C_mix, basis.localization_root, details)
            print('Mixed orbitals: rows = NOs, columns = MIX orbitals')
            print(details['mix_in_no'])
            print('Expansion weights:')
            print(np.abs(details['mix_in_no'])**2)
            if save_plots:
                self._plot_basis(result, path, show)
            self._save(path, name, localization_root=basis.localization_root, C_lmo=C_mix, **details)
            return result

    def mfh_observables(self, basis):
        """Optional MFH occupation-based comparison, separate from CAS observables."""
        self._basis(basis)
        indices = self.cas_data['CAS_indices']
        with self._stage(basis.name, 'mfh_observables') as path:
            result = LMO.compute_pm_lmo_observables_from_C_lmo_paired(
                self.model, basis.C_lmo, indices, indices)
            LMO.print_pm_lmo_observables_summary(result)
            self._save(path, 'mfh_observables', C_lmo=basis.C_lmo, observables=result)
            return result


def solve_canonical_cas(mfh_model, *, active_indices, U_site_param, n_elec_param,
                        n_roots_param=6, spin_param=0, result_folder=None,
                        basis_tolerance=1e-8):
    """Solve only CAS after inspecting TB; defer all root and basis selections."""
    parameters = CASWorkflowParameters(active_indices, U_site_param, n_elec_param,
        n_roots_param=n_roots_param, spin_param=spin_param, basis_tolerance=basis_tolerance)
    active, _ = _validate_model(mfh_model, parameters)
    output = None if result_folder is None else Path(result_folder) / f'CAS({n_elec_param},{len(active)})'
    context = nullcontext() if output is None else result_log(output / 'CAS' / 'solve_log.txt')
    with context:
        cas_data = CAS.run_hubbard_cas_from_mfh_eigensystem_canonical(
            mfh_model, U_site=U_site_param, CAS_indices=active, n_elec=n_elec_param,
            n_roots=n_roots_param, spin=spin_param)
        if output is not None:
            save_results_txt(output / 'CAS' / 'canonical_data.txt', cas_data=cas_data,
                active_indices=active, U_site_param=U_site_param, n_elec_param=n_elec_param,
                n_roots_param=n_roots_param, spin_param=spin_param)
    return CASSession(mfh_model, cas_data, output, basis_tolerance)

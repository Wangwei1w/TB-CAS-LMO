"""Small-system numerical and basis-consistency regressions for the public workflow."""
from dataclasses import replace
from types import SimpleNamespace
import ast
import io
import json
from pathlib import Path
import tokenize
import numpy as np
import pytest
from tb_mean_field_hubbard import CAS, LMO
from tb_mean_field_hubbard.cas_workflow import CASWorkflowParameters, PairAnalysis, run_cas_workflow


@pytest.fixture
def model():
    h = np.array([[-1.2, .3, .11], [.3, .45, -.2], [.11, -.2, 1.7]])
    energies, vectors = np.linalg.eigh(h)
    def plot_eigenvector(ax, evec, title=None):
        ax.scatter(np.arange(3), np.abs(evec)**2)
    return SimpleNamespace(evals=np.array([energies, energies]),
                           evecs=np.array([vectors.T, vectors.T]),
                           num_spin_el=[1, 1], one_electron_U=0,
                           pp=SimpleNamespace(plot_eigenvector=plot_eigenvector))


@pytest.fixture
def parameters():
    return CASWorkflowParameters(
        active_indices=(0, 1, 2), U_site_param=2.3, n_elec_param=2,
        n_roots_param=3, correlation_roots=(0, 1), report_roots=(0, 1),
        localization=dict(max_iter=10, n_grid=19, tol=1e-6, verbose=False),
        lmo_pairs=(PairAnalysis(0, 1, (2,), E_ref=4.5),),
        mix_pair=(0, 1), mix_pairs=(PairAnalysis(0, 1, (2,), E_ref=4.5),))


def test_workflow_matches_explicit_low_level_sequence(model, parameters):
    result = run_cas_workflow(model, parameters)
    cas_data = CAS.run_hubbard_cas_from_mfh_eigensystem_canonical(
        model, U_site=2.3, CAS_indices=(0, 1, 2), n_elec=2, n_roots=3,
        print_summary=False)
    np.testing.assert_allclose(result['cas_data']['result'].H, cas_data['result'].H)
    np.testing.assert_allclose(result['cas_data']['result'].energies, cas_data['result'].energies)
    C_no = cas_data['state_data'][0]['natural_orbitals_site'].real
    C_lmo, f_hist, angle_hist = LMO.pipek_mezey_localization_linesearch(C_no, **parameters.localization)
    np.testing.assert_allclose(result['lmo']['C_lmo'], C_lmo)
    np.testing.assert_allclose(result['lmo']['f_hist'], f_hist)
    np.testing.assert_allclose(result['lmo']['angle_hist'], angle_hist)
    for key, C in [('lmo', C_lmo), ('mix', LMO.mix_two_lmos_plus_minus(C_lmo, 0, 1))]:
        obs = CAS.build_cas_lmo_observables(cas_data, C, root=0)
        for name, value in obs.items():
            np.testing.assert_allclose(result[key]['observables'][name], value)
        pair = result[key]['pairs'][0]
        direct = LMO.downfold_hopping_between_two_lmos(
            obs, 0, 1, (2,), E_ref=4.5, C_lmo=C, hubbard_U=2.3)
        for name, value in direct.items():
            if isinstance(value, (np.ndarray, float, complex, int)):
                np.testing.assert_allclose(pair['downfolding'][name], value)
        channels = pair['channels']
        np.testing.assert_allclose(channels['t_bridge_sum_channels'], channels['t_bridge_exact'], atol=1e-12)
        for root in parameters.correlation_roots:
            expected = CAS.spin_correlation_from_cached_expectations(
                CAS.cache_active_spin_operator_expectations(cas_data, root),
                obs['U_active_to_lmo'], connected=True)
            for actual, value in zip(result[key]['spin_correlations'][root], expected):
                np.testing.assert_allclose(actual, value)
        np.testing.assert_allclose(C.conj().T @ C, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.trace(obs['rho']), 2, atol=1e-12)
    Q = C_lmo.conj().T @ result['mix']['C_lmo']
    for name in ('H_eff_avg', 'rho'):
        np.testing.assert_allclose(result['mix']['observables'][name],
            Q.conj().T @ result['lmo']['observables'][name] @ Q, atol=1e-12)
    mfh_obs = LMO.compute_pm_lmo_observables_from_C_lmo_paired(
        model, result['mix']['C_lmo'], (0, 1, 2), (0, 1, 2))
    np.testing.assert_allclose(result['mix']['mfh_observables']['B'], mfh_obs['B'])
    np.testing.assert_allclose(result['mix']['C_lmo'] @ result['mix']['no_in_mix'], C_no, atol=1e-12)


def test_u_zero_is_separate_from_cas_u(model, parameters):
    interacting = run_cas_workflow(model, parameters)
    # The preserved 4|t|^2/U proxy is singular at U=0; CAS energies remain defined.
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        noninteracting = run_cas_workflow(model, replace(parameters, U_site_param=0))
    assert not np.allclose(interacting['cas_data']['result'].energies,
                           noninteracting['cas_data']['result'].energies)
    assert model.one_electron_U == 0
    np.testing.assert_allclose(noninteracting['cas_data']['result'].energies[0], 2*model.evals[0, 0])
    model.one_electron_U = 2.3
    with pytest.raises(ValueError, match='MFH U'):
        run_cas_workflow(model, parameters)


def test_phase_invariance_and_spin_independence(model, parameters):
    model.evecs[1, 0] *= -1
    run_cas_workflow(model, parameters)
    model.evals[1, 0] += .1
    with pytest.raises(ValueError, match='spin-independent'):
        run_cas_workflow(model, parameters)


@pytest.mark.parametrize('changes', [
    dict(active_indices=(0, 0)), dict(active_indices=(-1, 1)),
    dict(root_param=3), dict(correlation_roots=(4,)), dict(n_elec_param=7),
    dict(n_roots_param=16), dict(mix_pair=(0, 0)), dict(mix_pair=None),
    dict(lmo_pairs=(PairAnalysis(0, 1, (1,)),)),
    dict(selected_no_indices=(0, 0)), dict(U_site_param=[1, 2]),
    dict(save_plots=True),
])
def test_invalid_parameters_fail_before_calculation(model, parameters, changes):
    with pytest.raises(ValueError):
        run_cas_workflow(model, replace(parameters, **changes))


def test_selected_subspace_and_nonzero_root(model, parameters):
    p = replace(parameters, root_param=1, selected_no_indices=(0, 1),
                lmo_pairs=(), mix_pairs=(), correlation_roots=(1,))
    result = run_cas_workflow(model, p)
    assert result['lmo']['C_lmo'].shape == (3, 2)
    expected = CAS.build_cas_lmo_observables(result['cas_data'], result['mix']['C_lmo'], root=1)
    np.testing.assert_allclose(result['mix']['observables']['rho'], expected['rho'])


def test_exports_and_plots(model, parameters, tmp_path):
    result = run_cas_workflow(model, replace(parameters, result_folder=tmp_path,
                                           save_plots=True, show_plots=False))
    output = tmp_path/'CAS(2,3)'
    assert (output/'CAS-LMO-MIX'/'mixed_analysis.txt').exists()
    assert (output/'CAS'/'canonical_data.txt').exists()
    assert len(list(output.rglob('*.pdf'))) == 10
    assert result['mix']['basis'] == 'CAS-LMO-MIX'


def test_comments_and_notebook_syntax():
    root = Path(__file__).resolve().parents[1]
    sources = [p.read_text(encoding='utf-8-sig') for p in (root/'tb_mean_field_hubbard').glob('*.py')]
    for path in root.glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        sources.extend(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    for source in sources:
        tree = ast.parse(source)
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                assert not any('\u4e00' <= char <= '\u9fff' for char in token.string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                assert not any('\u4e00' <= char <= '\u9fff' for char in (ast.get_docstring(node) or ''))


def test_degenerate_spin_eigensystems_are_allowed(model, parameters):
    model.evals[:] = [-1., -1., 2.]
    rotation = np.array([[.6, -.8, 0], [.8, .6, 0], [0, 0, 1]])
    model.evecs[1] = rotation @ model.evecs[0]
    result = run_cas_workflow(model, parameters)
    assert len(result["cas_data"]["result"].energies) == 3


def test_optional_mix_and_site_dependent_u(model, parameters):
    p = replace(parameters, U_site_param=np.array([1., 2., 3.]),
                lmo_pairs=(), mix_pair=None, mix_pairs=(), correlation_roots=())
    result = run_cas_workflow(model, p)
    assert result["mix"] is None
    assert result["lmo"]["pairs"] == []
    np.testing.assert_allclose(result["cas_data"]["result"].U_site, [1., 2., 3.])


def test_complex_localization_input_is_rejected(model, parameters):
    model.evecs = model.evecs.astype(complex)
    model.evecs[:, :, 0] *= 1j
    with pytest.raises(ValueError, match="real natural orbitals"):
        run_cas_workflow(model, parameters)


def test_real_mfh_model_smoke():
    from ase import Atoms
    from tb_mean_field_hubbard import MeanFieldHubbardModel
    atoms = Atoms('C4', positions=[[0, 0, 0], [1.42, 0, 0],
                                  [2.84, 0, 0], [4.26, 0, 0]])
    model = MeanFieldHubbardModel(atoms, t_list=[2.7], charge=0, multiplicity=1)
    model.run_mfh(u=0)
    p = CASWorkflowParameters(active_indices=(1, 2), U_site_param=3.5,
                              n_elec_param=2, n_roots_param=2, mix_pair=(0, 1),
                              localization=dict(max_iter=10, n_grid=19, verbose=False))
    result = run_cas_workflow(model, p)
    assert model.one_electron_U == 0
    assert result['mix']['C_lmo'].shape == (4, 2)
    np.testing.assert_allclose(np.trace(result['mix']['observables']['rho']), 2, atol=1e-12)


def test_interactive_stages_match_batch_without_recomputing(model, parameters, monkeypatch):
    from tb_mean_field_hubbard.cas_stages import solve_canonical_cas
    expected = run_cas_workflow(model, parameters)
    session = solve_canonical_cas(model, active_indices=(0, 1, 2), U_site_param=2.3,
                                  n_elec_param=2, n_roots_param=3)
    def forbid_solver(*args, **kwargs):
        raise AssertionError('An inspection stage must not rerun CAS.')
    monkeypatch.setattr(CAS, 'run_hubbard_cas_from_mfh_eigensystem_canonical', forbid_solver)
    reports = session.report_roots((0, 1))
    no = session.natural_orbitals(root_param=1)
    np.testing.assert_allclose(no['occupations'], reports[1])
    session.canonical_correlations((0,))
    basis = session.localize(0, (0, 1, 2), **parameters.localization)
    mixed = session.mix(basis, (0, 1))
    for key, current_basis in [('lmo', basis), ('mix', mixed)]:
        obs = session.observables(current_basis, root_param=0)
        np.testing.assert_allclose(obs.data['rho'], expected[key]['observables']['rho'])
        pair = parameters.lmo_pairs[0]
        down = session.downfold(obs, pair)
        channels = session.channels(obs, pair)
        np.testing.assert_allclose(channels['t_bridge_exact'], expected[key]['pairs'][0]['channels']['t_bridge_exact'])
        correlations = session.spin_correlations(current_basis, (0, 1))
        for root in correlations:
            for actual, target in zip(correlations[root], expected[key]['spin_correlations'][root]):
                np.testing.assert_allclose(actual, target)
    assert len(session._spin_cache) == 2
    # A new NO selection creates a separate basis without mutating prior observables.
    new_basis = session.localize(0, (0, 1), **parameters.localization)
    assert new_basis.C_lmo.shape == (3, 2)
    assert obs.basis.C_lmo.shape == (3, 3)
    np.testing.assert_allclose(session.mfh_observables(mixed)['B'], expected['mix']['mfh_observables']['B'])


def test_interactive_exports_and_foreign_basis(model, parameters, tmp_path):
    from tb_mean_field_hubbard.cas_stages import solve_canonical_cas, CASSession
    session = solve_canonical_cas(model, active_indices=(0, 1, 2), U_site_param=2.3,
        n_elec_param=2, n_roots_param=3, result_folder=tmp_path)
    session.report_roots((0, 1))
    session.natural_orbitals(0, save_plots=True)
    session.canonical_correlations((0,), save_plots=True)
    basis = session.localize(0, (0, 1, 2), save_plots=True, **parameters.localization)
    mixed = session.mix(basis, (0, 1), save_plots=True)
    for current in (basis, mixed):
        obs = session.observables(current, 0)
        session.spin_correlations(current, (0,))
        session.downfold(obs, parameters.lmo_pairs[0])
        session.channels(obs, parameters.lmo_pairs[0])
    session.mfh_observables(mixed)
    assert len(list(tmp_path.rglob('*.pdf'))) == 10
    assert (tmp_path/'CAS(2,3)'/'CAS-LMO-MIX'/'observables_root0_data.txt').exists()
    foreign = CASSession(model, dict(session.cas_data))
    with pytest.raises(ValueError, match='different CAS solution'):
        foreign.observables(basis)
    with pytest.raises(ValueError):
        session.localize(0, (3,))
    with pytest.raises(ValueError):
        session.report_roots((3,))
    with pytest.raises(ValueError):
        session.mix(basis, (0, 0))
    with pytest.raises(ValueError):
        session.downfold(obs, PairAnalysis(0, 1, (1,)))


def test_notebook_uses_separate_interactive_stages():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/'mfh_test.ipynb').read_text(encoding='utf-8'))
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'run_cas_workflow(' not in source
    assert 'CASWorkflowParameters(' not in source
    stages = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (
            isinstance(n.func, ast.Name) and n.func.id == 'solve_canonical_cas' or
            isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'cas_session')]
        if calls:
            assert len(calls) == 1
            stages.append(calls[0])
    assert len(stages) == 15


def test_pair_stages_reject_stale_basis(model, parameters):
    from tb_mean_field_hubbard.cas_stages import solve_canonical_cas
    session = solve_canonical_cas(model, active_indices=(0, 1, 2), U_site_param=2.3,
                                  n_elec_param=2, n_roots_param=3)
    first = session.localize(0, (0, 1, 2), **parameters.localization)
    old_obs = session.observables(first)
    current = session.localize(1, (0, 1, 2), **parameters.localization)
    pair = parameters.lmo_pairs[0]
    for method in (session.downfold, session.channels):
        with pytest.raises(ValueError, match='Rerun the observables cell'):
            method(old_obs, pair, expected_basis=current)
    current_obs = session.observables(current, root_param=1)
    session.downfold(current_obs, pair, expected_basis=current)
    session.channels(current_obs, pair, expected_basis=current)
    mixed = session.mix(current, (0, 1))
    with pytest.raises(ValueError, match='Rerun the observables cell'):
        session.downfold(current_obs, pair, expected_basis=mixed)
    # Advanced callers can still explicitly analyze an earlier basis.
    session.downfold(old_obs, pair, expected_basis=first)


def test_plot_configuration_fails_before_numerical_work(model, monkeypatch):
    from tb_mean_field_hubbard.cas_stages import solve_canonical_cas
    session = solve_canonical_cas(model, active_indices=(0, 1, 2), U_site_param=2.3,
                                  n_elec_param=2, n_roots_param=3)
    def forbid(*args, **kwargs):
        raise AssertionError('Plot configuration must be checked first.')
    monkeypatch.setattr(LMO, 'pipek_mezey_localization_linesearch', forbid)
    monkeypatch.setattr(CAS, 'print_spin_spin_correlation', forbid)
    monkeypatch.setattr(CAS, 'print_cas_natural_orbital_expansion', forbid)
    with pytest.raises(ValueError, match='result_folder'):
        session.localize(0, (0, 1, 2), save_plots=True)
    with pytest.raises(ValueError, match='result_folder'):
        session.canonical_correlations((0,), save_plots=True)
    with pytest.raises(ValueError, match='result_folder'):
        session.natural_orbitals(0, save_plots=True)


def test_execute_notebook_stages_with_nonzero_root(model, tmp_path, monkeypatch):
    """Execute actual notebook cells on a small model with only example inputs reduced."""
    notebook = json.loads((Path(__file__).resolve().parents[1]/'mfh_test.ipynb').read_text(encoding='utf-8'))
    state = {'mfh_model': model}
    solver_calls = []
    original_solver = CAS.run_hubbard_cas_from_mfh_eigensystem_canonical
    def counted_solver(*args, **kwargs):
        solver_calls.append(1)
        return original_solver(*args, **kwargs)
    monkeypatch.setattr(CAS, 'run_hubbard_cas_from_mfh_eigensystem_canonical', counted_solver)
    class SmallInputs(ast.NodeTransformer):
        def visit_Assign(self, node):
            values = dict(active_indices=[0, 1, 2], n_elec_param=2, n_roots_param=3,
                          report_roots=(0, 1), correlation_roots=(0, 1),
                          selected_no_indices=(0, 1, 2), max_iter_param=10,
                          n_grid_param=19, show_param=False, result_folder=str(tmp_path))
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in values:
                    node.value = ast.parse(repr(values[key]), mode='eval').body
                elif key == 'root_param' and isinstance(node.value, ast.Constant):
                    node.value = ast.Constant(1)
            return self.generic_visit(node)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'PairAnalysis':
                return ast.parse('PairAnalysis(0, 1, (2,), E_ref=4.5)', mode='eval').body
            for keyword in node.keywords:
                if keyword.arg == 'save_plots':
                    keyword.value = ast.Constant(False)
            return self.generic_visit(node)
    count = 0
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] != 'code' or not ('cas_session.' in source or 'cas_session = solve_canonical_cas(' in source):
            continue
        tree = ast.fix_missing_locations(SmallInputs().visit(ast.parse(source)))
        exec(compile(tree, '<notebook stage>', 'exec'), state)
        count += 1
    assert count == 15
    assert len(solver_calls) == 1
    assert state['no_analysis']['root'] == 1
    assert state['lmo_basis'].localization_root == 1
    assert state['lmo_observables'].root == 1
    assert state['mix_basis'].localization_root == 1
    assert state['mix_observables'].root == 1
    np.testing.assert_allclose(state['mix_channels']['t_bridge_sum_channels'],
                               state['mix_channels']['t_bridge_exact'])

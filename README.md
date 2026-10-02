# TB-CAS-LMO

Interactive tight-binding Hubbard CAS analysis for conjugated carbon systems:
canonical molecular orbitals, natural orbitals (NOs), localized molecular
orbitals (LMOs), spin correlations, and bridge-mediated hopping.

This project extends **[tb-mean-field-hubbard](https://github.com/eimrek/tb-mean-field-hubbard)**
by **Kristjan Eimre**. The original TB/MFH implementation remains the foundation;
this extension adds CAS/LMO analysis and a staged notebook interface. See
[Attribution and licensing](#attribution-and-licensing) for the original citation
and license information.

## Quick installation

For users already familiar with Python and Git:

`git clone https://github.com/Wangwei1w/TB-CAS-LMO.git`

`cd TB-CAS-LMO`

`python -m venv .venv`

`.venv\Scripts\Activate.ps1`

`python -m pip install --upgrade pip`

`python -m pip install -r requirements.txt`

`python -m pip install jupyterlab`

`jupyter lab mfh_cas_lmo.ipynb`

## System requirements

A Python 3 environment is required. The current TB-CAS-LMO workflow has been tested with **Python 3.12**.

Required Python libraries include:

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `ase`
- `pythtb` (**1.8.0**)
- `scikit-learn`
- `igor-tools`

The required libraries are listed in `requirements.txt` and can be installed automatically with `pip`.

JupyterLab is recommended for running the main notebook `mfh_cas_lmo.ipynb`.

## Installation

Open **Windows PowerShell** and move to the directory where you want to install the project, for example:

`cd C:\Users\YOUR_USERNAME\Documents`

Clone the repository:

`git clone https://github.com/Wangwei1w/TB-CAS-LMO.git`

Enter the repository directory:

`cd TB-CAS-LMO`

Create a dedicated Python environment:

`python -m venv .venv`

Activate it:

`.venv\Scripts\Activate.ps1`

Upgrade `pip`:

`python -m pip install --upgrade pip`

Install the required libraries:

`python -m pip install -r requirements.txt`

Install JupyterLab:

`python -m pip install jupyterlab`

Start the main notebook:

`jupyter lab mfh_cas_lmo.ipynb`

JupyterLab should open automatically in a web browser.

If the notebook does not use the same Python environment, install and register a Jupyter kernel:

`python -m pip install ipykernel`

`python -m ipykernel install --user --name tb-cas-lmo --display-name "Python (TB-CAS-LMO)"`

Then select `Python (TB-CAS-LMO)` as the notebook kernel.

## Optional: Visual Studio Code

Visual Studio Code is not required, but it can be used to edit the code and run `mfh_cas_lmo.ipynb`.

If using VS Code, install the **Python** and **Jupyter** extensions and select the `.venv` environment.

## Running the project later

After the initial installation, open PowerShell and run:

`cd C:\path\to\TB-CAS-LMO`

`.venv\Scripts\Activate.ps1`

`jupyter lab mfh_cas_lmo.ipynb`

The main notebook uses `geom/608M4.xyz`, a 44-site structure. Its two manual
hopping/onsite cells are special-purpose examples: **skip them for this
unmodified structure**, since their atom indices refer to a different system.
When manually overwriting a hopping term, the two molecular site indices must
be specified with the **larger index first**.
Geometry, orbital ranges, active spaces, and orbital pairs must be reviewed
when changing structures.

## Workflow: inspect, select, calculate

**Each CAS code cell contains its own parameters and one analysis call.**
Inspect each result before choosing the next stage's inputs. The notebook is
an interactive analysis, rather than a request to choose all parameters in
advance and run every cell without review.

| Stage | Choose after inspecting | Parameters to review | Call |
| --- | --- | --- | --- |
| Canonical-MO CAS | TB energies and orbitals | `active_indices`, `n_elec_param`, `n_roots_param`, `U_site_param` | `solve_canonical_cas` |
| Root report | CAS spectrum | `report_roots` | `report_roots` |
| Natural orbitals | Root reports | `root_param`, `cutoff_param` | `natural_orbitals` |
| CAS-LMO localization | NO plots and occupations | `selected_no_indices`, root, localization settings | `localize` |
| CAS-LMO observables | Localized orbitals | `root_param` | `observables` |
| CAS-LMO spin correlations | CAS roots and LMO basis | `correlation_roots`, `connected` | `spin_correlations` |
| CAS-LMO hopping downfolding | LMO plots and observables | Target pair, mediating subspace, `E_ref` | `downfold` |
| CAS-LMO bridge eigenchannels | Downfolding results | Same target/mediating partition | `channels` |
| Plus/minus mixing | LMO plots | `mix_pair` | `mix` |
| MIX observables | Mixed orbitals | `root_param` | `observables` |
| MIX hopping downfolding | MIX plots and observables | MIX target pair, mediating subspace, `E_ref` | `downfold` |
| MIX bridge eigenchannels | MIX downfolding results | Same MIX target/mediating partition | `channels` |
| Optional MIX spin correlations | CAS roots and MIX basis | `correlation_roots`, `connected` | `spin_correlations` |


All methods after the initial solver belong to the returned `cas_session`.
All orbital, site, and root indices are **zero-based**. NO indices refer to the
selected root's natural orbitals; LMO and MIX indices refer to columns of their
respective coefficient matrices. They are not interchangeable with the
original TB orbital indices.

### Reference-model U and CAS U

Prepare the one-electron reference with `mfh_model.run_mfh(u=0)`. This U is
separate from `U_site_param`, the site Hubbard interaction projected into the
CAS Hamiltonian. For example, a U=0 TB reference can be followed by CAS with
`U_site_param=3.5` eV.

The solver checks that the reconstructed up/down one-electron Hamiltonians
agree, allowing phases and rotations within degenerate orbital subspaces.
A recorded nonzero `one_electron_U` is rejected. For external models without
that metadata, spin independence alone does not establish a U=0 origin.

### Example stage calls

After inspecting TB, configure and solve CAS in one cell:

```python
from tb_mean_field_hubbard import PairAnalysis, solve_canonical_cas

active_indices = [19, 20, 21, 22, 23, 24]
n_elec_param = 6
n_roots_param = 6
U_site_param = 3.5

cas_session = solve_canonical_cas(
    mfh_model,
    active_indices=active_indices,
    U_site_param=U_site_param,
    n_elec_param=n_elec_param,
    n_roots_param=n_roots_param,
    spin_param=0,
    result_folder="results/example",
)
```

Inspect the spectrum, then select roots in a separate cell:

```python
report_roots = (0, 1)
root_report = cas_session.report_roots(report_roots=report_roots)
```

Select a root and inspect its NOs:

```python
root_param = 0
no_analysis = cas_session.natural_orbitals(
    root_param=root_param, save_plots=True, show=True,
)
```

Only then select the NO subspace for localization:

```python
root_param = no_analysis["root"]
selected_no_indices = (0, 1, 2, 3, 4, 5)
lmo_basis = cas_session.localize(
    root_param=root_param,
    selected_no_indices=selected_no_indices,
    save_plots=True,
    show=True,
)
```

The notebook provides the remaining cells. Build the LMO observables before
selecting a downfolding target pair and mediating subspace. Select a mixing
pair only after inspecting the localized orbitals. The localized-orbital
representation is not unique: for selected LMO pairs, symmetric and
antisymmetric linear combinations can yield a basis with a more intuitive
physical interpretation. Inspect the resulting MIX orbitals before choosing
the MIX targets and mediating subspace. The values above are example inputs,
not automatic active-space or orbital-selection rules.

## Results and repeated analysis

| Object | Contents |
| --- | --- |
| `cas_session.cas_data` | Canonical CAS Hamiltonian, energies, CI coefficients, and root data |
| `no_analysis` | Selected root, site-basis NO coefficients, occupations, and expansion table |
| `lmo_basis` / `mix_basis` | `OrbitalBasis` objects with `C_lmo`, `localization_root`, and `details` |
| `lmo_observables` / `mix_observables` | `BasisObservables` objects containing `.data`, `.root`, and their exact `.basis` |
| Downfolding / channel results | Dictionaries returned by the existing numerical helpers |

Root inspection and downstream analysis reuse the solved CAS data. Changing
`active_indices`, the electron count, the requested number of roots, or CAS U
requires a new solve. Changing the NO selection requires localization and its
dependent stages to be rerun. Changing only an analysis pair requires only the
corresponding downfolding/channel cells.

In the notebook, localization initially follows `no_analysis["root"]`, and
observables initially follow the basis's `localization_root`. These are
editable cell parameters; the method itself still accepts an explicit root
for comparisons in a fixed basis.

Each notebook pair-analysis call supplies `expected_basis=lmo_basis` or
`expected_basis=mix_basis`. If the basis was rebuilt but its observables were
not, the call asks for the observables cell to be rerun. Earlier observable
objects retain their earlier basis; they are not silently updated. Basis
objects from a different CAS solution are rejected. Spin-operator expectations
are cached by root and reused across localized analyses.

## Output files

With `result_folder="results/example"`, a CAS(6,6) calculation writes to:

```text
results/example/CAS(6,6)/
├── CAS/          # Solver, root reports, NOs, canonical correlations
├── CAS-LMO/      # Localization, observables, correlations, pair analyses
└── CAS-LMO-MIX/  # Mixed orbitals and their analyses
```

Stage calls save labeled text data and console logs. Plotting stages save PDFs
when `save_plots=True`; `show=True` also displays those plots. `result_folder=None`
disables file output, and plot export requires a result folder. Export settings
are checked before expensive localization or correlation work begins.

**Matching filenames are overwritten on rerun.** Use a different result folder
to preserve another parameter selection. Files from stages skipped on a later
run are not removed. Text exports are reports; keep the returned Python objects
for further computation.

## Physical conventions and interpretation

- **Hamiltonian:** Hubbard CAS uses selected canonical one-electron energies
  and the projected site interaction. It fixes total electron number, not a
  spin-adapted sector. No frozen-core or double-counting correction is added.
  Degenerate roots may have different eigenvector representations across
  numerical libraries.
- **Observables versus downfolding:** `observables` performs a basis
  transformation of the one-electron Hamiltonian and spin-summed 1-RDM into an
  LMO or MIX basis; it does not eliminate mediating orbitals. `downfold`
  separately eliminates the selected mediating orbitals through a one-electron
  resolvent. Bridge eigenchannel analysis decomposes the mediated contribution
  for the same target/mediating partition. In `downfold`, `E_ref=None` uses the
  mean target onsite energy.
- **Spin populations:** the existing CAS observable helper assigns
  `n_up = n_dn = n_tot/2`, `m=0`, and `J1_proxy=0`. These quantities do not provide
  the general spin-resolved density of an arbitrary CAS root. The separate
  spin-correlation methods evaluate spin operators from the CI state.
- **Correlation basis:** canonical correlations refer to active canonical MOs,
  not NOs. Localized correlations return `(connected_or_raw, raw, spin_vec, s2)`
  per root. If only a subset of NOs is localized, `s2` covers that subspace.
- **Localization:** the current implementation uses real coefficients.
  Significant imaginary components are rejected; localized columns are checked
  for orthonormality and membership in the CAS active subspace.
- **Exchange proxies:** hopping-derived proxies are not fitted many-body
  exchange constants. The expression `4|t|²/U` is singular at U=0 even when
  CAS energies are well-defined. Resonant bridge denominators retain the
  existing behavior; no broadening or regularization is applied.
- **Site-dependent U:** CAS supports a scalar or one U per site. The staged
  pair and orbital-plot helpers require uniform U and reduce it to a scalar.
  The older batch wrapper accepts only scalar U for those stages.
- **MFH comparison:** `mfh_observables` uses MFH occupations. Its output is
  separate from CAS observables and is not used by the notebook's CAS
  downfolding calls.

## Repository guide and tests

| File | Purpose |
| --- | --- |
| [mfh_cas_lmo.ipynb](mfh_cas_lmo.ipynb) | Main TB and staged CAS notebook |
| [cas_stages.py](tb_mean_field_hubbard/cas_stages.py) | Interactive solver and stage methods |
| [CAS.py](tb_mean_field_hubbard/CAS.py) | CAS, RDM, natural-orbital, and spin-correlation kernels |
| [LMO.py](tb_mean_field_hubbard/LMO.py) | Localization, mixing, hopping, and bridge-channel kernels |
| [cas_output.py](tb_mean_field_hubbard/cas_output.py) | Text exports and logging |
| [cas_workflow.py](tb_mean_field_hubbard/cas_workflow.py) | Optional batch interface for already-established selections |
| [tests/test_cas_workflow.py](tests/test_cas_workflow.py) | Numerical and workflow regression tests |

Advanced users may call the numerical kernels directly. `CAS1.py` remains a
legacy variant; it is not an alias of `CAS.py`. The independent examples are
not a sequential run-all workflow. The previous `Wanner_CAS` section is absent.

Run the test suite from the repository root:

```console
python -m pytest -q
```

The test suite covers numerical agreement with lower-level calls, basis
transformations, input validation, text/PDF exports, stale-result detection,
and execution of the notebook CAS stages on a small system. It checks
propagation of a nonzero root and verifies that downstream
analysis does not rerun CAS. A separate 44-site CAS(6,6) smoke run has also
validated the batch route, including LMO/MIX channels and plot export.

## Attribution and licensing

The original project is
[tb-mean-field-hubbard](https://github.com/eimrek/tb-mean-field-hubbard).
Its [package metadata](https://github.com/eimrek/tb-mean-field-hubbard/blob/e9c7dc0e833bb87cab9244987debc52e1b9036aa/setup.py)
identifies **Kristjan Eimre** as the author. This extension does not claim
authorship of the original TB/MFH implementation.

[README_original.md](README_original.md) is preserved unchanged and contains
the upstream software citation and Zenodo DOI information.

When citing the original TB/MFH implementation, use the upstream software
citation provided there. When reporting calculations performed with this
repository, also identify the version or commit of this extension that was
used. This repository currently has no separate Zenodo DOI.

[LICENSE](LICENSE) preserves the GNU GPL version 3 license text present in
the upstream repository at revision
`e9c7dc0e833bb87cab9244987debc52e1b9036aa`.

**The upstream licensing metadata is inconsistent:** the upstream
[LICENSE file](https://github.com/eimrek/tb-mean-field-hubbard/blob/e9c7dc0e833bb87cab9244987debc52e1b9036aa/LICENSE)
contains the GNU GPL version 3 text, whereas the `setup.py` metadata at the
same revision includes the classifier `License :: OSI Approved :: MIT License`.
This repository preserves the upstream GPL license text rather than assuming
that the metadata classifier supersedes it.

The CAS, CAS-LMO, effective-coupling, hopping-downfolding, bridge-channel, and
related workflow extensions in this repository were developed as modifications
and additions to the upstream codebase and are distributed as part of the
resulting GPL-3.0-covered work. Existing upstream copyright and attribution
notices are retained where applicable.

The exact upstream revision corresponding to the originally supplied local
source is not known. This documentation records, but does not attempt to
resolve, the inconsistency in the upstream licensing metadata.

Scientific references already present in
[utils.py](tb_mean_field_hubbard/utils.py) are retained:

- [Hydrogen-like wavefunctions and energies](https://en.wikipedia.org/wiki/Hydrogen-like_atom#Non-relativistic_wavefunction_and_energy)
- [Effective nuclear charge](https://en.wikipedia.org/wiki/Effective_nuclear_charge)
- [Slater's rules](https://en.wikipedia.org/wiki/Slater%27s_rules)
- [DOI 10.1038/s41557-019-0316-8](https://doi.org/10.1038/s41557-019-0316-8), cited in the source for the carbon 2pz effective-charge value.

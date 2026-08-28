# Project and Documentation Conventions

The public project name is **RIG-Hazard**, and the primary probability model is
`local_weather_hazard`. New files, configurations, result directories, and
user-facing reports use stable functional names instead of iteration or phase
numbers.

## Directory responsibilities

| Directory | Responsibility | Naming pattern |
| --- | --- | --- |
| `src/rig_hazard/` | Reusable model, controller, and evaluation code | Domain nouns such as `risk_trajectory.py` |
| `configs/` | Reproducible experiment contracts | `rig_hazard_<purpose>.json` |
| `scripts/` | Directly executable experiment and maintenance entry points | Verb prefixes such as `run_`, `export_`, and `inspect_` |
| `tests/` | Unit and regression tests | `test_<module>.py` |
| `requirements/` | Layered dependency specifications | Scope names such as `runtime.txt` |
| `docs/` | Maintained project documentation | Lowercase `snake_case.md` |
| `results/` | Immutable experiment artifacts | Model or experiment purpose |
| `visualization/` | Central manuscript figures and lightweight source data | Figure purpose |

Do not recreate historical source, artifact, tool, or log directories at the
repository root. Python packages belong under `src/`; experiment metrics,
predictions, checkpoints, and logs belong under `results/`; centrally exported
manuscript figures belong under `visualization/`. The `src/` source root and
`src/rig_hazard/` package directory must remain separate.

Raw data, frozen results, and checkpoints are not source-refactoring targets.
Compatibility with historical artifact identifiers is isolated in boundary
constants such as those in `src/rig_hazard/naming.py`; new interfaces must not
propagate legacy names.

## File and identifier rules

- Python, Shell, JSON, and Markdown filenames use lowercase `snake_case`.
  Python classes use `PascalCase`; functions, variables, and configuration keys
  use `snake_case`; constants use `UPPER_SNAKE_CASE`.
- Names describe the object and its role. Do not use chronological names such as
  `v2`, `v3`, `m0`, `p3`, `latest`, `new`, or `final2`.
- The primary model identifier is `local_weather_hazard`, and its trajectory
  directory is `local_weather_hazard_trajectory`.
- Record contract changes with a semantic contract ID, date, and content hash in
  a manifest rather than adding a version suffix to filenames.
- Distinguish repeated experimental runs with a `YYYYMMDD_HHMMSS` timestamp or a
  configuration hash. Do not clone scripts with incrementing suffixes.
- Boolean names start with `is_`, `has_`, `should_`, or `enable_`; collection
  names are plural; path variables end with `_path` or `_root`.

## Documentation standards

- English is the only language for maintained documentation, source comments,
  docstrings, command help, log messages, validation messages, and generated
  reports. Original-language field names needed to read source datasets are data
  contract literals, not user-facing terminology; isolate them in schema maps and
  translate them at the ingestion boundary.
- Write documentation as durable project reference material. Do not use
  conversational headings such as "files you asked about", copy question-and-answer
  exchanges into the repository, or address an individual reader.
- Keep a single source of truth for each result. Link to the canonical artifact
  instead of creating aliases or duplicating the same values in multiple files.
- Keep the root README concise: project scope, frozen evaluation contract, key
  findings, repository layout, asset retrieval, canonical results, installation,
  and reproduction commands. Detailed numerical interpretation belongs in
  `docs/results_analysis.md`; exhaustive evidence-tier and artifact routing
  belongs in `docs/results_catalog.md`.
- Every new top-level result directory or experiment family must be added to
  `docs/results_catalog.md` with its evidence tier, summary artifacts, result
  role, and precedence relative to the frozen primary contract.
- Separate scientific results from transient operations. Do not record local test
  counts, machine-specific paths, file-transfer summaries, temporary run status,
  or workstation inventories in maintained documentation. Put reproducibility
  metadata in machine-readable manifests and run logs.
- State the evidence contract beside every headline metric: selection year,
  frozen evaluation years, event queue, budget definition, uncertainty method,
  and whether the result is confirmatory or diagnostic.
- Use repository-relative paths in committed Markdown links and commands. Verify
  every referenced path after renames.
- Prefer short paragraphs, descriptive headings, and tables only when they make
  mappings or comparisons easier to scan. Define abbreviations on first use.
- Avoid mutable claims such as "all tests currently pass". Document the test
  command; let continuous integration or the current test run report its result.
- Generated reports follow the same language and provenance rules as handwritten
  documents and must identify their input contract or manifest.

## Canonical semantic mapping

| Concept | Canonical location or identifier |
| --- | --- |
| Python source | `src/rig_hazard/` |
| Executable scripts | `scripts/` |
| Experiment artifacts and logs | `results/` |
| Central visualization output | `visualization/` |
| Complete result-family inventory | `docs/results_catalog.md` |
| Default preprocessing configuration | `configs/rig_hazard_preprocessing.json` |
| Runtime dependencies | `requirements/runtime.txt` |
| Deep-learning dependencies | `requirements/deep_learning.txt` |
| Local weather hazard baselines | `results/local_weather_hazard_baselines/` |
| Deep-model contract audit | `results/deep_model_experiments/model_contract_audit/` |
| Primary-model trajectory | `results/dynamic_hard_budget/local_weather_hazard_trajectory/` |
| Spatial-generalization queue | `scripts/run_spatial_generalization_queue.sh` |
| Primary experiment resume script | `scripts/resume_main_experiments.sh` |

## Code readability baseline

- Put the module docstring before `from __future__` imports. Public functions
  document inputs, outputs, side effects, and important invariants.
- Prefer `pathlib.Path`, type annotations, data classes, and named constants over
  scattered paths, model identifiers, and magic numbers.
- Give each function one describable responsibility. Split long workflows into
  loading, validation, computation, and output stages.
- When a configuration path or script name changes, update CLI defaults, Shell
  invocations, transfer manifests, documentation, and tests in the same change.
- Before committing, run syntax compilation, naming checks, and the full
  `unittest` suite.

# Transfer From Old Codebase

Status:

- Updated uv instructions and dependencies in `pyproject.toml` and `README.md`.
- Fixed imports across the new `spider` package (replacing `retarget` with `spider`).
- Organized `config.py` and aligned `io.py`, `interp.py`, `optimizers/sampling.py`.
- Kept only the standard MJWP pipeline via `examples/run_mjwp.py`.
- Viewer helpers updated under `spider/viewers/*`.

Pending:

- Fill in docs for each module (design choices, usage, how to extend).
- Add `_test.py` per module colocated with implementation files.
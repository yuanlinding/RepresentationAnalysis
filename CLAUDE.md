# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`magirrep` does Bertaut representational analysis: given an mCIF (Bilbao MAGNDATA or hand-crafted, older CIF conventions supported) it finds which irrep(s) of the parent space group's little group G_k drive the magnetic ordering, and also does displacive/mechanical-representation analysis for any CIF. It is research code. Check results against Bilbao REPRES / BasIreps. The math, conventions and notation are in `docs/theory.md`.

## Commands

```bash
pip install -e ".[dev]"                      # install package + pytest
pytest                                       # all package tests (testpaths = tests/)
pytest tests/test_pipeline.py::test_name -q  # single test
pytest web/test_app.py                       # web service tests (needs fastapi, httpx, python-multipart)

python -m magirrep tests/data/0.15_MnF2.mcif             # combined magnetic + displacive report
python -m magirrep tests/data/1.6_NiO.mcif --magnetic    # magnetic only (faster)
python -m magirrep file.cif --displacive --kvector 0,1/2,0 [--distort 0.05 --out-dir DIR]
python -m magirrep file.mcif -v                          # debug output at each pipeline stage
```

Test fixtures are in `tests/data/` (MnF2, NiO, La2NiO4, MnSiN2). 

## Architecture

`src/magirrep/pipeline.py` holds most of the logic. It has two entry points that follow the same numbered stages. You can see them in the `# ── N. ...` comments.

- `run_analysis(mcif_path, verbose, output_file, displacive_pass)` handles magnetic analysis. It uses only atoms with nonzero moments and the axial-vector rep (det(R)·R). With `displacive_pass=True` it also runs an all-atom polar-vector pass.
- `run_displacive_analysis(path, kvector_str, ...)` handles displacive analysis. It uses all atoms with no det factor. It accepts plain CIFs, which have k from `--kvector`. It can optionally call `distort.generate_distorted_cifs`.

Stage flow and the module that owns each stage:
1. **Parse** (`parse_mcif.py`): gemmi reads the IT number, k-vector, and the child/parent transforms (`P,p` strings). pymatgen builds the `Structure`. gemmi also reads 3D crystal-axis moments directly, because pymatgen can read `_atom_site_moment` as a scalar. If the IT number is missing, spglib detects it. `_ensure_conventional_cell` expands ASU-only files. `_reject_multi_k` rejects multi-k mCIFs. If the file has no `_parent_space_group` block, the parent IT number comes from the BNS number. That is wrong for type-IV groups that have anti-translations `{1|t}'`. For those, `_derive_parent_for_type_iv` gets the parent from spglib (smaller cell) and solves for k from the anti-translations.
2. **Transform to the parent cell** (`mag_rep.map_atoms_to_parent_cell`), then dedupe and reduce to one atom per centering orbit (`_select_primitive_atoms`).
3. **Little group and irreps** (`little_group.py`, `irrep_decompose.py`): spgrep irreps come from a *reference crystal* (`build_reference_crystal`) built in the preferred Hall setting. That setting is origin choice 2 when it exists, to match Bilbao/MAGNDATA. This makes spgrep's (R,t) share a coordinate frame with the parent-cell atom positions. Setting mismatches here are a recurring source of bugs.
4. **Characters and decomposition** (`mag_rep.py`, `irrep_decompose.py`): atom matching is done modulo the *primitive* lattice, using the centering translations (`_match_with_centering`). If it is done only mod Z³, centered lattices produce fractional multiplicities. Projection operators give the basis vectors. `identify_active_irrep` projects the observed moments onto each irrep subspace, and it handles multi-irrep structures.
5. **Labels** (`irrep_label.py`, `bilbao_match.py`): k-point labels come from seekpath and are mapped to Bilbao names (GAMMA→GM). Irrep labels come from a live HTTP query to Bilbao REPRES (`fetch_repres`, cached in-process). That query matches characters by orthogonality after pairing ops by (R,t). If the network fails, labels fall back to spgrep-derived names and `_print_label_fallback_warning` is printed. Tests monkeypatch `fetch_repres`, so they need no network.
6. **Report**: the `_print_*` functions in `pipeline.py` write the numbered Bertaut-style sections that the README describes. Output goes to stdout. `_tee_stdout` also sends it to `output_file`.

Errors reach the user as `ValueError`/`RuntimeError`. The CLI catches them and exits 1.

## Web service

- `web/app.py` is a FastAPI app. `POST /analyze` takes a `.mcif`/`.cif` upload, a `mode` (`magnetic` or `displacive`), and an optional `kvector`. It runs the pipeline in a forked child process, which is killed after 180 s. One worker serializes requests.
- `magirrep.html` is the static frontend. It points at the Render deployment URL. A copy also lives in the separate website repo (`/gpfs/home/rrx4492/website`, `layouts/tools/magirrep.html`). Mirror frontend changes there.
- `web/Dockerfile` installs magirrep from GitHub **pinned to a commit SHA**. The convention is that each analysis fix is followed by a `chore(web): pin Docker image to ... commit` commit that bumps that SHA.

Commit messages use conventional-commit style with scopes (`fix(analysis):`, `feat(web):`, `chore(web):`).

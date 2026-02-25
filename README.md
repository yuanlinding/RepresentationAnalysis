# RepresentationAnalysis / magirrep

Irreducible representation analysis based on the theory of Bertaut. 
Determine which magnetic irreducible representation(s) drive the paramagnetic-to-magnetically-ordered phase transition, given an mCIF file from [Bilbao MAGNDATA](https://www.cryst.ehu.es/magndata/).

## Installation

```bash
conda activate findmagsym          # spglib, spgrep, pymatgen, numpy
pip install -e ".[dev]"            # installs magirrep + gemmi + pytest
```

## Usage

```bash
# Any of these work:
magirrep tests/data/0.222_CuMnAs.mcif
python -m magirrep tests/data/0.222_CuMnAs.mcif
```

## Validation targets

| File                | Expected irrep                | k-point | Parent SG |
| ------------------- | ----------------------------- | ------- | --------- |
| `0.222_CuMnAs.mcif` | **mGM5-** (dim=2)             | Gamma   | #129      |
| `1.6_NiO.mcif`      | **mL3+** (dim=8, small dim=2) | L       | #225      |

## Running tests

```bash
pytest -v
```

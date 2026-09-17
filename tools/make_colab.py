"""Generate Colab-ready copies of the course notebooks.

The notebooks in ``notebooks/`` are kept free of any setup code so that students
working locally are not confronted with it. This script writes copies that carry
a short setup cell, which clones the repository and installs it, so that a Colab
badge works with a single click. It is run by the ``colab`` GitHub workflow; the
result is published to the ``colab`` branch and is not meant to be edited by hand.
"""

import argparse
import json
import shutil
from pathlib import Path

REPO = "https://github.com/meyer-nils/structural_optimization.git"
CHECKOUT = "/content/structural_optimization"

SETUP = [
    "# Setup for Google Colab: fetch the course material and install the packages.\n",
    f"!git clone -q --depth 1 {REPO} {CHECKOUT}\n",
    f"%pip install -q {CHECKOUT}\n",
    f"%cd {CHECKOUT}/notebooks",
]

# Files the notebooks read at runtime; copied so the Colab checkout matches a local one.
DATA_SUFFIXES = (".vtu", ".vtk", ".step", ".stl", ".msh")


def setup_cell() -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(SETUP),
    }


def convert(source: Path, target: Path) -> None:
    nb = json.loads(source.read_text())
    nb["cells"].insert(0, setup_cell())
    target.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("notebooks"))
    parser.add_argument("--out", type=Path, default=Path("colab"))
    args = parser.parse_args()

    out = args.out / "notebooks"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    notebooks = sorted(args.source.glob("*.ipynb"))
    for path in notebooks:
        convert(path, out / path.name)
    for path in sorted(args.source.iterdir()):
        if path.suffix in DATA_SUFFIXES:
            shutil.copy2(path, out / path.name)

    print(f"wrote {len(notebooks)} Colab notebooks to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

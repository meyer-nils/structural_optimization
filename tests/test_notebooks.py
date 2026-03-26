from pathlib import Path

from testbook import testbook

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
NOTEBOOK_PATHS = sorted(
    path
    for path in NOTEBOOK_DIR.glob("*.ipynb")
    if "unsolved" not in path.stem.lower()
)


def _notebook_test_name(notebook_path: Path) -> str:
    return f"test_notebook__{notebook_path.stem.replace('-', '_').replace(' ', '_')}"


def _make_notebook_test(notebook_path: Path):
    @testbook(str(notebook_path), execute=False)
    def _test_notebook(tb):
        tb.inject(
            "from pathlib import Path\n"
            "import os\n"
            f"os.chdir({str(notebook_path.parent)!r})\n"
            "Path('../figures').mkdir(parents=True, exist_ok=True)\n"
        )
        tb.execute()

    _test_notebook.__name__ = _notebook_test_name(notebook_path)
    return _test_notebook


for notebook_path in NOTEBOOK_PATHS:
    globals()[_notebook_test_name(notebook_path)] = _make_notebook_test(notebook_path)

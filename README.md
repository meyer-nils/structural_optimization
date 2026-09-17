[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/-Python-4B8BBE?&logo=Python&logoColor=fff)
![Juypter](https://img.shields.io/badge/-Jupyter-F37626?&logo=Jupyter&logoColor=fff)
[![Build LaTeX](https://github.com/meyer-nils/structural_optimization/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/meyer-nils/structural_optimization/actions/workflows/main.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meyer-nils/structural_optimization)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/meyer-nils/structural_optimization/HEAD)
[![Static Badge](https://img.shields.io/badge/Download_PDF-1.3.0-blue)](https://github.com/meyer-nils/structural_optimization/releases/download/v1.3.0/structural_optimization.pdf)



# Structural Optimization
This is accompanying code for my *Structural Optimization* lecture MRM-0156. 

## Getting started

### Option 1: Run in your browser (nothing to install)
Click the [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/meyer-nils/structural_optimization/HEAD) badge. It opens the whole repository in a ready-to-use JupyterLab session with all packages installed — just open a notebook and start working. The first launch after a change to the material takes a few minutes while the environment is built; later launches are fast.

Note that Binder sessions are temporary and shut down after about ten minutes of inactivity, so download anything you want to keep. Alternatively, [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meyer-nils/structural_optimization) lets you pick any notebook of this repository in Google Colab, where you have to install the packages yourself by running

```python
!pip install git+https://github.com/meyer-nils/structural_optimization.git
```

in a new cell first.

### Option 2: Local installation with uv
We use [uv](https://docs.astral.sh/uv/) to manage Python and all packages. You do not need to install Python, Anaconda or anything else beforehand — uv takes care of it.

**Step 1.** Install uv by running this in a terminal:

*Windows (PowerShell):*
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

*macOS and Linux:*
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2.** Close and reopen the terminal, then run:
```bash
git clone https://github.com/meyer-nils/structural_optimization.git
cd structural_optimization
uv run jupyter lab
```

That's it. The last command downloads a suitable Python version, creates an isolated environment with the exact package versions from `uv.lock` and opens JupyterLab in your browser. Use it again any time you want to continue working.

<details>
<summary>Prefer Visual Studio Code?</summary>

Install [Visual Studio Code](https://code.visualstudio.com) with the "Python" and "Jupyter" extensions, run `uv sync` once in the repository, then open the folder in VS Code. Click "Select Kernel" in the top right of a notebook and choose the interpreter from the `.venv` folder.
</details>

## Contents

- 1 Introduction
  - [Theory](https://meyer-nils.github.io/structural_optimization/introduction.html)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_01_tensors_unsolved.html)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_01_tensors.html)
- 2 Unconstrained optimization
  - [Theory](https://meyer-nils.github.io/structural_optimization/unconstrained_optimization.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_02_unconstrained_optimization.html)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_02_unconstrained_optimization_unsolved.html)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_02_unconstrained_optimization.html)
- 3 Constrained optimization
  - [Theory](https://meyer-nils.github.io/structural_optimization/constrained_optimization.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_03_constrained_optimization.html)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_03_constrained_optimization_unsolved.html)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_03_constrained_optimization.html)
- 4 Optimization using local approximations
  - [Theory](https://meyer-nils.github.io/structural_optimization/approximation_optimization.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_04_approximations.html)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_04_approximations_unsolved.html)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_04_approximations.html)
- 5 Trusses in a nutshell
  - [Theory](https://meyer-nils.github.io/structural_optimization/truss.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_05_truss.html)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_05_sizing_unsolved.html)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_05_sizing.html)
- 6 Optimization of truss structures
  - [Theory](https://meyer-nils.github.io/structural_optimization/truss_optimization.html)
  - [Code (Size)](https://meyer-nils.github.io/structural_optimization/lecture_06_truss_size.html)
  - [Code (Topology)](https://meyer-nils.github.io/structural_optimization/lecture_06_truss_topology.html)
  - [Code (Shape)](https://meyer-nils.github.io/structural_optimization/lecture_06_truss_shape.html)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_06_shape_unsolved.html)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_06_shape.html)
- 7 Finite element analysis in a nutshell
  - [Theory](https://meyer-nils.github.io/structural_optimization/fem.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_07_fem.html)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_07_fem_unsolved.html)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_07_fem.html)
- 8 Optimization of continuum structures
  - [Theory](https://meyer-nils.github.io/structural_optimization/fem_optimization.html)
  - [Code (Topology)](https://meyer-nils.github.io/structural_optimization/lecture_08_topology.html)
  - [Code (Shape)](https://meyer-nils.github.io/structural_optimization/lecture_08_shape.html)
  - [Exercise (Size)](https://meyer-nils.github.io/structural_optimization/exercise_08_sizing_unsolved.html)
  - [Exercise (Topology)](https://meyer-nils.github.io/structural_optimization/exercise_09_topology_unsolved.html)
  - [Exercise (Shape)](https://meyer-nils.github.io/structural_optimization/exercise_10_shape_unsolved.html)
  - [Solution (Size)](https://meyer-nils.github.io/structural_optimization/exercise_08_sizing.html)
  - [Solution (Topology)](https://meyer-nils.github.io/structural_optimization/exercise_09_topology.html)
  - [Solution (Shape)](https://meyer-nils.github.io/structural_optimization/exercise_10_shape.html)

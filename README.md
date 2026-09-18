[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/-Python-4B8BBE?&logo=Python&logoColor=fff)
![Juypter](https://img.shields.io/badge/-Jupyter-F37626?&logo=Jupyter&logoColor=fff)
[![Build LaTeX](https://github.com/meyer-nils/structural_optimization/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/meyer-nils/structural_optimization/actions/workflows/main.yml)
[![Static Badge](https://img.shields.io/badge/Download_PDF-1.3.0-blue)](https://github.com/meyer-nils/structural_optimization/releases/download/v1.3.0/structural_optimization.pdf)



# Structural Optimization
This is accompanying code for my *Structural Optimization* lecture MRM-0156. 

## Getting started

### Option 1: Google Colab (nothing to install)
Every exercise and lecture notebook in the [contents](#contents) below carries a Colab badge that opens it in Google Colab, ready to run. The first cell installs the required packages; after that you can work through the notebook as usual. All you need is a Google account. Note that Colab discards your changes unless you save a copy to your own Google Drive.

### Option 2: Install locally with uv (recommended for the whole course)
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
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_01_tensors_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_01_tensors_unsolved.ipynb)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_01_tensors.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_01_tensors.ipynb)
- 2 Unconstrained optimization
  - [Theory](https://meyer-nils.github.io/structural_optimization/unconstrained_optimization.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_02_unconstrained_optimization.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_02_unconstrained_optimization.ipynb)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_02_unconstrained_optimization_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_02_unconstrained_optimization_unsolved.ipynb)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_02_unconstrained_optimization.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_02_unconstrained_optimization.ipynb)
- 3 Constrained optimization
  - [Theory](https://meyer-nils.github.io/structural_optimization/constrained_optimization.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_03_constrained_optimization.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_03_constrained_optimization.ipynb)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_03_constrained_optimization_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_03_constrained_optimization_unsolved.ipynb)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_03_constrained_optimization.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_03_constrained_optimization.ipynb)
- 4 Optimization using local approximations
  - [Theory](https://meyer-nils.github.io/structural_optimization/approximation_optimization.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_04_approximations.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_04_approximations.ipynb)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_04_approximations_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_04_approximations_unsolved.ipynb)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_04_approximations.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_04_approximations.ipynb)
- 5 Trusses in a nutshell
  - [Theory](https://meyer-nils.github.io/structural_optimization/truss.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_05_truss.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_05_truss.ipynb)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_05_sizing_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_05_sizing_unsolved.ipynb)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_05_sizing.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_05_sizing.ipynb)
- 6 Optimization of truss structures
  - [Theory](https://meyer-nils.github.io/structural_optimization/truss_optimization.html)
  - [Code (Size)](https://meyer-nils.github.io/structural_optimization/lecture_06_truss_size.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_06_truss_size.ipynb)
  - [Code (Topology)](https://meyer-nils.github.io/structural_optimization/lecture_06_truss_topology.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_06_truss_topology.ipynb)
  - [Code (Shape)](https://meyer-nils.github.io/structural_optimization/lecture_06_truss_shape.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_06_truss_shape.ipynb)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_06_shape_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_06_shape_unsolved.ipynb)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_06_shape.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_06_shape.ipynb)
- 7 Finite element analysis in a nutshell
  - [Theory](https://meyer-nils.github.io/structural_optimization/fem.html)
  - [Code](https://meyer-nils.github.io/structural_optimization/lecture_07_fem.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_07_fem.ipynb)
  - [Exercise](https://meyer-nils.github.io/structural_optimization/exercise_07_fem_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_07_fem_unsolved.ipynb)
  - [Solution](https://meyer-nils.github.io/structural_optimization/exercise_07_fem.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_07_fem.ipynb)
- 8 Optimization of continuum structures
  - [Theory](https://meyer-nils.github.io/structural_optimization/fem_optimization.html)
  - [Code (Topology)](https://meyer-nils.github.io/structural_optimization/lecture_08_topology.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_08_topology.ipynb)
  - [Code (Shape)](https://meyer-nils.github.io/structural_optimization/lecture_08_shape.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/lecture_08_shape.ipynb)
  - [Exercise (Size)](https://meyer-nils.github.io/structural_optimization/exercise_08_sizing_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_08_sizing_unsolved.ipynb)
  - [Exercise (Topology)](https://meyer-nils.github.io/structural_optimization/exercise_09_topology_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_09_topology_unsolved.ipynb)
  - [Exercise (Shape)](https://meyer-nils.github.io/structural_optimization/exercise_10_shape_unsolved.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_10_shape_unsolved.ipynb)
  - [Solution (Size)](https://meyer-nils.github.io/structural_optimization/exercise_08_sizing.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_08_sizing.ipynb)
  - [Solution (Topology)](https://meyer-nils.github.io/structural_optimization/exercise_09_topology.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_09_topology.ipynb)
  - [Solution (Shape)](https://meyer-nils.github.io/structural_optimization/exercise_10_shape.html) [![Colab](https://img.shields.io/badge/-Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/github/meyer-nils/structural_optimization/blob/colab/notebooks/exercise_10_shape.ipynb)


# Introduction

This repository contains algorithms to enable the approximation of the **Virtual Network Embedding Algorithm (VNEP)**.
It makes extensive use of our **[alib](https://github.com/vnep-approx/alib)** library. In particular, our algorithms employ
Gurobi to compute convex combinations of valid mappings to apply **randomized rounding**. We provide the following implementations: 
- A Linear Program (LP) implementation based on our papers [1,2] for cactus request graphs 
**[modelcreator_ecg_decomposition.py](vnep_approx/modelcreator_ecg_decomposition.py)**.
- A Linear Program based on our generalized extraction width concept for arbitrary requests based on our paper [3]: 
**[commutativity_model.py](vnep_approx/commutativity_model.py)**
- A Linear Program enabling the handling of decisions (outgoing edges represent choices) extending the formulation presented in [4]: **[gadget_model.py](vnep_approx/gadget_model.py)**.
- A implementation of randomized rounding for cactus request graphs **[randomized_rounding_triumvirate.py](vnep_approx/randomized_rounding_triumvirate.py)**. 
This randomized rounding procedure actually executed three different kinds of heuristics:
  - **Vanilla rounding**: simply round solutions and select the best one found within a fixed number of iterations (see [2]).
  - **Heuristic rounding**: perform the rounding while discarding selected (i.e. rounded) mappings whose addition would 
  exceed resource capacities. Accordingly: this heuristic always yields feasible solutions (see [2]).
  - **Multi-dimensional knapsack (MDK)**: given the decomposition into convex combinations of valid mappings for each request,
  the MDK computes the optimal rounding given all mapping possibilities found.
  
Note that our separate Github Repository [evaluation-ifip-networking-2018](https://github.com/vnep-approx/evaluation-ifip-networking-2018)
provides more explanation on how to generate scenarios and apply algorithms. 

## Papers

**[1]** Matthias Rost, Stefan Schmid: Service Chain and Virtual Network Embeddings: Approximations using Randomized Rounding. [CoRR abs/1604.02180](https://arxiv.org/abs/1604.02180) (2016)

**[2]** Matthias Rost, Stefan Schmid: Virtual Network Embedding Approximations: Leveraging Randomized Rounding. IFIP Networking 2018. (see [arXiv](https://arxiv.org/abs/1803.03622) for the corresponding technical report)

**[3]** Matthias Rost, Stefan Schmid: (FPT-)Approximation Algorithms for the Virtual Network Embedding Problem. [CoRR abs/1803.04452](https://arxiv.org/abs/1803.04452) (2018)

**[4]** Guy Even, Matthias Rost, Stefan Schmid: An Approximation Algorithm for Path Computation and Function Placement in SDNs. [SIROCCO 2016: 374-390](https://link.springer.com/chapter/10.1007%2F978-3-319-48314-6_24)

# Dependencies and Requirements

The **vnep_approx** library requires Python 2.7. Required python libraries: gurobipy, numpy, cPickle, networkx 1.9, matplotlib, and **[alib](https://github.com/vnep-approx/alib)**. 

Gurobi must be installed and the .../gurobi64/lib directory added to the environment variable LD_LIBRARY_PATH.

For generating and executing (etc.) experiments, the environment variable ALIB_EXPERIMENT_HOME must be set to a path,
such that the subfolders input/ output/ and log/ exist.

**Note**: Our source was only tested on Linux (specifically Ubuntu 14/16).  

# Overview

To install **vnep_approx**, we provide a setup script. Simply execute from within vnep_approx's root directory: 

```
pip install .
```

Furthermore, if the code base will be edited by you, we propose to install it as editable:
```
pip install -e .
```
When choosing this option, sources are not copied during the installation but the local sources are used: changes to
the sources are directly reflected in the installed package.

We generally propose to install **vnep_approx** into a virtual environment (together with **alib**).

# Usage

You may either use our code via our API by importing the library or via our command line interface:

```
python -m vnep_approx.cli
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  generate_scenarios
  start_experiment
```

# Tests


The test directory contains a large number of tests to check the correctness of our implementation and might also be useful
to understand the code. 

To execute the tests, simply execute pytest in the test directory.

```
pytest .
```

# API Documentation

We provide a basic template to create an API documentatio using **[Sphinx](http://www.sphinx-doc.org)**. 

To create the documentation, simply execute the makefile in **docs/**. Specifically, run for example

```
make html
```

to create the HTML documentation.

Note that **vnep_approx** must lie on the PYTHONPATH. If you use a virtual environment, we propose to install sphinx within the
virtual environment (using **pip install spinx**) and executing the above from within the virtual environment. 

# Contact

If you have any questions, simply write a mail to mrost(AT)inet.tu-berlin(DOT)de.

# Acknowledgement

Major parts of this code were developed under the support of the **German BMBF Software
Campus grant 01IS1205**.
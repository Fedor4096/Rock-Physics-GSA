# Scientific Computing for Anisotropic Rock Physics

### Research and performance engineering of the Generalized Singular Approximation method

This repository documents an academic research project at the intersection of **computational geophysics, numerical methods, tensor algebra, and performance engineering**.

The objective was not simply to reproduce an existing mathematical formula in code. The central challenge was to make a computationally intensive rock-physics model practical for repeated calculations by:

- understanding and decomposing the underlying mathematical model;
- identifying numerical and performance bottlenecks;
- designing an efficient integration strategy;
- implementing and validating the computational workflow;
- comparing optimisation approaches across Python, JIT compilation, and Rust.

> **Research project:** 2023 · **Documentation refreshed:** 2026  
> The original Python implementation and commit history have been preserved. This README was updated to provide clearer context about the research process, technical contribution, and limitations.

## Research problem

Forward and inverse seismic problems require mathematical models that can estimate the effective elastic properties of complex rocks.

This becomes particularly difficult for anisotropic, multicomponent materials. Their behaviour depends not only on the elastic properties and concentrations of individual components, but also on:

- particle geometry;
- inclusion orientation;
- pore connectivity;
- spatial distribution;
- local interactions between components.

The **Generalized Singular Approximation (GSA)** is an effective-medium method capable of representing both isotropic and anisotropic composites. Its flexibility comes at a computational cost: a practical implementation requires operations on fourth-rank tensors, repeated matrix inversion, and numerical integration of 81 tensor-component functions over angular coordinates.

The research question behind this project was:

> **How can the GSA method be translated from its mathematical formulation into a computationally efficient implementation suitable for repeated rock-physics calculations?**

## Research contribution

For this project, I:

- studied and compared established effective-medium approaches for isotropic and anisotropic rocks;
- performed a detailed decomposition of the GSA mathematical model;
- implemented fourth-rank elasticity and compliance tensor operations;
- developed conversions between full tensor representation and 6 × 6 Voigt notation;
- implemented the numerical calculation of the Green's-function derivative tensor **g**;
- investigated the behaviour of all 81 component integrands;
- designed component-specific angular integration grids;
- validated intermediate and final calculations against reference tabular values used in the research;
- profiled the implementation to identify its most computationally expensive stages;
- restructured the algorithm to avoid repeated matrix inversions;
- adapted the Python implementation for Numba JIT compilation;
- benchmarked Python, Python with JIT compilation, and Rust implementations.

The public repository contains the Python/Numba research prototype and the Rust implementation. The complete thesis is not currently included.

## Mathematical core

### Main GSA formula

In general, the effective elastic tensor for a multicomponent rock is calculated using the following expression:

<p align="center">
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/ccb5643f-4778-4ffc-9793-9ea6bb81424a" height="120" alt="Generalized Singular Approximation formula for the effective elastic tensor">
</p>

The calculation requires tensor multiplication and tensor inversion. Within the public implementation, the method is implemented for a **two-component composite with horizontally fixed inclusions**.

The main computational challenge is finding the components of tensor **g**, which describes the interaction of an inclusion with its local surroundings.

### Calculation of tensor g

Calculating tensor **g**, derived from the second derivative of the Green's function, is a multistage numerical process.

First, the components of an auxiliary non-symmetric tensor **a<sub>not sym</sub>** are calculated:

<p align="center">
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/28cae17e-b819-4b26-b6e7-9835a0228764" height="70" alt="Formula for a component of the auxiliary non-symmetric tensor">
</p>

where the required auxiliary terms are defined as:

<p align="center">
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/9b73ac8d-7960-486d-b6f8-7c70dbb42666" height="50" alt="Auxiliary term used in the tensor calculation">
</p>

<p align="center">
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/439c47ed-bd82-44b9-8cc3-ba4297fa70ca" height="50" alt="Lambda matrix used in the tensor calculation">
</p>

The resulting tensor is then symmetrized:

<p align="center">
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/5c555a87-96a5-4ef2-9012-bf419826c222" height="60" alt="Symmetrization of the auxiliary tensor">
</p>

Finally, tensor **g** is obtained by reassigning the components of the symmetrized tensor:

<p align="center">
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/bb17222f-aad9-421c-ae75-115152ba497f" height="70" alt="Calculation of tensor g from the symmetrized auxiliary tensor">
</p>

### Numerical integration

Each component of the auxiliary tensor requires numerical integration over a surface defined in angular coordinates.

The implementation uses predefined integration nodes and a two-dimensional mean rectangle rule:

<p align="center">
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/57fe586e-420c-4aee-8589-2204d0f499ae" height="70" alt="Two-dimensional numerical integration formula">
</p>

Selecting the integration nodes is itself a research problem. The optimal grid depends on the geometry of the inclusions and the behaviour of the individual integrand functions.

## Scientific computing challenge

A direct implementation of the mathematical model repeats many expensive operations. To calculate tensor **g**, the program must:

1. Build an inverse 3 × 3 matrix for every angular integration node.
2. Calculate 81 components of an auxiliary fourth-rank tensor.
3. Numerically integrate every component over a two-dimensional angular domain.
4. Symmetrize the resulting tensor.
5. Transform it into tensor **g**.
6. Use tensor **g** to calculate the effective elasticity tensor of the composite.

I decomposed this workflow and moved the matrix-inversion stage outside the component-level integration loop.

The inverse matrices are calculated once for every angular node, stored in memory, and reused across all 81 integrand calculations. This introduces a deliberate **compute–memory trade-off**: higher temporary memory consumption in exchange for substantially fewer repeated matrix inversions.

## Integration-grid research

The accuracy and performance of the algorithm depend strongly on the integration grid.

For approximately spherical inclusions with an aspect ratio of `(1, 1, 1)`, a uniform angular grid provides a reasonable representation.

For strongly flattened, crack-like inclusions with an aspect ratio of `(1000, 1000, 1)`, the integrands develop sharp features close to `θ = π/2`.

Using a dense uniform grid across the entire domain would significantly increase computation time and memory consumption. I therefore analysed three-dimensional integrand surfaces and two-dimensional angular slices to locate the high-gradient region.

Based on this analysis, I introduced a locally refined grid around approximately `1.555–1.590` radians. This concentrates integration nodes where they carry the most numerical value while maintaining a lower density in smoother regions.

The project therefore involved not only implementing a mathematical model, but also investigating the numerical behaviour of its internal functions and translating those observations into computational decisions.

## Performance engineering

The original implementation was developed in Python to support fast prototyping, visual inspection, and easier debugging of the tensor calculations.

After validating the computational workflow, I explored three levels of performance optimisation.

### 1. Algorithm-level optimisation

The calculation order was reorganised so that inverse matrices shared by multiple tensor components could be precomputed and reused. This reduced redundant calculations inside the most expensive nested loops.

### 2. JIT compilation

The Python code was significantly restructured to make its core numerical functions compatible with Numba's JIT compiler. This required:

- separating JIT-compatible computation from visualisation and interactive logic;
- replacing unsupported Python constructs;
- using explicit NumPy arrays and data types;
- decomposing larger functions into focused numerical operations;
- isolating the performance-critical execution path.

### 3. Compiled implementation

The final research stage explored a Rust implementation to evaluate the trade-off between development flexibility and execution speed.

The historical benchmark used:

- ten repeated effective-tensor calculations;
- approximately 60,000 integration nodes;
- an Apple M2 CPU;
- 8 GB of RAM.

| Implementation | Reported relative performance | Included in this repository |
|---|---:|:---:|
| Initial pure-Python prototype | Baseline | No |
| Python with Numba JIT | Approximately 60× faster | Yes |
| Rust release build | Approximately 600× faster | Yes |

The Rust release build was approximately ten times faster than the Python/Numba version in this historical experiment.

These figures are environment-specific research results, not universal performance guarantees. The initial pure-Python version and complete historical benchmark artefacts are not included in the public repository, so the complete comparison cannot currently be reproduced from this repository alone.

## Research cases

### Quartz matrix with water-filled inclusions

The public Python implementation models a two-component composite consisting of:

- an isotropic mineral matrix;
- strongly flattened water-filled inclusions;
- a configurable pore-connectivity parameter;
- horizontally aligned inclusion geometry.

This case was used to investigate the behaviour of the integrand functions, integration-grid selection, and calculation of the effective elasticity tensor.

### Clay–kerogen–quartz composite

The thesis also examined a more complex three-component composite consisting of:

- anisotropic VTI clay;
- kerogen;
- quartz.

The calculation was performed through sequential two-component mixing. This demonstrated how the method could be applied when one of the initial components was already anisotropic.

The three-component workflow is described in the thesis but is not implemented as a reusable workflow in the public Python script.

## Technical scope

The project includes hands-on work with:

- computational geophysics and rock-physics modelling;
- effective-medium theory;
- fourth-rank elasticity tensors and Voigt notation;
- Green's functions;
- multidimensional numerical integration;
- non-uniform integration grids;
- matrix inversion and tensor contraction;
- NumPy-based scientific computing;
- Numba JIT compilation;
- computational profiling;
- Python-to-Rust performance comparison;
- scientific visualisation with Matplotlib, Seaborn, and Plotly.

## Public implementation

The repository contains Python/Numba and Rust implementations for a two-component rock.

The main script:

- constructs component elasticity tensors;
- builds the comparison-body tensor;
- generates separate integration grids;
- precomputes inverse matrices;
- calculates and integrates all tensor components;
- constructs tensor **g**;
- calculates the effective stiffness tensor;
- returns the result as a 6 × 6 matrix in Voigt notation.

```text
.
├── README.md
├── LICENSE
├── CITATION.cff            # Software citation metadata
├── .gitignore
├── python/
│   ├── GSA.py              # Python/Numba scientific computing prototype
│   └── requirements.txt    # Python dependencies
└── rust/
    ├── Cargo.toml          # Rust package and dependencies
    ├── Cargo.lock          # Resolved Rust dependency versions
    └── src/
        └── main.rs        # Rust implementation
```

## Running the implementations

Run the following commands from the repository root.

### Python

Install the required Python dependencies:

```bash
python -m pip install -r python/requirements.txt
```

Run the default experiment:

```bash
python python/GSA.py
```

The script prints the effective stiffness tensor and total execution time.

The first execution includes JIT compilation overhead. For meaningful performance measurements, compilation time should be separated from repeated calculation time.

### Rust

Install the Rust toolchain with Cargo, then run the default experiment in release mode:

```bash
cargo run --release --locked --manifest-path rust/Cargo.toml
```

The executable prints the effective stiffness tensor and total execution time. Cargo stores generated build files in `rust/target/`, which is excluded from version control. The committed `Cargo.lock` records the resolved dependency versions.

## Limitations

This is an academic research prototype rather than a production-ready scientific library.

The public implementation currently has the following limitations:

- it implements the two-component calculation only;
- inclusion orientation is fixed horizontally;
- arbitrary orientation distributions are not available;
- integration grids must be configured manually;
- automated tests and continuous integration are not included;
- complete validation datasets and benchmark artefacts are not published.

The repository should therefore be used to examine the computational approach and research process rather than as a validated production package.

## Project context

This project represents an important part of my hands-on technical background before moving into Technical Product Management.

It taught me to work across several levels of a complex computational problem: from mathematical assumptions and numerical behaviour to software architecture, performance bottlenecks, implementation trade-offs, and validation.

That foundation now informs how I approach products across **AI, cloud computing, and complex infrastructure**: understanding the technology deeply enough to discuss real constraints and trade-offs while connecting technical capabilities to useful applications.

## License

Copyright (C) 2023 Fedor Lozovoi.

This project is distributed under the [GNU Affero General Public License v3.0](LICENSE), version 3 only (`AGPL-3.0-only`). The licence text is the unmodified [official GNU AGPLv3 text](https://www.gnu.org/licenses/agpl-3.0.txt).

Earlier revisions released under GPLv3 retain their original licence.

## Citation

If you use this software in your research, please cite the implementation using [CITATION.cff](CITATION.cff). GitHub provides a **Cite this repository** option with APA and BibTeX formats when this file is present on the default branch.

The citation credits the software implementation. Please also acknowledge the scientific publications underlying the GSA method where relevant. The citation request is separate from the licence terms.

# Devito: Fast Stencil Computation from Symbolic Specification

[![Build Status for the Core backend](https://github.com/devitocodes/devito/workflows/CI-core/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-core)
[![Build Status with MPI](https://github.com/devitocodes/devito/workflows/CI-mpi/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-mpi)
[![Build Status on GPU](https://github.com/devitocodes/devito/workflows/CI-gpu/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-gpu)
[![Code Coverage](https://codecov.io/gh/devitocodes/devito/branch/master/graph/badge.svg)](https://codecov.io/gh/devitocodes/devito)
[![Slack Status](https://img.shields.io/badge/chat-on%20slack-%2336C5F0)](https://join.slack.com/t/devitocodes/shared_invite/zt-2hgp6891e-jQDcepOWPQwxL5JJegYKSA)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](https://devitocodes.github.io/devito-performance)
[![PyPI version](https://badge.fury.io/py/devito.svg)](https://badge.fury.io/py/devito)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/devitocodes/devito/master)
[![Docker](https://img.shields.io/badge/dockerhub-images-important.svg?logo=Docker?color=blueviolet&label=docker&sort=semver)](https://hub.docker.com/r/devitocodes/devito)

[DEVITO SENAI CIMATEC](https://codigo-externo.petrobras.com.br/senai-cimatec-lde/devito) is a DSL
developed based on the open-source Devito package [Devito](http://www.devitoproject.org):
a Python package to implement optimized stencil computation (e.g., finite differences, image processing,
machine learning) from high-level symbolic problem definitions. This DSL builds
on [SymPy](http://www.sympy.org/en/index.html) and employs automated code
generation and just-in-time compilation to execute optimized computational
kernels on several computer platforms, including CPUs, GPUs, and clusters
thereof.

- [About Devito](#about-devito)
- [Disk Swap](#disk-swap)
- [CIMATEC Alocator](#cimatec-alocator)
- [CIMATEC Seismic API](#cimatec-seismic-api)
- [Installation](#installation)
- [Resources](#resources)
- [FAQs](https://github.com/devitocodes/devito/blob/master/FAQ.md)
- [Performance](#performance)
- [Get in touch](#get-in-touch)
- [Interactive jupyter notebooks](#interactive-jupyter-notebooks)

## About Devito

Devito provides a functional language to implement sophisticated operators that
can be made up of multiple stencil computations, boundary conditions, sparse
operations (e.g., interpolation), and much more.  A typical use case is
explicit finite difference methods for approximating partial differential
equations. For example, a 2D diffusion operator may be implemented with Devito
as follows

```python
>>> grid = Grid(shape=(10, 10))
>>> f = TimeFunction(name='f', grid=grid, space_order=2)
>>> eqn = Eq(f.dt, 0.5 * f.laplace)
>>> op = Operator(Eq(f.forward, solve(eqn, f.forward)))
```

An `Operator` generates low-level code from an ordered collection of `Eq` (the
example above being for a single equation). This code may also be compiled and
executed

```python
>>> op(t=timesteps, dt=dt)
```

There is virtually no limit to the complexity of an `Operator` -- the Devito
compiler will automatically analyze the input, detect and apply optimizations
(including single- and multi-node parallelism), and eventually generate code
with suitable loops and expressions.

Key features include:

* A functional language to express finite difference operators.
* Straightforward mechanisms to adjust the discretization.
* Constructs to express sparse operators (e.g., interpolation), classic linear
  operators (e.g., convolutions), and tensor contractions.
* Seamless support for boundary conditions and adjoint operators.
* A flexible API to define custom stencils, sub-domains, sub-sampling,
  and staggered grids.
* Generation of highly optimized parallel code (SIMD vectorization, CPU and
  GPU parallelism via OpenMP and OpenACC, multi-node parallelism via MPI,
  blocking, aggressive symbolic transformations for FLOP reduction, etc.).
* Distributed NumPy arrays over multi-node (MPI) domain decompositions.
* Inspection and customization of the generated code.
* Autotuning framework to ease performance tuning.
* Smooth integration with popular Python packages such as NumPy, SymPy, Dask,
  and SciPy, as well as machine learning frameworks such as TensorFlow and
  PyTorch.


## Disk Swap
The Disk Swap is an exclusive feature of DEVITO SENAI CIMATEC that enables
the full storage of TimeFunctions (commonly used to represent wavefields in 
various models) on high-performance devices, such as NVMe drives.

This functionality overcomes the field size limitations imposed by executions 
relying solely on RAM storage, as well as the poor performance of alternative 
approaches employing checkpointing techniques.

DEVITO SENAI CIMATEC provides an extremely simple and functional configuration 
interface for Disk Swap, allowing its setup and application in just 
a few lines of code.


```python
>>> grid = Grid(shape=(10, 10))
>>> f = TimeFunction(name='f', grid=grid, space_order=2)
>>> eqn = Eq(f.dt, 0.5 * f.laplace)
>>> ds_config = DiskSwapConfig(functions=[f],
                            mode="write",
                            path="path_to_device")
>>> op = Operator(Eq(f.forward, solve(eqn, f.forward)), opt=('advanced', {'disk-swap': ds_config})
```

The configuration, for example, for an operator that stores the wavefield to disk,
can be done intuitively and without increasing the code complexity for the user, as demonstrated above.


## CIMATEC Alocator
(KEEP IT?)

## CIMATEC SEISMIC API
(KEEP IT?)

## Installation
The use of virtual environments is recommended to isolate package dependencies, ensuring that the installation of DEVITO does not interfere with other projects or system configurations.

In addition to the installation of mandatory packages required for the basic functionality of the tool, DEVITO also supports the installation of additional dependencies for extended features, organized into four groups:

- **extras**: dependencies for Jupyter notebooks, plotting, and benchmarking.
- **tests**: dependencies for the testing infrastructure.
- **mpi**: dependencies for the MPI infrastructure.
- **nvidia**: dependencies to enable GPU execution.

**venv install**:
```
>>># Creation
>>>python -m venv <nome_do_ambiente>
>>>
>>># Activation
>>>source <nome_do_ambiente>/bin/activate  # Para sistemas Linux/macOS
>>>
>>># Installing additional dependencies
>>>pip install git+https://codigo-externo.petrobras.com.br/senai-cimatec-lde/devito.git
>>>
>>># ...or to install it with the additional dependencies already included:
>>>pip install devito[tests,extras,nvidia,mpi] @ git+https://codigo-externo.petrobras.com.br/senai-cimatec-lde/devito.git@main
```


**conda install**:
```
>>># Creation
>>>conda create --name <nome_do_ambiente>
>>>
>>># Activation
>>>conda activate <nome_do_ambiente>
>>>
>>># Download repository
>>>git clone https://codigo-externo.petrobras.com.br/senai-cimatec-lde/devito.git
>>>
>>># Install
>>>cd devito
>>>pip install -e .
>>>
>>># ...or to install it with the additional dependencies already included:
>>>pip install -e .[extras,mpi,nvidia,tests]
```

## Resources

To learn how to use the DEVITO SENAI CIMATEC,
[here](https://codigo-externo.petrobras.com.br/senai-cimatec-lde/devito/blob/master/examples) is a good
place to start, with lots of examples and tutorials.

The original Devito [website](https://www.devitoproject.org/) also provides access to other
information, including documentation and also a FAQs are discussed [here](FAQ.md).

## Performance

If you are interested in any of the following

* Generation of parallel code (CPU, GPU, multi-node via MPI);
* Performance tuning;
* Benchmarking operators;

then you should take a look at this
[README](https://codigo-externo.petrobras.com.br/senai-cimatec-lde/devito/blob/master/benchmarks/user).




## Get in touch
(KEEP IT?)

If you're using Devito, we would like to hear from you. Whether you
are facing issues or just trying it out, join the
[conversation](https://join.slack.com/t/devitocodes/shared_invite/zt-2hgp6891e-jQDcepOWPQwxL5JJegYKSA).

## Interactive jupyter notebooks
(KEEP IT?)
The tutorial jupyter notebook are available interactively at the public [binder](https://mybinder.org/v2/gh/devitocodes/devito/master) jupyterhub. 

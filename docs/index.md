---
title: Overview
description: A development and dispatch library for high-performance ML kernels
---

**Popcorn** is a development and dispatch library for high-performance machine learning
kernels. It unifies 96 kernels and more than 100 optimized implementations behind a single
API, then selects among eligible backends using hardware-specific benchmark records. Each
implementation is checked against a PyTorch reference; failed and unmeasured cases remain
visible evidence rather than being presented as successful validation.

Popcorn also ships a growing collection of first-party kernels, along with the environment
and harness to build, test, benchmark, and deploy new kernels and backends.

## Terminology

Some frequently used terms are overloaded; Popcorn uses them precisely:

| Term | Meaning |
| --- | --- |
| **Kernel** | A pure unit of work with a fixed signature and semantics defined by a *ground-truth* reference. |
| **Input** | A valid configuration of arguments for a kernel call. |
| **Case** | The set of all inputs with matching tensor metadata (shape, type) and other arguments — the data inside tensors is abstracted away. |
| **Implementation** | A function that matches the output of a *kernel's reference* for a subset of valid cases. |
| **Backend** | A library or collection of *kernel implementations*. |

## Guides

- [Quick start](./quickstart.md) — install and call your first kernel
- [Dispatching](./dispatching.md) — how a call resolves to one implementation
- [Registration](./registration.md) — define kernels and bind implementations
- [Validation](./validation.md) — compare implementations against the reference
- [Benchmarking](./benchmarking.md) — record timings and route on data
- [Tuning](./tuning.md) — select one implementation for a region of shapes
- [torch.compile](./torch-compile.md) — route compiled models through Popcorn
- [Optimization loop](./loop.md) — iterate on a candidate kernel with crash-isolated evals

Kernel cards with math, backends, and measured performance live in the
[kernel explorer](https://tilde-research.github.io/popcorn/kernels/).

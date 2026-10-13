# LLM Inference Performance Lab

**GPU and LLM inference performance engineering, with production-ready serving.**

How far can LLM inference performance on a GPU be pushed under explicit latency, throughput and memory
constraints, and can the bottleneck be explained at every layer?

> Status: Phase 0: benchmark infrastructure.

## Setup

```bash
uv sync --group dev
uv run pytest
uv run python -m benchmarks.env   # print the environment snapshot
```

## Documentation

- **[Project guide](docs/project-guide.md)**: goal, architecture, phases, every component's
  inputs and outputs, experiment catalog, decisions. Start here.
- [Benchmark methodology](docs/benchmark-methodology.md): how every metric is defined and measured

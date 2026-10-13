# Benchmark Methodology

## Principles

1. **Raw data first.** Each run stores per-request token timestamps and the full environment
   snapshot. Summaries are derived from raw data by code, never copied by hand.
2. **Reproducible.** A result file records the config, seed, git commit (and dirty flag), library
   versions, driver and GPU, and the environment variables that affect execution.
3. **Measured where it matters.** Only NVIDIA runs go into the performance report. Local Mac runs
   validate the harness.
4. **Enough samples.** Always report `n`. Don't quote P99 from fewer than ~100 samples.

## Timing

Timestamps come from host-side `time.perf_counter_ns()`, taken when a token becomes visible to the
caller. That is the latency a client experiences. Kernel-level timing (CUDA events, Nsight) is a
separate tool for explaining those numbers, not a replacement for them.

## Metric definitions

| Metric | Definition | What it mostly reflects |
|---|---|---|
| TTFT | first token time − submit time | queueing + prefill |
| ITL | gap between consecutive tokens, pooled over all requests | decode step time, stalls |
| TPOT | (last token − first token) / (output tokens − 1), one value per request | per-request decode speed |
| E2E | last token time − submit time | total request latency |
| Output tokens/s | total output tokens / wall time | system throughput |
| Requests/s | successful requests / wall time | system throughput |

Failed requests are counted in `num_errors` and excluded from latency distributions.

## Open items

- Warmup policy (number of warmup requests, discarding the first CUDA/compile runs)
- Device sampler (GPU utilization, memory, KV-cache usage) → `device_samples`
- Workload definitions (fixed vs distributed prompt/output lengths, arrival process)
- Run-to-run noise: repeat runs, and use the observed variance to set regression thresholds

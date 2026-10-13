"""Benchmark result schema.

A run stores raw per-request timestamps. Every latency metric is derived from those
timestamps, so a summary can always be recomputed and audited. Metric definitions
are documented in docs/benchmark-methodology.md.

All timestamps are host-side `time.perf_counter_ns()` values taken when a token
becomes visible to the caller, which is the latency a user experiences.
"""

from __future__ import annotations

import itertools
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.stats import distribution

SCHEMA_VERSION = 1
NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class BenchmarkConfig:
    runtime: str  # "pytorch", "vllm", "tensorrt_llm", "mock", ...
    model: str
    precision: str  # "fp32", "fp16", "bf16", ...
    quantization: str | None  # None, "int8-weight-only", "awq-int4", ...
    batch_size: int
    concurrency: int
    input_tokens: int
    output_tokens: int
    num_requests: int
    warmup_requests: int
    seed: int
    device: str  # "cuda:0", "mps", "cpu"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestRecord:
    request_id: str
    prompt_tokens: int
    t_submit_ns: int
    # Host arrival time of each output token, in order. For non-streaming runtimes
    # this holds only the completion time, and ITL cannot be measured.
    token_times_ns: list[int] = field(default_factory=list)
    error: str | None = None

    @property
    def output_tokens(self) -> int:
        return len(self.token_times_ns)

    @property
    def ttft_ms(self) -> float | None:
        if not self.token_times_ns:
            return None
        return (self.token_times_ns[0] - self.t_submit_ns) / NS_PER_MS

    @property
    def e2e_ms(self) -> float | None:
        if not self.token_times_ns:
            return None
        return (self.token_times_ns[-1] - self.t_submit_ns) / NS_PER_MS

    @property
    def itl_ms(self) -> list[float]:
        """Gaps between consecutive output tokens (the decode steps the user sees)."""
        t = self.token_times_ns
        return [(b - a) / NS_PER_MS for a, b in itertools.pairwise(t)]

    @property
    def tpot_ms(self) -> float | None:
        """Mean time per output token after the first: one number per request."""
        if len(self.token_times_ns) < 2:
            return None
        return (self.token_times_ns[-1] - self.token_times_ns[0]) / NS_PER_MS / (
            len(self.token_times_ns) - 1
        )


def summarize(records: list[RequestRecord], wall_time_s: float) -> dict[str, Any]:
    """Aggregate metrics over successful requests.

    ITL pools every token gap from every request, so it shows decode stalls (for example
    a decode step blocked behind another request's prefill). TPOT is one mean per request,
    so it shows how per-request decode speed is spread. They answer different questions.
    """
    ok = [r for r in records if r.error is None and r.token_times_ns]
    total_out = sum(r.output_tokens for r in ok)
    return {
        "num_requests": len(records),
        "num_errors": len(records) - len(ok),
        "wall_time_s": wall_time_s,
        "requests_per_s": len(ok) / wall_time_s if wall_time_s > 0 else None,
        "output_tokens_per_s": total_out / wall_time_s if wall_time_s > 0 else None,
        "total_prompt_tokens": sum(r.prompt_tokens for r in ok),
        "total_output_tokens": total_out,
        "ttft_ms": distribution([r.ttft_ms for r in ok]),
        "e2e_ms": distribution([r.e2e_ms for r in ok]),
        "itl_ms": distribution([gap for r in ok for gap in r.itl_ms]),
        "tpot_ms": distribution([r.tpot_ms for r in ok if r.tpot_ms is not None]),
    }


@dataclass
class RunResult:
    config: BenchmarkConfig
    environment: dict[str, Any]
    requests: list[RequestRecord]
    wall_time_s: float
    # Periodic device samples (utilization, memory, KV-cache usage, ...), filled in
    # once a GPU sampler exists. Kept raw, like requests.
    device_samples: list[dict[str, Any]] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": asdict(self.config),
            "environment": self.environment,
            "summary": summarize(self.requests, self.wall_time_s),
            "requests": [asdict(r) for r in self.requests],
            "device_samples": self.device_samples,
        }

    def save(self, results_dir: Path) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"{self.created_at[:10]}_{self.config.runtime}_{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> RunResult:
        data = json.loads(path.read_text())
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema v{data['schema_version']}, expected v{SCHEMA_VERSION}")
        return cls(
            config=BenchmarkConfig(**data["config"]),
            environment=data["environment"],
            requests=[RequestRecord(**r) for r in data["requests"]],
            wall_time_s=data["summary"]["wall_time_s"],
            device_samples=data["device_samples"],
            run_id=data["run_id"],
            created_at=data["created_at"],
        )

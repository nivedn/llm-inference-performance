import json

import pytest

from benchmarks.env import capture_environment
from benchmarks.schema import BenchmarkConfig, RequestRecord, RunResult, summarize
from benchmarks.stats import distribution, percentile

MS = 1_000_000


def _config(**overrides):
    base = {
        "runtime": "mock", "model": "none", "precision": "fp16", "quantization": None,
        "batch_size": 1, "concurrency": 1, "input_tokens": 8, "output_tokens": 4,
        "num_requests": 2, "warmup_requests": 0, "seed": 0, "device": "cpu",
    }
    return BenchmarkConfig(**(base | overrides))


def test_percentile_matches_linear_interpolation():
    xs = [1, 2, 3, 4]
    assert percentile(xs, 0) == 1
    assert percentile(xs, 100) == 4
    assert percentile(xs, 50) == 2.5
    assert percentile(xs, 90) == pytest.approx(3.7)


def test_percentile_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_empty_distribution_reports_zero_samples():
    d = distribution([])
    assert d["n"] == 0 and d["p99"] is None


def test_request_metrics_from_timestamps():
    # Submit at 0, tokens at 100, 110, 130, 140 ms.
    r = RequestRecord("r0", prompt_tokens=8, t_submit_ns=0,
                      token_times_ns=[100 * MS, 110 * MS, 130 * MS, 140 * MS])
    assert r.output_tokens == 4
    assert r.ttft_ms == 100
    assert r.e2e_ms == 140
    assert r.itl_ms == [10, 20, 10]
    assert r.tpot_ms == pytest.approx(40 / 3)


def test_single_token_request_has_no_decode_metrics():
    r = RequestRecord("r0", prompt_tokens=8, t_submit_ns=0, token_times_ns=[50 * MS])
    assert r.itl_ms == [] and r.tpot_ms is None


def test_summary_excludes_failed_requests():
    good = RequestRecord("a", 8, 0, [10 * MS, 20 * MS])
    failed = RequestRecord("b", 8, 0, [], error="timeout")
    s = summarize([good, failed], wall_time_s=1.0)
    assert s["num_requests"] == 2
    assert s["num_errors"] == 1
    assert s["total_output_tokens"] == 2
    assert s["output_tokens_per_s"] == 2.0
    assert s["ttft_ms"]["n"] == 1


def test_run_result_round_trip(tmp_path):
    run = RunResult(
        config=_config(),
        environment={"note": "test"},
        requests=[RequestRecord("a", 8, 0, [10 * MS, 20 * MS])],
        wall_time_s=0.5,
    )
    path = run.save(tmp_path)
    loaded = RunResult.load(path)
    assert loaded.to_dict() == run.to_dict()


def test_environment_is_json_serializable():
    env = capture_environment()
    json.dumps(env)
    assert env["python"] and env["cpu"]["logical_cores"]

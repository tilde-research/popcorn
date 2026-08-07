"""GPU-free tests for the submitit sweep: planning, budgets, sentinels, and drain plumbing."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from jaxtyping import Float, Float32
from torch import Tensor

import popcorn.bench.sweep as planning
from popcorn.bench.model import Case, Environment, Record, Result
from popcorn.bench.sweep import DEFAULT_NODES, MAX_NODES, PHASES, SweepConfig, phase_spec
from popcorn.core.dispatcher import Dispatcher


def _load_sweep():
    path = Path(__file__).parents[2] / "scripts" / "bench_sweep.py"
    spec = importlib.util.spec_from_file_location("bench_sweep_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_sweep()


def _record(status="pass", case_id="case"):
    result = Result(status=status, grad=False)
    environment = Environment("cpu", torch.__version__, None, "2026-01-01T00:00:00+00:00")
    config = {"dims": {"D": 4}, "batch": [], "dtype": "float32", "args": {}, "present": []}
    return Record("op", "fast", "D=4", case_id, config, environment, result)


def _scripted_popen(returncodes):
    remaining = iter(returncodes)
    started = []

    def popen(*_args, **_kwargs):
        returncode = next(remaining)
        started.append(returncode)
        return SimpleNamespace(returncode=returncode, poll=lambda: returncode)

    return started, popen


def test_sweep_phase_order_is_reference_then_isolated_backends():
    assert PHASES == (
        "reference",
        "popcorn",
        "cudnn",
        "fa3",
        "fla",
        "liger",
        "quack",
        "transformer_engine",
        "unsloth",
    )
    assert phase_spec("popcorn").target == ".[fla]"
    assert phase_spec("liger").packages == ("transformers>=4.52.0,<5",)


def test_reference_phase_covers_only_gradient_modes_backends_need():
    op = SimpleNamespace(
        _impls=[
            SimpleNamespace(name="torch", forward_only=False),
            SimpleNamespace(name="fla", forward_only=True),
            SimpleNamespace(name="popcorn", forward_only=False),
        ]
    )
    assert planning.reference_modes(op) == (True, False)
    assert planning.reference_modes(SimpleNamespace(_impls=[SimpleNamespace(name="torch", forward_only=False)])) == (True,)


def test_long_fp16_reductions_switch_to_forward_only():
    def reference(
        x: Float[Tensor, "tokens channels"],
        weight: Float[Tensor, "channels"],
    ):
        return x * weight

    op = Dispatcher(reference)
    safe = Case((("tokens", 16_384), ("channels", 8)), (), torch.float16, (), frozenset())
    boundary = Case((("tokens", 32_768), ("channels", 8)), (), torch.float16, (), frozenset())
    large = Case((("tokens", 65_536), ("channels", 8)), (), torch.float16, (), frozenset())
    wide_range = Case((("tokens", 65_536), ("channels", 8)), (), torch.bfloat16, (), frozenset())

    assert planning.backward_safe(op, safe)
    assert not planning.backward_safe(op, boundary)
    assert not planning.backward_safe(op, large)
    assert planning.backward_safe(op, wide_range)
    assert planning.reference_modes(op, boundary) == (False,)


def test_fp16_output_fanout_switches_to_forward_only():
    def fused(
        x: Float[Tensor, "tokens channels"],
        norm_weight: Float[Tensor, "channels"],
        linear_weight: Float[Tensor, "out_features channels"],
    ) -> Float[Tensor, "tokens out_features"]:
        return (x * norm_weight) @ linear_weight.T

    op = Dispatcher(fused)
    failed = Case(
        (("tokens", 8192), ("channels", 18_432), ("out_features", 28_672)),
        (),
        torch.float16,
        (),
        frozenset(),
    )
    small = Case(
        (("tokens", 128), ("channels", 4096), ("out_features", 128)),
        (),
        torch.float16,
        (),
        frozenset(),
    )
    wide_range = Case(failed.dims, (), torch.bfloat16, (), frozenset())

    assert planning.backward_safe(op, small)
    assert not planning.backward_safe(op, failed)
    assert planning.backward_safe(op, wide_range)
    assert planning.reference_modes(op, failed) == (False,)


def test_fp16_output_fanout_includes_named_batch_axes():
    def gate(
        x: Float[Tensor, "batch seq heads key_dim"],
        weight: Float[Tensor, "heads"],
    ) -> Float[Tensor, "batch seq heads key_dim"]:
        return x * weight.view(1, 1, -1, 1)

    case = Case(
        (("batch", 8), ("seq", 8192), ("heads", 32), ("key_dim", 128)),
        (),
        torch.float16,
        (),
        frozenset(),
    )

    assert not planning.backward_safe(Dispatcher(gate), case)


def test_static_adapter_gates_remove_impossible_reference_cases(monkeypatch):
    def reference(x: Float[Tensor, "D"]):
        return x

    op = Dispatcher(reference)

    @op.register("fla")
    def adapter(x: Float32[Tensor, "D"]):
        return x

    fp16 = Case((("D", 8),), (), torch.float16, (), frozenset())
    fp32 = Case((("D", 8),), (), torch.float32, (), frozenset())
    plan = SimpleNamespace(flatten=lambda: [fp16, fp32])
    monkeypatch.setattr(planning, "available", lambda _name: True)
    monkeypatch.setattr(planning, "case_plan", lambda _op: plan)

    assert planning.reference_modes(op, fp16) == ()
    assert planning.reference_modes(op, fp32) == (True,)
    assert [(case.dtype, grad) for _, _, case, grad in planning.phase_work([op], "fla")] == [(torch.float32, True)]


def test_lpt_distribution_is_deterministic_and_balanced():
    units = [(8, "a"), (7, "b"), (6, "c"), (5, "d"), (4, "e"), (3, "f")]
    bins, loads = sweep.distribute(units, workers=3)
    assert bins == [["a", "f"], ["b", "e"], ["c", "d"]]
    assert loads == [11, 11, 11]


def test_affinity_balancing_splits_only_heavy_implementation_bundles():
    groups = [{"op": "large", "impl": "popcorn", "cases": [index] * 3} for index in range(3)] + [
        {"op": "small", "impl": "popcorn", "cases": [0] * 3}
    ]
    units = sweep.affinity_units(groups, workers=4)
    flattened = [group for _, chunk in units for group in chunk]
    large_chunks = sum(any(group["op"] == "large" for group in chunk) for _, chunk in units)
    small_chunks = sum(any(group["op"] == "small" for group in chunk) for _, chunk in units)

    assert {id(group) for group in flattened} == {id(group) for group in groups}
    assert large_chunks == 3
    assert small_chunks == 1


def test_slurm_job_is_one_strict_node_capped_singleton():
    config = SweepConfig(nodes=3)
    parameters = sweep._slurm_parameters(config)
    assert parameters["nodes"] == 3
    assert parameters["ntasks_per_node"] == 8
    assert parameters["gpus_per_task"] == 1
    assert parameters["exclusive"] is True
    assert parameters["dependency"] == "singleton"
    assert parameters["job_name"] == "popcorn-sweep"
    assert parameters["time"] == 24 * 60
    assert "srun_args" not in parameters


def test_queued_sweeps_ignores_jobs_finishing_dynamic_node_cleanup(monkeypatch):
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="132619 CG\n132645 PD\n132646 R\n")

    monkeypatch.setattr(sweep.subprocess, "run", run)

    assert sweep._queued_sweeps() == ["132645", "132646"]
    assert calls[0][-2:] == ["popcorn-sweep", "--format=%i %t"]


def test_slurm_container_mounts_home_and_uses_the_repository_workdir():
    image = "nvcr.io#nvidia/pytorch:25.09-py3"
    parameters = sweep._slurm_parameters(SweepConfig(container_image=image))

    assert parameters["srun_args"] == (
        f"--container-image={image}",
        "--container-mount-home",
        f"--container-workdir={sweep.ROOT}",
    )


def test_sweep_config_round_trips_container_and_rejects_invalid_execution_modes():
    assert SweepConfig().nodes == DEFAULT_NODES == 6
    assert MAX_NODES == 10
    assert SweepConfig().workers == 48
    targeted = SweepConfig(
        nodes=2,
        curves_only=True,
        container_image="nvcr.io#nvidia/pytorch:25.09-py3",
    )
    assert SweepConfig.from_mapping(targeted.to_dict()) == targeted
    assert sweep._nodes("10") == 10
    with pytest.raises(sweep.argparse.ArgumentTypeError, match="must not exceed 10"):
        sweep._nodes("11")
    with pytest.raises(ValueError, match="between 1 and 10"):
        SweepConfig(nodes=11)
    with pytest.raises(ValueError, match="requires a Slurm sweep"):
        SweepConfig(local=True, container_image="cuda:13")


def test_walltime_parses_slurm_and_minute_forms():
    assert sweep._minutes("24:00:00") == 1440
    assert sweep._minutes("01:30:30") == 91
    assert sweep._minutes("90") == 90


def test_each_backend_phase_wipes_and_reinstalls_one_extra(monkeypatch, tmp_path):
    commands = []
    removed = []
    monkeypatch.setattr(sweep.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(sweep.shutil, "rmtree", lambda path, ignore_errors: removed.append(path))
    monkeypatch.setattr(sweep, "_run_logged", lambda command, log, env: commands.append((command, env)))

    environment = tmp_path / "environment"
    version = f"{sweep.sys.version_info.major}.{sweep.sys.version_info.minor}"
    assert sweep.install_environment("fla", environment, tmp_path / "install.log") == environment / "bin" / "python"
    assert removed == [environment]
    assert commands[0][0] == ["/usr/bin/uv", "venv", "--clear", "--python", version, str(environment)]
    assert commands[1][0][-2:] == ["-e", ".[fla]"]
    assert all(command[1]["POPCORN_SKIP_REPORTS"] == "1" for command in commands)
    assert all("FLASH_ATTENTION_DISABLE_SM80" not in command[1] for command in commands)


def test_popcorn_phase_allows_its_fla_support_dependency_without_measuring_it(monkeypatch):
    case = Case((("D", 4),), (), torch.float32, (), frozenset())
    op = SimpleNamespace(
        name="op",
        specs=(),
        _impls=[
            SimpleNamespace(name="popcorn", forward_only=False),
            SimpleNamespace(name="fla", forward_only=False),
            SimpleNamespace(name="torch", forward_only=False),
        ],
    )
    monkeypatch.setattr(planning, "available", lambda _name: True)
    monkeypatch.setattr(planning, "case_plan", lambda _op: SimpleNamespace(flatten=lambda: [case]))

    work = planning.phase_work([op], "popcorn")

    assert [(impl, planned) for _, impl, planned, _ in work] == [("popcorn", case)]


def test_fa3_phase_installs_build_requirements_before_the_extra(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sweep.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(sweep.shutil, "rmtree", lambda *args, **kwargs: None)
    monkeypatch.setattr(sweep, "_run_logged", lambda command, log, env: calls.append((command, env)))

    sweep.install_environment("fa3", tmp_path / "environment", tmp_path / "install.log")
    commands = [command for command, _ in calls]
    assert len(commands) == 3
    assert commands[1][-4:] == ["torch", "setuptools", "wheel", "packaging"]
    assert commands[2][-2:] == ["-e", ".[fa3]"]
    assert {
        name for name, value in calls[2][1].items() if name.startswith("FLASH_ATTENTION_DISABLE_") and value == "TRUE"
    } >= {
        "FLASH_ATTENTION_DISABLE_SM80",
        "FLASH_ATTENTION_DISABLE_FP8",
        "FLASH_ATTENTION_DISABLE_PAGEDKV",
        "FLASH_ATTENTION_DISABLE_APPENDKV",
        "FLASH_ATTENTION_DISABLE_LOCAL",
        "FLASH_ATTENTION_DISABLE_SOFTCAP",
        "FLASH_ATTENTION_DISABLE_HDIMDIFF64",
        "FLASH_ATTENTION_DISABLE_HDIMDIFF192",
    }


def test_environment_fingerprint_includes_phase_build_environment(monkeypatch):
    spec = phase_spec("fa3")
    original = sweep._lock_fingerprint("fa3", "cuda13")
    monkeypatch.setattr(
        sweep,
        "phase_spec",
        lambda _name: replace(spec, build_env=(*spec.build_env, ("FLASH_ATTENTION_DISABLE_SPLIT", "TRUE"))),
    )

    assert sweep._lock_fingerprint("fa3", "cuda13") != original


def test_cudnn_preflight_executes_the_direct_adapter(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", run)

    assert sweep.preflight_cudnn(tmp_path / "python") is None
    command, kwargs = calls[0]
    assert command[:2] == [str(tmp_path / "python"), "-c"]
    assert 'backend="cudnn"' in command[2]
    assert kwargs["env"]["POPCORN_SKIP_REPORTS"] == "1"


def test_cudnn_preflight_reports_the_last_error_line(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="traceback\nMultiple libcudart libraries found\n"),
    )

    assert sweep.preflight_cudnn(tmp_path / "python") == (
        "cuDNN frontend preflight failed: Multiple libcudart libraries found"
    )


def test_environment_reuse_skips_rebuild_until_the_lock_or_container_changes(monkeypatch, tmp_path):
    built = []

    def fake_install(phase_name, destination, log):
        (destination / "bin").mkdir(parents=True, exist_ok=True)
        (destination / "bin" / "python").touch()
        built.append(phase_name)
        return destination / "bin" / "python"

    monkeypatch.setattr(sweep, "install_environment", fake_install)
    monkeypatch.setattr(sweep, "_lock_fingerprint", lambda phase, image=None: f"{phase}-{image}-v1")
    environment = tmp_path / "environment"

    sweep._ensure_environment("fla", environment, tmp_path / "install.log", "cuda13")
    sweep._ensure_environment("fla", environment, tmp_path / "install.log", "cuda13")
    assert built == ["fla"]

    sweep._ensure_environment("fla", environment, tmp_path / "install.log", "cuda13-new")
    assert built == ["fla", "fla"]

    monkeypatch.setattr(sweep, "_lock_fingerprint", lambda phase, image=None: f"{phase}-{image}-v2")
    sweep._ensure_environment("fla", environment, tmp_path / "install.log", "cuda13-new")
    assert built == ["fla", "fla", "fla"]


def test_case_budget_floors_at_timeout_and_scales_with_bytes():
    def reference(x: Float[Tensor, "seq hidden"]):
        return x

    op = Dispatcher(reference)
    small = Case((("hidden", 8), ("seq", 8)), (), torch.float32, (), frozenset())
    huge = Case((("hidden", 65536), ("seq", 65536)), (), torch.float32, (), frozenset())
    assert sweep._budget(op, small, timeout=300) == 300
    assert sweep._budget(op, huge, timeout=300) == 900  # capped at 3x
    assert sweep._budget(op, huge, timeout=300, phase_name="reference") == 60
    assert sweep._budget(op, huge, timeout=30, phase_name="reference") == 30
    assert 300 < sweep._budget(op, Case((("hidden", 16384), ("seq", 16384)), (), torch.float32, (), frozenset()), 300) < 900


def test_infeasibility_prices_correctness_copies_and_never_prunes_unpriceable_cases():
    def reference(x: Float[Tensor, "seq hidden"]):
        return x

    op = Dispatcher(reference)
    case = Case((("hidden", 1024), ("seq", 1024)), (), torch.float32, (), frozenset())
    bytes_in = 1024 * 1024 * 4
    assert sweep._infeasible(op, case, grad=True, capacity=bytes_in * 10)
    assert not sweep._infeasible(op, case, grad=True, capacity=bytes_in * 11)
    assert sweep._infeasible(op, case, grad=False, capacity=bytes_in * 6)
    assert not sweep._infeasible(op, case, grad=False, capacity=bytes_in * 7)
    unpriceable = Case((("mystery", 4),), (), torch.float32, (), frozenset())
    assert not sweep._infeasible(op, unpriceable, grad=True, capacity=1)


def test_infeasibility_rejects_reference_case_without_cudnn_workspace():
    def convolution(
        scores: Float[Tensor, "batch channels seq seq"],
        weight: Float[Tensor, "out_channels channels kernel_size kernel_size"],
        bias: Float[Tensor, "out_channels"] | None = None,
    ):
        del weight, bias
        return scores

    op = Dispatcher(convolution)
    case = Case(
        (("batch", 1), ("channels", 12_288), ("kernel_size", 11), ("out_channels", 2048), ("seq", 37)),
        (),
        torch.bfloat16,
        (),
        frozenset({"bias"}),
    )

    assert sweep._infeasible(op, case, grad=True, capacity=80 * 2**30)


def test_shard_reader_tolerates_the_torn_line_a_killed_worker_leaves(tmp_path):
    shard = tmp_path / "0.jsonl"
    good = _record()
    shard.write_text(json.dumps(good.to_dict(), sort_keys=True) + "\n" + '{"op": "torn", "impl"')
    records = sweep._read_records(shard)
    assert [record.case_id for record in records] == ["case"]
    assert sweep._read_records(tmp_path / "absent.jsonl") == []


def test_sentinel_converts_the_killer_case_to_a_crash_row_once(tmp_path):
    def wedge(x: Float[Tensor, "D"]):
        return x + 1

    op = Dispatcher(wedge)
    case = Case((("D", 4),), (), torch.float32, (), frozenset())
    sentinel = tmp_path / "0.json"
    payload = {"op": "wedge", "impl": "torch", "grad": True, "config": case.config()}

    sweep._write_json(sentinel, payload)
    record = sweep._convert_sentinel(sentinel, {"wedge": op}, "cpu", done=set())
    assert record is not None
    assert record.result.status == "crash"
    assert record.result.grad is True
    assert record.case_id == case.case_id
    assert not sentinel.exists()
    assert sweep._convert_sentinel(sentinel, {"wedge": op}, "cpu", done=set()) is None

    sweep._write_json(sentinel, payload)
    already_done = {("wedge", "torch", case.case_id, True)}
    assert sweep._convert_sentinel(sentinel, {"wedge": op}, "cpu", done=already_done) is None
    assert not sentinel.exists()


def test_poison_marks_match_the_async_cuda_failures_and_nothing_routine():
    poisoned = (
        "inputs: AcceleratorError: CUDA error: an illegal memory access was encountered",
        "reference: RuntimeError: CUDA error: device-side assert triggered",
        "RuntimeError: CUDA error: misaligned address",
        "RuntimeError: CUDA error: unspecified launch failure",
    )
    routine = ("reference: RuntimeError: shape mismatch", "CUDA out of memory", "no result within 300s")
    for reason in poisoned:
        assert any(mark in reason for mark in sweep.POISON_MARKS), reason
    for reason in routine:
        assert not any(mark in reason for mark in sweep.POISON_MARKS), reason


@pytest.mark.parametrize("returncode", [sweep.WATCHDOG_EXIT, sweep.POISON_EXIT])
def test_supervise_does_not_count_controlled_recycles(monkeypatch, tmp_path, returncode):
    started, popen = _scripted_popen([returncode, returncode, returncode, 0])
    monkeypatch.setattr(sweep, "MAX_RESTARTS", 2)
    monkeypatch.setattr(sweep.subprocess, "Popen", popen)
    monkeypatch.setattr(sweep.time, "sleep", lambda _seconds: None)

    sweep._supervise(
        tmp_path,
        "reference",
        3,
        tmp_path / "python",
        reps=10,
        timeout=300,
        hardware=None,
        drain=tmp_path / "drain",
    )

    assert started == [returncode, returncode, returncode, 0]
    assert not (tmp_path / "reference" / "progress" / "3.json").exists()


def test_supervise_still_caps_unexpected_worker_deaths(monkeypatch, tmp_path):
    started, popen = _scripted_popen([137, 137])
    monkeypatch.setattr(sweep, "MAX_RESTARTS", 2)
    monkeypatch.setattr(sweep.subprocess, "Popen", popen)
    monkeypatch.setattr(sweep.time, "sleep", lambda _seconds: None)

    sweep._supervise(
        tmp_path,
        "reference",
        3,
        tmp_path / "python",
        reps=10,
        timeout=300,
        hardware=None,
        drain=tmp_path / "drain",
    )

    progress = sweep._read_json(tmp_path / "reference" / "progress" / "3.json")
    assert started == [137, 137]
    assert progress == {"error": "worker kept dying (exit 137 after 2 restarts)", "status": "failed"}


def test_reference_quality_gate_rejects_bad_statuses_and_timing_errors():
    assert sweep._quality_error({"pass": 8, "oom": 2, "timeout": 1}) == ""
    assert sweep._quality_error({"pass": 8, "fail": 2}, bench_errors=3) == "fail=2, bench_error=3"


def test_cached_quality_gate_rejects_only_bad_reference_evidence():
    passing = _record()
    failed = _record("fail")
    untimed = _record()
    untimed.result.bench_error = "timing failed"

    assert sweep._cached_quality_error("reference", [passing]) == ""
    assert sweep._cached_quality_error("reference", [passing, failed, untimed]) == "fail=1, bench_error=1"
    assert sweep._cached_quality_error("popcorn", [passing, failed, untimed]) == ""


def test_followers_start_each_announced_phase_exactly_once():
    control = {"job": "7", "status": "ready", "phase": "fla", "python": "p", "reps": 10, "timeout": 300}
    assert sweep._ready_phase(control, "7", handled=set()) == "fla"
    assert sweep._ready_phase(control, "7", handled={"fla"}) is None
    assert sweep._ready_phase(control, "8", handled=set()) is None
    assert sweep._ready_phase({"job": "7", "status": "done"}, "7", handled=set()) is None
    assert sweep._ready_phase(None, "7", handled=set()) is None


def test_silent_rank_drains_and_requeues_the_unfinished_phase(monkeypatch, tmp_path):
    state = sweep._new_state(tmp_path, ["fla"])
    phase = state["phases"][0]
    aggregate = {
        "measured": 3,
        "pruned": 0,
        "bench_errors": 0,
        "statuses": {"pass": 3},
        "failed": [],
        "terminal": 1,
    }
    monkeypatch.setattr(sweep, "_supervise", lambda *args, **kwargs: None)
    monkeypatch.setattr(sweep, "_ranks_settled", lambda *args, **kwargs: (True, "1 worker went silent"))
    monkeypatch.setattr(sweep, "_aggregate_progress", lambda *args, **kwargs: aggregate)
    monkeypatch.setattr(sweep, "_merge_shards", lambda *args, **kwargs: 3)
    drain, requeue = tmp_path / "drain", tmp_path / "requeue"

    sweep._run_phase(
        tmp_path,
        "fla",
        state,
        phase,
        2,
        tmp_path / "python",
        SweepConfig(local=True, nodes=1),
        drain,
        requeue,
    )

    assert drain.exists()
    assert requeue.exists()
    assert phase["status"] == "paused"
    assert "1 worker went silent" in phase["message"]


def test_fa3_preflight_names_the_toolchain_gap():
    assert sweep._cuda_mismatch(None, "13.0") == "nvcc is not on PATH; flash-attn-3 builds from source"
    assert "older than torch's CUDA 13.0" in sweep._cuda_mismatch("release 12.9, V12.9.86", "13.0")
    assert sweep._cuda_mismatch("release 13.0, V13.0.48", "13.0") is None
    assert sweep._cuda_mismatch("release 13.1, V13.1.2", "13.0") is None
    assert "no CUDA runtime" in sweep._cuda_mismatch("release 12.9", None)


def test_progress_view_reports_cache_measurements_pruning_and_statuses(tmp_path):
    state = sweep._new_state(tmp_path, list(sweep.PHASES))
    state.update(job_id="123", status="running", workers=80, attempts=2)
    phase = state["phases"][0]
    phase.update(
        status="running",
        total=100,
        cached=20,
        prior_pruned=5,
        infeasible=3,
        measured=47,
        pruned=10,
        bench_errors=1,
        statuses={"pass": 45, "oom": 2},
    )
    rendered = sweep.render(state)
    assert "80 GPU workers, attempt 2" in rendered
    assert "85/100" in rendered
    assert "cached 20" in rendered
    assert "ran 47" in rendered
    assert "pruned 15" in rendered
    assert "infeasible 3" in rendered
    assert "bench_error=1" in rendered
    assert "oom=2 pass=45" in rendered


def test_watch_tolerates_transient_missing_state(monkeypatch, tmp_path):
    completed = sweep._new_state(tmp_path, ["reference"])
    completed["status"] = "completed"
    states = iter((None, completed))
    monkeypatch.setattr(sweep, "_read_json", lambda *args, **kwargs: next(states))
    monkeypatch.setattr(sweep.time, "sleep", lambda _seconds: None)

    sweep.watch(tmp_path)


def test_manual_resume_revalidates_settled_phases(monkeypatch, tmp_path):
    state = sweep._new_state(tmp_path, ["reference", "popcorn", "fa3"])
    state["phases"][0]["status"] = "completed"
    state["phases"][1]["status"] = "skipped"
    state["phases"][2].update(status="installing", message="building isolated environment")
    sweep._write_json(tmp_path / sweep.STATE, state)
    monkeypatch.setattr(sweep, "expected_totals", lambda *args: {"reference": 3, "popcorn": 5, "fa3": 7})

    config = SweepConfig(phases=("reference", "popcorn", "fa3"), local=True)
    sweep._submit(tmp_path, config, dry_run=True, revalidate=True)

    resumed = sweep._read_json(tmp_path / sweep.STATE)
    assert [phase["status"] for phase in resumed["phases"]] == ["pending", "pending", "pending"]
    assert [phase["total"] for phase in resumed["phases"]] == [3, 5, 7]
    assert resumed["phases"][2]["message"] == "resuming unfinished phase"

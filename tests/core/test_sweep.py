"""GPU-free tests for the fixed-allocation sequential sweep coordinator."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_sweep():
    path = Path(__file__).parents[2] / "scripts" / "bench_sweep.py"
    spec = importlib.util.spec_from_file_location("bench_sweep_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_sweep()


def test_sweep_phase_order_is_reference_then_isolated_backends():
    assert sweep.PHASES == ("reference", "popcorn", "fa3", "fla", "liger", "quack", "unsloth")


def test_reference_phase_covers_only_gradient_modes_backends_need():
    op = SimpleNamespace(
        _impls=[
            SimpleNamespace(name="torch", forward_only=False),
            SimpleNamespace(name="fla", forward_only=True),
            SimpleNamespace(name="popcorn", forward_only=False),
        ]
    )
    assert sweep.reference_modes(op) == (True, False)
    assert sweep.reference_modes(SimpleNamespace(_impls=[SimpleNamespace(name="torch", forward_only=False)])) == (True,)


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


def test_job_script_has_one_strict_ten_node_allocation(tmp_path):
    text = sweep._job_script(tmp_path, "staff-prod", "24:00:00", reps=10, timeout=300)
    assert "#SBATCH --nodes=10" in text
    assert "#SBATCH --ntasks=80" in text
    assert "#SBATCH --ntasks-per-node=8" in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --dependency=singleton" in text
    assert "--array" not in text


def test_each_backend_phase_wipes_and_reinstalls_one_extra(monkeypatch, tmp_path):
    commands = []
    removed = []
    monkeypatch.setattr(sweep.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(sweep.shutil, "rmtree", lambda path, ignore_errors: removed.append(path))
    monkeypatch.setattr(sweep, "_run_logged", lambda command, log, env: commands.append((command, env)))

    environment = tmp_path / "environment"
    assert sweep.install_environment("fla", environment, tmp_path / "install.log") == environment / "bin" / "python"
    assert removed == [environment]
    assert commands[0][0] == ["/usr/bin/uv", "venv", "--clear", str(environment)]
    assert commands[1][0][-2:] == ["-e", ".[fla]"]
    assert all(command[1]["POPCORN_SKIP_REPORTS"] == "1" for command in commands)


def test_fa3_phase_installs_build_requirements_before_the_extra(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(sweep.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(sweep.shutil, "rmtree", lambda *args, **kwargs: None)
    monkeypatch.setattr(sweep, "_run_logged", lambda command, log, env: commands.append(command))

    sweep.install_environment("fa3", tmp_path / "environment", tmp_path / "install.log")
    assert len(commands) == 3
    assert commands[1][-4:] == ["torch", "setuptools", "wheel", "packaging"]
    assert commands[2][-2:] == ["-e", ".[fa3]"]


def test_progress_view_reports_cache_measurements_pruning_and_statuses(tmp_path):
    state = sweep._new_state(tmp_path)
    state["job_id"] = "123"
    state["status"] = "running"
    phase = state["phases"][0]
    phase.update(
        status="running",
        total=100,
        cached=20,
        prior_pruned=5,
        measured=50,
        pruned=10,
        statuses={"pass": 48, "oom": 2},
    )
    rendered = sweep.render(state)
    assert "fixed 10 nodes / 80 H100 workers" in rendered
    assert "85/100" in rendered
    assert "cached 20" in rendered
    assert "ran 50" in rendered
    assert "OOM-pruned 15" in rendered
    assert "oom=2 pass=48" in rendered

import sys
from types import SimpleNamespace

import pytest

from popcorn import cli


def test_top_level_help_lists_lazy_commands(capsys):
    cli.main([])

    output = capsys.readouterr().out
    assert "usage: popcorn <command> [args]" in output
    assert all(command in output for command in ("bench", "sweep", "loop", "profile"))


@pytest.mark.parametrize(
    ("command", "module_name"),
    [
        ("bench", "popcorn.bench.__main__"),
        ("loop", "popcorn.bench.loop"),
        ("profile", "popcorn.impls._profile"),
    ],
)
def test_package_commands_import_lazily_and_forward_arguments(monkeypatch, command, module_name):
    called = {}

    def run(*, prog):
        called.update(prog=prog, argv=list(sys.argv))

    monkeypatch.setattr(cli.importlib, "import_module", lambda name: called.update(module=name) or SimpleNamespace(main=run))
    original = list(sys.argv)

    cli.main([command, "target", "--flag"])

    assert called == {
        "module": module_name,
        "prog": f"popcorn {command}",
        "argv": [f"popcorn {command}", "target", "--flag"],
    }
    assert sys.argv == original


def test_sweep_forwards_to_checkout_script(monkeypatch):
    called = {}

    def run():
        called["argv"] = list(sys.argv)

    monkeypatch.setattr(cli, "_load_sweep", lambda: SimpleNamespace(main=run))
    original = list(sys.argv)

    cli.main(["sweep", "status", "logs/sweeps/run"])

    assert called["argv"] == ["popcorn sweep", "status", "logs/sweeps/run"]
    assert sys.argv == original


def test_sweep_explains_that_it_needs_a_source_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "SWEEP_SCRIPT", tmp_path / "missing.py")

    with pytest.raises(SystemExit, match="available only from a Popcorn source checkout"):
        cli.main(["sweep", "status"])


def test_unknown_command_exits_with_usage(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["unknown"])

    assert error.value.code == 2
    assert "unknown command 'unknown'" in capsys.readouterr().err

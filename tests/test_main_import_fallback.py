import subprocess
import sys
from pathlib import Path


def test_main_fallback_import_uses_local_modules(tmp_path) -> None:
    """Ensure importing `main` from within the `src` directory triggers the
    fallback (script-style) imports rather than package-style imports.

    The test runs a short Python snippet with the current working directory set
    to `src/` so that `from src.cli` will raise ModuleNotFoundError and the
    fallback `import cli` path is exercised. We pre-import and stub
    `ollama.check_and_start_ollama` to avoid side effects during import.
    """

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    assert src_dir.exists(), "expected src directory to exist for test"

    # Create a temporary minimal set of top-level modules (cli/config/agent/ollama)
    # so the fallback import path imports these lightweight stubs rather than
    # the repository's heavier modules that expect `src` to be a package.
    temp_pkg = tmp_path / "fallback_src"
    temp_pkg.mkdir()
    (temp_pkg / "cli.py").write_text(
        "def build_arg_parser():\n    class P:\n        def parse_args(self):\n            return type('NS', (), {})()\n    return P()\n\ndef configure_logging(*a, **k):\n    pass\n"
    )
    (temp_pkg / "config.py").write_text("class AgentConfig:\n    def __init__(self, **kwargs):\n        pass\n")
    (temp_pkg / "agent.py").write_text(
        "class Agent:\n    def __init__(self, cfg):\n        pass\n    def answer_once(self, q):\n        return q\n    def run(self):\n        return None\n"
    )
    (temp_pkg / "ollama.py").write_text("def check_and_start_ollama(exit_on_failure=True):\n    return True\n")

    python = sys.executable
    cmd = [
        python,
        "-c",
        (
            "import runpy, sys; sys.path.insert(0,'.');"
            f"runpy.run_path('{src_dir}/main.py', run_name='main'); print('FALLBACK_OK')"
        ),
    ]

    proc = subprocess.run(cmd, cwd=str(temp_pkg), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "FALLBACK_OK" in proc.stdout

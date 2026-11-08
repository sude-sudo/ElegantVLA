"""Dependency-free checks for the two backend source layouts."""

import ast
import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import symtable
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceLayoutTest(unittest.TestCase):
    def test_python_syntax_and_local_import_targets(self):
        for path in (ROOT / "elegantvla").rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    modules = []
                    if isinstance(node, ast.ImportFrom) and node.module:
                        modules.append(node.module)
                    elif isinstance(node, ast.Import):
                        modules.extend(alias.name for alias in node.names)
                    for module in modules:
                        if module == "elegantvla" or module.startswith("elegantvla."):
                            target = ROOT.joinpath(*module.split("."))
                            self.assertTrue(
                                target.with_suffix(".py").is_file()
                                or (target / "__init__.py").is_file(),
                                f"{path}: unresolved import {module}",
                            )
                            if isinstance(node, ast.ImportFrom):
                                source_path = (
                                    target.with_suffix(".py") if target.with_suffix(".py").is_file()
                                    else target / "__init__.py"
                                )
                                symbols = symtable.symtable(source_path.read_text(), str(source_path), "exec")
                                exports = {
                                    symbol.get_name() for symbol in symbols.get_symbols()
                                    if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
                                }
                                for alias in node.names:
                                    self.assertTrue(
                                        alias.name == "*" or alias.name in exports
                                        or (target / f"{alias.name}.py").is_file(),
                                        f"{path}: {module} has no import target {alias.name}",
                                    )

    def test_core_global_names_resolve(self):
        directories = [ROOT / "elegantvla" / backend / "core" for backend in ("gr00t", "openpi")]
        directories.extend([ROOT / "elegantvla/scheduling", ROOT / "elegantvla/openpi/legacy"])
        for directory in directories:
            for path in directory.glob("*.py"):
                with self.subTest(path=str(path.relative_to(ROOT))):
                    table = symtable.symtable(path.read_text(), str(path), "exec")
                    available = set(vars(builtins))
                    available.update(
                        symbol.get_name() for symbol in table.get_symbols()
                        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
                    )
                    pending = [table]
                    while pending:
                        scope = pending.pop()
                        pending.extend(scope.get_children())
                        for symbol in scope.get_symbols():
                            if symbol.is_global() and symbol.is_referenced():
                                self.assertIn(
                                    symbol.get_name(), available,
                                    f"{path}: unbound global in {scope.get_name()}: {symbol.get_name()}",
                                )

    def test_entrypoints_are_discoverable(self):
        commands = {
            "gr00t": ("evaluate", "benchmark", "train_rl", "eval_rl", "train_bc"),
            "openpi": ("evaluate", "benchmark", "serve"),
        }
        for backend, names in commands.items():
            for name in names:
                module = f"elegantvla.{backend}.entrypoints.{name}"
                with self.subTest(module=module):
                    self.assertIsNotNone(importlib.util.find_spec(module))

    def test_backends_share_the_same_core_roles(self):
        for backend in ("gr00t", "openpi"):
            for name in ("types", "rule_policy", "rl_policy", "wrapper", "adapter", "action_head"):
                self.assertTrue((ROOT / "elegantvla" / backend / "core" / f"{name}.py").is_file())

    def test_launchers_resolve_from_an_external_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHON_BIN="/bin/echo", MPLCONFIGDIR=directory)
            env.pop("SIMPLERENV_ROOT", None)
            env.pop("SIMPLERENV_SITE_PACKAGES", None)
            for backend in ("gr00t", "openpi"):
                script = ROOT / "elegantvla" / backend / "scripts" / "run.sh"
                with self.subTest(backend=backend):
                    result = subprocess.run(
                        ["sh", str(script), "evaluate", "--help"],
                        cwd=directory,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=10,
                    )
                    self.assertEqual(
                        result.stdout.strip(),
                        f"-m elegantvla.{backend}.entrypoints.evaluate --help",
                    )

    def test_shell_syntax(self):
        scripts = list((ROOT / "scripts").glob("*.sh"))
        scripts.extend((ROOT / "elegantvla").rglob("*.sh"))
        for script in scripts:
            with self.subTest(script=str(script.relative_to(ROOT))):
                subprocess.run(["bash", "-n", str(script)], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()

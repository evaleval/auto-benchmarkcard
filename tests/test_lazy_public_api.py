"""Fresh-process facade regressions; no providers, network, or real credentials."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
EXPORTS = [
    ("Config", "auto_benchmarkcard.config"),
    ("GraphState", "auto_benchmarkcard.state"),
    ("build_workflow", "auto_benchmarkcard.workflow"),
    ("OutputManager", "auto_benchmarkcard.output"),
    ("sanitize_benchmark_name", "auto_benchmarkcard.output"),
    ("extract_card", "auto_benchmarkcard.card_utils"),
    ("extract_missing_fields", "auto_benchmarkcard.card_utils"),
    ("run_eee_pipeline", "auto_benchmarkcard.eee_workflow"),
    ("process_single_benchmark", "auto_benchmarkcard.eee_workflow"),
    ("unitxt_benchmark_lookup", "auto_benchmarkcard.tools.unitxt"),
    ("UnitxtMetadata", "auto_benchmarkcard.tools.unitxt"),
    ("hf_dataset_metadata", "auto_benchmarkcard.tools.hf"),
    ("extract_ids", "auto_benchmarkcard.tools.extractor"),
    ("RAGRetriever", "auto_benchmarkcard.tools.rag"),
    ("MetadataIndexer", "auto_benchmarkcard.tools.rag"),
    ("atomize_benchmark_card", "auto_benchmarkcard.tools.rag"),
    ("compose_benchmark_card", "auto_benchmarkcard.tools.composer"),
    ("extract_paper_with_docling", "auto_benchmarkcard.tools.docling"),
]

GUARD = r'''
import atexit, os, sys
from pathlib import Path
_forbidden_attempts = []
def _audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        _forbidden_attempts.append(event)
        raise RuntimeError("network forbidden in lazy-import fixture")
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = Path(os.fsdecode(args[0])).name
        if name == ".env" or name.startswith(".env.") or name in {"token", "stored_tokens"}:
            _forbidden_attempts.append("credential-file-open")
            raise RuntimeError("credential file forbidden in lazy-import fixture")
sys.addaudithook(_audit)
def _assert_no_forbidden_attempts():
    if _forbidden_attempts:
        sys.stderr.write("forbidden I/O attempts: " + repr(_forbidden_attempts) + "\n")
        os._exit(93)
atexit.register(_assert_no_forbidden_attempts)
'''

LIGHT_GUARD = r'''
import importlib.abc
_heavy = (
    "auto_benchmarkcard.workflow", "auto_benchmarkcard.workers",
    "auto_benchmarkcard.eee_workflow", "auto_benchmarkcard.tools.composer.composer_tool",
    "auto_benchmarkcard.tools.docling", "auto_benchmarkcard.tools.factreasoner",
    "auto_benchmarkcard.tools.unitxt", "auto_benchmarkcard.tools.rag",
    "docling", "fact_reasoner", "unitxt", "pgmpy", "torch", "litellm",
)
class _NoHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in _heavy):
            raise AssertionError("unrelated heavy module imported: " + fullname)
sys.meta_path.insert(0, _NoHeavy())
'''


def _fresh(code, *, light=False):
    # Do not forward credentials or discover the real checkout's .env.
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TMPDIR")
           if key in os.environ}
    env.update(PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", PYTHON_DOTENV_DISABLED="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               LITELLM_LOCAL_MODEL_COST_MAP="True", ANONYMIZED_TELEMETRY="False",
               OTEL_SDK_DISABLED="true")
    assertion = (
        "\nimport auto_benchmarkcard as _abc\n"
        f"assert Path(_abc.__file__).resolve().is_relative_to(Path({str(SRC)!r}).resolve())\n"
    )
    # Some optional libraries explicitly read cwd/.env via dotenv_values,
    # independently of PYTHON_DOTENV_DISABLED. Exercise the real source tree
    # from an empty working directory, retaining the strict I/O guard.
    with tempfile.TemporaryDirectory(prefix="abc-public-api-") as fixture_cwd:
        result = subprocess.run(
            [sys.executable, "-B", "-c", GUARD + (LIGHT_GUARD if light else "")
             + textwrap.dedent(code) + assertion],
            cwd=fixture_cwd, env=env, capture_output=True, text=True, timeout=60,
        )
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


def test_metadata_all_and_dir_preserve_eighteen_names_without_heavy_imports():
    _fresh(f'''
        import auto_benchmarkcard as abc
        from auto_benchmarkcard.tools import composer
        expected = {json.dumps([name for name, _ in EXPORTS])}
        assert abc.__all__ == expected
        assert abc.__version__ == "0.1.0" and abc.__author__ == "Aris Hofmann"
        assert set(expected).issubset(dir(abc))
        assert composer.__all__ == ["compose_benchmark_card"]
        assert "compose_benchmark_card" in dir(composer)
        assert "compose_benchmark_card" not in vars(composer)
        assert set(expected) & set(vars(abc)) == {{"Config"}}
        assert "auto_benchmarkcard.config" in sys.modules
        assert "auto_benchmarkcard.workflow" not in sys.modules
        assert "auto_benchmarkcard.tools.composer.composer_tool" not in sys.modules
    ''', light=True)


@pytest.mark.parametrize("module", ["auto_benchmarkcard", "auto_benchmarkcard.tools.composer"])
def test_unknown_attributes_remain_attribute_errors_and_do_not_load_exports(module):
    _fresh(f'''
        import importlib
        pkg = importlib.import_module({module!r})
        before = set(sys.modules)
        for name in ("not_a_public_export", "__missing_attribute__"):
            try:
                getattr(pkg, name)
            except AttributeError as exc:
                assert name in str(exc) and pkg.__name__ in str(exc)
            else:
                raise AssertionError("unknown name accepted")
            assert not hasattr(pkg, name)
        assert set(sys.modules) == before
        assert dir(pkg) == sorted(set(dir(pkg)))
    ''', light=True)


@pytest.mark.parametrize("statement", [
    "from auto_benchmarkcard import Config, GraphState, OutputManager, sanitize_benchmark_name",
    "from auto_benchmarkcard.tools.composer.field_spec import CardSchema",
    "from auto_benchmarkcard.tools.composer import field_spec, evidence, validator, docstructure",
])
def test_representative_lightweight_from_imports_do_not_initialize_tools(statement):
    _fresh(statement, light=True)


def test_every_public_name_is_the_original_object_and_cached_without_wrapping():
    _fresh(f'''
        import importlib
        import auto_benchmarkcard as abc
        expected = {json.dumps(EXPORTS)}
        for name, module_name in expected:
            resolved = getattr(abc, name)
            assert resolved is getattr(importlib.import_module(module_name), name)
            assert vars(abc)[name] is resolved and getattr(abc, name) is resolved
        from auto_benchmarkcard.tools.composer import compose_benchmark_card
        from auto_benchmarkcard.tools.composer.composer_tool import compose_benchmark_card as direct
        assert abc.compose_benchmark_card is compose_benchmark_card is direct
        assert callable(direct.func) and direct.name == "compose_benchmark_card"
        for name, mod in list(sys.modules.items()):
            if name.startswith("auto_benchmarkcard") and getattr(mod, "__file__", None):
                assert Path(mod.__file__).resolve().is_relative_to(Path({str(SRC)!r}).resolve())
    ''')


def test_star_import_preserves_exact_public_names_and_objects():
    _fresh('''
        import auto_benchmarkcard as abc
        scope = {}
        exec("from auto_benchmarkcard import *", scope)
        assert set(scope) - {"__builtins__"} == set(abc.__all__)
        assert all(scope[name] is getattr(abc, name) for name in abc.__all__)
        from auto_benchmarkcard.tools import composer
        composer_scope = {}
        exec("from auto_benchmarkcard.tools.composer import *", composer_scope)
        assert set(composer_scope) - {"__builtins__"} == {"compose_benchmark_card"}
        assert composer_scope["compose_benchmark_card"] is abc.compose_benchmark_card
    ''')


@pytest.mark.parametrize("first", [
    "from auto_benchmarkcard.tools.composer import field_spec, composer_tool",
    "from auto_benchmarkcard import workflow, eee_workflow",
    "from auto_benchmarkcard import compose_benchmark_card",
])
def test_submodule_import_orders_preserve_modules_and_do_not_shadow_exports(first):
    _fresh(first + '''
import importlib, types
import auto_benchmarkcard as abc
import auto_benchmarkcard.tools.composer as composer_pkg
import auto_benchmarkcard.tools.composer.composer_tool as composer_module
from auto_benchmarkcard import workflow
from auto_benchmarkcard.tools.composer import field_spec, evidence
assert isinstance(workflow, types.ModuleType)
assert isinstance(composer_pkg, types.ModuleType)
assert isinstance(composer_module, types.ModuleType)
assert isinstance(field_spec, types.ModuleType) and isinstance(evidence, types.ModuleType)
assert composer_pkg.composer_tool is composer_module
assert abc.compose_benchmark_card is composer_pkg.compose_benchmark_card
assert abc.compose_benchmark_card is composer_module.compose_benchmark_card
assert abc.build_workflow is workflow.build_workflow
assert hasattr(abc.build_workflow(), "invoke")  # Build only; never invoke workers.
assert importlib.import_module("auto_benchmarkcard.config").Config is abc.Config
''')


def test_root_import_failure_is_not_cached_and_the_next_access_can_retry():
    _fresh('''
        import auto_benchmarkcard as abc
        real_import = abc._import_module
        def unavailable(name):
            raise ModuleNotFoundError("synthetic missing optional dependency")
        abc._import_module = unavailable
        try:
            abc.sanitize_benchmark_name
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("original import error was swallowed")
        assert "sanitize_benchmark_name" not in vars(abc)
        abc._import_module = real_import
        from auto_benchmarkcard.output import sanitize_benchmark_name
        assert abc.sanitize_benchmark_name is sanitize_benchmark_name
    ''', light=True)


def test_composer_import_failure_does_not_cache_and_can_retry_without_a_wrapper():
    _fresh('''
        import importlib.abc, types
        from auto_benchmarkcard.tools import composer
        class Missing(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "auto_benchmarkcard.tools.composer.composer_tool":
                    raise ModuleNotFoundError("synthetic unavailable composer dependency")
        missing = Missing(); sys.meta_path.insert(0, missing)
        try:
            composer.compose_benchmark_card
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("composer import failure was swallowed")
        assert "compose_benchmark_card" not in vars(composer)
        sys.meta_path.remove(missing)
        module = types.ModuleType("auto_benchmarkcard.tools.composer.composer_tool")
        sentinel = object(); module.compose_benchmark_card = sentinel
        sys.modules[module.__name__] = module
        assert composer.compose_benchmark_card is sentinel
        assert composer.compose_benchmark_card is sentinel
    ''')


def test_reload_discards_resolved_facade_cache_but_preserves_child_module_identity():
    _fresh('''
        import importlib, types
        import auto_benchmarkcard as abc
        from auto_benchmarkcard import output
        original = abc.sanitize_benchmark_name
        replacement = object(); output.sanitize_benchmark_name = replacement
        importlib.reload(abc)
        assert "sanitize_benchmark_name" not in vars(abc)
        assert abc.output is output and abc.sanitize_benchmark_name is replacement
        output.sanitize_benchmark_name = original
        importlib.reload(abc)
        assert abc.sanitize_benchmark_name is original
        from auto_benchmarkcard.tools import composer
        module = types.ModuleType("auto_benchmarkcard.tools.composer.composer_tool")
        one, two = object(), object(); module.compose_benchmark_card = one
        sys.modules[module.__name__] = module
        assert composer.compose_benchmark_card is one
        module.compose_benchmark_card = two
        importlib.reload(composer)
        assert "compose_benchmark_card" not in vars(composer)
        assert composer.compose_benchmark_card is two
    ''', light=True)


def test_concurrent_first_access_and_deleted_cache_keep_identical_objects():
    _fresh('''
        from concurrent.futures import ThreadPoolExecutor
        import auto_benchmarkcard as abc
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(lambda _: abc.sanitize_benchmark_name, range(32)))
        assert all(value is values[0] for value in values)
        del abc.sanitize_benchmark_name
        assert abc.sanitize_benchmark_name is values[0]
        assert values[0]("allenai/OLMo-2-1124-7B-7df9a82518af") == "allenai-olmo-2-1124-7b-7df9a82518af"
    ''', light=True)


def test_config_environment_snapshot_still_occurs_on_root_import():
    _fresh('''
        os.environ["LLM_ENGINE_TYPE"] = "hf"
        import auto_benchmarkcard as abc
        assert "auto_benchmarkcard.config" in sys.modules
        os.environ["LLM_ENGINE_TYPE"] = "vllm"
        assert abc.Config.LLM_ENGINE_TYPE == "hf"
        from auto_benchmarkcard.config import Config
        assert abc.Config is Config
    ''', light=True)

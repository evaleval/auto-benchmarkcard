#!/usr/bin/env python3
"""BenchmarkCard CLI for benchmark metadata extraction and validation."""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, Optional, Union

import typer
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Warning filters are centralized in logging_setup.py (imported via Config).
import auto_benchmarkcard.logging_setup  # noqa: F401  ensure filters are active

original_level = logging.root.level
logging.root.setLevel(logging.WARNING)
try:
    from auto_benchmarkcard.config import Config
except ImportError as e:
    Console(stderr=True).print(f"[red]Import Error: {e}[/red]")
    Console(stderr=True).print("[red]Please ensure all dependencies are installed and the project is properly set up.[/red]")
    sys.exit(1)

logging.root.setLevel(original_level)

console = Console(log_time=False, log_time_format="[%X]")
error_console = Console(stderr=True, style="bold red")
app = typer.Typer(
    name="benchmarkcard",
    help="Benchmark Metadata Extraction & Validation",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    add_completion=False,
)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with Rich console output and optional file handler."""
    log_level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger("benchmarkcard")
    logger.setLevel(log_level)
    logger.handlers.clear()

    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=verbose,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=verbose,
    )
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def enable_debug_logging() -> None:
    """Re-enable debug-level logging for all suppressed external libraries."""
    debug_loggers = [
        "faiss.loader",
        "faiss",
        "vllm",
        "vllm.config",
        "vllm.utils.import_utils",
        "transformers",
        "httpx",
        "httpcore",
        "litellm",
        "LiteLLM",
        "litellm.llms",
        "litellm.utils",
        "litellm.cost_calculator",
        "openai",
        "urllib3",
        "huggingface_hub",
        "docling",
        "unitxt",
        "chromadb",
        "sentence_transformers",
        "fact_reasoner",
        "FactReasoner",
        "ai_atlas_nexus",
    ]

    for logger_name in debug_loggers:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)

    warnings.resetwarnings()


def display_banner() -> None:
    """Print the application banner."""
    title = Text("Auto-BenchmarkCard", style="bold cyan")
    subtitle = Text("Benchmark Metadata Extraction & Validation", style="dim italic")

    banner_content = Align.center(
        Columns(
            [
                Panel(
                    Align.center(f"{title}\n{subtitle}"),
                    border_style="cyan",
                    padding=(1, 2),
                    title="[bold white]Welcome[/bold white]",
                )
            ],
            expand=True,
        )
    )

    console.print(banner_content)
    console.print()


def display_workflow_summary(
    benchmark: str, execution_time: float, step_results: dict, output_manager=None
):
    """Render a summary table of workflow step results and timing."""
    console.print("\n" + "=" * 60)
    console.print(f"[bold cyan]Workflow Summary: {benchmark}[/bold cyan]")
    console.print("=" * 60)

    results_table = Table(border_style="green", title="Processing Results")
    results_table.add_column("Step", style="cyan", width=25)
    results_table.add_column("Status", style="green", width=10)
    results_table.add_column("Details", style="white")

    for step_name, result in step_results.items():
        status_icon = "[OK]" if result.get("success", False) else "[FAIL]"
        details = result.get("details", "No details available")
        results_table.add_row(step_name, status_icon, details)

    console.print(results_table)

    successful_steps = sum(1 for r in step_results.values() if r.get("success", False))
    total_steps = len(step_results)

    console.print(f"\n[bold]Execution Summary:[/bold]")
    console.print(f"• Total execution time: [cyan]{format_duration(execution_time)}[/cyan]")
    console.print(f"• Steps completed: [green]{successful_steps}/{total_steps}[/green]")

    if output_manager:
        summary = output_manager.get_summary()
        output_dir = os.path.abspath(summary["session_directory"])
        console.print(f"• Output directory: [cyan]{output_dir}[/cyan]")
        console.print(f"• Generation timestamp: [cyan]{summary['timestamp']}[/cyan]")

    console.print("=" * 60)


def create_progress_display() -> Progress:
    """Create a Rich progress bar with spinner and time tracking."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


@contextmanager
def workflow_step(step_name: str, step_number: int = None, total_steps: int = None):
    """Context manager that tracks and displays workflow step timing."""
    if step_number and total_steps:
        step_indicator = f"[dim]Step {step_number}/{total_steps}[/dim] "
    else:
        step_indicator = ""

    console.print(f"\n{step_indicator}[bold blue]{step_name}[/bold blue]")

    with Status(
        f"[blue]{step_name}...[/blue]",
        console=console,
        spinner="dots12",
    ) as status:
        start_time = time.time()
        try:
            yield status
            elapsed = time.time() - start_time
            console.print(f"[green][OK] {step_name} completed[/green] [dim]({elapsed:.1f}s)[/dim]")
        except Exception:
            elapsed = time.time() - start_time
            console.print(f"[red][FAIL] {step_name} failed[/red] [dim]({elapsed:.1f}s)[/dim]")
            raise


@contextmanager
def workflow_substep(substep_name: str, show_completion: bool = True):
    """Context manager that tracks and displays a workflow sub-step."""
    with Status(
        f"[dim]{substep_name}...[/dim]",
        console=console,
        spinner="dots",
    ) as status:
        start_time = time.time()
        try:
            yield status
            if show_completion:
                elapsed = time.time() - start_time
                console.print(
                    f"  [green]• {substep_name} completed[/green] [dim]({elapsed:.1f}s)[/dim]"
                )
        except Exception as e:
            elapsed = time.time() - start_time
            console.print(
                f"  [red]• {substep_name} failed: {str(e)}[/red] [dim]({elapsed:.1f}s)[/dim]"
            )
            raise


def display_error(message: str, details: Optional[str] = None) -> None:
    """Display an error message in a styled panel."""
    error_panel = Panel(
        f"[bold red][FAIL] Error[/bold red]\n\n{message}"
        + (f"\n\n[dim]{details}[/dim]" if details else ""),
        border_style="red",
        title="[bold red]Execution Failed[/bold red]",
    )
    error_console.print(error_panel)


def display_success(message: str, details: Optional[str] = None) -> None:
    """Display a success message in a styled panel."""
    success_panel = Panel(
        f"[bold green][OK] Success[/bold green]\n\n{message}"
        + (f"\n\n[dim]{details}[/dim]" if details else ""),
        border_style="green",
        title="[bold green]Execution Completed[/bold green]",
    )
    console.print(success_panel)


def validate_benchmark_name(benchmark: str) -> str:
    """Validate and sanitize a benchmark name for filesystem use."""
    if not benchmark or not benchmark.strip():
        raise typer.BadParameter(
            "[red]Benchmark name cannot be empty[/red]\n"
            "[dim]Example: 'glue', 'safety.truthful_qa', 'ethos_binary'[/dim]"
        )

    sanitized = benchmark.strip()

    if len(sanitized) > 100:
        raise typer.BadParameter(
            f"[red]Benchmark name too long ({len(sanitized)} chars, max 100)[/red]"
        )

    invalid_chars = set(sanitized) - set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    )
    if invalid_chars:
        raise typer.BadParameter(
            f"[red]Invalid characters in benchmark name: {''.join(invalid_chars)}[/red]\n"
            "[dim]Only letters, numbers, dots, hyphens, and underscores are allowed[/dim]"
        )

    return sanitized


def validate_path(path: str, must_exist: bool = False) -> Path:
    """Resolve a path, optionally asserting it exists."""
    path_obj = Path(path).resolve()

    if must_exist and not path_obj.exists():
        raise typer.BadParameter(f"[red]Path does not exist: {path_obj}[/red]")

    return path_obj


def format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_session_info(session_dir: Path) -> Dict[str, Union[str, int, bool]]:
    """Extract session metadata (benchmark, status, file stats) from a session directory."""
    try:
        parts = session_dir.name.rsplit("_", 2)
        benchmark = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
        timestamp = "_".join(parts[-2:]) if len(parts) >= 2 else "unknown"

        benchmark_card_dir = session_dir / "benchmarkcard"
        tool_output_dir = session_dir / "tool_output"

        completed = benchmark_card_dir.exists() and any(benchmark_card_dir.glob("*.json"))
        tool_count = len(list(tool_output_dir.iterdir())) if tool_output_dir.exists() else 0

        total_size = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())
        file_count = len(list(session_dir.rglob("*")))

        return {
            "benchmark": benchmark,
            "timestamp": timestamp,
            "completed": completed,
            "tool_count": tool_count,
            "total_size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "modified_time": session_dir.stat().st_mtime,
        }
    except Exception:
        return {
            "benchmark": "unknown",
            "timestamp": "unknown",
            "completed": False,
            "tool_count": 0,
            "total_size_mb": 0.0,
            "file_count": 0,
            "modified_time": 0,
        }


@app.command("generate-unitxt")
def generate_unitxt(
    benchmark: Annotated[
        str,
        typer.Argument(
            help="Benchmark name from UnitXT catalog (e.g., 'glue', 'safety.truthful_qa')",
            callback=validate_benchmark_name,
        ),
    ],
    catalog: Annotated[
        Optional[str],
        typer.Option("--catalog", "-c", help="Path to custom UnitXT catalog directory"),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Custom output directory for results"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug mode with full tool logging"),
    ] = False,
    log_file: Annotated[
        Optional[str],
        typer.Option("--log-file", help="Save logs to file"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing output directory"),
    ] = False,
) -> None:
    """
    Generate a benchmark card from UnitXT catalog (alternative to 'generate').

    Processes a single benchmark through the full pipeline starting from a
    UnitXT catalog entry. Use 'generate' instead if you have evaluation data.

    [bold green]Examples:[/bold green]

        [dim]# Basic usage[/dim]
        benchmarkcard generate-unitxt glue

        [dim]# With custom output[/dim]
        benchmarkcard generate-unitxt safety.truthful_qa --output ./results -v
    """
    logger = setup_logging(verbose=verbose, log_file=log_file)

    if debug:
        enable_debug_logging()

    display_banner()

    if catalog:
        catalog_path = validate_path(catalog, must_exist=True)

    if output_dir:
        output_path = validate_path(output_dir)
        if output_path.exists() and not force:
            error_console.print(
                f"[red]Output directory already exists: {output_path}[/red]\n"
                "[dim]Use --force to overwrite or choose a different path[/dim]"
            )
            raise typer.Exit(1)

    start_time = time.time()

    try:
        Config.validate_config()

        from auto_benchmarkcard.workflow import (
            build_workflow, OutputManager, setup_logging_suppression, sanitize_benchmark_name,
        )
        setup_logging_suppression(debug_mode=debug)

        safe_name = sanitize_benchmark_name(benchmark)
        output_manager = OutputManager(safe_name, str(output_path) if output_dir else None)

        console.print(f"\n[bold cyan]Generating BenchmarkCard from UnitXT[/bold cyan]")
        console.print(f"[dim]Benchmark: {benchmark}[/dim]")

        initial_state = {
            "query": benchmark,
            "catalog_path": str(catalog_path) if catalog else None,
            "output_manager": output_manager,
            "unitxt_json": None,
            "extracted_ids": None,
            "hf_repo": None,
            "hf_json": None,
            "docling_output": None,
            "composed_card": None,
            "risk_enhanced_card": None,
            "completed": [],
            "errors": [],
            "hf_extraction_attempted": False,
            "rag_results": None,
            "factuality_results": None,
            "eee_metadata": None,
        }

        workflow = build_workflow()
        state = workflow.invoke(initial_state)

        execution_time = time.time() - start_time

        if state.get("errors"):
            for error in state["errors"]:
                console.print(f"[red]  {error}[/red]")
            raise typer.Exit(1)

        display_success(
            f"Benchmark '{benchmark}' processed successfully",
            f"Time: {format_duration(execution_time)}\n"
            f"Output: {output_manager.benchmarkcard_dir}",
        )

    except KeyboardInterrupt:
        execution_time = time.time() - start_time
        console.print(f"\n[yellow]Interrupted[/yellow] [dim]({format_duration(execution_time)})[/dim]")
        raise typer.Exit(130)

    except typer.Exit:
        raise

    except Exception as e:
        execution_time = time.time() - start_time
        display_error(
            f"Workflow failed for '{benchmark}'",
            f"Error: {e}\nTime: {format_duration(execution_time)}",
        )
        logger.error("Workflow failed: %s", e, exc_info=verbose)
        raise typer.Exit(1)


@app.command("generate")
def generate(
    eee_path: Annotated[
        str,
        typer.Argument(
            help="Path to evaluation data directory (e.g., ./eee_data or path to HF clone)",
        ),
    ],
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Custom output directory path for results",
        ),
    ] = None,
    benchmarks: Annotated[
        Optional[str],
        typer.Option(
            "--benchmarks",
            "-b",
            help="Comma-separated list of benchmark names to process (default: all)",
        ),
    ] = None,
    max_files: Annotated[
        int,
        typer.Option(
            "--max-files",
            help="Max eval files to sample per benchmark folder",
        ),
    ] = 50,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug mode"),
    ] = False,
) -> None:
    """
    Generate benchmark cards from evaluation data.

    Scans evaluation JSONs, discovers benchmarks, resolves HuggingFace repos,
    and generates fact-checked BenchmarkCards for each.

    [bold green]Examples:[/bold green]

        [dim]# Generate cards for all benchmarks[/dim]
        benchmarkcard generate ./eee_data -o ./output

        [dim]# Generate specific benchmarks only[/dim]
        benchmarkcard generate ./eee_data -b "MMLU,TruthfulQA,BBH" -o ./output

        [dim]# From HuggingFace dataset clone[/dim]
        benchmarkcard generate ~/datasets/EEE_datastore/data -v
    """
    logger = setup_logging(verbose=verbose)

    if debug:
        enable_debug_logging()

    display_banner()

    console.print(f"\n[bold cyan]Starting EEE-to-BenchmarkCard Pipeline[/bold cyan]")
    console.print(f"[dim]EEE data path: {eee_path}[/dim]")

    start_time = time.time()

    try:
        from auto_benchmarkcard.eee_workflow import run_eee_pipeline

        benchmarks_filter = None
        if benchmarks:
            benchmarks_filter = [b.strip() for b in benchmarks.split(",")]
            console.print(f"[dim]Filter: {', '.join(benchmarks_filter)}[/dim]")

        output_path = str(Path(output_dir).resolve()) if output_dir else None

        summary = run_eee_pipeline(
            eee_path=eee_path,
            output_path=output_path,
            max_files_per_benchmark=max_files,
            benchmarks_filter=benchmarks_filter,
            debug=debug,
        )

        execution_time = time.time() - start_time

        console.print(f"\n[bold green]EEE Pipeline Complete[/bold green]")
        console.print(f"[dim]Time: {format_duration(execution_time)}[/dim]")

        result_table = Table(title="Results", border_style="green")
        result_table.add_column("Status", style="bold")
        result_table.add_column("Count", justify="right")
        result_table.add_column("Benchmarks")

        successful = summary.get("successful", [])
        failed = summary.get("failed", [])
        skipped = summary.get("skipped", [])

        if successful:
            result_table.add_row("Success", str(len(successful)), ", ".join(successful[:5]))
        if failed:
            result_table.add_row("[red]Failed[/red]", str(len(failed)), ", ".join(failed[:5]))
        if skipped:
            skip_names = [s["benchmark"] for s in skipped]
            result_table.add_row("[yellow]Skipped[/yellow]", str(len(skipped)), ", ".join(skip_names[:5]))

        console.print(result_table)

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Pipeline interrupted[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        execution_time = time.time() - start_time
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        logger.error(f"EEE pipeline failed after {format_duration(execution_time)}: {e}", exc_info=verbose)
        raise typer.Exit(1)


@app.command("list")
def list_outputs(
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Output directory to scan (default: ./output)",
            rich_help_panel="Directory Options",
        ),
    ] = None,
    recent: Annotated[
        int,
        typer.Option(
            "--recent",
            "-n",
            help="Show only the N most recent sessions",
            rich_help_panel="Display Options",
            min=1,
            max=100,
        ),
    ] = 10,
    format_type: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: table, json, or tree",
            rich_help_panel="Display Options",
        ),
    ] = "table",
    filter_completed: Annotated[
        bool,
        typer.Option(
            "--completed-only",
            help="Show only completed sessions",
            rich_help_panel="Filter Options",
        ),
    ] = False,
) -> None:
    """
    List recent benchmark processing sessions and their outputs.

    [bold green]Examples:[/bold green]

        [dim]# Show recent sessions[/dim]
        benchmarkcard list

        [dim]# Show only completed sessions in JSON format[/dim]
        benchmarkcard list --completed-only --format json

        [dim]# Show last 20 sessions as tree structure[/dim]
        benchmarkcard list --recent 20 --format tree
    """
    setup_logging()

    valid_formats = {"table", "json", "tree"}
    if format_type not in valid_formats:
        display_error(
            f"Invalid format type: {format_type}",
            f"Valid options: {', '.join(valid_formats)}",
        )
        raise typer.Exit(1)

    output_path = validate_path(output_dir or "output")

    if not output_path.exists():
        display_error(
            f"Output directory not found: {output_path}",
            "Run 'benchmarkcard generate <path>' to create output sessions",
        )
        return

    with workflow_substep("Scanning processing sessions", show_completion=False):
        sessions = []
        session_count = 0
        for item in output_path.iterdir():
            if item.is_dir() and "_" in item.name:
                session_info = get_session_info(item)
                if session_info["benchmark"] != "unknown":
                    session_info["path"] = item
                    sessions.append(session_info)
                    session_count += 1

        console.print(f"  [green]• Found {session_count} processing sessions[/green]")

    if not sessions:
        console.print(
            Panel(
                "[yellow]No benchmark processing sessions found[/yellow]\n\n"
                "[dim]Create sessions by running:[/dim]\n"
                '[cyan]benchmarkcard generate ./eee_data -b "MMLU"[/cyan]',
                title="[bold]No Sessions Found[/bold]",
                border_style="yellow",
            )
        )
        return

    if filter_completed:
        sessions = [s for s in sessions if s["completed"]]
        if not sessions:
            console.print("[yellow]No completed sessions found[/yellow]")
            return

    sessions.sort(key=lambda x: x["modified_time"], reverse=True)
    sessions = sessions[:recent]

    if format_type == "json":
        json_data = [
            {
                "benchmark": s["benchmark"],
                "timestamp": s["timestamp"],
                "completed": s["completed"],
                "tool_count": s["tool_count"],
                "file_count": s["file_count"],
                "size_mb": round(s["total_size_mb"], 2),
                "path": str(s["path"]),
            }
            for s in sessions
        ]
        console.print_json(data=json_data)

    elif format_type == "tree":
        tree = Tree(
            f"[bold cyan]Processing Sessions[/bold cyan] ([dim]{len(sessions)} sessions[/dim])",
            guide_style="dim",
        )

        for session in sessions:
            status_icon = "[OK]" if session["completed"] else "[WARN]"
            node = tree.add(
                f"{status_icon} [cyan]{session['benchmark']}[/cyan] "
                f"[dim]({session['timestamp'].replace('_', ' ').replace('-', ':')}) "
                f"• {session['file_count']} files • {session['total_size_mb']:.1f}MB[/dim]"
            )

            node.add(f"Path: [blue]{session['path']}[/blue]")
            node.add(f"Tools: [green]{session['tool_count']}[/green]")
            node.add(
                f"Status: {'[green]Complete[/green]' if session['completed'] else '[yellow]Incomplete[/yellow]'}"
            )

        console.print(tree)

    else:
        table = Table(
            title=f"Recent Benchmark Processing Sessions ({len(sessions)} sessions)",
            border_style="cyan",
            title_style="bold cyan",
        )
        table.add_column("Status", style="green", width=8, justify="center")
        table.add_column("Benchmark", style="cyan", no_wrap=True)
        table.add_column("Timestamp", style="blue", width=16)
        table.add_column("Tools", style="yellow", width=6, justify="right")
        table.add_column("Files", style="magenta", width=6, justify="right")
        table.add_column("Size", style="green", width=8, justify="right")
        table.add_column("Path", style="dim", overflow="ellipsis")

        for session in sessions:
            status = "[OK] Done" if session["completed"] else "[WARN] Partial"
            timestamp_fmt = session["timestamp"].replace("_", " ").replace("-", ":")
            size_fmt = (
                f"{session['total_size_mb']:.1f}MB" if session["total_size_mb"] > 0 else "0MB"
            )

            table.add_row(
                status,
                session["benchmark"],
                timestamp_fmt,
                str(session["tool_count"]),
                str(session["file_count"]),
                size_fmt,
                str(session["path"].relative_to(Path.cwd())),
            )

        console.print(table)

    completed_count = sum(1 for s in sessions if s["completed"])
    total_size = sum(s["total_size_mb"] for s in sessions)

    console.print(
        f"\n[dim]Summary: {completed_count}/{len(sessions)} completed • "
        f"Total size: {total_size:.1f}MB[/dim]"
    )


@app.command("show")
def show_session(
    session_path: Annotated[
        str,
        typer.Argument(help="Path to the benchmark session directory", metavar="SESSION_PATH"),
    ],
    detailed: Annotated[
        bool,
        typer.Option(
            "--detailed",
            "-d",
            help="Show detailed file information and content previews",
            rich_help_panel="Display Options",
        ),
    ] = False,
) -> None:
    """
    Show details about a benchmark processing session.

    [bold green]Examples:[/bold green]

        [dim]# Show basic session info[/dim]
        benchmarkcard show output/glue_2025-01-08_14-30

        [dim]# Show detailed file information[/dim]
        benchmarkcard show output/glue_2025-01-08_14-30 --detailed
    """
    setup_logging()

    session_dir = validate_path(session_path, must_exist=True)

    if not session_dir.is_dir():
        display_error(
            f"Path is not a directory: {session_dir}",
            "Please provide a valid session directory path",
        )
        raise typer.Exit(1)

    session_info = get_session_info(session_dir)

    console.print(Rule(f"[bold cyan]Session Details: {session_dir.name}[/bold cyan]", style="cyan"))
    console.print()

    overview_table = Table(border_style="blue", title="Session Overview")
    overview_table.add_column("Property", style="cyan", width=20)
    overview_table.add_column("Value", style="green")

    status_text = (
        "[bold green][OK] Completed[/bold green]"
        if session_info["completed"]
        else "[bold yellow][WARN] Incomplete[/bold yellow]"
    )
    timestamp_fmt = session_info["timestamp"].replace("_", " ").replace("-", ":")

    overview_table.add_row("Benchmark", f"[bold]{session_info['benchmark']}[/bold]")
    overview_table.add_row("Status", status_text)
    overview_table.add_row("Timestamp", timestamp_fmt)
    overview_table.add_row("Tools Used", str(session_info["tool_count"]))
    overview_table.add_row("Total Files", str(session_info["file_count"]))
    overview_table.add_row("Total Size", f"{session_info['total_size_mb']:.2f} MB")
    overview_table.add_row("Full Path", str(session_dir.absolute()))

    console.print(overview_table)
    console.print()

    tool_output_dir = session_dir / "tool_output"
    benchmark_card_dir = session_dir / "benchmarkcard"

    if tool_output_dir.exists():
        console.print("[bold cyan]Tool Outputs[/bold cyan]")

        tools_table = Table(border_style="cyan")
        tools_table.add_column("Tool", style="cyan", width=20)
        tools_table.add_column("Files", style="yellow", width=8, justify="right")
        tools_table.add_column("Size", style="green", width=10, justify="right")
        tools_table.add_column("Description", style="dim")

        tool_descriptions = {
            "unitxt": "UnitXT benchmark metadata",
            "extractor": "Extracted IDs and URLs",
            "hf": "HuggingFace dataset info",
            "docling": "Processed academic papers",
            "risk_enhanced": "Risk-enhanced benchmark cards",
            "ai_atlas_nexus": "AI risk assessment results",
            "rag": "Evidence retrieval results",
            "factreasoner": "Factuality verification scores",
        }

        for tool_dir in sorted(tool_output_dir.iterdir()):
            if tool_dir.is_dir():
                files = list(tool_dir.glob("*"))
                file_count = len(files)
                total_size = sum(f.stat().st_size for f in files if f.is_file()) / 1024  # KB
                description = tool_descriptions.get(tool_dir.name, "Tool output")

                size_str = f"{total_size:.1f} KB" if total_size > 0 else "0 KB"
                tools_table.add_row(tool_dir.name, str(file_count), size_str, description)

                if detailed and files:
                    console.print(f"\n[dim]Files in {tool_dir.name}:[/dim]")
                    for file in sorted(files):
                        if file.is_file():
                            size_kb = file.stat().st_size / 1024
                            mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime(
                                "%Y-%m-%d %H:%M"
                            )
                            console.print(
                                f"  [blue]{file.name}[/blue] ({size_kb:.1f} KB, {mtime})"
                            )

        console.print(tools_table)
        console.print()

    if benchmark_card_dir.exists():
        console.print("[bold green]Benchmark Cards[/bold green]")

        cards_table = Table(border_style="green")
        cards_table.add_column("File", style="green", width=40)
        cards_table.add_column("Size", style="yellow", width=10, justify="right")
        cards_table.add_column("Modified", style="blue", width=16)

        for card_file in sorted(benchmark_card_dir.glob("*.json")):
            size_kb = card_file.stat().st_size / 1024
            mtime = datetime.fromtimestamp(card_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            cards_table.add_row(card_file.name, f"{size_kb:.1f} KB", mtime)

            if detailed:
                try:
                    with open(card_file) as f:
                        data = json.load(f)

                    if "benchmark_card" in data:
                        card = data["benchmark_card"]
                        details = card.get("benchmark_details", {})
                        console.print(f"\n[dim]Preview of {card_file.name}:[/dim]")
                        console.print(f"  Name: [cyan]{details.get('name', 'N/A')}[/cyan]")
                        console.print(f"  Domains: {', '.join(details.get('domains', []))}")
                        console.print(f"  Languages: {', '.join(details.get('languages', []))}")

                        overview = details.get("overview", "")
                        if overview:
                            preview = overview[:150] + "..." if len(overview) > 150 else overview
                            console.print(f"  Overview: [dim]{preview}[/dim]")
                except Exception as e:
                    console.print(f"  [red]Error reading {card_file.name}: {e}[/red]")

        console.print(cards_table)
        console.print()

    console.print(
        Panel(
            "[bold]Helpful Commands:[/bold]\n\n"
            f"[cyan]cd {session_dir}[/cyan] - Navigate to session directory\n"
            f"[cyan]benchmarkcard show {session_path} --detailed[/cyan] - Show detailed info",
            border_style="dim",
            title="[dim]Quick Actions[/dim]",
        )
    )


@app.command("validate")
def validate_setup(
    fix_issues: Annotated[
        bool,
        typer.Option(
            "--fix",
            help="Automatically fix issues where possible",
            rich_help_panel="Repair Options",
        ),
    ] = False,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Run live checks: HF auth, LLM inference, Merlin execution",
            rich_help_panel="Repair Options",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed validation information",
            rich_help_panel="Display Options",
        ),
    ] = False,
) -> None:
    """
    Validate system setup: env vars, dependencies, external tools, and directories.

    [bold green]Examples:[/bold green]

        [dim]# Basic validation[/dim]
        benchmarkcard validate

        [dim]# Full live check (HF token, LLM, Merlin)[/dim]
        benchmarkcard validate --live
    """
    setup_logging(verbose=verbose)

    display_banner()

    console.print(Rule("[bold yellow]System Validation[/bold yellow]", style="yellow"))
    console.print()

    issues = []
    warnings = []
    fixed_issues = []

    with create_progress_display() as progress:
        env_task = progress.add_task("[cyan]Checking environment variables...", total=4)

        try:
            Config.validate_config()
            progress.console.print("[green][OK] Environment configuration valid[/green]")
        except ValueError as e:
            progress.console.print(f"[red][FAIL] Configuration error: {e}[/red]")
            issues.append(
                (
                    "Environment Configuration",
                    str(e),
                    "Set required environment variables",
                )
            )

        progress.update(env_task, advance=1)

        engine = Config.LLM_ENGINE_TYPE.lower()
        engine_required = Config._ENGINE_REQUIRED_VARS.get(engine, [])
        env_vars = [(var, f"Required for {engine.upper()} engine") for var in engine_required]

        for var, desc in env_vars:
            value = Config.get_env_var(var)
            if value:
                if verbose:
                    preview = f"{value[:10]}..." if len(value) > 10 else value
                    progress.console.print(f"[green][OK] {var}:[/green] [dim]{preview}[/dim]")
            else:
                progress.console.print(f"[red][FAIL] {var} not set[/red]")
                issues.append(
                    (
                        f"Environment Variable: {var}",
                        "Not set",
                        f"Set {var} in .env file",
                    )
                )
            progress.update(env_task, advance=1)

        deps_task = progress.add_task("[cyan]Validating Python dependencies...", total=6)

        critical_imports = [
            ("auto_benchmarkcard.workflow", "Main workflow orchestrator"),
            ("auto_benchmarkcard.config", "Configuration management"),
            ("auto_benchmarkcard.tools.unitxt.unitxt_tool", "UnitXT benchmark lookup"),
            ("auto_benchmarkcard.tools.factreasoner.factreasoner_tool", "FactReasoner validation"),
            ("ai_atlas_nexus.library", "AI Atlas Nexus integration"),
            ("typer", "CLI framework"),
        ]

        for module, description in critical_imports:
            try:
                __import__(module)
                if verbose:
                    progress.console.print(f"[green][OK] {description}[/green]")
            except ImportError as e:
                progress.console.print(f"[red][FAIL] {description}: {e}[/red]")
                issues.append((f"Python Import: {module}", str(e), "Install missing dependencies"))
            progress.update(deps_task, advance=1)

        tools_task = progress.add_task("[cyan]Checking external tools...", total=1)

        merlin_path = Config.MERLIN_BIN
        if merlin_path.exists():
            if os.access(merlin_path, os.X_OK):
                progress.console.print(f"[green][OK] Merlin binary: {merlin_path}[/green]")
            else:
                progress.console.print(f"[red][FAIL] Merlin binary not executable: {merlin_path}[/red]")
                issues.append(("Merlin Binary", "Not executable", "Check file permissions"))
        else:
            progress.console.print(f"[red][FAIL] Merlin binary not found: {merlin_path}[/red]")
            issues.append(
                (
                    "Merlin Binary",
                    "File not found",
                    "Build Merlin following README instructions",
                )
            )

        progress.update(tools_task, advance=1)

        dirs_task = progress.add_task("[cyan]Validating directories...", total=3)

        directories_to_check = [
            (Config.FACTREASONER_CACHE_DIR, "FactReasoner cache", True),
            ("output", "Output directory", True),
            (".", "Current directory", False),
        ]

        for dir_path, desc, can_create in directories_to_check:
            path_obj = Path(dir_path)

            if path_obj.exists():
                if verbose:
                    progress.console.print(f"[green][OK] {desc}: {path_obj.absolute()}[/green]")
            elif can_create:
                if fix_issues:
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                        progress.console.print(
                            f"[yellow]Created {desc}: {path_obj.absolute()}[/yellow]"
                        )
                        fixed_issues.append(f"Created directory: {path_obj}")
                    except Exception as e:
                        progress.console.print(f"[red][FAIL] Cannot create {desc}: {e}[/red]")
                        issues.append((f"Directory: {desc}", str(e), "Create directory manually"))
                else:
                    progress.console.print(
                        f"[yellow][WARN] {desc} missing: {path_obj.absolute()}[/yellow]"
                    )
                    warnings.append(
                        (
                            f"Directory: {desc}",
                            "Does not exist",
                            "Will be created automatically",
                        )
                    )
            else:
                progress.console.print(f"[red][FAIL] {desc} not found: {path_obj.absolute()}[/red]")
                issues.append(
                    (
                        f"Directory: {desc}",
                        "Required directory missing",
                        "Check installation",
                    )
                )

            progress.update(dirs_task, advance=1)

        if live:
            live_task = progress.add_task("[cyan]Running live infrastructure checks...", total=3)

            # 1. HF authentication
            try:
                from huggingface_hub import HfApi
                hf_api = HfApi()
                user_info = hf_api.whoami()
                username = user_info.get("name", "unknown")
                progress.console.print(f"[green]  HF auth OK (user: {username})[/green]")
            except Exception as e:
                progress.console.print(f"[red]  HF auth failed: {e}[/red]")
                issues.append(("HF Authentication", str(e)[:80], "Check HF_TOKEN in .env"))
            progress.update(live_task, advance=1)

            # 2. LLM inference (short prompt to composer model)
            try:
                from auto_benchmarkcard.config import get_llm_handler
                llm = get_llm_handler(Config.COMPOSER_MODEL)
                response = llm.generate("Reply with exactly: OK")
                if response and len(response.strip()) > 0:
                    progress.console.print(
                        f"[green]  LLM inference OK ({Config.COMPOSER_MODEL})[/green]"
                    )
                else:
                    progress.console.print("[red]  LLM returned empty response[/red]")
                    issues.append(("LLM Inference", "Empty response", "Check model availability and API key"))
            except Exception as e:
                progress.console.print(f"[red]  LLM inference failed: {e}[/red]")
                issues.append(("LLM Inference", str(e)[:80], "Check LLM engine config and API key"))
            progress.update(live_task, advance=1)

            # 3. Merlin binary execution
            if merlin_path.exists() and os.access(merlin_path, os.X_OK):
                try:
                    import subprocess
                    result = subprocess.run(
                        [str(merlin_path), "--help"],
                        capture_output=True, timeout=10,
                    )
                    if result.returncode == 0:
                        progress.console.print(f"[green]  Merlin execution OK[/green]")
                    else:
                        stderr = result.stderr.decode()[:80] if result.stderr else "unknown error"
                        progress.console.print(f"[red]  Merlin returned exit code {result.returncode}: {stderr}[/red]")
                        issues.append(("Merlin Execution", f"Exit code {result.returncode}", "Check Merlin build"))
                except subprocess.TimeoutExpired:
                    progress.console.print("[red]  Merlin timed out[/red]")
                    issues.append(("Merlin Execution", "Timed out after 10s", "Check Merlin binary"))
                except Exception as e:
                    progress.console.print(f"[red]  Merlin execution failed: {e}[/red]")
                    issues.append(("Merlin Execution", str(e)[:80], "Check Merlin build"))
            else:
                progress.console.print("[yellow]  Merlin skipped (binary not found or not executable)[/yellow]")
            progress.update(live_task, advance=1)

    console.print()

    if issues or warnings or fixed_issues:
        if issues:
            issues_table = Table(title="[FAIL] Issues Found", border_style="red")
            issues_table.add_column("Component", style="red", width=25)
            issues_table.add_column("Problem", style="yellow", width=30)
            issues_table.add_column("Solution", style="green")

            for component, problem, solution in issues:
                issues_table.add_row(component, problem, solution)

            console.print(issues_table)
            console.print()

        if warnings:
            warnings_table = Table(title="[WARN] Warnings", border_style="yellow")
            warnings_table.add_column("Component", style="yellow", width=25)
            warnings_table.add_column("Issue", style="white", width=30)
            warnings_table.add_column("Recommendation", style="dim")

            for component, issue, recommendation in warnings:
                warnings_table.add_row(component, issue, recommendation)

            console.print(warnings_table)
            console.print()

        if fixed_issues:
            console.print("[bold green]Issues Fixed:[/bold green]")
            for fix in fixed_issues:
                console.print(f"  [OK] {fix}")
            console.print()

    if issues:
        display_error(
            f"Validation failed with {len(issues)} critical issue(s)",
            "Please resolve the issues above before proceeding.",
        )
        raise typer.Exit(1)

    elif warnings:
        display_success(
            f"Validation completed with {len(warnings)} warning(s)",
            "System is functional but some optimizations are recommended.",
        )

    else:
        display_success(
            "All validation checks passed!",
            "System is fully configured and ready for use.",
        )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
) -> None:
    """Benchmark metadata extraction and validation CLI."""
    if ctx.invoked_subcommand is None:
        display_banner()
        console.print(ctx.get_help())

        console.print("\n[bold green]Quick Examples:[/bold green]\n")
        examples = [
            ("Validate system setup", "benchmarkcard validate"),
            ("Generate benchmark cards", "benchmarkcard generate ./eee_data -b MMLU"),
            ("List recent sessions", "benchmarkcard list --recent 5"),
        ]

        for desc, cmd in examples:
            console.print(f"  [dim]{desc}:[/dim] [cyan]{cmd}[/cyan]")

        console.print(
            f"\n[dim]For detailed help: [/dim][cyan]benchmarkcard <command> --help[/cyan]"
        )


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow][WARN] Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        error_console.print(f"\n[bold red]Unexpected error: {e}[/bold red]")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback

            error_console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)

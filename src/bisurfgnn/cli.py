from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from .config import load_config
from .training import run_experiment


DEFAULT_CELL_CONFIG = Path(__file__).resolve().parents[2] / "configs/cell_context.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bisurfgnn",
        description="Train and audit BiSurfGNN dose-response surface models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Run one configured experiment")
    train_parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    verify_nci = subparsers.add_parser(
        "verify-nci", help="Verify frozen NCI-ALMANAC source files and ZIP metadata"
    )
    verify_nci.add_argument("--config", required=True)
    audit_nci = subparsers.add_parser(
        "audit-nci", help="Run independent consistency checks on prepared NCI outputs"
    )
    audit_nci.add_argument("--config", required=True)
    prepare_nci = subparsers.add_parser(
        "prepare-nci", help="Normalize NCI-ALMANAC points and build molecular graph caches"
    )
    prepare_nci.add_argument("--config", required=True)
    prepare_nci.add_argument(
        "--download", action="store_true", help="Download missing official source files"
    )
    prepare_nci.add_argument(
        "--overwrite", action="store_true", help="Replace generated processed outputs only"
    )
    prepare_nci.add_argument(
        "--max-chunks", type=int, default=None, help="Partial engineering run; not for results"
    )
    smoke_nci = subparsers.add_parser(
        "smoke-nci", help="Run a lineage-CellOut forward/backward audit on real NCI records"
    )
    smoke_nci.add_argument("--config", required=True)
    smoke_nci.add_argument(
        "--cell-config", default=str(DEFAULT_CELL_CONFIG), help="Cell context YAML"
    )
    prepare_benchmark = subparsers.add_parser(
        "prepare-benchmark",
        help="Freeze the five-fold One-new-scaffold benchmark manifest",
    )
    prepare_benchmark.add_argument("--config", required=True)
    for command, help_text in (
        (
            "prepare-benchmark-context",
            "Fit and cache CellMiner context for every frozen chemical fold",
        ),
        (
            "audit-benchmark-context",
            "Verify the frozen benchmark-specific CellMiner context caches",
        ),
    ):
        context_parser = subparsers.add_parser(command, help=help_text)
        context_parser.add_argument("--config", required=True)
        context_parser.add_argument(
            "--cell-config",
            default=str(DEFAULT_CELL_CONFIG),
            help="Cell context YAML",
        )
    benchmark_nci = subparsers.add_parser(
        "benchmark-nci",
        help="Evaluate a frozen NCI-ALMANAC benchmark baseline",
    )
    benchmark_nci.add_argument("--config", required=True)
    benchmark_nci.add_argument(
        "--model",
        choices=("observed_bliss", "observed_hsa"),
        default="observed_bliss",
        help="Frozen benchmark implementation to evaluate",
    )
    prepare_learned = subparsers.add_parser(
        "prepare-learned-benchmark",
        help="Physically partition frozen train/validation/test response shards",
    )
    prepare_learned.add_argument("--config", required=True)
    prepare_learned.add_argument(
        "--overwrite", action="store_true", help="Replace learned role shards"
    )
    for command, help_text in (
        (
            "select-learned-benchmark",
            "Train and freeze validation-selected checkpoints without opening test shards",
        ),
        (
            "evaluate-learned-benchmark",
            "Evaluate only after all selected checkpoints pass the global barrier",
        ),
    ):
        learned_parser = subparsers.add_parser(command, help=help_text)
        learned_parser.add_argument("--config", required=True)
        learned_parser.add_argument(
            "--cell-config", default=str(DEFAULT_CELL_CONFIG), help="Cell context YAML"
        )
        learned_parser.add_argument(
            "--fold",
            type=int,
            action="append",
            default=None,
            help="Outer fold to select; repeat for multiple folds (selection only)",
        )
        learned_parser.add_argument(
            "--max-train-points",
            type=int,
            default=None,
            help="Complete-surface training subset for an engineering smoke",
        )
        learned_parser.add_argument(
            "--output-dir", default=None, help="Explicit learned-run directory"
        )
        learned_parser.add_argument(
            "--device", default="auto", help="auto, cpu, cuda, or a CUDA device"
        )
        learned_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace an existing selection run (selection only)",
        )
    independent_graph = subparsers.add_parser(
        "select-independent-gnn-benchmark",
        help=(
            "Train and freeze validation-only IndependentPooledGNN checkpoints; "
            "no outer-test entrypoint is provided"
        ),
    )
    independent_graph.add_argument("--config", required=True)
    independent_graph.add_argument(
        "--cell-config", default=str(DEFAULT_CELL_CONFIG), help="Cell context YAML"
    )
    independent_graph.add_argument(
        "--fold",
        type=int,
        action="append",
        default=None,
        help="Outer fold to select; repeat for multiple folds",
    )
    independent_graph.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Configured seed to run; repeat for multiple seeds",
    )
    independent_graph.add_argument(
        "--max-train-points",
        type=int,
        default=None,
        help="Complete-surface training subset for an engineering smoke",
    )
    independent_graph.add_argument(
        "--output-dir", default=None, help="Explicit graph-selection directory"
    )
    independent_graph.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a CUDA device"
    )
    independent_graph.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known validation-selection artifacts only",
    )
    noise_floor = subparsers.add_parser(
        "replicate-noise-floor",
        help=(
            "Estimate the surface-macro RMSE measurement-noise floor from technical "
            "replicates; reads train/validation shards only"
        ),
    )
    noise_floor.add_argument(
        "--shard-dir",
        required=True,
        help="Directory holding fold_*/train.parquet and fold_*/validation.parquet",
    )
    noise_floor.add_argument("--output", required=True, help="Manifest path to write")
    noise_floor.add_argument(
        "--fold",
        type=int,
        action="append",
        default=None,
        help="Fold to estimate; repeat for multiple folds",
    )
    noise_floor.add_argument("--seed", type=int, default=41, help="Bootstrap seed")
    noise_floor.add_argument(
        "--resamples", type=int, default=2000, help="Bootstrap resample count"
    )
    noise_floor.add_argument(
        "--anchor-reference",
        type=float,
        default=None,
        help=(
            "Optional cohort-wide anchor for orientation only; it is not mixed "
            "with replicate-subset floors to report headroom"
        ),
    )
    noise_floor.add_argument(
        "--relative-gate",
        type=float,
        default=0.05,
        help="Relative-improvement target for matched-subset comparisons",
    )
    warm_pairout = subparsers.add_parser(
        "prepare-warm-pairout-probe",
        help=(
            "Freeze a train/validation-only exact-pair holdout discriminator; "
            "never opens test response shards"
        ),
    )
    warm_pairout.add_argument("--config", required=True)
    warm_pairout.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known warm-probe artifacts",
    )
    warm_morgan = subparsers.add_parser(
        "select-warm-morgan-probe",
        help=(
            "Train validation-only Morgan controls on frozen warm PairOut shards; "
            "there is no test or Phase B entrypoint"
        ),
    )
    warm_morgan.add_argument("--warm-config", required=True)
    warm_morgan.add_argument("--benchmark-config", required=True)
    warm_morgan.add_argument(
        "--cell-config", default=str(DEFAULT_CELL_CONFIG), help="Cell context YAML"
    )
    warm_morgan.add_argument("--fold", type=int, action="append", default=None)
    warm_morgan.add_argument("--seed", type=int, action="append", default=None)
    warm_morgan.add_argument("--max-train-points", type=int, default=None)
    warm_morgan.add_argument(
        "--output-dir",
        default="runs/warm_pairout/morgan_hallmark_residual_mlp",
    )
    warm_morgan.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a CUDA device"
    )
    warm_morgan.add_argument("--overwrite", action="store_true")
    for command, help_text in (
        ("verify-cell-context", "Verify frozen CellMiner and MSigDB source files"),
        ("prepare-cell-context", "Prepare expression, crosswalk, and Hallmark topology"),
        ("audit-cell-context", "Audit prepared CellMiner and Hallmark assets"),
    ):
        cell_parser = subparsers.add_parser(command, help=help_text)
        cell_parser.add_argument("--config", required=True)
        if command == "prepare-cell-context":
            cell_parser.add_argument(
                "--download", action="store_true", help="Download missing frozen inputs"
            )
    return parser


def _load_cell_context_extras(path: str | Path) -> tuple[Path, dict[str, object]]:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Cell context configuration must be a mapping")
    configured_root = raw.get("project_root")
    if configured_root is None:
        project_root = config_path.parent.parent
    else:
        candidate = Path(str(configured_root))
        project_root = (
            candidate.resolve()
            if candidate.is_absolute()
            else (config_path.parent / candidate).resolve()
        )
    cells_value = Path(str(raw.get("almanac_cells", "data/processed/nci_almanac/cells.csv")))
    almanac_cells = (
        cells_value.resolve()
        if cells_value.is_absolute()
        else (project_root / cells_value).resolve()
    )
    smoke = raw.get("cellout_smoke", {})
    if not isinstance(smoke, dict):
        raise ValueError("cellout_smoke must be a mapping")
    return almanac_cells, dict(smoke)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        config = load_config(args.config)
        result = run_experiment(config)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command in {"verify-nci", "prepare-nci", "audit-nci", "smoke-nci"}:
        from .data.nci_almanac import (
            audit_prepared_nci,
            download_source_files,
            load_nci_almanac_config,
            prepare_nci_almanac,
            verify_source_files,
        )

        config = load_nci_almanac_config(args.config)
        if args.command == "verify-nci":
            result = verify_source_files(config)
        elif args.command == "audit-nci":
            result = audit_prepared_nci(config)
        elif args.command == "prepare-nci":
            if args.download:
                download_source_files(config)
            result = prepare_nci_almanac(
                config,
                overwrite=args.overwrite,
                max_chunks=args.max_chunks,
            )
        else:
            from .data.cellminer import load_cellminer_config
            from .data.real_dataset import run_real_data_smoke

            cell_config = load_cellminer_config(args.cell_config)
            _, smoke = _load_cell_context_extras(args.cell_config)
            result = run_real_data_smoke(config, cell_config, smoke)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command in {
        "prepare-benchmark",
        "prepare-benchmark-context",
        "audit-benchmark-context",
        "benchmark-nci",
    }:
        from .benchmark_config import load_benchmark_config

        config = load_benchmark_config(args.config)
        if args.command == "prepare-benchmark":
            from .data.chemical_splits import prepare_one_new_scaffold_benchmark

            result = prepare_one_new_scaffold_benchmark(config)
        elif args.command in {
            "prepare-benchmark-context",
            "audit-benchmark-context",
        }:
            from .data.benchmark_context import (
                load_benchmark_cell_context,
                prepare_benchmark_cell_context,
            )
            from .data.cellminer import load_cellminer_config

            cell_config = load_cellminer_config(args.cell_config)
            if args.command == "prepare-benchmark-context":
                prepared = prepare_benchmark_cell_context(config, cell_config)
            else:
                prepared = load_benchmark_cell_context(config, cell_config)
            result = prepared.manifest
        else:
            from .benchmark import run_reference_benchmark

            result = run_reference_benchmark(config, reference_model=args.model)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command in {
        "prepare-learned-benchmark",
        "select-learned-benchmark",
        "evaluate-learned-benchmark",
    }:
        from .learned_benchmark import (
            evaluate_learned_morgan_benchmark,
            prepare_learned_point_shards,
            select_learned_morgan_benchmark,
        )

        if args.command == "prepare-learned-benchmark":
            result = prepare_learned_point_shards(
                args.config,
                overwrite=args.overwrite,
            )
        elif args.command == "select-learned-benchmark":
            result = select_learned_morgan_benchmark(
                args.config,
                cell_config=args.cell_config,
                folds=args.fold,
                max_train_points=args.max_train_points,
                output_dir=args.output_dir,
                device=args.device,
                overwrite=args.overwrite,
            )
        else:
            if args.fold is not None or args.max_train_points is not None or args.overwrite:
                raise ValueError(
                    "Evaluation uses the frozen selection manifest; fold/subset/overwrite "
                    "options are selection-only"
                )
            result = evaluate_learned_morgan_benchmark(
                args.config,
                cell_config=args.cell_config,
                output_dir=args.output_dir,
                device=args.device,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "select-independent-gnn-benchmark":
        from .independent_graph_benchmark import select_independent_graph_benchmark

        result = select_independent_graph_benchmark(
            args.config,
            cell_config=args.cell_config,
            folds=args.fold,
            seeds=args.seed,
            max_train_points=args.max_train_points,
            output_dir=args.output_dir,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "replicate-noise-floor":
        from .replicate_noise import estimate_all_folds

        result = estimate_all_folds(
            args.shard_dir,
            args.output,
            folds=tuple(args.fold) if args.fold else (0, 1, 2, 3, 4),
            seed=args.seed,
            resamples=args.resamples,
            anchor_reference=args.anchor_reference,
            relative_gate=args.relative_gate,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "prepare-warm-pairout-probe":
        from .warm_pairout import prepare_warm_pairout_probe

        result = prepare_warm_pairout_probe(
            args.config,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "select-warm-morgan-probe":
        from .warm_morgan import select_warm_morgan_probe

        result = select_warm_morgan_probe(
            args.warm_config,
            args.benchmark_config,
            cell_config=args.cell_config,
            folds=args.fold,
            seeds=args.seed,
            max_train_points=args.max_train_points,
            output_dir=args.output_dir,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command in {
        "verify-cell-context",
        "prepare-cell-context",
        "audit-cell-context",
    }:
        from .data.cellminer import (
            audit_prepared_cellminer,
            download_cellminer_sources,
            load_cellminer_config,
            prepare_cellminer_assets,
            verify_cellminer_source,
            verify_hallmark_source,
        )

        config = load_cellminer_config(args.config)
        almanac_cells, _ = _load_cell_context_extras(args.config)
        if args.command == "verify-cell-context":
            result = {
                "status": "passed",
                "cellminer": verify_cellminer_source(config),
                "hallmark": verify_hallmark_source(config),
            }
        elif args.command == "prepare-cell-context":
            if args.download:
                download_cellminer_sources(config)
            prepared = prepare_cellminer_assets(config, almanac_cells)
            result = {"status": "completed", **prepared["audit"]}
        else:
            result = audit_prepared_cellminer(config, almanac_cells)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

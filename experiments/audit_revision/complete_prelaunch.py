"""Run startup acceptance, then prepare (never launch) the full matrix."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import run_formal as formal


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True)
    root=parser.parse_args().output_dir.resolve()
    allowed=(formal.ROOT/"runs/audit_revision").resolve()
    if not root.is_relative_to(allowed) or root==allowed:
        raise ValueError("Fresh audit output directory required")
    root.mkdir(parents=True,exist_ok=False)
    sources=[formal.learned._file_record(p) for p in formal.source_paths()]
    test_sources=[formal.learned._file_record(p) for p in sorted((formal.ROOT/"tests").glob("*.py"))]
    print("Running full regression suite before any acceptance training",flush=True)
    tests=subprocess.run([sys.executable,"-m","pytest"],cwd=formal.ROOT,capture_output=True,text=True,
                          encoding="utf-8",errors="replace",timeout=900)
    log=root/"pytest.log"; log.write_text(tests.stdout+tests.stderr,encoding="utf-8")
    if tests.returncode != 0:
        raise RuntimeError(f"Regression failed: {log}")
    print(tests.stdout.splitlines()[-1],flush=True)
    plan=formal.prepare(root/"acceptance",preflight=True)
    plan_path=root/"acceptance/plan.json"
    for i,job in enumerate(plan["inventory"]["jobs"]):
        print(f"Acceptance {i+1}/{len(plan['inventory']['jobs'])}: {job['id']}",flush=True)
        formal.worker(plan_path,job["id"])
    print("Validating complete point inventories and aggregate statistical path",flush=True)
    formal.aggregate(plan_path)
    formal.verify(sources+test_sources)
    final=formal.prepare(root/"prepared_formal")
    if final["sources"] != sources:
        raise RuntimeError("Prepared source version differs from accepted version")
    ready=formal.seal(root/"readiness.json",dict(status="passed",sources=sources,
        evidence=[formal.learned._file_record(log),formal.learned._file_record(plan_path),
                  formal.learned._file_record(root/"acceptance/aggregate.json"),
                  formal.learned._file_record(root/"prepared_formal/plan.json"),*test_sources],
        actual_acceptance_jobs=len(plan["inventory"]["jobs"]), acceptance_aliases=len(plan["inventory"]["aliases"]),
        tests_exit_code=tests.returncode, full_experiment_started=False))
    print(f"READY: {len(final['inventory']['jobs'])} full jobs prepared, NONE launched. Report: {root/'readiness.json'}",flush=True)


if __name__=="__main__": main()

"""Execute every skill in the catalog and write a reproducible run report.

    python src/run_all.py            # run all 46 skills
    python src/run_all.py --pack agent-ml-skills
    python src/run_all.py --skill model-evaluation

Writes `reports/skill_run_report.md` (human-readable, every headline and metric)
and `reports/skill_run_summary.json` (machine-readable status + timings).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from skills.base import SkillResult  # noqa: E402
from skills.registry import SKILLS, BY_ID  # noqa: E402
from src.data_prep import get_data  # noqa: E402
from src.model import get_model  # noqa: E402

REPORTS = ROOT / "reports"


def build_context() -> dict:
    bundle = get_data()
    ctx = dict(bundle)
    ctx["model"] = get_model(bundle["df"])
    return ctx


def run_skills(skill_list, ctx: dict, verbose: bool = True) -> list[dict]:
    records: list[dict] = []
    # analysis-retrospective reports on the batch, so it runs last with timings.
    ordered = sorted(skill_list, key=lambda s: s.id == "analysis-retrospective")
    for skill in ordered:
        t0 = time.perf_counter()
        try:
            result = skill.run(ctx)
            status, error = "ok", ""
        except Exception as exc:  # a failing skill must not abort the batch
            result, status, error = None, "failed", f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        records.append({"id": skill.id, "name": skill.name, "pack": skill.pack,
                        "category": skill.category, "crisp_dm": skill.crisp_dm,
                        "status": status, "seconds": round(elapsed, 3),
                        "headline": result.headline if result else "",
                        "blocks": len(result.blocks) if result else 0,
                        "error": error, "result": result})
        if verbose:
            mark = "ok  " if status == "ok" else "FAIL"
            print(f"[{mark}] {skill.id:38s} {elapsed:6.2f}s  "
                  f"{(result.headline if result else error)[:70]}")
        done = [r for r in records if r["status"] == "ok"]
        ctx["timings"] = {
            "count": len(done),
            "total": sum(r["seconds"] for r in records),
            "slowest": [(r["id"], r["seconds"])
                        for r in sorted(records, key=lambda r: -r["seconds"])[:5]],
        }
    return records


def _block_to_markdown(block) -> str:
    if block.kind == "metrics":
        cells = " | ".join(f"**{m.label}**: {m.value}" for m in block.payload)
        return f"{cells}\n"
    if block.kind == "table":
        df: pd.DataFrame = block.payload
        head = df.head(8)
        return (f"*{block.title}*\n\n{head.to_markdown(index=False)}\n"
                + (f"\n_({len(df)} rows total)_\n" if len(df) > 8 else ""))
    if block.kind == "chart":
        return (f"*Chart — {block.title}* "
                f"(`{block.meta['kind']}`, x=`{block.meta['x']}`, y=`{block.meta['y']}`)\n")
    if block.kind in ("narrative", "artifact"):
        return f"{block.payload}\n"
    if block.kind == "code":
        return f"```{block.meta.get('language', 'text')}\n{block.payload}\n```\n"
    return ""


def write_report(records: list[dict], ctx: dict) -> tuple[Path, Path]:
    REPORTS.mkdir(exist_ok=True)
    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]

    lines = [
        "# Skill Run Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Dataset: {ctx['source']} — {ctx['rows_raw']:,} raw rows, "
        f"{ctx['rows_clean']:,} after cleaning",
        f"Skills executed: **{len(ok)}/{len(records)}** in "
        f"{sum(r['seconds'] for r in records):.1f}s",
        "",
        "## Summary",
        "",
        "| # | Skill | Pack | CRISP-DM phase | Status | Seconds | Headline |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(records, 1):
        lines.append(f"| {i} | `{r['id']}` | {r['pack']} | {r['crisp_dm']} | "
                     f"{r['status']} | {r['seconds']:.2f} | "
                     f"{r['headline'].replace('|', '/')} |")
    if failed:
        lines += ["", "## Failures", ""]
        lines += [f"- `{r['id']}`: {r['error']}" for r in failed]

    lines += ["", "## Skill output detail", ""]
    for r in records:
        lines += [f"### {r['name']}  (`{r['id']}`)", "",
                  f"*{r['pack']} / {r['category']} — {r['crisp_dm']}*", ""]
        if r["status"] != "ok":
            lines += [f"**FAILED:** {r['error']}", ""]
            continue
        lines += [f"**{r['headline']}**", ""]
        for block in r["result"].blocks:
            md = _block_to_markdown(block)
            if md.strip():
                lines += [md, ""]

    md_path = REPORTS / "skill_run_report.md"
    md_path.write_text("\n".join(lines))

    json_path = REPORTS / "skill_run_summary.json"
    json_path.write_text(json.dumps(
        {"generated": datetime.now().isoformat(timespec="seconds"),
         "dataset_source": ctx["source"],
         "rows_raw": ctx["rows_raw"], "rows_clean": ctx["rows_clean"],
         "skills_total": len(records), "skills_ok": len(ok),
         "total_seconds": round(sum(r["seconds"] for r in records), 2),
         "skills": [{k: r[k] for k in
                     ("id", "name", "pack", "category", "crisp_dm",
                      "status", "seconds", "headline", "blocks", "error")}
                    for r in records]},
        indent=2))
    return md_path, json_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the skills lab end to end")
    ap.add_argument("--pack", help="run only one pack")
    ap.add_argument("--skill", help="run one skill by id")
    args = ap.parse_args()

    selected = SKILLS
    if args.skill:
        if args.skill not in BY_ID:
            print(f"unknown skill: {args.skill}")
            return 2
        selected = [BY_ID[args.skill]]
    elif args.pack:
        selected = [s for s in SKILLS if s.pack == args.pack]
        if not selected:
            print(f"unknown pack: {args.pack}")
            return 2

    print(f"Preparing data and baseline model ...")
    ctx = build_context()
    print(f"Dataset: {ctx['source']} — {ctx['rows_clean']:,} rows, "
          f"held-out ROC-AUC {ctx['model']['metrics']['roc_auc']:.4f}\n")

    records = run_skills(selected, ctx)
    md_path, json_path = write_report(records, ctx)

    ok = sum(r["status"] == "ok" for r in records)
    print(f"\n{ok}/{len(records)} skills succeeded in "
          f"{sum(r['seconds'] for r in records):.1f}s")
    print(f"report:  {md_path.relative_to(ROOT)}")
    print(f"summary: {json_path.relative_to(ROOT)}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

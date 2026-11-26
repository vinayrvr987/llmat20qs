#!/usr/bin/env python3
"""Evaluate the archived v1 (free-retry guess bug) results without trusting them.

Under the v1 rule a wrong GUESS naming a listed candidate was silently retried
within the same turn. A game with no such hidden guess is identical to what the
corrected rule would have produced (with v1 prompts), so its outcome is exact.
A game with a hidden guess diverges at that turn, so its outcome is unknown:
the win rate is bounded below by counting it as a loss and above by counting it
as a win.

Caveat: v1 logged the guesser's attempts only on guess and invalid-move turns,
not on question turns, so a wrong guess retried before a question is invisible.
"Clean" therefore means *no visible* hidden guess. Replaying v1 games under the
fixed rule found that 11 of 102 visibly clean Qwen3 games had such invisible
retries, versus 0 of 64 Gemma games. Treat the bounds as approximate for Qwen3
and Granite; for Gemma and Mistral they are supported by the replay evidence. Measures that do not depend on guess handling up to the divergence
(question repetition, invalid moves, answerer behavior) are also reported.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from src.data.paths import PROJECT_ROOT
from src.experiments import e1


ARCHIVE = PROJECT_ROOT / "archive" / "guess-bug-v1" / "results"
REPORTS = PROJECT_ROOT / "reports"
GUESS = re.compile(r"^\s*GUESS\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
FILES = {
    "E1 cross-play": ("e1-cross-play-r1.jsonl", True),
    "E1 self-play": ("e1-self-play-r1.jsonl", True),
    "E2 predictable": ("e2-predictable-games-s519.jsonl", False),
    "E2 scouting (partial)": ("e2-scouting-s519.jsonl", False),
    "E3 liar (partial)": ("e3-liar-s519.jsonl", False),
    "E4 noisy": ("e4-noisy-s519.jsonl", False),
}


def condition_label(record: dict) -> str:
    condition = record.get("condition") or {}
    if "flip_probability" in condition:
        return f"flip={condition['flip_probability']:.0%}"
    if "lie_count" in condition:
        return f"lies={condition['lie_count']},{'warned' if condition['warned'] else 'unwarned'}"
    if "favorite_picks" in condition:
        return "scouting report" if condition["favorite_picks"] else "no report"
    return "all"


def first_hidden_guess_turn(record: dict, listed: set[str]) -> int | None:
    """Turn of the first wrong listed-candidate guess that v1 retried for free."""
    for turn in record["turns"]:
        responses = turn.get("raw_responses", [])
        retried = responses[:-1] if turn["type"] == "guess" else responses
        for response in retried:
            match = GUESS.match(response["text"].strip().splitlines()[0] if response["text"].strip() else "")
            if match and e1.normalize(match.group(1)) in listed:
                return turn["turn"]
    return None


def game_row(record: dict, listed: set[str]) -> dict:
    questions = [turn for turn in record["turns"] if turn["type"] == "question"]
    texts = [e1.normalize(turn["text"]) for turn in questions]
    answers = [turn.get("answer") for turn in questions]
    divergence = first_hidden_guess_turn(record, listed)
    return {
        "guesser": record["guesser"]["key"],
        "answerer": record["answerer"]["key"],
        "condition": condition_label(record),
        "won": record["won"],
        "clean": divergence is None,
        "divergence_turn": divergence,
        "questions": len(questions),
        "repeated_questions": len(texts) - len(set(texts)),
        "invalid_moves": record["invalid_moves"],
        "invalid_answers": record["invalid_answers"],
        "yes_answers": sum(answer == "Yes" for answer in answers),
        "valid_answers": sum(answer in ("Yes", "No") for answer in answers),
        "applied_flips": record.get("applied_flips", 0),
    }


def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    clean = [row for row in rows if row["clean"]]
    clean_wins = sum(row["won"] for row in clean)
    affected = n - len(clean)
    questions = sum(row["questions"] for row in rows)
    valid_answers = sum(row["valid_answers"] for row in rows)
    return {
        "games": n,
        "reported_v1_win_rate": sum(row["won"] for row in rows) / n,
        "win_rate_lower_bound": clean_wins / n,
        "win_rate_upper_bound": (clean_wins + affected) / n,
        "clean_games": len(clean),
        "clean_game_win_rate": clean_wins / len(clean) if clean else None,
        "affected_games": affected,
        "median_divergence_turn": (
            statistics.median(row["divergence_turn"] for row in rows if not row["clean"])
            if affected else None
        ),
        "mean_questions": questions / n,
        "repeated_question_rate": (
            sum(row["repeated_questions"] for row in rows) / questions if questions else 0.0
        ),
        "invalid_moves_per_game": sum(row["invalid_moves"] for row in rows) / n,
        "answerer_yes_rate": (
            sum(row["yes_answers"] for row in rows) / valid_answers if valid_answers else None
        ),
        "invalid_answers_per_game": sum(row["invalid_answers"] for row in rows) / n,
        "mean_applied_flips": sum(row["applied_flips"] for row in rows) / n,
    }


def grouped(rows: list[dict], *keys: str) -> dict[str, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return {" | ".join(key): summarize_rows(value) for key, value in sorted(groups.items())}


def load_rows(archive: Path = ARCHIVE) -> dict[str, list[dict]]:
    """One row per archived game, keyed by study name."""
    _, targets = e1.load_targets()
    pools = {
        True: e1.candidate_names([target for target in targets if target.e1]),
        False: e1.candidate_names(targets),
    }
    rows_by_study = {}
    for study, (filename, e1_pool) in FILES.items():
        path = archive / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
        rows_by_study[study] = [
            {"study": study, **game_row(record, pools[e1_pool])} for record in records
        ]
    return rows_by_study


def evaluate(archive: Path = ARCHIVE) -> dict:
    result = {
        "schema_version": 1,
        "source": str(archive.relative_to(PROJECT_ROOT)),
        "method": __doc__.strip(),
        "studies": {},
    }
    for study, rows in load_rows(archive).items():
        entry = {
            "overall": summarize_rows(rows),
            "by_guesser": grouped(rows, "guesser"),
            "by_answerer": grouped(rows, "answerer"),
        }
        if any(row["condition"] != "all" for row in rows):
            entry["by_guesser_condition"] = grouped(rows, "guesser", "condition")
        result["studies"][study] = entry
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def render_markdown(result: dict) -> str:
    lines = [
        "# Archived v1 results: what survives the guess bug",
        "",
        "Generated by `python -m src.analysis.v1_salvage` from "
        f"`{result['source']}`. These runs retried wrong guesses for free. "
        "*Clean* games contain no visible retry; affected games diverge at the first "
        "retried guess, so win rates are given as bounds (affected games counted as "
        "losses / as wins). v1 did not log guesser attempts on question turns, so some "
        "visibly clean games were affected too: on replay, 11 of 102 visibly clean "
        "Qwen3 games differed (0 of 64 Gemma games). Bounds are approximate for Qwen3 "
        "and Granite. Not for headline claims; use the v2 re-runs.",
        "",
    ]
    for study, entry in result["studies"].items():
        lines += [
            f"## {study}",
            "",
            "| Guesser | Games | v1 reported | Bounds (low–high) | Clean games | Clean win rate "
            "| Repeated Qs | Invalid moves/game |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key, s in entry["by_guesser"].items():
            lines.append(
                f"| {key} | {s['games']} | {_pct(s['reported_v1_win_rate'])} | "
                f"{_pct(s['win_rate_lower_bound'])}–{_pct(s['win_rate_upper_bound'])} | "
                f"{s['clean_games']} | {_pct(s['clean_game_win_rate'])} | "
                f"{_pct(s['repeated_question_rate'])} | {s['invalid_moves_per_game']:.2f} |"
            )
        lines += [
            "",
            "| Answerer | Yes rate | Invalid answers/game |",
            "|---|---:|---:|",
        ]
        for key, s in entry["by_answerer"].items():
            lines.append(
                f"| {key} | {_pct(s['answerer_yes_rate'])} | {s['invalid_answers_per_game']:.2f} |"
            )
        if "by_guesser_condition" in entry:
            lines += [
                "",
                "| Guesser | Condition | Games | Bounds (low–high) | Clean win rate | Applied flips |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for key, s in entry["by_guesser_condition"].items():
                guesser, condition = key.split(" | ")
                lines.append(
                    f"| {guesser} | {condition} | {s['games']} | "
                    f"{_pct(s['win_rate_lower_bound'])}–{_pct(s['win_rate_upper_bound'])} | "
                    f"{_pct(s['clean_game_win_rate'])} | {s['mean_applied_flips']:.2f} |"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    args = parser.parse_args(argv)
    result = evaluate(args.archive)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / "v1-salvage.summary.json"
    md_path = args.reports_dir / "v1-salvage.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"Summary: {json_path}\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

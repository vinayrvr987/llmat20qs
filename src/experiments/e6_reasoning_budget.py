#!/usr/bin/env python3
"""Run E6: does more Qwen3 reasoning buy more bits per question, and at what cost?

Qwen3 8B plays the same 40 E5 targets against the same deterministic oracle and
question bank under three conditions: thinking disabled, a 256-token thinking
budget, and a 1024-token thinking budget. Budgets are enforced by budget forcing:
if the model has not closed its reasoning block when the budget runs out, the
controller closes it and allows a short answer. Conditions share targets and
sampling seeds, so the analysis compares them target by target.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.paths import RESULTS_DIR
from src.experiments import e1, e5_information as e5


DEFAULT_RESULTS = RESULTS_DIR
EXPERIMENT = "E6_reasoning_budget"
MODEL = next(model for model in e1.MODELS if model.key == "qwen3-8b")
BOOTSTRAP_REPLICATES = 2000
NO_THINKING_MAX_TOKENS = e1.GENERATION_CONFIG["max_new_tokens"]
ANSWER_MAX_TOKENS = 32
ACTION_LINE = re.compile(
    r"(?im)^[ \t`*]*(QUESTION[ \t]*:[ \t]*Q\d+|GUESS[ \t]*:[ \t]*[^`*\n]+?)[ \t`*]*$"
)
GAME_METRICS = (
    "won",
    "turns",
    "questions",
    "invalid_actions",
    "expected_information_bits",
    "realized_information_bits",
    "mean_question_information_bits",
    "mean_optimality_ratio",
    "output_tokens",
    "thinking_tokens",
    "budget_exhausted_rate",
    "generation_seconds",
)
PAIRED_METRICS = (
    "won",
    "turns",
    "invalid_actions",
    "mean_question_information_bits",
    "mean_optimality_ratio",
    "output_tokens",
    "generation_seconds",
)


class ReasoningExperimentError(RuntimeError):
    """An expected E6 input, schedule, or execution error."""


@dataclass(frozen=True)
class ReasoningCondition:
    key: str
    thinking_budget: int  # 0 disables thinking

    @property
    def enable_thinking(self) -> bool:
        return self.thinking_budget > 0

    def record(self) -> dict:
        return {
            "key": self.key,
            "enable_thinking": self.enable_thinking,
            "thinking_budget_tokens": self.thinking_budget,
            "answer_max_tokens": (
                ANSWER_MAX_TOKENS if self.enable_thinking else NO_THINKING_MAX_TOKENS
            ),
        }


CONDITIONS = (
    ReasoningCondition("thinking_off", 0),
    ReasoningCondition("thinking_256", 256),
    ReasoningCondition("thinking_1024", 1024),
)


@dataclass(frozen=True)
class GamePlan:
    game: e5.GamePlan
    condition: ReasoningCondition


def build_schedule(targets: list[e1.Target], seed: int) -> list[GamePlan]:
    """Target-major schedule: a partial run still yields complete matched triples."""
    base_games = [plan for plan in e5.build_schedule(targets, seed) if plan.model.key == MODEL.key]
    schedule = []
    for base in base_games:
        # Shared randomization ID -> identical sampling seeds across conditions.
        randomization_id = f"e6-reasoning|seed={seed}|model={MODEL.key}|target={base.target.id}"
        for condition in CONDITIONS:
            schedule.append(GamePlan(
                game=e5.GamePlan(
                    game_id=f"{randomization_id}|condition={condition.key}",
                    seed=seed,
                    model=MODEL,
                    target=base.target,
                    randomization_id=randomization_id,
                ),
                condition=condition,
            ))
    return schedule


def extract_action(text: str) -> str | None:
    matches = list(ACTION_LINE.finditer(text))
    return matches[-1].group(1).strip() if matches else None


class ReasoningBackend:
    """Route E5 generation calls through the active reasoning condition.

    Only the extracted action (or the raw answer, if none parses) is returned to
    the game, so reasoning traces never enter the conversation history. Per-call
    token and budget statistics are collected for the current game.
    """

    def __init__(self, inner):
        self.inner = inner
        self.mode = getattr(inner, "mode", "real")
        self.condition: ReasoningCondition | None = None
        self.calls: list[dict] = []

    def prepare_pair(self, answerer: e1.ModelConfig, guesser: e1.ModelConfig) -> None:
        self.inner.prepare_pair(answerer, guesser)

    def start_game(self, condition: ReasoningCondition) -> None:
        self.condition = condition
        self.calls = []

    def generate(
        self,
        model: e1.ModelConfig,
        messages: list[dict[str, str]],
        seed: int,
    ) -> e1.Generation:
        if self.condition is None:
            raise ReasoningExperimentError("No reasoning condition is active.")
        if self.condition.enable_thinking:
            generation = self.inner.generate_with_thinking_budget(
                model,
                messages,
                seed,
                thinking_budget=self.condition.thinking_budget,
                answer_max_tokens=ANSWER_MAX_TOKENS,
            )
            thinking_tokens = generation.thinking_tokens
            budget_exhausted = generation.budget_exhausted
        else:
            generation = self.inner.generate_with_options(
                model,
                messages,
                seed,
                enable_thinking=False,
                max_new_tokens=NO_THINKING_MAX_TOKENS,
            )
            thinking_tokens = 0
            budget_exhausted = False
        action = extract_action(generation.text)
        self.calls.append({
            "elapsed_seconds": generation.elapsed_seconds,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "thinking_tokens": thinking_tokens,
            "budget_exhausted": budget_exhausted,
            "answer_text": generation.text,
            "action_extracted": action is not None,
        })
        return e1.Generation(
            text=action if action is not None else generation.text,
            elapsed_seconds=generation.elapsed_seconds,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
        )

    def close(self) -> None:
        self.inner.close()

    def metadata(self) -> dict:
        return {
            **self.inner.metadata(),
            "reasoning_condition": self.condition.record() if self.condition else None,
            "budget_enforcement": "budget forcing: close </think> at the budget, then answer",
            "reasoning_trace_recorded": False,
        }


class MockBackend:
    """Deterministic stand-in for the transformers backend (no GPU)."""

    mode = "mock"
    NATURAL_THINKING_TOKENS = 400

    def prepare_pair(self, answerer: e1.ModelConfig, guesser: e1.ModelConfig) -> None:
        del answerer, guesser

    @staticmethod
    def _answer(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        question = re.search(r"^(Q\d+):", prompt, re.MULTILINE)
        if question:
            return f"QUESTION: {question.group(1)}"
        candidates = re.search(r"candidates:\n([^\n]+)", prompt, re.IGNORECASE)
        return f"GUESS: {candidates.group(1).split(', ')[0] if candidates else 'chair'}"

    def generate_with_options(self, model, messages, seed, *, enable_thinking, max_new_tokens):
        del model, seed, max_new_tokens
        if enable_thinking:
            raise ReasoningExperimentError("Mock: thinking-off path received enable_thinking.")
        return e1.Generation(self._answer(messages), 0.0, input_tokens=20, output_tokens=6)

    def generate_with_thinking_budget(
        self, model, messages, seed, *, thinking_budget, answer_max_tokens
    ):
        del model, seed, answer_max_tokens
        thinking = min(thinking_budget, self.NATURAL_THINKING_TOKENS)
        return e1.ThinkingGeneration(
            text=self._answer(messages),
            elapsed_seconds=0.0,
            input_tokens=20,
            output_tokens=thinking + 6,
            thinking_tokens=thinking,
            answer_tokens=6,
            budget_exhausted=thinking_budget < self.NATURAL_THINKING_TOKENS,
        )

    def close(self) -> None:
        return None

    def metadata(self) -> dict:
        return {"inference_mode": self.mode}


def game_metrics(record: dict) -> dict[str, float]:
    question_turns = [turn for turn in record["turns"] if turn["action"] == "question"]
    calls = record["reasoning_calls"]
    return {
        "won": float(record["won"]),
        "turns": float(record["turns_used"]),
        "questions": float(record["questions_asked"]),
        "invalid_actions": float(record["invalid_actions"]),
        "expected_information_bits": sum(
            turn["expected_information_gain_bits"] for turn in question_turns
        ),
        "realized_information_bits": sum(
            turn["realized_information_gain_bits"] for turn in question_turns
        ),
        "mean_question_information_bits": (
            statistics.mean(turn["expected_information_gain_bits"] for turn in question_turns)
            if question_turns else 0.0
        ),
        "mean_optimality_ratio": (
            statistics.mean(turn["optimality_ratio"] for turn in question_turns)
            if question_turns else 0.0
        ),
        "output_tokens": float(sum(call["output_tokens"] or 0 for call in calls)),
        "thinking_tokens": float(sum(call["thinking_tokens"] for call in calls)),
        "budget_exhausted_rate": (
            statistics.mean(float(call["budget_exhausted"]) for call in calls) if calls else 0.0
        ),
        "generation_seconds": sum(call["elapsed_seconds"] for call in calls),
    }


def bootstrap_interval(values: list[float], seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return [
        means[int(0.025 * BOOTSTRAP_REPLICATES)],
        means[int(0.975 * BOOTSTRAP_REPLICATES) - 1],
    ]


def summarize(records: list[dict]) -> dict:
    metrics_by_condition: dict[str, dict[str, dict[str, float]]] = {
        condition.key: {} for condition in CONDITIONS
    }
    for record in records:
        metrics_by_condition[record["reasoning_condition"]["key"]][record["target"]["id"]] = (
            game_metrics(record)
        )

    groups = []
    for condition in CONDITIONS:
        by_target = metrics_by_condition[condition.key]
        group = {**condition.record(), "games": len(by_target)}
        for metric in GAME_METRICS:
            values = [item[metric] for item in by_target.values()]
            group[f"mean_{metric}"] = statistics.mean(values) if values else 0.0
            group[f"mean_{metric}_target_bootstrap_95_ci"] = bootstrap_interval(
                values, e1.game_seed("e6-summary", condition.key, GAME_METRICS.index(metric), 0)
            )
        groups.append(group)

    paired = []
    for reference, condition in itertools.combinations(CONDITIONS, 2):
        ref, cur = metrics_by_condition[reference.key], metrics_by_condition[condition.key]
        matched = sorted(ref.keys() & cur.keys())
        differences = {
            metric: [cur[target][metric] - ref[target][metric] for target in matched]
            for metric in PAIRED_METRICS
        }
        means = {
            metric: statistics.mean(values) if values else 0.0
            for metric, values in differences.items()
        }
        token_delta = means["output_tokens"]
        paired.append({
            "condition": condition.key,
            "reference_condition": reference.key,
            "matched_targets": len(matched),
            "mean_paired_differences": means,
            "target_bootstrap_95_ci": {
                metric: bootstrap_interval(
                    values,
                    e1.game_seed(
                        "e6-paired", f"{reference.key}->{condition.key}",
                        PAIRED_METRICS.index(metric), 0,
                    ),
                )
                for metric, values in differences.items()
            },
            "question_bits_gained_per_1k_extra_tokens_per_game": (
                1000 * means["mean_question_information_bits"] / token_delta
                if token_delta > 0 else None
            ),
        })

    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "model": asdict(MODEL),
        "design": {
            "targets": len({record["target"]["id"] for record in records}),
            "conditions": [condition.record() for condition in CONDITIONS],
            "generation_config": e1.GENERATION_CONFIG,
            "matched_on": ["target", "per-turn sampling seed", "question bank", "oracle"],
            "budget_enforcement": (
                "Thinking is capped at the budget; if </think> is missing at the cap the "
                "controller inserts it and allows up to "
                f"{ANSWER_MAX_TOKENS} answer tokens."
            ),
            "reasoning_trace_saved": False,
            "uncertainty": "95% percentile bootstrap over targets (paired for differences)",
        },
        "games": len(records),
        "groups": groups,
        "paired_comparisons": paired,
    }


def read_existing(path: Path, schedule: list[GamePlan]) -> dict[str, dict]:
    if not path.exists():
        return {}
    scheduled = {plan.game.game_id for plan in schedule}
    existing = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReasoningExperimentError(
                    f"Malformed JSON in {path} line {line_number}: {error}."
                ) from error
            game_id = record.get("game_id")
            if record.get("experiment") != EXPERIMENT or game_id not in scheduled:
                raise ReasoningExperimentError(f"Unexpected E6 record in {path} line {line_number}.")
            if record.get("run_config", {}).get("dialogue") != e5.DIALOGUE:
                raise ReasoningExperimentError(
                    f"{path} line {line_number} uses an older dialogue protocol; move the "
                    "old results file aside and start a new one."
                )
            if game_id in existing:
                raise ReasoningExperimentError(f"Duplicate E6 game ID {game_id} in {path}.")
            existing[game_id] = record
    return existing


def write_summary(path: Path, summary: dict) -> Path:
    output = path.with_suffix(".summary.json")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def run(
    schedule: list[GamePlan],
    targets: list[e1.Target],
    questions: list[e5.Question],
    results_path: Path,
    backend: ReasoningBackend,
    max_games: int | None = None,
) -> list[dict]:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing(results_path, schedule)
    pending = [plan for plan in schedule if plan.game.game_id not in existing]
    if max_games is not None:
        pending = pending[:max_games]
    print(
        f"E6 reasoning budget: {len(schedule)} games; {len(existing)} complete; "
        f"{len(pending)} to run.",
        flush=True,
    )
    records = dict(existing)
    if pending:
        backend.prepare_pair(MODEL, MODEL)
    with results_path.open("a", encoding="utf-8") as output:
        for index, plan in enumerate(pending, start=1):
            backend.start_game(plan.condition)
            record = e5.play_game(plan.game, targets, questions, backend)
            record["experiment"] = EXPERIMENT
            record["reasoning_condition"] = plan.condition.record()
            record["reasoning_calls"] = backend.calls
            record["run_config"]["reasoning_condition"] = plan.condition.record()
            record["baselines"] = {
                "greedy": e5.simulate_baseline(
                    plan.game.target, targets, questions, "greedy", plan.game.seed
                ),
                "random": e5.simulate_baseline(
                    plan.game.target, targets, questions, "random",
                    e1.game_seed(plan.game.randomization_id, "random-baseline", 0, 0),
                ),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            records[plan.game.game_id] = record
            if index % 15 == 0 or index == len(pending):
                print(f"Saved game {len(records)}/{len(schedule)}.", flush=True)
    return [records[plan.game.game_id] for plan in schedule if plan.game.game_id in records]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=519)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--question-bank", type=Path, default=e5.QUESTION_BANK)
    parser.add_argument("--max-games", type=int, help="limit games for mock runs only")
    parser.add_argument("--mock", action="store_true", help="use a deterministic fake backend")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.max_games is not None and (not args.mock or args.max_games < 1):
            raise ReasoningExperimentError("--max-games requires --mock and must be positive.")
        _, all_targets = e1.load_targets()
        targets = [target for target in all_targets if target.e1]
        questions = e5.load_question_bank(targets, args.question_bank)
        schedule = build_schedule(all_targets, args.seed)
        output_path = args.results_dir / f"e6-reasoning-budget-s{args.seed}.jsonl"
        backend = ReasoningBackend(MockBackend() if args.mock else e1.HuggingFaceBackend())
        try:
            records = run(schedule, targets, questions, output_path, backend, args.max_games)
            summary_path = write_summary(output_path, summarize(records))
        finally:
            backend.close()
        print(f"Game results: {output_path}", flush=True)
        print(f"Summary: {summary_path}", flush=True)
        return 0
    except (e1.ExperimentError, e5.InformationExperimentError, ReasoningExperimentError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

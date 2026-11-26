#!/usr/bin/env python3
"""Run E5: model question selection with a deterministic exact-information oracle."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.paths import PROJECT_ROOT, RESULTS_DIR, TARGET_MANIFEST
from src.experiments import e1


ROOT = PROJECT_ROOT
QUESTION_BANK = TARGET_MANIFEST.with_name("e5_question_bank_v1.json")
DEFAULT_RESULTS = RESULTS_DIR
MAX_TURNS = e1.MAX_TURNS
EXPERIMENT = "E5_exact_information"
# Each turn sends only the system prompt and the current state prompt, which
# already lists every previous answer. Accumulating full state prompts across
# turns ran Gemma 3 (float32, ~45 GiB weights) out of GPU memory.
DIALOGUE = "current-state-only-v2"
QUESTION_ACTION = re.compile(r"^\s*QUESTION\s*:\s*(Q\d+)\s*$", re.IGNORECASE)
GUESS_ACTION = re.compile(r"^\s*GUESS\s*:\s*(.+?)\s*$", re.IGNORECASE)


class InformationExperimentError(RuntimeError):
    """An expected E5 input, schedule, or execution error."""


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    yes_target_ids: frozenset[str]


@dataclass(frozen=True)
class GamePlan:
    game_id: str
    seed: int
    model: e1.ModelConfig
    target: e1.Target
    randomization_id: str | None = None


def _entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (
        1.0 - probability
    ) * math.log2(1.0 - probability)


def expected_information_gain(
    candidates: set[str] | frozenset[str],
    question: Question,
) -> float:
    if not candidates:
        raise InformationExperimentError("Information gain requires candidates.")
    yes_count = len(candidates & question.yes_target_ids)
    if yes_count == 0 or yes_count == len(candidates):
        return 0.0
    return _entropy(yes_count / len(candidates))


def load_question_bank(
    targets: list[e1.Target],
    path: Path = QUESTION_BANK,
) -> list[Question]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InformationExperimentError(f"Cannot read question bank {path}: {error}") from error
    if payload.get("schema_version") != 1:
        raise InformationExperimentError("Unsupported E5 question-bank schema.")

    target_ids = {target.id for target in targets}
    questions = []
    seen_ids = set()
    for item in payload.get("questions", []):
        question_id = item.get("id")
        if not isinstance(question_id, str) or question_id in seen_ids:
            raise InformationExperimentError(f"Invalid or duplicate question ID: {question_id!r}.")
        seen_ids.add(question_id)
        yes_ids = set(item.get("yes_target_ids", []))
        categories = set(item.get("yes_categories", []))
        if categories - set(e1.CATEGORIES):
            raise InformationExperimentError(f"Unknown category in question {question_id}.")
        yes_ids.update(target.id for target in targets if target.category in categories)
        unknown_ids = yes_ids - target_ids
        if unknown_ids:
            raise InformationExperimentError(
                f"Question {question_id} references unknown targets: {sorted(unknown_ids)}."
            )
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise InformationExperimentError(f"Question {question_id} has no prompt text.")
        if not yes_ids or yes_ids == target_ids:
            raise InformationExperimentError(
                f"Question {question_id} must have both yes and no targets."
            )
        questions.append(Question(question_id, item["text"], frozenset(yes_ids)))
    if not questions:
        raise InformationExperimentError("The E5 question bank is empty.")
    return questions


def build_schedule(
    targets: list[e1.Target],
    seed: int,
) -> list[GamePlan]:
    selected = [target for target in targets if target.e1]
    if len(selected) != 40:
        raise InformationExperimentError(f"Expected 40 E1 targets; found {len(selected)}.")
    shuffled = list(selected)
    random.Random(seed).shuffle(shuffled)
    return [
        GamePlan(
            game_id=f"e5-information|seed={seed}|model={model.key}|target={target.id}",
            seed=seed,
            model=model,
            target=target,
        )
        for model in e1.MODELS
        for target in shuffled
    ]


def available_questions(
    candidates: set[str],
    questions: list[Question],
    asked: set[str],
) -> list[Question]:
    return [
        question for question in questions
        if question.question_id not in asked
        and 0 < len(candidates & question.yes_target_ids) < len(candidates)
    ]


def _guess_target(value: str, candidates: set[str], target_by_id: dict[str, e1.Target]) -> str | None:
    normalized = e1.normalize(value)
    for target_id in candidates:
        target = target_by_id[target_id]
        if normalized in {e1.normalize(name) for name in target.accepted_names}:
            return target_id
    return None


def parse_action(
    text: str,
    questions: list[Question],
    candidates: set[str],
    target_by_id: dict[str, e1.Target],
) -> tuple[str, str] | None:
    question_match = QUESTION_ACTION.fullmatch(text)
    if question_match:
        question_id = question_match.group(1).upper()
        if any(question.question_id == question_id for question in questions):
            return "question", question_id
        return None
    guess_match = GUESS_ACTION.fullmatch(text)
    if guess_match:
        target_id = _guess_target(guess_match.group(1), candidates, target_by_id)
        if target_id is not None:
            return "guess", target_id
    return None


def _state_prompt(
    candidates: set[str],
    questions: list[Question],
    target_by_id: dict[str, e1.Target],
    history: list[str],
    turn: int,
) -> str:
    names = sorted((target_by_id[target_id].name for target_id in candidates), key=str.casefold)
    menu = "\n".join(
        f"{question.question_id}: {question.text}"
        for question in available_questions(candidates, questions, set())
    )
    history_text = "\n".join(history) if history else "No questions have been asked yet."
    guess_rule = (
        "You may guess one remaining candidate with `GUESS: <exact concept name>`."
        if candidates else "No candidates remain."
    )
    return (
        f"Turn {turn} of {MAX_TURNS}. The secret is one of these {len(candidates)} candidates:\n"
        f"{', '.join(names)}\n\n"
        f"Previous answers:\n{history_text}\n\n"
        f"Questions that split the remaining candidates:\n{menu or '(none)'}\n\n"
        "Choose exactly one action. Ask an available question using "
        "`QUESTION: <question ID>`, or guess using `GUESS: <exact concept name>`. "
        "A question gets a truthful Yes/No answer from a deterministic oracle. "
        "Each question or guess uses one turn. Do not add any other text. "
        f"{guess_rule}"
    )


def _system_prompt() -> str:
    return (
        "You are playing a controlled version of 20 Questions. Select questions that "
        "efficiently distinguish the listed candidates. The oracle's answers are exact "
        "according to the experiment's fixed question bank. Do not infer a secret from "
        "information outside the candidate list or question answers."
    )


def play_game(
    plan: GamePlan,
    targets: list[e1.Target],
    questions: list[Question],
    backend: e1.Backend,
) -> dict:
    target_by_id = {target.id: target for target in targets}
    candidate_ids = {target.id for target in targets if target.e1}
    candidates = set(candidate_ids)
    if plan.target.id not in candidates:
        raise InformationExperimentError(f"Target {plan.target.id} is outside the E5 set.")

    system_message = {"role": "system", "content": _system_prompt()}
    history = []
    asked: set[str] = set()
    turns = []
    won = False

    for turn_number in range(1, MAX_TURNS + 1):
        if len(candidates) == 1:
            guess = next(iter(candidates))
            won = guess == plan.target.id
            turns.append({
                "turn": turn_number,
                "action": "forced_guess",
                "target_id": guess,
                "correct": won,
            })
            break
        choices = available_questions(candidates, questions, asked)
        prompt = _state_prompt(
            candidates, choices, target_by_id, history, turn_number
        )
        messages = [system_message, {"role": "user", "content": prompt}]
        raw_responses = []
        parsed = None
        for attempt in range(e1.MAX_FORMAT_ATTEMPTS):
            generation = backend.generate(
                plan.model,
                messages,
                e1.game_seed(
                    plan.randomization_id or plan.game_id,
                    "questioner",
                    turn_number,
                    attempt,
                ),
            )
            raw_responses.append({
                "text": generation.text,
                "elapsed_seconds": generation.elapsed_seconds,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
            })
            messages.append({"role": "assistant", "content": generation.text})
            parsed = parse_action(generation.text, choices, candidates, target_by_id)
            if parsed:
                break
            if attempt + 1 < e1.MAX_FORMAT_ATTEMPTS:
                messages.append({
                    "role": "user",
                    "content": (
                        "Invalid action. Choose an available question ID or an exact "
                        "remaining candidate, and use only the required syntax."
                    ),
                })

        if parsed is None:
            turns.append({
                "turn": turn_number,
                "action": "invalid",
                "raw_responses": raw_responses,
                "candidate_count_before": len(candidates),
                "candidate_count_after": len(candidates),
            })
            history.append("The action was invalid; the turn was spent.")
            continue

        action, value = parsed
        if action == "guess":
            correct = value == plan.target.id
            turns.append({
                "turn": turn_number,
                "action": "guess",
                "target_id": value,
                "correct": correct,
                "raw_responses": raw_responses,
                "candidate_count_before": len(candidates),
                "candidate_count_after": len(candidates) if correct else len(candidates) - 1,
            })
            if correct:
                won = True
                break
            candidates.remove(value)
            history.append(f"No, the secret is not {target_by_id[value].name}.")
            continue

        question = next(question for question in choices if question.question_id == value)
        before = len(candidates)
        gain = expected_information_gain(candidates, question)
        best_gain = max(expected_information_gain(candidates, candidate_question)
                        for candidate_question in choices)
        answer = plan.target.id in question.yes_target_ids
        asked.add(question.question_id)
        candidates = (
            candidates & question.yes_target_ids
            if answer else candidates - question.yes_target_ids
        )
        turns.append({
            "turn": turn_number,
            "action": "question",
            "question_id": question.question_id,
            "question": question.text,
            "answer": "Yes" if answer else "No",
            "expected_information_gain_bits": gain,
            "best_available_information_gain_bits": best_gain,
            "optimality_ratio": gain / best_gain if best_gain else 1.0,
            "realized_information_gain_bits": math.log2(before / len(candidates)),
            "candidate_count_before": before,
            "candidate_count_after": len(candidates),
            "raw_responses": raw_responses,
        })
        history.append(f"{question.text} — {turns[-1]['answer']}.")

    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "inference_mode": getattr(backend, "mode", "real"),
        "game_id": plan.game_id,
        "seed": plan.seed,
        "model": asdict(plan.model),
        "target": e1.target_record(plan.target),
        "target_manifest": "twenty-questions-targets-v1",
        "question_bank": "e5-exact-information-v1",
        "run_config": {
            **e1.RUN_CONFIG,
            "max_turns": MAX_TURNS,
            "oracle": "deterministic binary labels from e5-exact-information-v1",
            "dialogue": DIALOGUE,
        },
        "runtime": backend.metadata() if hasattr(backend, "metadata") else {
            "inference_mode": getattr(backend, "mode", "real")
        },
        "won": won,
        "turns_used": len(turns),
        "questions_asked": sum(turn["action"] == "question" for turn in turns),
        "invalid_actions": sum(turn["action"] == "invalid" for turn in turns),
        "turns": turns,
    }


def simulate_baseline(
    target: e1.Target,
    targets: list[e1.Target],
    questions: list[Question],
    strategy: str,
    seed: int,
) -> dict:
    candidates = {candidate.id for candidate in targets if candidate.e1}
    asked: set[str] = set()
    rng = random.Random(seed)
    questions_asked = 0
    bits = 0.0
    won = False
    turns = 0
    while turns < MAX_TURNS:
        if len(candidates) == 1:
            won = target.id in candidates
            turns += 1
            break
        choices = available_questions(candidates, questions, asked)
        if not choices:
            break
        if strategy == "greedy":
            question = min(
                choices,
                key=lambda option: (
                    -expected_information_gain(candidates, option),
                    option.question_id,
                ),
            )
        elif strategy == "random":
            question = rng.choice(choices)
        else:
            raise InformationExperimentError(f"Unknown baseline strategy: {strategy}.")
        gain = expected_information_gain(candidates, question)
        bits += gain
        questions_asked += 1
        turns += 1
        asked.add(question.question_id)
        candidates = (
            candidates & question.yes_target_ids
            if target.id in question.yes_target_ids
            else candidates - question.yes_target_ids
        )
    return {
        "won": won,
        "turns_used": turns,
        "questions_asked": questions_asked,
        "expected_information_gain_bits": bits,
    }


def summarize(records: list[dict]) -> dict:
    groups = []
    for model_key in sorted({record["model"]["key"] for record in records}):
        subset = [record for record in records if record["model"]["key"] == model_key]
        question_turns = [
            turn for record in subset for turn in record["turns"]
            if turn["action"] == "question"
        ]
        group = {
            "model": model_key,
            "games": len(subset),
            "win_rate": statistics.mean(record["won"] for record in subset),
            "mean_turns": statistics.mean(record["turns_used"] for record in subset),
            "mean_questions": statistics.mean(record["questions_asked"] for record in subset),
            "mean_expected_information_gain_per_question_bits": (
                statistics.mean(turn["expected_information_gain_bits"] for turn in question_turns)
                if question_turns else 0.0
            ),
            "mean_question_optimality_ratio": (
                statistics.mean(turn["optimality_ratio"] for turn in question_turns)
                if question_turns else 0.0
            ),
            "invalid_actions": sum(record["invalid_actions"] for record in subset),
        }
        for baseline in ("greedy", "random"):
            baseline_records = [record["baselines"][baseline] for record in subset]
            group[f"{baseline}_baseline_win_rate"] = statistics.mean(
                record["won"] for record in baseline_records
            )
            group[f"{baseline}_baseline_mean_turns"] = statistics.mean(
                record["turns_used"] for record in baseline_records
            )
            group[f"{baseline}_baseline_mean_information_gain_bits"] = statistics.mean(
                record["expected_information_gain_bits"] for record in baseline_records
            )
        groups.append(group)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "games": len(records),
        "groups": groups,
    }


def _read_existing(path: Path, schedule: list[GamePlan]) -> dict[str, dict]:
    if not path.exists():
        return {}
    scheduled = {plan.game_id for plan in schedule}
    completed = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise InformationExperimentError(
                    f"Malformed JSON in {path} line {line_number}: {error}."
                ) from error
            game_id = record.get("game_id")
            if record.get("run_config", {}).get("dialogue") != DIALOGUE:
                raise InformationExperimentError(
                    f"{path} line {line_number} uses an older dialogue protocol; move the "
                    "old results file aside and start a new one."
                )
            if record.get("experiment") != EXPERIMENT or game_id not in scheduled:
                raise InformationExperimentError(
                    f"Unexpected E5 record in {path} line {line_number}."
                )
            if game_id in completed:
                raise InformationExperimentError(f"Duplicate E5 game ID {game_id} in {path}.")
            completed[game_id] = record
    return completed


def _write_summary(path: Path, summary: dict) -> Path:
    output = path.with_suffix(".summary.json")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def _run(
    backend: e1.Backend,
    schedule: list[GamePlan],
    targets: list[e1.Target],
    questions: list[Question],
    output_path: Path,
    max_games: int | None = None,
) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(output_path, schedule)
    pending = [plan for plan in schedule if plan.game_id not in existing]
    if max_games is not None:
        pending = pending[:max_games]
    print(
        f"E5 exact-information: {len(schedule)} games; {len(existing)} complete; "
        f"{len(pending)} to run.",
        flush=True,
    )
    records = dict(existing)
    with output_path.open("a", encoding="utf-8") as output:
        active_model = None
        for index, plan in enumerate(pending, start=1):
            if plan.model.key != active_model:
                backend.prepare_pair(plan.model, plan.model)
                active_model = plan.model.key
            record = play_game(plan, targets, questions, backend)
            greedy = simulate_baseline(plan.target, targets, questions, "greedy", plan.seed)
            random_baseline = simulate_baseline(
                plan.target,
                targets,
                questions,
                "random",
                e1.game_seed(plan.game_id, "random-baseline", 0, 0),
            )
            record["baselines"] = {"greedy": greedy, "random": random_baseline}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            records[plan.game_id] = record
            if index % 20 == 0 or index == len(pending):
                print(f"Saved game {len(records)}/{len(schedule)}.", flush=True)
    return [records[plan.game_id] for plan in schedule if plan.game_id in records]


class _MockBackend:
    mode = "mock"

    def prepare_pair(self, answerer: e1.ModelConfig, guesser: e1.ModelConfig) -> None:
        del answerer, guesser

    def generate(self, model: e1.ModelConfig, messages: list[dict[str, str]], seed: int) -> e1.Generation:
        del model, seed
        prompt = messages[-1]["content"]
        question = re.search(r"^(Q\d+):", prompt, re.MULTILINE)
        if question:
            return e1.Generation(f"QUESTION: {question.group(1)}", 0.0)
        candidates = re.search(r"candidates:\n([^\n]+)", prompt, re.IGNORECASE)
        if candidates:
            return e1.Generation(f"GUESS: {candidates.group(1).split(', ')[0]}", 0.0)
        return e1.Generation("INVALID", 0.0)

    def close(self) -> None:
        return None

    def metadata(self) -> dict:
        return {"inference_mode": self.mode}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=519)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--question-bank", type=Path, default=QUESTION_BANK)
    parser.add_argument("--max-games", type=int, help="limit games for mock runs only")
    parser.add_argument("--mock", action="store_true", help="use a deterministic fake backend")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.max_games is not None and (not args.mock or args.max_games < 1):
            raise InformationExperimentError("--max-games requires --mock and must be positive.")
        _, all_targets = e1.load_targets()
        targets = [target for target in all_targets if target.e1]
        questions = load_question_bank(targets, args.question_bank)
        schedule = build_schedule(all_targets, args.seed)
        backend = _MockBackend() if args.mock else e1.HuggingFaceBackend()
        output_path = args.results_dir / f"e5-exact-information-s{args.seed}.jsonl"
        try:
            records = _run(
                backend, schedule, targets, questions, output_path, args.max_games
            )
            summary_path = _write_summary(output_path, summarize(records))
            print(f"Game results: {output_path}", flush=True)
            print(f"Summary: {summary_path}", flush=True)
        finally:
            backend.close()
        return 0
    except (e1.ExperimentError, InformationExperimentError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run E2 predictable-minds/scouting and E3/E4 controlled-answer studies."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.paths import PROJECT_ROOT, RESULTS_DIR
from src.experiments import e1


ROOT = PROJECT_ROOT
DEFAULT_RESULTS = RESULTS_DIR
PICK_ATTEMPTS_PER_MODEL = 100
PICK_RETRIES = 3
PREDICTABLE_GAMES_PER_PAIR = 40
SCOUT_GAMES_PER_CONDITION = 80
LIAR_GAME_TOTAL = 500
NOISY_GAMES_PER_RATE = 25
NOISE_RATES = (0.05, 0.10, 0.17)
LIE_COUNTS = (0, 1, 2)
MAX_TURNS = e1.MAX_TURNS
SCOUT_TOP_K = 5


class FollowupError(RuntimeError):
    """An expected input, schedule, or execution error."""


@dataclass(frozen=True)
class GamePlan:
    experiment: str
    game_id: str
    seed: int
    answerer: e1.ModelConfig
    guesser: e1.ModelConfig
    target: e1.Target
    condition: dict
    scheduled_lie_turns: tuple[int, ...] = ()
    source_pick_attempt: int | None = None
    randomization_id: str = ""


def _stable_seed(seed: int, *parts: object) -> int:
    value = "|".join((str(seed), *(str(part) for part in parts)))
    return e1.game_seed(value, "followup", 0, 0)


def _shuffle_targets(targets: list[e1.Target], seed: int) -> list[e1.Target]:
    shuffled = list(targets)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def balanced_target(targets: list[e1.Target], index: int, seed: int) -> e1.Target:
    strata: dict[tuple[str, str], list[e1.Target]] = defaultdict(list)
    for target in targets:
        strata[(target.category, target.frequency)].append(target)
    keys = [
        (category, frequency)
        for category in e1.CATEGORIES
        for frequency in ("common", "rare")
    ]
    shuffled_strata = []
    for key_index, key in enumerate(keys):
        shuffled_strata.append(_shuffle_targets(strata[key], seed + key_index))
    ordered = [
        shuffled_strata[stratum_index][target_index]
        for target_index in range(len(shuffled_strata[0]))
        for stratum_index in range(len(shuffled_strata))
    ]
    return ordered[index % len(ordered)]


def _selected_targets(targets: list[e1.Target]) -> list[e1.Target]:
    if len(targets) != 128:
        raise FollowupError(f"Expected 128 targets, found {len(targets)}.")
    return targets


def build_predictable_schedule(
    targets: list[e1.Target],
    picks: list[dict],
    seed: int,
) -> list[GamePlan]:
    targets = _selected_targets(targets)
    picks_by_model: dict[str, list[dict]] = defaultdict(list)
    for pick in picks:
        picks_by_model[pick["model_key"]].append(pick)

    schedule: list[GamePlan] = []
    for answerer in e1.MODELS:
        model_picks = picks_by_model[answerer.key]
        if len(model_picks) != PICK_ATTEMPTS_PER_MODEL:
            raise FollowupError(
                f"Expected {PICK_ATTEMPTS_PER_MODEL} secret picks for "
                f"{answerer.key}, found {len(model_picks)}."
            )
        chosen = random.Random(_stable_seed(seed, answerer.key, "game-picks")).sample(
            model_picks, PREDICTABLE_GAMES_PER_PAIR
        )
        for guesser in e1.MODELS:
            for pick in chosen:
                target = next(
                    (item for item in targets if item.id == pick["target_id"]), None
                )
                if target is None:
                    raise FollowupError(
                        f"Pick {pick['pick_id']} refers to unknown target {pick['target_id']}."
                    )
                game_id = (
                    f"e2-predictable|seed={seed}|answerer={answerer.key}|"
                    f"guesser={guesser.key}|pick={pick['pick_id']}"
                )
                schedule.append(GamePlan(
                    experiment="E2_predictable_minds",
                    game_id=game_id,
                    seed=seed,
                    answerer=answerer,
                    guesser=guesser,
                    target=target,
                    condition={"secret_source": "answerer_model_pick"},
                    source_pick_attempt=pick["attempt"],
                    randomization_id=(
                        f"e2-predictable|seed={seed}|answerer={answerer.key}|"
                        f"pick={pick['pick_id']}"
                    ),
                ))
    _validate_unique_schedule(schedule, expected=640)
    return schedule


def top_model_picks(picks: list[dict], model_key: str, top_k: int = SCOUT_TOP_K) -> list[str]:
    counts = Counter(
        pick["target_id"] for pick in picks if pick["model_key"] == model_key
    )
    targets_by_id = {target.id: target.name for target in e1.load_targets()[1]}
    ordered = sorted(counts, key=lambda target_id: (-counts[target_id], target_id))
    return [targets_by_id[target_id] for target_id in ordered[:top_k]]


def build_scouting_schedule(
    targets: list[e1.Target],
    picks: list[dict],
    seed: int,
) -> list[GamePlan]:
    targets = _selected_targets(targets)
    picks_by_model: dict[str, list[dict]] = defaultdict(list)
    for pick in picks:
        picks_by_model[pick["model_key"]].append(pick)

    schedule: list[GamePlan] = []
    for model in e1.MODELS:
        model_picks = picks_by_model[model.key]
        if len(model_picks) != PICK_ATTEMPTS_PER_MODEL:
            raise FollowupError(
                f"Expected {PICK_ATTEMPTS_PER_MODEL} secret picks for "
                f"{model.key}, found {len(model_picks)}."
            )
        chosen = random.Random(_stable_seed(seed, model.key, "scout-picks")).sample(
            model_picks, SCOUT_GAMES_PER_CONDITION
        )
        favorites = top_model_picks(picks, model.key)
        for informed in (False, True):
            condition = {
                "scouting_report": informed,
                "report_top_k": SCOUT_TOP_K if informed else 0,
            }
            for pick in chosen:
                target = next(
                    (item for item in targets if item.id == pick["target_id"]), None
                )
                if target is None:
                    raise FollowupError(
                        f"Pick {pick['pick_id']} refers to unknown target {pick['target_id']}."
                    )
                game_id = (
                    f"e2-scouting|seed={seed}|model={model.key}|"
                    f"scouting={int(informed)}|pick={pick['pick_id']}"
                )
                schedule.append(GamePlan(
                    experiment="E2_self_secret_scouting",
                    game_id=game_id,
                    seed=seed,
                    answerer=model,
                    guesser=model,
                    target=target,
                    condition={**condition, "favorite_picks": favorites if informed else []},
                    source_pick_attempt=pick["attempt"],
                    randomization_id=(
                        f"e2-scouting|seed={seed}|model={model.key}|pick={pick['pick_id']}"
                    ),
                ))
    _validate_unique_schedule(schedule, expected=640)
    return schedule


def _validate_unique_schedule(schedule: list[GamePlan], expected: int) -> None:
    game_ids = [plan.game_id for plan in schedule]
    if len(game_ids) != expected or len(set(game_ids)) != expected:
        raise FollowupError(
            f"Expected {expected} unique games; found {len(game_ids)} "
            f"records and {len(set(game_ids))} game IDs."
        )


def build_liar_schedule(targets: list[e1.Target], seed: int) -> list[GamePlan]:
    targets = _selected_targets(targets)
    answerer = next(model for model in e1.MODELS if model.key == "qwen3-8b")
    extra_replicates = {
        e1.MODELS[0].key: {0, 1, 2},
        e1.MODELS[1].key: {0, 1, 2},
        e1.MODELS[2].key: {0, 1},
        e1.MODELS[3].key: {1, 2},
    }
    schedule: list[GamePlan] = []
    for guesser in e1.MODELS:
        for lie_count in LIE_COUNTS:
            replicates = 20 + int(lie_count in extra_replicates[guesser.key])
            for replicate in range(replicates):
                turns = tuple(sorted(random.Random(
                    _stable_seed(seed, guesser.key, lie_count, replicate, "lie-turns")
                ).sample(range(1, MAX_TURNS + 1), lie_count)))
                target = balanced_target(
                    targets, replicate, _stable_seed(seed, lie_count, "targets")
                )
                for warned in (False, True):
                    game_id = (
                        f"e3-liar|seed={seed}|guesser={guesser.key}|"
                        f"lies={lie_count}|warned={int(warned)}|rep={replicate:02d}"
                    )
                    schedule.append(GamePlan(
                        experiment="E3_liars_20_questions",
                        game_id=game_id,
                        seed=seed,
                        answerer=answerer,
                        guesser=guesser,
                        target=target,
                        condition={
                            "lie_count": lie_count,
                            "warned": warned,
                            "lie_turns": list(turns),
                            "warning_text": (
                                "Some answers may be changed by the game master; "
                                "you are not told which answers or how many."
                                if warned else None
                            ),
                        },
                        scheduled_lie_turns=turns,
                        randomization_id=(
                            f"e3-liar|seed={seed}|guesser={guesser.key}|"
                            f"lies={lie_count}|rep={replicate:02d}"
                        ),
                    ))
    _validate_unique_schedule(schedule, expected=LIAR_GAME_TOTAL)
    return schedule


def build_noisy_schedule(targets: list[e1.Target], seed: int) -> list[GamePlan]:
    targets = _selected_targets(targets)
    answerer = next(model for model in e1.MODELS if model.key == "qwen3-8b")
    schedule: list[GamePlan] = []
    for guesser in e1.MODELS:
        for probability in NOISE_RATES:
            for replicate in range(NOISY_GAMES_PER_RATE):
                target = balanced_target(
                    targets, replicate, _stable_seed(seed, "noisy-targets")
                )
                game_id = (
                    f"e4-noisy|seed={seed}|guesser={guesser.key}|"
                    f"prob={probability:.2f}|rep={replicate:02d}"
                )
                schedule.append(GamePlan(
                    experiment="E4_noisy_oracle",
                    game_id=game_id,
                    seed=seed,
                    answerer=answerer,
                    guesser=guesser,
                    target=target,
                    condition={
                        "flip_probability": probability,
                        "oracle_model": answerer.key,
                    },
                    randomization_id=(
                        f"e4-noisy|seed={seed}|guesser={guesser.key}|rep={replicate:02d}"
                    ),
                ))
    _validate_unique_schedule(schedule, expected=4 * len(NOISE_RATES) * NOISY_GAMES_PER_RATE)
    return schedule


def _pick_file(results_dir: Path, seed: int) -> Path:
    return results_dir / f"e2-predictable-picks-s{seed}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record must be an object")
                    records.append(record)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise FollowupError(f"Could not read {path}: {error}") from error
    return records


def _validate_picks(picks: list[dict]) -> None:
    expected_ids = set()
    model_keys = {model.key for model in e1.MODELS}
    for model in e1.MODELS:
        for attempt in range(PICK_ATTEMPTS_PER_MODEL):
            expected_ids.add(f"{model.key}|{attempt:03d}")
    seen = set()
    targets_by_id = {target.id: target for target in e1.load_targets()[1]}
    for pick in picks:
        pick_id = pick.get("pick_id")
        if pick_id not in expected_ids or pick_id in seen:
            raise FollowupError(f"Invalid or duplicate pick record: {pick_id!r}.")
        if pick.get("model_key") not in model_keys:
            raise FollowupError(f"Unknown model for pick {pick_id}.")
        model_key, attempt_text = pick_id.split("|", 1)
        try:
            attempt = int(attempt_text)
        except ValueError as error:
            raise FollowupError(f"Invalid attempt number for pick {pick_id}.") from error
        if pick["model_key"] != model_key or pick.get("attempt") != attempt:
            raise FollowupError(f"Model/attempt metadata mismatch for pick {pick_id}.")
        target = targets_by_id.get(pick.get("target_id"))
        expected_category = e1.CATEGORIES[attempt % len(e1.CATEGORIES)]
        if (
            target is None
            or pick.get("category") != expected_category
            or target.category != expected_category
        ):
            raise FollowupError(
                f"Target/category metadata mismatch for pick {pick_id}."
            )
        seen.add(pick_id)


def _parse_pick(text: str, category_targets: list[e1.Target]) -> e1.Target | None:
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if line.lower().startswith("pick:"):
        line = line.split(":", 1)[1].strip()
    normalized = e1.normalize(line)
    for target in category_targets:
        if normalized in {e1.normalize(name) for name in target.accepted_names}:
            return target
    return None


def collect_picks(
    backend: e1.HuggingFaceBackend,
    targets: list[e1.Target],
    path: Path,
    seed: int,
) -> list[dict]:
    existing = read_jsonl(path)
    _validate_picks(existing)
    by_id = {pick["pick_id"]: pick for pick in existing}
    if len(by_id) == len(e1.MODELS) * PICK_ATTEMPTS_PER_MODEL:
        return list(by_id.values())
    category_targets = {
        category: [target for target in targets if target.category == category]
        for category in e1.CATEGORIES
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for model in e1.MODELS:
            missing_attempts = [
                attempt for attempt in range(PICK_ATTEMPTS_PER_MODEL)
                if f"{model.key}|{attempt:03d}" not in by_id
            ]
            if not missing_attempts:
                continue
            backend.prepare_pair(model, model)
            for attempt in missing_attempts:
                pick_id = f"{model.key}|{attempt:03d}"
                category = e1.CATEGORIES[attempt % len(e1.CATEGORIES)]
                choices = category_targets[category]
                names = ", ".join(target.name for target in choices)
                messages = [{
                    "role": "system",
                    "content": (
                        "Choose one concept uniformly at random from the provided list. "
                        "Do not choose based on personal preference. Reply exactly as "
                        "'PICK: <one exact list item>' and nothing else."
                    ),
                }, {
                    "role": "user",
                    "content": f"Choose a {category} from this list:\n{names}",
                }]
                selected = None
                raw_responses = []
                for retry in range(PICK_RETRIES):
                    generation = backend.generate(
                        model,
                        messages,
                        _stable_seed(seed, pick_id, retry),
                    )
                    raw_responses.append({
                        "text": generation.text,
                        "elapsed_seconds": generation.elapsed_seconds,
                        "input_tokens": generation.input_tokens,
                        "output_tokens": generation.output_tokens,
                    })
                    selected = _parse_pick(generation.text, choices)
                    if selected:
                        break
                    if retry + 1 < PICK_RETRIES:
                        messages.extend([
                            {"role": "assistant", "content": generation.text},
                            {
                                "role": "user",
                                "content": "Invalid choice. Reply exactly "
                                "'PICK: <one exact list item>'.",
                            },
                        ])
                if selected is None:
                    raise FollowupError(
                        f"{model.name} failed to choose a valid {category} for {pick_id}; "
                        f"raw replies: {raw_responses!r}"
                    )
                record = {
                    "schema_version": 1,
                    "experiment": "E2_predictable_minds_pick",
                    "pick_id": pick_id,
                    "seed": seed,
                    "model_key": model.key,
                    "model": asdict(model),
                    "attempt": attempt,
                    "category": category,
                    "target_id": selected.id,
                    "target": e1.target_record(selected),
                    "raw_responses": raw_responses,
                    "inference_mode": "transformers",
                    "runtime": backend.metadata(),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                by_id[pick_id] = record
                if len(by_id) % 20 == 0:
                    print(f"Saved {len(by_id)}/400 model secret picks.", flush=True)
    result = list(by_id.values())
    _validate_picks(result)
    return result


def _game_messages(
    plan: GamePlan,
    targets: list[e1.Target],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    allowed = ", ".join(target.name for target in targets)
    answerer_messages = [{
        "role": "system",
        "content": e1.ANSWERER_SYSTEM.format(secret=plan.target.name),
    }]
    guesser_system = (
        "You are playing 20 Questions as the guesser. The secret is exactly one of "
        f"these {len(targets)} concepts: {allowed}. Ask useful yes/no questions and "
        "guess when ready. Each turn, reply with exactly one line:\n"
        "QUESTION: <one yes/no question ending in ?>\n"
        "GUESS: <one exact candidate concept name>\n"
        f"A guess uses a turn. You have at most {MAX_TURNS} turns. Do not add explanations."
    )
    if plan.condition.get("favorite_picks"):
        favorites = ", ".join(plan.condition["favorite_picks"])
        guesser_system += (
            "\n\nScouting report: this opponent has frequently selected these secret "
            f"concepts in prior random-pick trials: {favorites}. This is only a tendency, "
            "not a guarantee about the current secret."
        )
    warning = plan.condition.get("warning_text")
    if warning:
        first_prompt = (
            "I am thinking of one of those concepts. Ask your first question.\n\n"
            f"{warning}"
        )
    else:
        first_prompt = "I am thinking of one of those concepts. Ask your first question."
    guesser_messages = [
        {"role": "system", "content": guesser_system},
        {"role": "user", "content": first_prompt},
    ]
    return answerer_messages, guesser_messages


def _is_flipped(plan: GamePlan, answer_turn: int, rng: random.Random) -> bool:
    if plan.experiment == "E3_liars_20_questions":
        return answer_turn in plan.scheduled_lie_turns
    if plan.experiment == "E4_noisy_oracle":
        return rng.random() < float(plan.condition["flip_probability"])
    return False


def play_followup_game(
    plan: GamePlan,
    targets: list[e1.Target],
    backend: e1.Backend,
) -> dict:
    accepted = {e1.normalize(name) for name in plan.target.accepted_names}
    listed = e1.candidate_names(targets)
    answerer_messages, guesser_messages = _game_messages(plan, targets)
    turns: list[dict] = []
    invalid_answers = 0
    invalid_moves = 0
    answer_turns = 0
    valid_answer_turns = 0
    applied_flips = 0
    won = False
    rng = random.Random(_stable_seed(plan.seed, plan.randomization_id, "noise"))

    for turn_number in range(1, MAX_TURNS + 1):
        raw_moves = []
        parsed_move = None
        for attempt in range(e1.MAX_FORMAT_ATTEMPTS):
            generation = backend.generate(
                plan.guesser,
                guesser_messages,
                e1.game_seed(plan.randomization_id, "guesser", turn_number, attempt),
            )
            raw_moves.append({
                "seed": e1.game_seed(plan.randomization_id, "guesser", turn_number, attempt),
                "text": generation.text,
                "elapsed_seconds": generation.elapsed_seconds,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
            })
            guesser_messages.append({"role": "assistant", "content": generation.text})
            parsed_move = e1.parse_move(generation.text)
            if parsed_move:
                kind, body = parsed_move
                if kind == "GUESS" and e1.normalize(body) not in listed:
                    parsed_move = None
                else:
                    break
            if attempt + 1 < e1.MAX_FORMAT_ATTEMPTS:
                guesser_messages.append({
                    "role": "user",
                    "content": (
                        "Malformed move. Reply with exactly one line: "
                        "QUESTION: <yes/no question ending in ?> or "
                        "GUESS: <one exact candidate concept name>."
                    ),
                })

        if parsed_move is None:
            invalid_moves += 1
            turns.append({
                "turn": turn_number,
                "type": "invalid_move",
                "raw_responses": raw_moves,
                "format_retries": max(0, len(raw_moves) - 1),
            })
            guesser_messages.append({
                "role": "user",
                "content": "Invalid move; this turn is spent. Continue with the next turn.",
            })
            continue

        kind, body = parsed_move
        if kind == "GUESS":
            correct = e1.normalize(body) in accepted
            turns.append({
                "turn": turn_number,
                "type": "guess",
                "text": body,
                "correct": correct,
                "raw_responses": raw_moves,
            })
            if correct:
                won = True
                break
            guesser_messages.append({
                "role": "user",
                "content": f"No, the secret is not {body}. Ask another question or guess.",
            })
            continue

        answer_turns += 1
        answerer_messages.append({"role": "user", "content": body})
        raw_answers = []
        answer = None
        for attempt in range(e1.MAX_FORMAT_ATTEMPTS):
            generation = backend.generate(
                plan.answerer,
                answerer_messages,
                e1.game_seed(plan.randomization_id, "answerer", turn_number, attempt),
            )
            raw_answers.append({
                "seed": e1.game_seed(plan.randomization_id, "answerer", turn_number, attempt),
                "text": generation.text,
                "elapsed_seconds": generation.elapsed_seconds,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
            })
            answerer_messages.append({"role": "assistant", "content": generation.text})
            answer = e1.parse_answer(generation.text)
            if answer:
                break
            if attempt + 1 < e1.MAX_FORMAT_ATTEMPTS:
                answerer_messages.append({
                    "role": "user",
                    "content": "Malformed answer. Reply with exactly one word: Yes or No.",
                })

        if answer is None:
            invalid_answers += 1
            turns.append({
                "turn": turn_number,
                "answer_request": answer_turns,
                "type": "question",
                "text": body,
                "answer": None,
                "answer_status": "invalid",
                "raw_answers": raw_answers,
                "format_retries": max(0, len(raw_answers) - 1),
                "raw_moves": raw_moves,
            })
            guesser_messages.append({
                "role": "user",
                "content": "The answerer gave no valid Yes/No answer; this turn is spent.",
            })
            continue

        truthful_answer = answer
        valid_answer_turns += 1
        flipped = _is_flipped(plan, valid_answer_turns, rng)
        delivered_answer = (
            "No" if truthful_answer == "Yes" else "Yes"
        ) if flipped else truthful_answer
        applied_flips += int(flipped)
        turns.append({
            "turn": turn_number,
            "answer_turn": valid_answer_turns,
            "type": "question",
            "text": body,
            "answer": delivered_answer,
            "truthful_answer": truthful_answer,
            "answer_flipped": flipped,
            "answer_status": "valid",
            "raw_answers": raw_answers,
            "format_retries": max(0, len(raw_answers) - 1),
            # Guesser attempts for this turn, including any rejected retries.
            "raw_moves": raw_moves,
        })
        guesser_messages.append({"role": "user", "content": delivered_answer})

    return {
        "schema_version": 1,
        "experiment": plan.experiment,
        "inference_mode": getattr(backend, "mode", "test"),
        "run_config": {
            **e1.RUN_CONFIG,
            "protocol": plan.experiment,
            "max_turns": MAX_TURNS,
        },
        "runtime": backend.metadata() if hasattr(backend, "metadata") else {"inference_mode": "test"},
        "game_id": plan.game_id,
        "randomization_id": plan.randomization_id,
        "seed": plan.seed,
        "answerer": asdict(plan.answerer),
        "guesser": asdict(plan.guesser),
        "target": e1.target_record(plan.target),
        "condition": plan.condition,
        "source_pick_attempt": plan.source_pick_attempt,
        "scheduled_lie_turns": list(plan.scheduled_lie_turns),
        "applied_flips": applied_flips,
        "won": won,
        "turns_used": len(turns),
        "answer_turns": answer_turns,
        "invalid_answers": invalid_answers,
        "invalid_moves": invalid_moves,
        "turns": turns,
    }


def _read_existing_games(
    path: Path,
    experiment: str,
    schedule: list[GamePlan],
) -> dict[str, dict]:
    records = read_jsonl(path)
    expected = {plan.game_id: plan for plan in schedule}
    by_id = {}
    for record in records:
        game_id = record.get("game_id")
        if game_id not in expected:
            raise FollowupError(f"Unexpected game in {path}: {game_id!r}.")
        if game_id in by_id:
            raise FollowupError(f"Duplicate game ID in {path}: {game_id}.")
        if record.get("experiment") != experiment:
            raise FollowupError(f"Wrong experiment label in {path} for {game_id}.")
        if record.get("inference_mode") != "transformers":
            raise FollowupError(f"Non-model record found in {path} for {game_id}.")
        plan = expected[game_id]
        if (
            record.get("schema_version") != 1
            or record.get("seed") != plan.seed
            or record.get("answerer") != asdict(plan.answerer)
            or record.get("guesser") != asdict(plan.guesser)
            or record.get("target", {}).get("id") != plan.target.id
            or record.get("condition") != plan.condition
            or record.get("scheduled_lie_turns") != list(plan.scheduled_lie_turns)
            or record.get("randomization_id") != plan.randomization_id
        ):
            raise FollowupError(f"Schedule metadata mismatch in {path} for {game_id}.")
        run_config = record.get("run_config", {})
        if (
            run_config.get("guess_rule") != e1.GUESS_RULE_VERSION
            or run_config.get("prompt_version") != e1.PROMPT_VERSION
        ):
            raise FollowupError(
                f"{game_id} in {path} was recorded under a different guess rule or prompt version; move the "
                "old results file aside and start a new one."
            )
        by_id[game_id] = record
    return by_id


def run_schedule(
    backend: e1.HuggingFaceBackend,
    schedule: list[GamePlan],
    targets: list[e1.Target],
    path: Path,
    experiment: str,
) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_existing_games(path, experiment, schedule)
    pending = [plan for plan in schedule if plan.game_id not in completed]
    print(
        f"{experiment}: {len(schedule)} games; {len(completed)} complete; "
        f"{len(pending)} to run.",
        flush=True,
    )
    active_pair = None
    with path.open("a", encoding="utf-8") as output:
        for index, plan in enumerate(pending, start=1):
            pair = (plan.answerer.key, plan.guesser.key)
            if pair != active_pair:
                backend.prepare_pair(plan.answerer, plan.guesser)
                active_pair = pair
            record = play_followup_game(plan, targets, backend)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            completed[plan.game_id] = record
            if index % 20 == 0 or index == len(pending):
                print(
                    f"Saved game {len(completed)}/{len(schedule)} "
                    f"({plan.answerer.key} vs {plan.guesser.key}).",
                    flush=True,
                )
    return [completed[plan.game_id] for plan in schedule]


def summarize_games(records: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        condition = record["condition"]
        if record["experiment"] == "E2_predictable_minds":
            key = ("answerer", record["answerer"]["key"], "guesser", record["guesser"]["key"])
        elif record["experiment"] == "E2_self_secret_scouting":
            key = ("model", record["guesser"]["key"], "scouting", condition["scouting_report"])
        elif record["experiment"] == "E3_liars_20_questions":
            key = (
                "guesser", record["guesser"]["key"],
                "lies", condition["lie_count"],
                "warned", condition["warned"],
            )
        else:
            key = (
                "guesser", record["guesser"]["key"],
                "flip_probability", condition["flip_probability"],
            )
        groups[key].append(record)
    def wilson_interval(wins: int, total: int) -> list[float]:
        if total == 0:
            return [0.0, 0.0]
        z = 1.959963984540054
        rate = wins / total
        denominator = 1 + z * z / total
        center = (rate + z * z / (2 * total)) / denominator
        margin = (
            z
            * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
            / denominator
        )
        return [max(0.0, center - margin), min(1.0, center + margin)]

    return {
        "schema_version": 1,
        "experiment": records[0]["experiment"] if records else None,
        "games": len(records),
        "groups": [
            {
                "condition": dict(zip(key[::2], key[1::2])),
                "games": len(group),
                "win_rate": statistics.mean(float(record["won"]) for record in group),
                "win_rate_wilson_95_ci": wilson_interval(
                    sum(bool(record["won"]) for record in group), len(group)
                ),
                "mean_turns": statistics.mean(record["turns_used"] for record in group),
                "mean_answer_turns": statistics.mean(
                    record["answer_turns"] for record in group
                ),
                "mean_applied_flips": statistics.mean(
                    record["applied_flips"] for record in group
                ),
                "flipped_answer_rate": (
                    sum(record["applied_flips"] for record in group)
                    / sum(
                        turn.get("answer_status") == "valid"
                        for record in group
                        for turn in record["turns"]
                    )
                    if sum(
                        turn.get("answer_status") == "valid"
                        for record in group
                        for turn in record["turns"]
                    )
                    else 0.0
                ),
            }
            for key, group in sorted(groups.items(), key=lambda item: repr(item[0]))
        ],
    }


def _write_summary(path: Path, summary: dict) -> Path:
    destination = ROOT / "reports" / path.with_suffix(".summary.json").name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values() if count
    )


def summarize_picks(picks: list[dict], seed: int, replicates: int = 2000) -> dict:
    all_targets = e1.load_targets()[1]
    model_stats = []
    baseline_rng = random.Random(_stable_seed(seed, "uniform-pick-baseline"))
    baseline_targets = [target.id for target in all_targets]
    for model in e1.MODELS:
        model_picks = [pick["target_id"] for pick in picks if pick["model_key"] == model.key]
        counts = Counter(model_picks)
        observed_entropy = _entropy(counts)
        lower_tail = 0
        simulated_entropies = []
        for _ in range(replicates):
            baseline_counts = Counter(
                baseline_rng.choice(baseline_targets) for _ in range(PICK_ATTEMPTS_PER_MODEL)
            )
            entropy = _entropy(baseline_counts)
            simulated_entropies.append(entropy)
            lower_tail += entropy <= observed_entropy
        model_stats.append({
            "model": model.key,
            "draws": len(model_picks),
            "distinct_picks": len(counts),
            "shannon_entropy_bits": observed_entropy,
            "maximum_entropy_bits_for_128_targets": math.log2(len(all_targets)),
            "top_picks": [
                {"target_id": target_id, "count": count}
                for target_id, count in counts.most_common(10)
            ],
            "uniform_baseline_mean_entropy_bits": statistics.mean(simulated_entropies),
            "uniform_baseline_entropy_2_5_97_5": [
                sorted(simulated_entropies)[int(0.025 * (replicates - 1))],
                sorted(simulated_entropies)[int(0.975 * (replicates - 1))],
            ],
            "uniform_baseline_lower_tail_fraction": lower_tail / replicates,
        })
    return {
        "schema_version": 1,
        "experiment": "E2_predictable_minds_pick_distribution",
        "games": len(picks),
        "uniform_baseline": (
            "Monte Carlo uniform draws from the fixed 128-target manifest; "
            "the small 100-draw sample is descriptive and not a high-powered "
            "test of uniformity."
        ),
        "models": model_stats,
    }


def _output_path(results_dir: Path, experiment: str, seed: int) -> Path:
    filenames = {
        "predictable": f"e2-predictable-games-s{seed}.jsonl",
        "scouting": f"e2-scouting-s{seed}.jsonl",
        "liar": f"e3-liar-s{seed}.jsonl",
        "noisy": f"e4-noisy-s{seed}.jsonl",
    }
    return results_dir / filenames[experiment]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("predictable", "scouting", "liar", "noisy"),
        required=True,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=519)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _, targets = e1.load_targets()
        targets = _selected_targets(targets)
        backend = e1.HuggingFaceBackend()
        try:
            if args.experiment in ("predictable", "scouting"):
                pick_path = _pick_file(args.results_dir, args.seed)
                picks = collect_picks(backend, targets, pick_path, args.seed)
                if args.experiment == "predictable":
                    pick_summary = summarize_picks(picks, args.seed)
                    saved_picks_summary = _write_summary(pick_path, pick_summary)
                    print(f"Pick distribution summary: {saved_picks_summary}", flush=True)
                    schedule = build_predictable_schedule(targets, picks, args.seed)
                    experiment = "E2_predictable_minds"
                else:
                    schedule = build_scouting_schedule(targets, picks, args.seed)
                    experiment = "E2_self_secret_scouting"
            elif args.experiment == "liar":
                schedule = build_liar_schedule(targets, args.seed)
                experiment = "E3_liars_20_questions"
            else:
                schedule = build_noisy_schedule(targets, args.seed)
                experiment = "E4_noisy_oracle"

            path = _output_path(args.results_dir, args.experiment, args.seed)
            records = run_schedule(backend, schedule, targets, path, experiment)
            summary = summarize_games(records)
            summary_file = _write_summary(path, summary)
            print(f"Game results: {path}", flush=True)
            print(f"Summary: {summary_file}", flush=True)
        finally:
            backend.close()
        return 0
    except (e1.ExperimentError, FollowupError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

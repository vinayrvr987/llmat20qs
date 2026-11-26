#!/usr/bin/env python3
"""Run and analyze the assigned-secret E1 20 Questions round robin."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from src.data.paths import PROJECT_ROOT, RESULTS_DIR, TARGET_MANIFEST
from src.models.registry import MODELS, ModelConfig

ROOT = PROJECT_ROOT
DEFAULT_TARGETS = TARGET_MANIFEST
DEFAULT_RESULTS = RESULTS_DIR / "e1_round_robin.jsonl"
MAX_TURNS = 20
MAX_FORMAT_ATTEMPTS = 3
BOOTSTRAP_REPLICATES = 2000
PERMUTATION_REPLICATES = 10000
CATEGORIES = ("object", "animal", "place", "person")
# Prompts are unchanged from the first run. A turns-left reminder (v2) was tried
# and dropped: it made Gemma delay guessing until the final turn.
PROMPT_VERSION = "e1-prompts-v1"
GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 128,
}


# v2: a guess naming any listed candidate is a scored move that spends a turn.
# v1 (buggy) silently retried wrong guesses as format errors, giving free guesses.
GUESS_RULE_VERSION = "listed-candidate-guess-spends-turn-v2"

RUN_CONFIG = {
    "prompt_version": PROMPT_VERSION,
    "guess_rule": GUESS_RULE_VERSION,
    "model_dtypes": {model.key: model.inference_dtype for model in MODELS},
    "max_turns": MAX_TURNS,
    "max_format_attempts": MAX_FORMAT_ATTEMPTS,
    **GENERATION_CONFIG,
}

ANSWERER_SYSTEM = (
    "You are playing 20 Questions as the answerer. The secret concept is {secret!r}. "
    "Answer each question truthfully according to its ordinary meaning. Reply with exactly "
    "one word: Yes or No. Do not reveal or hint at the secret."
)


class ExperimentError(RuntimeError):
    """An expected experiment, input, or model-execution failure."""


@dataclass(frozen=True)
class Target:
    id: str
    name: str
    aliases: tuple[str, ...]
    category: str
    frequency: str
    e1: bool

    @property
    def accepted_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    seed: int
    target_pool: str
    answerer: ModelConfig
    guesser: ModelConfig
    target: Target


@dataclass(frozen=True)
class Generation:
    text: str
    elapsed_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


THINK_END_TOKEN = "</think>"
FORCED_THINK_CLOSE = "\n</think>\n\n"


@dataclass(frozen=True)
class ThinkingGeneration(Generation):
    """A budgeted thinking-mode generation; ``text`` holds only the final answer."""

    thinking_tokens: int = 0
    answer_tokens: int = 0
    budget_exhausted: bool = False


@dataclass(frozen=True)
class ThinkingStep:
    action: str  # "done", "continue_answer", or "force_close"
    thinking_tokens: int


def plan_thinking_continuation(
    phase_one_ids: list[int],
    end_think_id: int,
    eos_ids: set[int],
    thinking_budget: int,
) -> ThinkingStep:
    """Decide how to finish a budgeted generation after the thinking phase.

    ``thinking_tokens`` counts the reasoning tokens up to and including the
    closing tag (or all phase-one tokens when the tag is forced).
    """
    if end_think_id in phase_one_ids:
        thinking_tokens = phase_one_ids.index(end_think_id) + 1
        finished = bool(phase_one_ids) and (
            phase_one_ids[-1] in eos_ids or len(phase_one_ids) < thinking_budget
        )
        return ThinkingStep("done" if finished else "continue_answer", thinking_tokens)
    return ThinkingStep("force_close", len(phase_one_ids))


class Backend(Protocol):
    def prepare_pair(self, answerer: ModelConfig, guesser: ModelConfig) -> None: ...

    def generate(
        self, model: ModelConfig, messages: list[dict[str, str]], seed: int
    ) -> Generation: ...

    def close(self) -> None: ...

    def metadata(self) -> dict: ...


class MockBackend:
    """Deterministic mock backend for schedule and protocol tests only."""

    mode = "mock"

    def prepare_pair(self, answerer: ModelConfig, guesser: ModelConfig) -> None:
        return None

    def generate(
        self, model: ModelConfig, messages: list[dict[str, str]], seed: int
    ) -> Generation:
        del model, seed
        start = time.perf_counter()
        if messages[0]["content"].startswith("You are playing 20 Questions as the answerer"):
            text = "Yes"
        else:
            text = "QUESTION: Is it alive?"
        return Generation(text=text, elapsed_seconds=time.perf_counter() - start)

    def close(self) -> None:
        return None

    def metadata(self) -> dict:
        return {"inference_mode": self.mode}


class HuggingFaceBackend:
    """Load at most the two active players onto separate allocated GPUs."""

    mode = "transformers"

    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ExperimentError(
                "Real inference requires torch, transformers, and accelerate. "
                "Load the AI3 pytorch-cuda12 module and install requirements.txt."
            ) from error

        if not torch.cuda.is_available():
            raise ExperimentError(
                "CUDA is not available. Run real inference inside a GPU Slurm allocation."
            )
        self.torch = torch
        self.model_class = AutoModelForCausalLM
        self.tokenizer_class = AutoTokenizer
        self.loaded: dict[str, tuple[object, object]] = {}
        self.devices: dict[str, int] = {}
        self.active_pair: frozenset[str] = frozenset()
        import transformers

        self.transformers_version = transformers.__version__

    def prepare_pair(self, answerer: ModelConfig, guesser: ModelConfig) -> None:
        configs = {model.key: model for model in (answerer, guesser)}
        needed = frozenset(configs)
        if needed == self.active_pair and needed <= self.loaded.keys():
            return
        self._release_models()

        if len(needed) > self.torch.cuda.device_count():
            raise ExperimentError(
                f"This pairing needs {len(needed)} GPU(s), but Slurm exposed "
                f"{self.torch.cuda.device_count()}. Request one GPU per distinct player model."
            )

        for config in configs.values():
            try:
                available = [
                    index for index in range(self.torch.cuda.device_count())
                    if index not in self.devices.values()
                ]
                device = max(
                    available,
                    key=lambda index: self.torch.cuda.mem_get_info(index)[0],
                )
                tokenizer = self.tokenizer_class.from_pretrained(
                    config.model_id, revision=config.revision
                )
                model = self.model_class.from_pretrained(
                    config.model_id,
                    revision=config.revision,
                    dtype=getattr(self.torch, config.inference_dtype),
                    device_map={"": device},
                    low_cpu_mem_usage=True,
                )
            except Exception as error:
                hint = (
                    " Accept Google's terms for Gemma 3 and authenticate with "
                    "`hf auth login` or HF_TOKEN."
                    if config.model_id == "google/gemma-3-12b-it" else ""
                )
                raise ExperimentError(
                    f"Could not load {config.model_id}@{config.revision}: "
                    f"{type(error).__name__}: {error}.{hint}"
                ) from error
            self.loaded[config.key] = (tokenizer, model)
            self.devices[config.key] = device
            allocated = self.torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(
                f"Loaded {config.name} on CUDA:{device} "
                f"({allocated:.1f} GiB allocated).",
                flush=True,
            )
        self.active_pair = needed

    def generate(
        self, model: ModelConfig, messages: list[dict[str, str]], seed: int
    ) -> Generation:
        return self.generate_with_options(
            model,
            messages,
            seed,
            enable_thinking=False,
            max_new_tokens=None,
        )

    def generate_with_options(
        self,
        model: ModelConfig,
        messages: list[dict[str, str]],
        seed: int,
        *,
        enable_thinking: bool,
        max_new_tokens: int | None,
    ) -> Generation:
        if model.key not in self.loaded:
            raise ExperimentError(f"{model.name} was not loaded for this pairing.")
        tokenizer, language_model = self.loaded[model.key]
        try:
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError as error:
                if enable_thinking:
                    raise ExperimentError(
                        f"{model.name}'s tokenizer does not support enabled thinking."
                    ) from error
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            inputs = inputs.to(language_model.device)
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)
            start = time.perf_counter()
            generation_config = dict(GENERATION_CONFIG)
            if max_new_tokens is not None:
                if max_new_tokens < 1:
                    raise ExperimentError("max_new_tokens must be positive.")
                generation_config["max_new_tokens"] = max_new_tokens
            with self.torch.inference_mode():
                output = language_model.generate(
                    **inputs,
                    **generation_config,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - start
            prompt_tokens = inputs["input_ids"].shape[-1]
            text = tokenizer.decode(
                output[0][prompt_tokens:], skip_special_tokens=True
            ).strip()
            return Generation(
                text=text,
                elapsed_seconds=elapsed,
                input_tokens=int(prompt_tokens),
                output_tokens=int(output.shape[-1] - prompt_tokens),
            )
        except ExperimentError:
            raise
        except Exception as error:
            raise ExperimentError(
                f"Generation failed for {model.model_id}: "
                f"{type(error).__name__}: {error}"
            ) from error

    def generate_with_thinking_budget(
        self,
        model: ModelConfig,
        messages: list[dict[str, str]],
        seed: int,
        *,
        thinking_budget: int,
        answer_max_tokens: int,
    ) -> ThinkingGeneration:
        """Think for at most ``thinking_budget`` tokens, then force a final answer.

        Phase 1 samples up to the budget with thinking enabled. If the model has not
        closed its ``</think>`` block, the controller appends one and phase 2 samples
        at most ``answer_max_tokens`` answer tokens. Only the post-reasoning answer is
        returned as ``text``; the reasoning trace itself is discarded.
        """
        if thinking_budget < 1 or answer_max_tokens < 1:
            raise ExperimentError("Thinking budget and answer cap must be positive.")
        if model.key not in self.loaded:
            raise ExperimentError(f"{model.name} was not loaded for this pairing.")
        tokenizer, language_model = self.loaded[model.key]
        end_think_id = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
        if end_think_id is None or end_think_id == tokenizer.unk_token_id:
            raise ExperimentError(f"{model.name}'s tokenizer has no {THINK_END_TOKEN} token.")
        eos_ids = language_model.generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        try:
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    enable_thinking=True,
                )
            except TypeError as error:
                raise ExperimentError(
                    f"{model.name}'s tokenizer does not support enabled thinking."
                ) from error
            inputs = inputs.to(language_model.device)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)
            start = time.perf_counter()
            with self.torch.inference_mode():
                output = language_model.generate(
                    **inputs,
                    **{**GENERATION_CONFIG, "max_new_tokens": thinking_budget},
                    pad_token_id=tokenizer.eos_token_id,
                )
                phase_one = output[0][prompt_tokens:].tolist()
                step = plan_thinking_continuation(
                    phase_one, end_think_id, eos_ids, thinking_budget
                )
                if step.action != "done":
                    sequence = output
                    if step.action == "force_close":
                        if phase_one and phase_one[-1] in eos_ids:
                            sequence = sequence[:, :-1]
                        closing = tokenizer.encode(
                            FORCED_THINK_CLOSE, add_special_tokens=False, return_tensors="pt"
                        ).to(output.device)
                        sequence = self.torch.cat([sequence, closing], dim=-1)
                    output = language_model.generate(
                        input_ids=sequence,
                        attention_mask=self.torch.ones_like(sequence),
                        **{**GENERATION_CONFIG, "max_new_tokens": answer_max_tokens},
                        pad_token_id=tokenizer.eos_token_id,
                    )
            elapsed = time.perf_counter() - start
            generated = output[0][prompt_tokens:].tolist()
            answer_ids = generated[generated.index(end_think_id) + 1:]
            answer_ids = [token for token in answer_ids if token not in eos_ids]
            return ThinkingGeneration(
                text=tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
                elapsed_seconds=elapsed,
                input_tokens=prompt_tokens,
                output_tokens=len(generated),
                thinking_tokens=step.thinking_tokens,
                answer_tokens=len(answer_ids),
                budget_exhausted=step.action == "force_close",
            )
        except ExperimentError:
            raise
        except Exception as error:
            raise ExperimentError(
                f"Generation failed for {model.model_id}: "
                f"{type(error).__name__}: {error}"
            ) from error

    def _release_models(self) -> None:
        if not self.loaded:
            self.active_pair = frozenset()
            self.devices.clear()
            return
        self.loaded.clear()
        self.devices.clear()
        self.active_pair = frozenset()
        import gc

        gc.collect()
        self.torch.cuda.empty_cache()

    def close(self) -> None:
        self._release_models()

    def metadata(self) -> dict:
        return {
            "inference_mode": self.mode,
            "python_version": sys.version.split()[0],
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers_version,
            "cuda_version": self.torch.version.cuda,
            "devices": {
                key: {
                    "cuda_index": index,
                    "name": self.torch.cuda.get_device_name(index),
                    "inference_dtype": next(
                        model.inference_dtype for model in MODELS if model.key == key
                    ),
                }
                for key, index in self.devices.items()
            },
        }


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.lower()))


def load_targets(path: Path = DEFAULT_TARGETS) -> tuple[str, list[Target]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentError(f"Could not read target manifest {path}: {error}") from error

    if not isinstance(document, dict):
        raise ExperimentError(f"Target manifest {path} must contain a JSON object.")
    if document.get("schema_version") != 1:
        raise ExperimentError(f"Unsupported target manifest schema in {path}.")
    pool_id = document.get("pool_id")
    raw_targets = document.get("targets")
    if not isinstance(pool_id, str) or not isinstance(raw_targets, list):
        raise ExperimentError(f"Target manifest {path} is missing pool_id or targets.")
    provenance = document.get("provenance_by_category")
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(category), str) or not provenance[category]
        for category in CATEGORIES
    ):
        raise ExperimentError(f"Target manifest {path} is missing category provenance.")

    targets: list[Target] = []
    seen_ids: set[str] = set()
    seen_names: dict[str, str] = {}
    for raw in raw_targets:
        if not isinstance(raw, dict):
            raise ExperimentError("Each target manifest entry must be an object.")
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ExperimentError(f"Invalid aliases for target {raw.get('id')!r}.")
        try:
            target = Target(
                id=raw["id"],
                name=raw["name"],
                aliases=tuple(aliases),
                category=raw["category"],
                frequency=raw["frequency"],
                e1=raw["e1"],
            )
        except KeyError as error:
            raise ExperimentError(f"Target entry is missing {error.args[0]!r}.") from error
        if not all(isinstance(value, str) and value for value in (
            target.id, target.name, target.category, target.frequency
        )):
            raise ExperimentError(f"Invalid target fields for {target.id!r}.")
        if target.id in seen_ids:
            raise ExperimentError(f"Duplicate target ID: {target.id}.")
        if target.category not in CATEGORIES or target.frequency not in ("common", "rare"):
            raise ExperimentError(f"Invalid category or frequency label for {target.id}.")
        if not isinstance(target.e1, bool):
            raise ExperimentError(f"Target {target.id} has a non-boolean e1 flag.")
        seen_ids.add(target.id)
        for name in target.accepted_names:
            key = normalize(name)
            if not key:
                raise ExperimentError(f"Target {target.id} has an empty name or alias.")
            if key in seen_names and seen_names[key] != target.id:
                raise ExperimentError(
                    f"Name or alias {name!r} collides between {seen_names[key]} and {target.id}."
                )
            seen_names[key] = target.id
        targets.append(target)

    validate_target_pool(targets)
    return pool_id, targets


def validate_target_pool(targets: list[Target]) -> None:
    if len(targets) != 128:
        raise ExperimentError(f"Expected 128 pool concepts, found {len(targets)}.")
    for category in CATEGORIES:
        category_targets = [target for target in targets if target.category == category]
        if len(category_targets) != 32:
            raise ExperimentError(
                f"Expected 32 {category} concepts, found {len(category_targets)}."
            )
        for frequency in ("common", "rare"):
            stratum = [target for target in category_targets if target.frequency == frequency]
            if len(stratum) != 16:
                raise ExperimentError(
                    f"Expected 16 {frequency} {category} concepts, found {len(stratum)}."
                )
            selected = [target for target in stratum if target.e1]
            if len(selected) != 5:
                raise ExperimentError(
                    f"E1 requires 5 selected {frequency} {category} targets, "
                    f"found {len(selected)}."
                )
    if sum(target.e1 for target in targets) != 40:
        raise ExperimentError("E1 target selection must contain exactly 40 concepts.")


def build_schedule(
    targets: list[Target],
    seed: int,
    pool_id: str = "twenty-questions-targets-v1",
) -> list[GameSpec]:
    selected = [target for target in targets if target.e1]
    if len(selected) != 40:
        raise ExperimentError(f"Expected 40 E1 targets, found {len(selected)}.")
    selected.sort(
        key=lambda target: hashlib.sha256(f"{seed}|{target.id}".encode("utf-8")).digest()
    )

    schedule: list[GameSpec] = []
    for answerer in MODELS:
        for guesser in MODELS:
            for target in selected:
                game_id = (
                    f"e1|pool={pool_id}|seed={seed}|answerer={answerer.key}|"
                    f"guesser={guesser.key}|target={target.id}"
                )
                schedule.append(GameSpec(game_id, seed, pool_id, answerer, guesser, target))
    validate_schedule(schedule)
    return schedule


def validate_schedule(schedule: list[GameSpec]) -> None:
    if len({game.game_id for game in schedule}) != len(schedule):
        raise ExperimentError("Schedule contains duplicate game IDs.")
    pairings = {(game.answerer.key, game.guesser.key) for game in schedule}
    if len(pairings) != 16:
        raise ExperimentError(f"Expected 16 directed pairings, found {len(pairings)}.")
    for answerer in MODELS:
        for guesser in MODELS:
            games = [
                game for game in schedule
                if game.answerer.key == answerer.key and game.guesser.key == guesser.key
            ]
            if (answerer.key, guesser.key) not in pairings:
                continue
            if len(games) != 40 or len({game.target.id for game in games}) != 40:
                raise ExperimentError(
                    f"Pairing {answerer.key}/{guesser.key} must cover the same 40 unique targets."
                )


def select_pairings(schedule: list[GameSpec], group: str) -> list[GameSpec]:
    if group == "all":
        return schedule
    if group == "self-play":
        selected = [game for game in schedule if game.answerer.key == game.guesser.key]
    elif group == "cross-play":
        selected = [game for game in schedule if game.answerer.key != game.guesser.key]
    else:
        raise ExperimentError(f"Unknown pairing group: {group}.")
    expected = 160 if group == "self-play" else 480
    if len(selected) != expected:
        raise ExperimentError(f"Expected {expected} {group} games, found {len(selected)}.")
    return selected


def render_guesser_system(targets: list[Target]) -> str:
    names = ", ".join(target.name for target in targets if target.e1)
    return (
        "You are playing 20 Questions as the guesser. The secret is exactly one of these "
        f"40 concepts: {names}. Ask useful yes/no questions and make a guess when ready. "
        "Each turn, reply with exactly one line in one of these formats:\n"
        "QUESTION: <one yes/no question ending in ?>\n"
        "GUESS: <one candidate concept name>\n"
        "A guess uses up a turn. You have at most 20 turns. Do not add explanations."
    )


def candidate_names(targets: list[Target]) -> set[str]:
    """Normalized accepted names of every candidate a guesser may name."""
    return {normalize(name) for target in targets for name in target.accepted_names}


def parse_answer(text: str) -> str | None:
    match = re.match(r"^\s*(yes|no)\b", text, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def parse_move(text: str) -> tuple[str, str] | None:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    match = re.fullmatch(r"(QUESTION|GUESS)\s*:\s*(.+)", first_line, flags=re.IGNORECASE)
    if not match:
        return None
    kind, body = match.group(1).upper(), match.group(2).strip()
    if kind == "QUESTION":
        if len(re.findall(r"[A-Za-z0-9]+", body)) < 3 or not body.endswith("?"):
            return None
    elif not normalize(body):
        return None
    return kind, body


def game_seed(game_id: str, role: str, turn: int, attempt: int) -> int:
    payload = f"{game_id}|{role}|{turn}|{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


def target_record(target: Target) -> dict:
    record = asdict(target)
    record["aliases"] = list(target.aliases)
    return record


def play_game(
    spec: GameSpec,
    targets: list[Target],
    backend: Backend,
) -> dict:
    accepted = {normalize(name) for name in spec.target.accepted_names}
    listed = candidate_names([target for target in targets if target.e1])
    answerer_messages = [{
        "role": "system",
        "content": ANSWERER_SYSTEM.format(secret=spec.target.name),
    }]
    guesser_messages = [
        {"role": "system", "content": render_guesser_system(targets)},
        {"role": "user", "content": "I am thinking of one of those concepts. Ask your first question."},
    ]
    turns: list[dict] = []
    invalid_answers = 0
    invalid_moves = 0
    answer_turns = 0
    won = False

    for turn_number in range(1, MAX_TURNS + 1):
        raw_moves: list[dict] = []
        parsed_move: tuple[str, str] | None = None
        for attempt in range(MAX_FORMAT_ATTEMPTS):
            generation_seed = game_seed(spec.game_id, "guesser", turn_number, attempt)
            generation = backend.generate(
                spec.guesser,
                guesser_messages,
                generation_seed,
            )
            raw_moves.append({
                "seed": generation_seed,
                "text": generation.text,
                "elapsed_seconds": generation.elapsed_seconds,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
            })
            guesser_messages.append({"role": "assistant", "content": generation.text})
            parsed_move = parse_move(generation.text)
            if parsed_move:
                kind, body = parsed_move
                if kind == "GUESS" and normalize(body) not in listed:
                    parsed_move = None
                else:
                    break
            if attempt + 1 < MAX_FORMAT_ATTEMPTS:
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
            correct = normalize(body) in accepted
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
        raw_answers: list[dict] = []
        answer: str | None = None
        for attempt in range(MAX_FORMAT_ATTEMPTS):
            generation_seed = game_seed(spec.game_id, "answerer", turn_number, attempt)
            generation = backend.generate(
                spec.answerer,
                answerer_messages,
                generation_seed,
            )
            raw_answers.append({
                "seed": generation_seed,
                "text": generation.text,
                "elapsed_seconds": generation.elapsed_seconds,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
            })
            answerer_messages.append({"role": "assistant", "content": generation.text})
            answer = parse_answer(generation.text)
            if answer:
                break
            if attempt + 1 < MAX_FORMAT_ATTEMPTS:
                answerer_messages.append({
                    "role": "user",
                    "content": "Malformed answer. Reply with exactly one word: Yes or No.",
                })

        if answer is None:
            invalid_answers += 1
            turns.append({
                "turn": turn_number,
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

        turns.append({
            "turn": turn_number,
            "type": "question",
            "text": body,
            "answer": answer,
            "answer_status": "valid",
            "raw_answers": raw_answers,
            "format_retries": max(0, len(raw_answers) - 1),
            # Guesser attempts for this turn, including any rejected retries.
            "raw_moves": raw_moves,
        })
        guesser_messages.append({"role": "user", "content": answer})

    return {
        "schema_version": 1,
        "experiment": "E1",
        "inference_mode": getattr(backend, "mode", "test"),
        "run_config": RUN_CONFIG,
        "runtime": backend.metadata() if hasattr(backend, "metadata") else {"inference_mode": "test"},
        "game_id": spec.game_id,
        "seed": spec.seed,
        "target_pool": spec.target_pool,
        "answerer": asdict(spec.answerer),
        "guesser": asdict(spec.guesser),
        "target": asdict(spec.target),
        "won": won,
        "turns_used": len(turns),
        "answer_turns": answer_turns,
        "invalid_answers": invalid_answers,
        "invalid_moves": invalid_moves,
        "turns": turns,
    }


def _record_is_for_spec(record: dict, spec: GameSpec) -> bool:
    expected = {
        "game_id": spec.game_id,
        "experiment": "E1",
        "run_config": RUN_CONFIG,
        "seed": spec.seed,
        "answerer": asdict(spec.answerer),
        "guesser": asdict(spec.guesser),
        "target_pool": spec.target_pool,
        "target": target_record(spec.target),
    }
    return all(record.get(key) == value for key, value in expected.items())


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ExperimentError(
                        f"Invalid JSON at {path}:{line_number}: {error}. "
                        "Preserve the file and repair or choose a fresh output path."
                    ) from error
                if not isinstance(record, dict):
                    raise ExperimentError(f"Expected a JSON object at {path}:{line_number}.")
                records.append(record)
    except OSError as error:
        raise ExperimentError(f"Could not read results {path}: {error}") from error
    return records


def validate_records(
    records: list[dict],
    schedule: list[GameSpec],
    require_complete: bool,
) -> None:
    specs = {spec.game_id: spec for spec in schedule}
    seen: set[str] = set()
    inference_modes: set[str] = set()
    for record in records:
        game_id = record.get("game_id")
        if not isinstance(game_id, str):
            raise ExperimentError("Every results row must have a string game_id.")
        if game_id not in specs:
            raise ExperimentError(f"Unexpected game_id in results: {game_id!r}.")
        if game_id in seen:
            raise ExperimentError(f"Duplicate game_id in results: {game_id}.")
        if not _record_is_for_spec(record, specs[game_id]):
            raise ExperimentError(f"Model, seed, or target metadata mismatch for {game_id}.")
        if record.get("schema_version") != 1 or record.get("experiment") != "E1":
            raise ExperimentError(f"Unsupported result schema for {game_id}.")
        run_config = record.get("run_config", {})
        if (
            run_config.get("guess_rule") != GUESS_RULE_VERSION
            or run_config.get("prompt_version") != PROMPT_VERSION
        ):
            raise ExperimentError(
                f"{game_id} was recorded under a different guess rule or prompt version; move the old results "
                "file aside and start a new one."
            )
        inference_mode = record.get("inference_mode")
        if inference_mode not in ("mock", "transformers", "test"):
            raise ExperimentError(f"Missing or unsupported inference mode for {game_id}.")
        inference_modes.add(inference_mode)
        if not isinstance(record.get("won"), bool):
            raise ExperimentError(f"Missing boolean game outcome for {game_id}.")
        turns = record.get("turns")
        turns_used = record.get("turns_used")
        if (
            not isinstance(turns, list)
            or not isinstance(turns_used, int)
            or not 1 <= turns_used <= MAX_TURNS
        ):
            raise ExperimentError(f"Invalid turn log for {game_id}.")
        if len(turns) != turns_used:
            raise ExperimentError(f"Turn count does not match turn log for {game_id}.")
        seen.add(game_id)

    if len(inference_modes) > 1:
        raise ExperimentError("Results mix mock, test, or Transformers inference records.")
    if require_complete and seen != set(specs):
        missing = len(set(specs) - seen)
        raise ExperimentError(
            f"Incomplete E1 results: {len(seen)}/{len(specs)} games, {missing} missing."
        )


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ExperimentError("Cannot calculate a percentile from no observations.")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def cluster_bootstrap_ci(
    records: list[dict],
    group_key,
    outcome,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        clusters[record["target"]["id"]].append(record)
    cluster_values = list(clusters.values())
    if not cluster_values:
        raise ExperimentError("Cannot calculate a confidence interval with no records.")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sample = [rng.choice(cluster_values) for _ in cluster_values]
        selected = [
            record
            for cluster in sample
            for record in cluster
            if group_key(record)
        ]
        if selected:
            estimates.append(sum(outcome(record) for record in selected) / len(selected))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_model_comparisons(
    records: list[dict],
    role: str,
    seed: int,
) -> list[dict]:
    if role not in ("guesser", "answerer"):
        raise ExperimentError(f"Unknown comparison role: {role}.")
    model_field = role
    targets = sorted({record["target"]["id"] for record in records})
    scores: dict[str, dict[str, float]] = {}
    for model in MODELS:
        scores[model.key] = {}
        for target_id in targets:
            matched = [
                record for record in records
                if record[model_field]["key"] == model.key
                and record["target"]["id"] == target_id
            ]
            if not matched:
                raise ExperimentError(
                    f"Missing {role} observations for {model.key} on {target_id}."
                )
            outcomes = [
                record["won"] if role == "guesser" else not record["won"]
                for record in matched
            ]
            scores[model.key][target_id] = sum(outcomes) / len(outcomes)

    comparisons: list[dict] = []
    for first_index, first in enumerate(MODELS):
        for second in MODELS[first_index + 1:]:
            differences = [
                scores[first.key][target_id] - scores[second.key][target_id]
                for target_id in targets
            ]
            mean_difference = statistics.mean(differences)
            rng = random.Random(seed ^ (len(comparisons) + 1))
            bootstrap = sorted(
                statistics.mean(rng.choice(differences) for _ in differences)
                for _ in range(BOOTSTRAP_REPLICATES)
            )
            ci = [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]

            permutation_rng = random.Random(seed ^ (1000 + len(comparisons)))
            observed = abs(mean_difference)
            extreme = 0
            for _ in range(PERMUTATION_REPLICATES):
                permuted = abs(
                    sum(value if permutation_rng.getrandbits(1) else -value
                        for value in differences) / len(differences)
                )
                if permuted >= observed:
                    extreme += 1
            p_value = (extreme + 1) / (PERMUTATION_REPLICATES + 1)
            comparisons.append({
                "model_a": first.key,
                "model_b": second.key,
                "mean_rate_difference_a_minus_b": mean_difference,
                "paired_target_bootstrap_95_ci": ci,
                "permutation_p_value": p_value,
            })

    ordered = sorted(
        enumerate(comparisons),
        key=lambda item: item[1]["permutation_p_value"],
    )
    adjusted_so_far = 0.0
    total = len(ordered)
    for rank, (original_index, comparison) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * comparison["permutation_p_value"])
        adjusted_so_far = max(adjusted_so_far, adjusted)
        comparisons[original_index]["holm_adjusted_p_value"] = adjusted_so_far
    return comparisons


def summarize_results(records: list[dict], seed: int) -> dict:
    if not records:
        raise ExperimentError("No completed game records to summarize.")
    if any(record.get("inference_mode") != "transformers" for record in records):
        raise ExperimentError("Mock or test records cannot be summarized as experimental results.")

    pairings: list[dict] = []
    for answerer in MODELS:
        for guesser in MODELS:
            subset = [
                record for record in records
                if record["answerer"]["key"] == answerer.key
                and record["guesser"]["key"] == guesser.key
            ]
            if not subset:
                continue
            rate = sum(record["won"] for record in subset) / len(subset)
            low, high = cluster_bootstrap_ci(
                subset,
                lambda record: record["answerer"]["key"] == answerer.key
                and record["guesser"]["key"] == guesser.key,
                lambda record: float(record["won"]),
                seed ^ (len(pairings) + 1),
            )
            pairings.append({
                "answerer": answerer.key,
                "guesser": guesser.key,
                "games": len(subset),
                "guesser_wins": sum(record["won"] for record in subset),
                "guesser_win_rate": rate,
                "target_clustered_95_ci": [low, high],
                "mean_turns": statistics.mean(record["turns_used"] for record in subset),
                "invalid_answers": sum(record["invalid_answers"] for record in subset),
                "invalid_moves": sum(record["invalid_moves"] for record in subset),
            })

    roles: dict[str, dict] = {}
    for model in MODELS:
        guesser_games = [record for record in records if record["guesser"]["key"] == model.key]
        answerer_games = [record for record in records if record["answerer"]["key"] == model.key]
        if guesser_games:
            low, high = cluster_bootstrap_ci(
                guesser_games,
                lambda record: record["guesser"]["key"] == model.key,
                lambda record: float(record["won"]),
                seed ^ (100 + len(roles)),
            )
            answerer_low, answerer_high = cluster_bootstrap_ci(
                answerer_games,
                lambda record: record["answerer"]["key"] == model.key,
                lambda record: float(not record["won"]),
                seed ^ (200 + len(roles)),
            )
            roles[model.key] = {
                "guesser_games": len(guesser_games),
                "guesser_wins": sum(record["won"] for record in guesser_games),
                "guesser_win_rate": sum(record["won"] for record in guesser_games) / len(guesser_games),
                "guesser_target_clustered_95_ci": [low, high],
                "answerer_games": len(answerer_games),
                "answerer_nonloss_rate": sum(not record["won"] for record in answerer_games)
                / len(answerer_games),
                "answerer_target_clustered_95_ci": [answerer_low, answerer_high],
            }

    return {
        "schema_version": 1,
        "experiment": "E1",
        "seed": seed,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_replicates": PERMUTATION_REPLICATES,
        "games": len(records),
        "models": [asdict(model) for model in MODELS],
        "pairings": pairings,
        "roles": roles,
        "paired_model_comparisons": {
            "guesser": paired_model_comparisons(records, "guesser", seed ^ 300),
            "answerer": paired_model_comparisons(records, "answerer", seed ^ 400),
            "method": (
                "Two-sided paired sign-flip permutation tests across the 40 shared targets; "
                f"{PERMUTATION_REPLICATES} Monte Carlo replicates per comparison. "
                "95% intervals are paired target bootstrap intervals. Holm correction is "
                "applied separately within each role's six comparisons."
            ),
        },
        "interpretation_note": (
            "Intervals use a percentile bootstrap clustered by target. "
            "No answer accuracy or question information gain is reported; "
            "the free-form questions do not yet have a reviewed answer key."
        ),
    }


def print_summary(summary: dict) -> None:
    print(f"\nE1 summary: {summary['games']} games")
    print("\nGuesser win rate by directed pairing (target-clustered 95% CI):")
    print("| Answerer / Guesser | " + " | ".join(model.key for model in MODELS) + " |")
    for answerer in MODELS:
        cells = []
        for guesser in MODELS:
            entry = next((
                row for row in summary["pairings"]
                if row["answerer"] == answerer.key and row["guesser"] == guesser.key
            ), None)
            if entry is None:
                cells.append("n/a")
            else:
                low, high = entry["target_clustered_95_ci"]
                cells.append(
                    f"{entry['guesser_wins']}/{entry['games']} "
                    f"({entry['guesser_win_rate']:.1%}, {low:.1%}-{high:.1%})"
                )
        print(f"| {answerer.key} | " + " | ".join(cells) + " |")
    print("\nRole summaries:")
    for model in MODELS:
        role = summary["roles"].get(model.key)
        if role:
            low, high = role["guesser_target_clustered_95_ci"]
            print(
                f"  {model.name}: guesser {role['guesser_wins']}/{role['guesser_games']} "
                f"({role['guesser_win_rate']:.1%}, 95% CI {low:.1%}-{high:.1%}); "
                f"answerer non-loss {role['answerer_nonloss_rate']:.1%} "
                f"({role['answerer_target_clustered_95_ci'][0]:.1%}-"
                f"{role['answerer_target_clustered_95_ci'][1]:.1%})"
            )
    print("\nPaired model comparisons (mean rate difference A - B):")
    for role in ("guesser", "answerer"):
        print(f"  {role.capitalize()}:")
        for comparison in summary["paired_model_comparisons"][role]:
            low, high = comparison["paired_target_bootstrap_95_ci"]
            print(
                f"    {comparison['model_a']} vs {comparison['model_b']}: "
                f"{comparison['mean_rate_difference_a_minus_b']:+.1%} "
                f"(95% CI {low:+.1%} to {high:+.1%}); "
                f"Holm-adjusted p={comparison['holm_adjusted_p_value']:.4f}"
            )
    print(f"\n{summary['interpretation_note']}")


def summary_path(results_path: Path, reports_dir: Path = ROOT / "reports") -> Path:
    filename = (
        f"{results_path.stem}.summary.json"
        if results_path.suffix == ".jsonl"
        else f"{results_path.name}.summary.json"
    )
    return reports_dir / filename


def write_summary(
    summary: dict,
    results_path: Path,
    reports_dir: Path = ROOT / "reports",
) -> Path:
    destination = summary_path(results_path, reports_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def merge_result_files(paths: list[Path], destination: Path, schedule: list[GameSpec]) -> int:
    if destination.exists():
        raise ExperimentError(
            f"Merge destination already exists: {destination}. Choose a new output file."
        )
    records = [record for path in paths for record in read_records(path)]
    validate_records(records, schedule, require_complete=True)
    if any(record.get("inference_mode") != "transformers" for record in records):
        raise ExperimentError("Only real Transformers inference results can be merged for E1.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise ExperimentError(f"Could not write merged results to {destination}: {error}") from error
    return len(records)


def run_smoke(backend: HuggingFaceBackend) -> int:
    print("Smoke test: loading each selected model, then a two-model pairing.")
    failures: list[str] = []
    for model in MODELS:
        try:
            backend.prepare_pair(model, model)
            response = backend.generate(
                model,
                [{"role": "user", "content": "Reply with exactly the word OK."}],
                game_seed("smoke", model.key, 1, 0),
            )
        except ExperimentError as error:
            failures.append(model.name)
            print(f"FAILED {model.name}: {error}", flush=True)
            backend.close()
            continue
        print(f"OK {model.name}: {response.text[:100]!r}", flush=True)

    pair = (MODELS[1], MODELS[2])
    try:
        backend.prepare_pair(*pair)
        for model in pair:
            response = backend.generate(
                model,
                [{"role": "user", "content": "Reply with exactly the word OK."}],
                game_seed("smoke-pair", model.key, 1, 0),
            )
            print(f"OK paired {model.name}: {response.text[:100]!r}", flush=True)
    except ExperimentError as error:
        failures.append("Gemma 3 + Mistral Nemo pair")
        print(f"FAILED paired load/generation: {error}", flush=True)
        backend.close()

    if failures:
        print(
            "Smoke test incomplete: " + ", ".join(failures)
            + ". Resolve these issues before starting E1.",
            flush=True,
        )
        return 1
    print("All selected models and the paired-model inference check passed.", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E1: the assigned-secret 20 Questions round robin.")
    parser.add_argument("--targets-file", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=519)
    parser.add_argument("--mock", action="store_true", help="use mock responses; never use for research results")
    parser.add_argument("--max-games", type=int, help="run at most N scheduled games (mock mode only)")
    parser.add_argument("--smoke-models", action="store_true", help="load and generate with each selected model")
    output_actions = parser.add_mutually_exclusive_group()
    output_actions.add_argument("--validate-only", action="store_true", help="validate a complete results file")
    output_actions.add_argument("--report-only", action="store_true", help="summarize a complete results file")
    output_actions.add_argument(
        "--merge-files",
        nargs="+",
        type=Path,
        metavar="JSONL",
        help="merge complete self-play and cross-play files into --results-file",
    )
    parser.add_argument(
        "--pairing-group",
        choices=("all", "self-play", "cross-play"),
        default="all",
        help="run or validate all pairings, just self-play, or just cross-play",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pool_id, targets = load_targets(args.targets_file)
        full_schedule = build_schedule(targets, args.seed, pool_id)
        schedule = select_pairings(full_schedule, args.pairing_group)
        if args.max_games is not None and (not args.mock or args.max_games < 1):
            raise ExperimentError("--max-games requires --mock and must be positive.")
        if args.smoke_models and args.mock:
            raise ExperimentError("--smoke-models cannot be combined with --mock.")
        if args.smoke_models and (args.validate_only or args.report_only or args.merge_files):
            raise ExperimentError("--smoke-models cannot be combined with result-file actions.")
        if args.merge_files and args.max_games is not None:
            raise ExperimentError("--merge-files cannot be combined with --max-games.")
        if args.merge_files:
            if args.pairing_group != "all":
                raise ExperimentError("--merge-files cannot be combined with --pairing-group.")
            count = merge_result_files(args.merge_files, args.results_file, full_schedule)
            print(f"Merged and validated {count} E1 games in {args.results_file}.")
            return 0
        if args.validate_only or args.report_only:
            if args.report_only and args.pairing_group != "all":
                raise ExperimentError("--report-only requires the complete E1 schedule.")
            records = read_records(args.results_file)
            validate_records(records, schedule, require_complete=True)
            print(f"Validated {len(records)} E1 games for {pool_id}.")
            if args.report_only:
                summary = summarize_results(records, args.seed)
                print_summary(summary)
                print(f"Summary: {write_summary(summary, args.results_file)}")
            return 0

        if args.smoke_models:
            backend = HuggingFaceBackend()
            try:
                return run_smoke(backend)
            finally:
                backend.close()

        records = read_records(args.results_file)
        validate_records(records, schedule, require_complete=False)
        expected_mode = "mock" if args.mock else "transformers"
        if any(record["inference_mode"] != expected_mode for record in records):
            raise ExperimentError(
                f"Existing results use a different inference mode than this run ({expected_mode}). "
                "Choose a separate output path."
            )
        completed = {record["game_id"] for record in records}
        pending = [spec for spec in schedule if spec.game_id not in completed]
        if args.max_games is not None:
            pending = pending[:args.max_games]
        print(
            f"E1 {args.pairing_group} schedule: {len(schedule)} games; "
            f"{len(completed)} already complete; "
            f"{len(pending)} to run. Target pool: {pool_id}.",
            flush=True,
        )
        backend: Backend = MockBackend() if args.mock else HuggingFaceBackend()
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.results_file.open("a", encoding="utf-8") as results:
                active_pair: tuple[str, str] | None = None
                for index, spec in enumerate(pending, start=1):
                    pair = (spec.answerer.key, spec.guesser.key)
                    if pair != active_pair:
                        backend.prepare_pair(spec.answerer, spec.guesser)
                        active_pair = pair
                    game = play_game(spec, targets, backend)
                    results.write(json.dumps(game, ensure_ascii=False) + "\n")
                    results.flush()
                    os.fsync(results.fileno())
                    if index % 20 == 0 or index == len(pending):
                        print(
                            f"Saved game {len(completed) + index}/{len(schedule)} "
                            f"({spec.answerer.key} vs {spec.guesser.key}, {spec.target.name}).",
                            flush=True,
                        )
        finally:
            backend.close()

        records = read_records(args.results_file)
        complete = len(records) == len(schedule)
        validate_records(records, schedule, require_complete=complete)
        if args.mock:
            print(
                f"Mock run only: {len(records)}/{len(schedule)} games saved; "
                "these are not experimental results."
            )
            return 0
        if not complete:
            print(f"Partial run saved: {len(records)}/{len(schedule)} games.")
            return 0

        if args.pairing_group != "all":
            print(
                f"Completed {args.pairing_group}: {len(records)} games. "
                "Merge the self-play and cross-play result files to validate the full E1 schedule."
            )
            return 0

        summary = summarize_results(records, args.seed)
        summary_file = write_summary(summary, args.results_file)
        print_summary(summary)
        print(f"Full game logs: {args.results_file}")
        print(f"Summary: {summary_file}")
        return 0
    except ExperimentError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

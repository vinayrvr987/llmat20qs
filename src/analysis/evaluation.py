"""Evaluation of the final v2 runs (E1-E6) against the proposal's research questions.

RQ1 guesser/answerer/pairing effects (E1, E2 predictable), RQ2 randomness of
model picks (E2 picks, scouting), RQ3 question information (E5, E6, E1 proxies),
RQ4 answer accuracy and the cost of wrong answers (category-keyed answers, E3,
E4). The experiment runners' own summaries are reused where they exist; this
module adds the remaining analyses the proposal defines. Everything here is
deterministic given the result files and ``SEED``.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import PROJECT_ROOT, RESULTS_DIR
from src.experiments import e1, followups, e5_information as e5, e6_reasoning_budget as e6


SEED = 519
REPLICATES = 2000
PERMUTATIONS = 10000
MODELS = [model.key for model in e1.MODELS]
LABELS = {model.key: model.name for model in e1.MODELS}
FILES = {
    "e1_cross": "e1-cross-play-r1.jsonl",
    "e1_self": "e1-self-play-r1.jsonl",
    "e2_picks": "e2-predictable-picks-s519.jsonl",
    "e2_predictable": "e2-predictable-games-s519.jsonl",
    "e2_scouting": "e2-scouting-s519.jsonl",
    "e3": "e3-liar-s519.jsonl",
    "e4": "e4-noisy-s519.jsonl",
    "e5": "e5-exact-information-s519.jsonl",
    "e6": "e6-reasoning-budget-s519.jsonl",
}
EXPECTED = {
    "e1_cross": 480, "e1_self": 160, "e2_picks": 400, "e2_predictable": 640,
    "e2_scouting": 640, "e3": 500, "e4": 300, "e5": 160, "e6": 120,
}
POOL_SIZE = 128          # candidates listed to E2-E4 guessers
E1_POOL_SIZE = 40        # candidates listed to E1/E5/E6 guessers


# --------------------------------------------------------------------------- data

@lru_cache(maxsize=None)
def load(name: str, results_dir: Path = RESULTS_DIR) -> tuple[dict, ...]:
    path = results_dir / FILES[name]
    with path.open(encoding="utf-8") as source:
        return tuple(json.loads(line) for line in source if line.strip())


def e1_records() -> list[dict]:
    return [*load("e1_cross"), *load("e1_self")]


def inventory() -> pd.DataFrame:
    """Record counts and protocol versions per file (the data-validity check)."""
    rows = []
    for name, filename in FILES.items():
        records = load(name)
        config = Counter(
            (r.get("run_config") or {}).get("guess_rule")
            or (r.get("run_config") or {}).get("dialogue")
            or "-"
            for r in records
        )
        rows.append({
            "dataset": name,
            "file": filename,
            "records": len(records),
            "expected": EXPECTED[name],
            "unique_ids": len({r.get("game_id") or r.get("pick_id") for r in records}),
            "protocol": ", ".join(sorted(config)),
            "mock_records": sum(r.get("inference_mode") != "transformers" for r in records),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- statistics

def wilson(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    rate = wins / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(values, seed: int, replicates: int = REPLICATES) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def sign_flip_p(differences, seed: int, permutations: int = PERMUTATIONS) -> float:
    """Two-sided paired sign-flip permutation p-value for a mean difference of zero."""
    d = np.asarray(list(differences), dtype=float)
    if len(d) == 0 or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(permutations, len(d)))
    extreme = np.sum(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12)
    return float((extreme + 1) / (permutations + 1))


def holm(p_values) -> list[float]:
    p_values = list(p_values)
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def paired_test(differences, seed: int) -> dict:
    differences = list(differences)
    low, high = bootstrap_ci(differences, seed)
    return {
        "n_pairs": len(differences),
        "mean_difference": float(np.mean(differences)) if differences else math.nan,
        "ci_low": low,
        "ci_high": high,
        "p_value": sign_flip_p(differences, seed + 1),
    }


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def ols_clustered(design: pd.DataFrame, outcome, clusters) -> pd.DataFrame:
    """OLS with CR1 cluster-robust standard errors and normal-approximation p-values."""
    x = design.to_numpy(dtype=float)
    y = np.asarray(outcome, dtype=float)
    groups = np.asarray(clusters)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    labels = np.unique(groups)
    for label in labels:
        score = x[groups == label].T @ residual[groups == label]
        meat += np.outer(score, score)
    n, k, g = len(y), x.shape[1], len(labels)
    correction = g / (g - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.diag(correction * xtx_inv @ meat @ xtx_inv))
    z = beta / np.where(se > 0, se, np.nan)
    p = np.array([math.erfc(abs(value) / math.sqrt(2)) if np.isfinite(value) else math.nan
                  for value in z])
    return pd.DataFrame({
        "estimate": beta, "se": se,
        "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se, "p_value": p,
    }, index=design.columns)


# --------------------------------------------------------------------------- question keys

_PREFIX = re.compile(
    r"^(?:is|was) (?:it|this|the (?:secret|concept|thing|answer)|your (?:secret|concept))"
    r"(?: (?:concept|secret|word))? (.+)$"
)
CATEGORY_PREDICATES = {
    "an animal": "animal",
    "a person": "person", "a human": "person", "a human being": "person",
    "a place": "place", "a location": "place", "a geographic location": "place",
    "a geographical location": "place", "a geographic place": "place",
    "an object": "object", "an inanimate object": "object", "a man-made object": "object",
}
TAXON_PREDICATES = {"a mammal": "Q06", "a bird": "Q07"}


def normalize_question(text: str) -> str:
    text = re.sub(r"[^a-z0-9' -]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _taxon_ids() -> dict[str, frozenset[str]]:
    bank = json.loads(e5.QUESTION_BANK.read_text(encoding="utf-8"))
    return {q["id"]: frozenset(q["yes_target_ids"])
            for q in bank["questions"] if q["id"] in TAXON_PREDICATES.values()}


def question_predicate(text: str) -> str | None:
    """The keyed predicate a free-form question asks, or None if it is not keyed."""
    match = _PREFIX.match(normalize_question(text))
    if not match:
        return None
    phrase = match.group(1).strip()
    if phrase in CATEGORY_PREDICATES:
        return CATEGORY_PREDICATES[phrase]
    if phrase in TAXON_PREDICATES:
        return TAXON_PREDICATES[phrase]
    return None


def keyed_answer(text: str, target: dict) -> str | None:
    """Correct Yes/No for unambiguous category/taxon questions, else None.

    Category questions follow from the target's category. "Mammal"/"bird" use the
    E5 question bank labels, which exist only for E1 animals; persons are not
    keyed for taxon questions because "Is it a mammal?" is ambiguous for people.
    Object questions are not keyed for places: "Is it an (inanimate) object?"
    about a city or landmark has no single defensible answer.
    """
    predicate = question_predicate(text)
    if predicate is None:
        return None
    if predicate == "object" and target["category"] == "place":
        return None
    if predicate in ("animal", "person", "place", "object"):
        return "Yes" if target["category"] == predicate else "No"
    if target["category"] in ("object", "place"):
        return "No"
    if target["category"] == "animal" and target.get("e1"):
        return "Yes" if target["id"] in _taxon_ids()[predicate] else "No"
    return None


def question_turns(record: dict) -> list[dict]:
    return [turn for turn in record["turns"] if turn["type"] == "question"]


# --------------------------------------------------------------------------- RQ1: E1

def e1_summary() -> dict:
    return e1.summarize_results(e1_records(), SEED)


def e1_win_matrix(summary: dict) -> pd.DataFrame:
    rows = [{"answerer": p["answerer"], "guesser": p["guesser"],
             "win_rate": p["guesser_win_rate"]} for p in summary["pairings"]]
    matrix = pd.DataFrame(rows).pivot(index="guesser", columns="answerer", values="win_rate")
    return matrix.reindex(index=MODELS, columns=MODELS)


def e1_pairings(summary: dict) -> pd.DataFrame:
    rows = []
    for p in summary["pairings"]:
        rows.append({
            "guesser": p["guesser"], "answerer": p["answerer"],
            "self_play": p["guesser"] == p["answerer"], "games": p["games"],
            "win_rate": p["guesser_win_rate"],
            "ci_low": p["target_clustered_95_ci"][0], "ci_high": p["target_clustered_95_ci"][1],
            "mean_turns": p["mean_turns"],
            "invalid_moves_per_game": p["invalid_moves"] / p["games"],
            "invalid_answers_per_game": p["invalid_answers"] / p["games"],
        })
    return pd.DataFrame(rows)


def e1_roles(summary: dict) -> pd.DataFrame:
    rows = []
    for model, r in summary["roles"].items():
        rows.append({
            "model": model,
            "guesser_win_rate": r["guesser_win_rate"],
            "guesser_ci": tuple(r["guesser_target_clustered_95_ci"]),
            "answerer_nonloss_rate": r["answerer_nonloss_rate"],
            "answerer_ci": tuple(r["answerer_target_clustered_95_ci"]),
        })
    return pd.DataFrame(rows).set_index("model")


def e1_role_comparisons(summary: dict, role: str) -> pd.DataFrame:
    rows = summary["paired_model_comparisons"][role]
    return pd.DataFrame([{
        "model_a": r["model_a"], "model_b": r["model_b"],
        "difference_a_minus_b": r["mean_rate_difference_a_minus_b"],
        "ci_low": r["paired_target_bootstrap_95_ci"][0],
        "ci_high": r["paired_target_bootstrap_95_ci"][1],
        "p_value": r["permutation_p_value"], "holm_p": r["holm_adjusted_p_value"],
    } for r in rows])


def e1_design(records: list[dict] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Additive guesser + answerer + same-model + target-stratum design for E1."""
    records = records or e1_records()
    frame = pd.DataFrame([{
        "guesser": r["guesser"]["key"], "answerer": r["answerer"]["key"],
        "category": r["target"]["category"], "frequency": r["target"]["frequency"],
        "target": r["target"]["id"], "won": float(r["won"]),
    } for r in records])
    design = pd.DataFrame({"intercept": 1.0}, index=frame.index)
    for model in MODELS[1:]:
        design[f"guesser: {model}"] = (frame["guesser"] == model).astype(float)
    for model in MODELS[1:]:
        design[f"answerer: {model}"] = (frame["answerer"] == model).astype(float)
    design["same model (self-play)"] = (frame["guesser"] == frame["answerer"]).astype(float)
    for category in ("animal", "place", "person"):
        design[f"category: {category}"] = (frame["category"] == category).astype(float)
    design["rare target"] = (frame["frequency"] == "rare").astype(float)
    return design, frame["won"].to_numpy(), frame["target"].to_numpy()


def e1_regression() -> pd.DataFrame:
    """Linear probability model of a win; reference: Qwen3 guesser, Qwen3 answerer, object, common."""
    design, won, clusters = e1_design()
    return ols_clustered(design, won, clusters)


def e1_self_play_matched() -> pd.DataFrame:
    """Per guesser: self-play win minus mean cross-play win on the same target."""
    wins = defaultdict(dict)
    for r in e1_records():
        wins[(r["guesser"]["key"], r["answerer"]["key"])][r["target"]["id"]] = float(r["won"])
    rows = []
    for index, guesser in enumerate(MODELS):
        targets = sorted(wins[(guesser, guesser)])
        differences = [
            wins[(guesser, guesser)][t]
            - statistics.mean(wins[(guesser, a)][t] for a in MODELS if a != guesser)
            for t in targets
        ]
        rows.append({"guesser": guesser,
                     "self_play_win_rate": statistics.mean(wins[(guesser, guesser)].values()),
                     **paired_test(differences, SEED + 10 * index)})
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table["p_value"])
    return table


def cumulative_wins(records, group: str = "guesser", max_turns: int = 20) -> pd.DataFrame:
    """Share of games won by turn t (no censoring before the 20-turn cap)."""
    by_group = defaultdict(list)
    for r in records:
        by_group[r[group]["key"]].append(r["turns_used"] if r["won"] else math.inf)
    return pd.DataFrame({
        key: [sum(t <= turn for t in values) / len(values) for turn in range(1, max_turns + 1)]
        for key, values in by_group.items()
    }, index=range(1, max_turns + 1))


def win_by_stratum(records, stratum: str) -> pd.DataFrame:
    rows = defaultdict(list)
    for r in records:
        rows[(r["guesser"]["key"], r["target"][stratum])].append(r["won"])
    table = pd.Series({key: statistics.mean(values) for key, values in rows.items()}).unstack()
    return table.reindex(MODELS)


def guesser_behavior(records) -> pd.DataFrame:
    """Move-level behavior: guessing, repetition, and redundant category questions."""
    stats = defaultdict(Counter)
    first_guess = defaultdict(list)
    for r in records:
        key = r["guesser"]["key"]
        s = stats[key]
        s["games"] += 1
        s["invalid_moves"] += r["invalid_moves"]
        guesses = [t for t in r["turns"] if t["type"] == "guess"]
        s["guesses"] += len(guesses)
        s["correct_guesses"] += sum(bool(t.get("correct")) for t in guesses)
        s["games_without_guess"] += not guesses
        if guesses:
            first_guess[key].append(guesses[0]["turn"])
        seen: set[str] = set()
        allowed = {"animal", "person", "place", "object"}
        for turn in question_turns(r):
            text = normalize_question(turn["text"])
            s["questions"] += 1
            s["repeated_questions"] += text in seen
            seen.add(text)
            predicate = question_predicate(turn["text"])
            if predicate not in ("animal", "person", "place", "object") or not allowed:
                continue
            s["category_questions"] += 1
            s["determined_category_questions"] += predicate not in allowed or allowed == {predicate}
            if turn.get("answer") == "Yes":
                allowed &= {predicate}
            elif turn.get("answer") == "No":
                allowed -= {predicate}
    rows = []
    for key in [m for m in MODELS if m in stats]:
        s = stats[key]
        rows.append({
            "guesser": key,
            "games": s["games"],
            "invalid_moves_per_game": s["invalid_moves"] / s["games"],
            "guesses_per_game": s["guesses"] / s["games"],
            "guess_precision": s["correct_guesses"] / s["guesses"] if s["guesses"] else math.nan,
            "games_without_guess": s["games_without_guess"] / s["games"],
            "median_first_guess_turn": statistics.median(first_guess[key]) if first_guess[key] else math.nan,
            "repeated_question_rate": s["repeated_questions"] / s["questions"],
            "category_question_share": s["category_questions"] / s["questions"],
            "determined_category_question_rate": (
                s["determined_category_questions"] / s["category_questions"]
                if s["category_questions"] else math.nan
            ),
        })
    return pd.DataFrame(rows).set_index("guesser")


# --------------------------------------------------------------------------- RQ2: E2

def _entropy_bits(counts) -> float:
    total = sum(counts)
    return -sum(c / total * math.log2(c / total) for c in counts if c)


def pick_uniformity(replicates: int = PERMUTATIONS) -> pd.DataFrame:
    """Entropy, Miller-Madow correction, and a Monte Carlo chi-squared test vs uniform.

    Picks are constrained to 25 per category, so the uniform reference draws 25
    uniform picks from each category's 32 concepts.
    """
    _, targets = e1.load_targets()
    by_category = defaultdict(list)
    for target in targets:
        by_category[target.category].append(target.id)
    frequency = {target.id: target.frequency for target in targets}
    picks = load("e2_picks")
    rng = np.random.default_rng(SEED)

    def chi_square(per_category_counts: dict[str, np.ndarray]) -> float:
        total = 0.0
        for counts in per_category_counts.values():
            expected = counts.sum() / len(counts)
            total += float(((counts - expected) ** 2 / expected).sum())
        return total

    rows = []
    for model in MODELS:
        model_picks = [p["target_id"] for p in picks if p["model_key"] == model]
        counts = Counter(model_picks)
        observed = {
            category: np.array([counts[t] for t in ids], dtype=float)
            for category, ids in by_category.items()
        }
        sizes = {c: int(v.sum()) for c, v in observed.items()}
        statistic = chi_square(observed)
        simulated = np.array([
            chi_square({c: np.bincount(rng.integers(0, len(by_category[c]), sizes[c]),
                                       minlength=len(by_category[c])).astype(float)
                        for c in by_category})
            for _ in range(replicates)
        ])
        plug_in = _entropy_bits(counts.values())
        miller_madow = plug_in + (len(counts) - 1) / (2 * len(model_picks) * math.log(2))
        rare = sum(frequency[t] == "rare" for t in model_picks)
        rows.append({
            "model": model,
            "picks": len(model_picks),
            "distinct": len(counts),
            "entropy_bits": plug_in,
            "entropy_mm_bits": miller_madow,
            "gap_log2N_minus_H": math.log2(POOL_SIZE) - miller_madow,
            "chi_square": statistic,
            "chi_square_mc_p": float((np.sum(simulated >= statistic) + 1) / (replicates + 1)),
            "rare_share": rare / len(model_picks),
            "rare_ci": wilson(rare, len(model_picks)),
            "top_picks": ", ".join(f"{t.split('-', 1)[1]} ({n})" for t, n in counts.most_common(5)),
        })
    return pd.DataFrame(rows).set_index("model")


def uniform_entropy_reference(draws: int = 100, replicates: int = REPLICATES) -> tuple[float, float, float]:
    """Plug-in entropy of 25-per-category uniform draws: mean and 95% range."""
    rng = np.random.default_rng(SEED + 1)
    per_category = draws // 4
    values = []
    for _ in range(replicates):
        counts = np.concatenate([np.bincount(rng.integers(0, 32, per_category), minlength=32)
                                 for _ in range(4)])
        values.append(_entropy_bits(counts))
    return (float(np.mean(values)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def scouting_effect() -> pd.DataFrame:
    """Matched with-report minus without-report differences per model."""
    pairs = defaultdict(dict)
    for r in load("e2_scouting"):
        pairs[r["randomization_id"]][r["condition"]["scouting_report"]] = r
    rows = []
    for index, model in enumerate(MODELS):
        matched = [p for p in pairs.values()
                   if len(p) == 2 and p[True]["guesser"]["key"] == model]
        win_d = [float(p[True]["won"]) - float(p[False]["won"]) for p in matched]
        turn_d = [p[True]["turns_used"] - p[False]["turns_used"] for p in matched]
        win = paired_test(win_d, SEED + 100 + index)
        turns = paired_test(turn_d, SEED + 200 + index)
        rows.append({
            "model": model, "pairs": len(matched),
            "win_rate_no_report": statistics.mean(p[False]["won"] for p in matched),
            "win_rate_report": statistics.mean(p[True]["won"] for p in matched),
            "win_diff": win["mean_difference"], "win_ci": (win["ci_low"], win["ci_high"]),
            "win_p": win["p_value"],
            "turns_saved": -turns["mean_difference"],
            "turns_saved_ci": (-turns["ci_high"], -turns["ci_low"]),
            "turns_p": turns["p_value"],
            "secret_in_report_share": statistics.mean(
                e1.normalize(p[True]["target"]["name"])
                in {e1.normalize(x) for x in p[True]["condition"]["favorite_picks"]}
                for p in matched
            ),
        })
    table = pd.DataFrame(rows).set_index("model")
    table["win_holm_p"] = holm(table["win_p"])
    table["turns_holm_p"] = holm(table["turns_p"])
    return table


def e2_predictable_matrix() -> pd.DataFrame:
    rows = defaultdict(list)
    for r in load("e2_predictable"):
        rows[(r["guesser"]["key"], r["answerer"]["key"])].append(r["won"])
    table = pd.Series({k: statistics.mean(v) for k, v in rows.items()}).unstack()
    return table.reindex(index=MODELS, columns=MODELS)


def e2_predictable_guesser_comparisons() -> pd.DataFrame:
    """Paired guesser comparisons; the same answerer picks are played by every guesser."""
    outcome = defaultdict(lambda: defaultdict(list))
    for r in load("e2_predictable"):
        outcome[r["guesser"]["key"]][r["randomization_id"]].append(float(r["won"]))
    rows = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            ids = sorted(outcome[a].keys() & outcome[b].keys())
            d = [statistics.mean(outcome[a][k]) - statistics.mean(outcome[b][k]) for k in ids]
            rows.append({"model_a": a, "model_b": b, **paired_test(d, SEED + 300 + len(rows))})
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table["p_value"])
    return table


# --------------------------------------------------------------------------- RQ4: answers

def answer_accuracy() -> pd.DataFrame:
    """Error rate of each answerer's own Yes/No on keyed category/taxon questions.

    E2-E4 use ``truthful_answer`` (before any controller flip), so this measures
    the model, not the injected noise.
    """
    sources = {
        "E1": (e1_records(), "answer"),
        "E2": ([*load("e2_predictable"), *load("e2_scouting")], "truthful_answer"),
        "E3/E4": ([*load("e3"), *load("e4")], "truthful_answer"),
    }
    counts = defaultdict(Counter)
    for source, (records, field) in sources.items():
        for r in records:
            for turn in question_turns(r):
                truth = keyed_answer(turn["text"], r["target"])
                answer = turn.get(field)
                if truth is None or answer not in ("Yes", "No"):
                    continue
                for key in ((r["answerer"]["key"], source), (r["answerer"]["key"], "all")):
                    counts[key]["n"] += 1
                    counts[key]["errors"] += answer != truth
    rows = []
    for (model, source), c in counts.items():
        epsilon = c["errors"] / c["n"]
        low, high = wilson(c["errors"], c["n"])
        rows.append({
            "answerer": model, "source": source, "keyed_answers": c["n"], "errors": c["errors"],
            "error_rate": epsilon, "ci_low": low, "ci_high": high,
            "capacity_bits": 1 - binary_entropy(epsilon),
        })
    order = {m: i for i, m in enumerate(MODELS)}
    return pd.DataFrame(rows).sort_values(
        ["source", "answerer"], key=lambda s: s.map(order) if s.name == "answerer" else s
    ).reset_index(drop=True)


def answer_errors_by_predicate() -> pd.DataFrame:
    counts = defaultdict(Counter)
    for r in e1_records():
        for turn in question_turns(r):
            truth = keyed_answer(turn["text"], r["target"])
            if truth is None or turn.get("answer") not in ("Yes", "No"):
                continue
            key = (r["answerer"]["key"], question_predicate(turn["text"]), r["target"]["category"])
            counts[key]["n"] += 1
            counts[key]["errors"] += turn["answer"] != truth
    return pd.DataFrame([
        {"answerer": a, "predicate": p, "target_category": c, "n": v["n"],
         "errors": v["errors"], "error_rate": v["errors"] / v["n"]}
        for (a, p, c), v in counts.items() if v["errors"]
    ]).sort_values("errors", ascending=False).reset_index(drop=True)


def answer_error_cost() -> pd.DataFrame:
    """E1 win rate in games with vs without a keyed answer error (observational)."""
    rows = defaultdict(lambda: defaultdict(list))
    for r in e1_records():
        error = any(
            (truth := keyed_answer(t["text"], r["target"])) is not None
            and t.get("answer") in ("Yes", "No") and t["answer"] != truth
            for t in question_turns(r)
        )
        rows[r["answerer"]["key"]][error].append(float(r["won"]))
    return pd.DataFrame([{
        "answerer": model,
        "games_with_error": len(groups[True]),
        "win_rate_with_error": statistics.mean(groups[True]) if groups[True] else math.nan,
        "games_without_error": len(groups[False]),
        "win_rate_without_error": statistics.mean(groups[False]) if groups[False] else math.nan,
    } for model, groups in ((m, rows[m]) for m in MODELS)]).set_index("answerer")


# --------------------------------------------------------------------------- lies & noise

def ulam_questions(n: int, lies: int) -> int:
    """Volume (Berlekamp) lower bound on questions to find one of n with up to `lies` lies."""
    q = 0
    while 2 ** q < n * sum(math.comb(q, i) for i in range(lies + 1)):
        q += 1
    return q


def noisy_questions(n: int, epsilon: float) -> float:
    """Shannon reference: log2 n / (1 - h(epsilon)) questions for a BSC answerer."""
    return math.log2(n) / (1 - binary_entropy(epsilon))


def theory_table() -> pd.DataFrame:
    rows = [{"condition": f"{k} lie{'s' if k != 1 else ''} (adversarial bound)",
             "min_questions": ulam_questions(POOL_SIZE, k),
             "extra_questions": ulam_questions(POOL_SIZE, k) - math.log2(POOL_SIZE)}
            for k in followups.LIE_COUNTS]
    for epsilon in followups.NOISE_RATES:
        q = noisy_questions(POOL_SIZE, epsilon)
        rows.append({"condition": f"{epsilon:.0%} random flips (capacity {1 - binary_entropy(epsilon):.2f} bits)",
                     "min_questions": q, "extra_questions": q - math.log2(POOL_SIZE)})
    return pd.DataFrame(rows).set_index("condition")


def liar_table() -> pd.DataFrame:
    groups = defaultdict(list)
    for r in load("e3"):
        groups[(r["condition"]["lie_count"], r["condition"]["warned"])].append(r)
    rows = []
    for (lies, warned), records in sorted(groups.items()):
        wins = sum(r["won"] for r in records)
        low, high = wilson(wins, len(records))
        rows.append({
            "lies": lies, "warned": warned, "games": len(records), "win_rate": wins / len(records),
            "ci_low": low, "ci_high": high,
            "delivered_share": (sum(r["applied_flips"] for r in records) / (lies * len(records))
                                if lies else math.nan),
            "mean_turns": statistics.mean(r["turns_used"] for r in records),
        })
    return pd.DataFrame(rows)


def liar_by_guesser() -> pd.DataFrame:
    groups = defaultdict(list)
    for r in load("e3"):
        groups[(r["guesser"]["key"], r["condition"]["lie_count"])].append(r["won"])
    return pd.Series({k: statistics.mean(v) for k, v in groups.items()}).unstack().reindex(MODELS)


def warning_effect() -> pd.DataFrame:
    """Warned minus unwarned, paired on target and flip schedule."""
    pairs = defaultdict(dict)
    for r in load("e3"):
        pairs[r["randomization_id"]][r["condition"]["warned"]] = r
    rows = []
    for lies in followups.LIE_COUNTS:
        matched = [p for p in pairs.values()
                   if len(p) == 2 and p[True]["condition"]["lie_count"] == lies]
        d = [float(p[True]["won"]) - float(p[False]["won"]) for p in matched]
        rows.append({"lies": lies, **paired_test(d, SEED + 400 + lies)})
    return pd.DataFrame(rows)


def stratified_trend_test(records, level, seed: int, permutations: int = PERMUTATIONS) -> dict:
    """Permutation test of the win/level association, permuting levels within guesser."""
    frame = pd.DataFrame({"guesser": [r["guesser"]["key"] for r in records],
                          "level": [float(level(r)) for r in records],
                          "won": [float(r["won"]) for r in records]})
    def statistic(levels):
        return float(np.corrcoef(levels, frame["won"])[0, 1])
    observed = statistic(frame["level"].to_numpy())
    rng = np.random.default_rng(seed)
    blocks = [frame.index[frame["guesser"] == g].to_numpy() for g in frame["guesser"].unique()]
    levels = frame["level"].to_numpy()
    extreme = 0
    for _ in range(permutations):
        shuffled = levels.copy()
        for block in blocks:
            shuffled[block] = rng.permutation(levels[block])
        extreme += abs(statistic(shuffled)) >= abs(observed) - 1e-12
    return {"correlation": observed, "p_value": (extreme + 1) / (permutations + 1)}


def noise_table() -> pd.DataFrame:
    """E4 by flip rate, with E3's unwarned 0-lie games (same answerer) as the 0% row."""
    groups = defaultdict(list)
    for r in load("e3"):
        if r["condition"]["lie_count"] == 0 and not r["condition"]["warned"]:
            groups[0.0].append(r)
    for r in load("e4"):
        groups[r["condition"]["flip_probability"]].append(r)
    rows = []
    for p, records in sorted(groups.items()):
        wins = sum(r["won"] for r in records)
        answers = sum(t.get("answer_status") == "valid" for r in records for t in r["turns"])
        flips = sum(r["applied_flips"] for r in records)
        low, high = wilson(wins, len(records))
        rows.append({
            "flip_probability": p, "games": len(records), "win_rate": wins / len(records),
            "ci_low": low, "ci_high": high, "realized_flip_rate": flips / answers if answers else 0.0,
            "capacity_bits": 1 - binary_entropy(p),
            "reference_questions": noisy_questions(POOL_SIZE, p),
            "source": "E3 (0 lies, unwarned)" if p == 0 else "E4",
        })
    return pd.DataFrame(rows)


def recheck_rates() -> pd.DataFrame:
    """How often a guesser re-asks a question later in the game, after a flipped vs a true answer."""
    counts = defaultdict(Counter)
    for name in ("e3", "e4"):
        for r in load(name):
            turns = question_turns(r)
            texts = [normalize_question(t["text"]) for t in turns]
            for i, turn in enumerate(turns):
                if turn.get("answer_status") != "valid":
                    continue
                key = (name.upper(), r["guesser"]["key"],
                       "flipped" if turn.get("answer_flipped") else "unflipped")
                counts[key]["answers"] += 1
                counts[key]["rechecked"] += texts[i] in texts[i + 1:]
    rows = [{"experiment": e, "guesser": g, "answer": a, "answers": c["answers"],
             "recheck_rate": c["rechecked"] / c["answers"]}
            for (e, g, a), c in counts.items()]
    return (pd.DataFrame(rows)
            .pivot_table(index=["experiment", "guesser"], columns="answer",
                         values=["recheck_rate", "answers"])
            .reindex(MODELS, level="guesser"))


def outcome_after_flip() -> pd.DataFrame:
    """Win rate by whether a scheduled lie was actually delivered (E3, observational)."""
    groups = defaultdict(list)
    for r in load("e3"):
        lies = r["condition"]["lie_count"]
        if lies:
            groups[(lies, r["applied_flips"])].append(r)
    return pd.DataFrame([{
        "scheduled_lies": lies, "delivered": delivered, "games": len(v),
        "win_rate": statistics.mean(r["won"] for r in v),
        "mean_turns": statistics.mean(r["turns_used"] for r in v),
    } for (lies, delivered), v in sorted(groups.items())])


# --------------------------------------------------------------------------- RQ3: E5/E6

def e5_summary() -> dict:
    return e5.summarize(list(load("e5")))


def e5_question_quality() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        records = [r for r in load("e5") if r["model"]["key"] == model]
        questions = [t for r in records for t in r["turns"] if t["action"] == "question"]
        guesses = [t for r in records for t in r["turns"] if t["action"] == "guess"]
        actions = [t for r in records for t in r["turns"]]
        rows.append({
            "model": model,
            "games": len(records),
            "win_rate": statistics.mean(r["won"] for r in records),
            "mean_turns": statistics.mean(r["turns_used"] for r in records),
            "questions_per_game": len(questions) / len(records),
            "bits_per_question": statistics.mean(t["expected_information_gain_bits"] for t in questions),
            "best_available_bits": statistics.mean(t["best_available_information_gain_bits"] for t in questions),
            "optimality_ratio": statistics.mean(t["optimality_ratio"] for t in questions),
            # The menu only offers questions that split the candidates, so 0-bit or
            # <0.1-bit questions cannot occur; count clearly sub-par choices instead.
            "below_half_best_rate": statistics.mean(t["optimality_ratio"] < 0.5 for t in questions),
            "invalid_action_rate": sum(t["action"] == "invalid" for t in actions) / len(actions),
            "voluntary_guesses_per_game": len(guesses) / len(records),
            "voluntary_guess_accuracy": (statistics.mean(bool(t.get("correct")) for t in guesses)
                                         if guesses else math.nan),
            "greedy_turns": statistics.mean(r["baselines"]["greedy"]["turns_used"] for r in records),
            "random_turns": statistics.mean(r["baselines"]["random"]["turns_used"] for r in records),
        })
    return pd.DataFrame(rows).set_index("model")


def e5_game_metrics() -> pd.DataFrame:
    rows = []
    for r in load("e5"):
        questions = [t for t in r["turns"] if t["action"] == "question"]
        rows.append({
            "model": r["model"]["key"], "target": r["target"]["id"],
            "turns": r["turns_used"], "won": float(r["won"]),
            "optimality": (statistics.mean(t["optimality_ratio"] for t in questions)
                           if questions else math.nan),
            "greedy_turns": r["baselines"]["greedy"]["turns_used"],
            "random_turns": r["baselines"]["random"]["turns_used"],
        })
    return pd.DataFrame(rows)


def e5_model_comparisons(metric: str = "turns") -> pd.DataFrame:
    games = e5_game_metrics().pivot(index="target", columns="model", values=metric)
    rows = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            d = (games[a] - games[b]).dropna()
            rows.append({"model_a": a, "model_b": b, "metric": metric,
                         **paired_test(d, SEED + 500 + len(rows))})
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table["p_value"])
    return table


def e5_vs_greedy() -> pd.DataFrame:
    games = e5_game_metrics()
    rows = []
    for index, model in enumerate(MODELS):
        subset = games[games["model"] == model]
        rows.append({"model": model, "baseline": "greedy",
                     **paired_test(subset["turns"] - subset["greedy_turns"], SEED + 600 + index)})
        rows.append({"model": model, "baseline": "random",
                     **paired_test(subset["turns"] - subset["random_turns"], SEED + 700 + index)})
    return pd.DataFrame(rows)


def e6_summary() -> dict:
    return e6.summarize(list(load("e6")))


def e6_groups(summary: dict) -> pd.DataFrame:
    keys = ["games", "mean_won", "mean_turns", "mean_mean_question_information_bits",
            "mean_mean_optimality_ratio", "mean_invalid_actions", "mean_output_tokens",
            "mean_thinking_tokens", "mean_budget_exhausted_rate", "mean_generation_seconds"]
    return pd.DataFrame([{k: g[k] for k in keys} for g in summary["groups"]],
                        index=[g["key"] for g in summary["groups"]])


def e6_paired(summary: dict) -> pd.DataFrame:
    rows = []
    for p in summary["paired_comparisons"]:
        for metric, value in p["mean_paired_differences"].items():
            low, high = p["target_bootstrap_95_ci"][metric]
            rows.append({"comparison": f"{p['condition']} − {p['reference_condition']}",
                         "metric": metric, "mean_difference": value, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def e6_game_metrics() -> pd.DataFrame:
    return pd.DataFrame([{"condition": r["reasoning_condition"]["key"], "target": r["target"]["id"],
                          **e6.game_metrics(r)} for r in load("e6")])


def e6_paired_p_values() -> pd.DataFrame:
    """Sign-flip p-values for the paired E6 differences (the runner reports only intervals)."""
    games = e6_game_metrics()
    rows = []
    conditions = [c.key for c in e6.CONDITIONS]
    for i, a in enumerate(conditions):
        for b in conditions[i + 1:]:
            for metric in ("mean_question_information_bits", "mean_optimality_ratio", "turns",
                           "invalid_actions", "generation_seconds"):
                wide = games.pivot(index="target", columns="condition", values=metric)
                d = (wide[b] - wide[a]).dropna()
                rows.append({"comparison": f"{b} − {a}", "metric": metric,
                             **paired_test(d, SEED + 800 + len(rows))})
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table["p_value"])
    return table

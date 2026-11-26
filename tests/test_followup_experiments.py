import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.experiments import e1, followups


class ScriptedBackend:
    mode = "test"

    def __init__(self, plan, target):
        self.plan = plan
        self.target = target
        self.guesser_calls = 0

    def prepare_pair(self, answerer, guesser):
        del answerer, guesser

    def generate(self, model, messages, seed):
        del messages, seed
        if model.key == self.plan.guesser.key:
            self.guesser_calls += 1
            if self.guesser_calls == 1:
                return e1.Generation("QUESTION: Is it a living thing?", 0.001)
            return e1.Generation(f"GUESS: {self.target.name}", 0.001)
        return e1.Generation("Yes", 0.001)

    def close(self):
        return None

    def metadata(self):
        return {"inference_mode": self.mode}


def make_picks(targets):
    picks = []
    for model in e1.MODELS:
        for attempt in range(followups.PICK_ATTEMPTS_PER_MODEL):
            category = e1.CATEGORIES[attempt % len(e1.CATEGORIES)]
            category_targets = [target for target in targets if target.category == category]
            target = category_targets[attempt % len(category_targets)]
            picks.append({
                "pick_id": f"{model.key}|{attempt:03d}",
                "model_key": model.key,
                "attempt": attempt,
                "category": category,
                "target_id": target.id,
            })
    return picks


class FollowupExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.targets = e1.load_targets()
        cls.picks = make_picks(cls.targets)

    def test_predictable_minds_has_640_ordered_pair_games_and_matched_targets(self):
        schedule = followups.build_predictable_schedule(self.targets, self.picks, 519)
        self.assertEqual(len(schedule), 640)
        for answerer in e1.MODELS:
            targets_by_guesser = []
            for guesser in e1.MODELS:
                games = [
                    plan for plan in schedule
                    if plan.answerer.key == answerer.key and plan.guesser.key == guesser.key
                ]
                self.assertEqual(len(games), 40)
                targets_by_guesser.append([plan.target.id for plan in games])
            self.assertTrue(all(items == targets_by_guesser[0] for items in targets_by_guesser))

    def test_scouting_has_640_matched_self_play_games(self):
        schedule = followups.build_scouting_schedule(self.targets, self.picks, 519)
        self.assertEqual(len(schedule), 640)
        for model in e1.MODELS:
            conditions = {}
            for informed in (False, True):
                games = [
                    plan for plan in schedule
                    if plan.guesser.key == model.key
                    and plan.condition["scouting_report"] is informed
                ]
                self.assertEqual(len(games), 80)
                conditions[informed] = {plan.source_pick_attempt for plan in games}
                if informed:
                    self.assertTrue(games[0].condition["favorite_picks"])
                else:
                    self.assertEqual(games[0].condition["favorite_picks"], [])
            self.assertEqual(conditions[False], conditions[True])

    def test_liar_schedule_is_balanced_to_500_with_paired_warning_conditions(self):
        schedule = followups.build_liar_schedule(self.targets, 519)
        self.assertEqual(len(schedule), 500)
        self.assertTrue(all(plan.answerer.key == "qwen3-8b" for plan in schedule))
        groups = {}
        for plan in schedule:
            key = (plan.guesser.key, plan.condition["lie_count"], plan.game_id.rsplit("=", 1)[-1])
            groups.setdefault(key, {})[plan.condition["warned"]] = plan
            self.assertEqual(len(set(plan.scheduled_lie_turns)), len(plan.scheduled_lie_turns))
            self.assertEqual(len(plan.scheduled_lie_turns), plan.condition["lie_count"])
            self.assertTrue(1 <= min(plan.scheduled_lie_turns, default=1) <= 20)
        self.assertEqual(len(groups), 250)
        for conditions in groups.values():
            self.assertEqual(set(conditions), {False, True})
            unwarned, warned = conditions[False], conditions[True]
            self.assertEqual(unwarned.target.id, warned.target.id)
            self.assertEqual(unwarned.scheduled_lie_turns, warned.scheduled_lie_turns)
            self.assertEqual(unwarned.randomization_id, warned.randomization_id)
        counts = {}
        for plan in schedule:
            key = (plan.guesser.key, plan.condition["lie_count"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(sum(counts.values()), 500)
        self.assertEqual(set(counts.values()), {40, 42})

    def test_warned_game_keeps_gemma_chat_roles_alternating(self):
        plan = next(
            plan for plan in followups.build_liar_schedule(self.targets, 519)
            if plan.guesser.key == "gemma3-12b-it" and plan.condition["warned"]
        )
        _, messages = followups._game_messages(plan, self.targets)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("I am thinking of one of those concepts", messages[1]["content"])
        self.assertIn(plan.condition["warning_text"], messages[1]["content"])

    def test_noisy_schedule_has_300_games_and_matched_targets_by_rep(self):
        schedule = followups.build_noisy_schedule(self.targets, 519)
        self.assertEqual(len(schedule), 300)
        for model in e1.MODELS:
            for probability in followups.NOISE_RATES:
                games = [
                    plan for plan in schedule
                    if plan.guesser.key == model.key
                    and plan.condition["flip_probability"] == probability
                ]
                self.assertEqual(len(games), 25)
        targets_by_rate = {
            probability: {
                plan.game_id.rsplit("=", 1)[-1]: plan.target.id
                for plan in schedule if plan.condition["flip_probability"] == probability
            }
            for probability in followups.NOISE_RATES
        }
        self.assertTrue(
            all(targets_by_rate[probability] == targets_by_rate[followups.NOISE_RATES[0]]
                for probability in followups.NOISE_RATES[1:])
        )

    def test_one_scheduled_lie_flips_answer_delivered_to_guesser(self):
        plan = next(
            plan for plan in followups.build_liar_schedule(self.targets, 519)
            if plan.guesser.key == "gemma3-12b-it"
            and plan.condition["lie_count"] == 1
            and not plan.condition["warned"]
        )
        plan = replace(
            plan,
            condition={**plan.condition, "lie_turns": [1]},
            scheduled_lie_turns=(1,),
        )
        backend = ScriptedBackend(plan, plan.target)
        record = followups.play_followup_game(plan, self.targets, backend)
        answer_turn = next(turn for turn in record["turns"] if turn["type"] == "question")
        self.assertEqual(answer_turn["truthful_answer"], "Yes")
        self.assertEqual(answer_turn["answer"], "No")
        self.assertTrue(answer_turn["answer_flipped"])
        self.assertEqual(record["applied_flips"], 1)
        self.assertTrue(record["won"])

    def test_noisy_probability_one_flips_each_valid_answer(self):
        plan = next(
            plan for plan in followups.build_noisy_schedule(self.targets, 519)
            if plan.guesser.key == "gemma3-12b-it"
        )
        plan = replace(
            plan,
            condition={**plan.condition, "flip_probability": 1.0},
        )
        backend = ScriptedBackend(plan, plan.target)
        record = followups.play_followup_game(plan, self.targets, backend)
        answer_turn = next(turn for turn in record["turns"] if turn["type"] == "question")
        self.assertNotEqual(answer_turn["truthful_answer"], answer_turn["answer"])
        self.assertEqual(record["applied_flips"], 1)

    def test_wrong_listed_guess_spends_a_turn_in_followups(self):
        plan = next(
            plan for plan in followups.build_noisy_schedule(self.targets, 519)
            if plan.guesser.key == "qwen3-8b"
        )
        other = next(target for target in self.targets if target.id != plan.target.id)

        class WrongThenRight(ScriptedBackend):
            def generate(self, model, messages, seed):
                if model.key == self.plan.guesser.key:
                    self.guesser_calls += 1
                    name = other.name if self.guesser_calls == 1 else self.target.name
                    return e1.Generation(f"GUESS: {name}", 0.001)
                return e1.Generation("Yes", 0.001)

        record = followups.play_followup_game(plan, self.targets, WrongThenRight(plan, plan.target))
        self.assertTrue(record["won"])
        self.assertEqual(record["turns_used"], 2)
        self.assertEqual(record["invalid_moves"], 0)
        self.assertEqual([t["correct"] for t in record["turns"]], [False, True])
        self.assertEqual(record["run_config"]["guess_rule"], e1.GUESS_RULE_VERSION)

    def test_question_turn_logs_rejected_guesser_attempts(self):
        plan = next(
            plan for plan in followups.build_noisy_schedule(self.targets, 519)
            if plan.guesser.key == "granite-3.3-8b"
        )
        plan = replace(plan, condition={**plan.condition, "flip_probability": 0.0})

        class RetryThenAsk(ScriptedBackend):
            def generate(self, model, messages, seed):
                if model.key == self.plan.guesser.key:
                    self.guesser_calls += 1
                    if self.guesser_calls == 1:
                        return e1.Generation("GUESS: spaceship", 0.001)
                    if self.guesser_calls == 2:
                        return e1.Generation("QUESTION: Is it a living thing?", 0.001)
                    return e1.Generation(f"GUESS: {self.target.name}", 0.001)
                return e1.Generation("Yes", 0.001)

        record = followups.play_followup_game(plan, self.targets, RetryThenAsk(plan, plan.target))
        question = record["turns"][0]
        self.assertEqual(question["type"], "question")
        self.assertEqual([m["text"] for m in question["raw_moves"]],
                         ["GUESS: spaceship", "QUESTION: Is it a living thing?"])

    def test_resume_rejects_records_from_old_guess_rule(self):
        schedule = followups.build_noisy_schedule(self.targets, 519)
        plan = schedule[0]
        record = followups.play_followup_game(plan, self.targets, ScriptedBackend(plan, plan.target))
        record["inference_mode"] = "transformers"
        record["run_config"].pop("guess_rule")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(followups.FollowupError, "different guess rule"):
                followups._read_existing_games(path, plan.experiment, schedule)

    def test_pick_validation_rejects_category_mismatch(self):
        picks = make_picks(self.targets)
        picks[0]["category"] = "place"
        with self.assertRaisesRegex(followups.FollowupError, "metadata mismatch"):
            followups._validate_picks(picks)

    def test_summary_reports_wilson_interval_and_actual_flip_rate(self):
        records = [{
            "experiment": "E3_liars_20_questions",
            "answerer": {"key": "qwen3-8b"},
            "guesser": {"key": "gemma3-12b-it"},
            "condition": {"lie_count": 1, "warned": False},
            "won": True,
            "turns_used": 3,
            "answer_turns": 2,
            "applied_flips": 1,
            "turns": [
                {"answer_status": "valid"},
                {"answer_status": "valid"},
                {"answer_status": "guess"},
            ],
        }]
        summary = followups.summarize_games(records)
        group = summary["groups"][0]
        self.assertEqual(group["win_rate"], 1.0)
        self.assertAlmostEqual(group["win_rate_wilson_95_ci"][0], 0.2065493)
        self.assertEqual(group["win_rate_wilson_95_ci"][1], 1.0)
        self.assertEqual(group["flipped_answer_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from src.experiments import e1, e5_information as e5


class ScriptedBackend:
    mode = "test"

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def prepare_pair(self, answerer, guesser):
        del answerer, guesser

    def generate(self, model, messages, seed):
        del model, messages, seed
        return e1.Generation(next(self.outputs), 0.001)

    def close(self):
        return None

    def metadata(self):
        return {"inference_mode": self.mode}


class ExactInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.all_targets = e1.load_targets()
        cls.targets = [target for target in cls.all_targets if target.e1]
        cls.questions = e5.load_question_bank(cls.targets)

    def test_question_bank_covers_e1_targets_with_valid_binary_partitions(self):
        self.assertEqual(len(self.targets), 40)
        self.assertGreaterEqual(len(self.questions), 50)
        target_ids = {target.id for target in self.targets}
        self.assertTrue(all(question.yes_target_ids <= target_ids for question in self.questions))
        self.assertTrue(all(0 < len(question.yes_target_ids) < len(target_ids)
                            for question in self.questions))
        signatures = {
            tuple(target.id in question.yes_target_ids for question in self.questions)
            for target in self.targets
        }
        self.assertEqual(len(signatures), len(self.targets))

    def test_schedule_has_160_model_target_games(self):
        schedule = e5.build_schedule(self.all_targets, 519)
        self.assertEqual(len(schedule), 160)
        for model in e1.MODELS:
            model_games = [plan for plan in schedule if plan.model.key == model.key]
            self.assertEqual(len(model_games), 40)
            self.assertEqual({plan.target.id for plan in model_games},
                             {target.id for target in self.targets})

    def test_information_gain_matches_known_split(self):
        question = next(question for question in self.questions if question.question_id == "Q01")
        self.assertAlmostEqual(
            e5.expected_information_gain({target.id for target in self.targets}, question),
            e5._entropy(0.25),
        )

    def test_parser_accepts_only_available_question_and_remaining_guess(self):
        chair = next(target for target in self.targets if target.id == "object-chair")
        target_by_id = {target.id: target for target in self.targets}
        q01 = [question for question in self.questions if question.question_id == "Q01"]
        candidates = {target.id for target in self.targets}
        self.assertEqual(
            e5.parse_action("QUESTION: Q01", q01, candidates, target_by_id),
            ("question", "Q01"),
        )
        self.assertIsNone(e5.parse_action("QUESTION: Q02", q01, candidates, target_by_id))
        self.assertEqual(
            e5.parse_action(f"GUESS: {chair.name}", q01, candidates, target_by_id),
            ("guess", chair.id),
        )
        self.assertIsNone(
            e5.parse_action("GUESS: not a target", q01, candidates, target_by_id)
        )

    def test_game_uses_exact_oracle_answer_and_records_information_gain(self):
        target = next(target for target in self.targets if target.id == "object-chair")
        plan = e5.GamePlan("test-game", 519, e1.MODELS[0], target)
        backend = ScriptedBackend(["QUESTION: Q01", f"GUESS: {target.name}"])
        record = e5.play_game(plan, self.targets, self.questions, backend)
        self.assertTrue(record["won"])
        self.assertEqual(record["questions_asked"], 1)
        question_turn = record["turns"][0]
        self.assertEqual(question_turn["answer"], "Yes")
        self.assertEqual(question_turn["candidate_count_after"], 10)
        self.assertGreater(question_turn["expected_information_gain_bits"], 0)
        self.assertEqual(record["turns"][1]["action"], "guess")

    def test_each_turn_sends_only_system_and_current_state(self):
        target = next(target for target in self.targets if target.id == "object-chair")
        plan = e5.GamePlan("test-game", 519, e1.MODELS[0], target)
        seen = []

        class Recording(ScriptedBackend):
            def generate(self, model, messages, seed):
                seen.append([message["role"] for message in messages])
                return super().generate(model, messages, seed)

        backend = Recording(["QUESTION: Q01", "nonsense", f"GUESS: {target.name}"])
        record = e5.play_game(plan, self.targets, self.questions, backend)
        self.assertTrue(record["won"])
        self.assertEqual(record["run_config"]["dialogue"], e5.DIALOGUE)
        self.assertEqual(seen[0], ["system", "user"])
        self.assertEqual(seen[1], ["system", "user"])  # fresh context on turn 2
        self.assertEqual(seen[2], ["system", "user", "assistant", "user"])  # in-turn retry

    def test_greedy_baseline_identifies_every_target_within_turn_limit(self):
        for target in self.targets:
            result = e5.simulate_baseline(
                target, self.targets, self.questions, "greedy", 519
            )
            self.assertTrue(result["won"], target.name)
            self.assertLessEqual(result["turns_used"], e5.MAX_TURNS)

    def test_runner_writes_resumable_jsonl_and_summary_for_mock_games(self):
        schedule = e5.build_schedule(self.all_targets, 519)
        chair_plan = next(plan for plan in schedule if plan.target.id == "object-chair")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "e5.jsonl"
            records = e5._run(
                ScriptedBackend(["GUESS: chair"]),
                [chair_plan],
                self.targets,
                self.questions,
                path,
                max_games=1,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len(e5._read_existing(path, [chair_plan])), 1)
            resumed = e5._run(
                ScriptedBackend([]),
                [chair_plan],
                self.targets,
                self.questions,
                path,
                max_games=1,
            )
            self.assertEqual(len(resumed), 1)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
            summary = e5.summarize(records)
            self.assertEqual(summary["games"], 1)
            self.assertEqual(summary["groups"][0]["greedy_baseline_win_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

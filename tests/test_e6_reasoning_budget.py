import json
import tempfile
import unittest
from pathlib import Path

from src.experiments import e1, e5_information as e5, e6_reasoning_budget as e6


END = 99
EOS = {0}


class ThinkingContinuationTests(unittest.TestCase):
    def test_closed_and_finished_within_budget_is_done(self):
        step = e1.plan_thinking_continuation([5, 6, END, 7, 0], END, EOS, 256)
        self.assertEqual((step.action, step.thinking_tokens), ("done", 3))

    def test_closed_but_cut_at_budget_continues_answer(self):
        step = e1.plan_thinking_continuation([5, END, 7, 8], END, EOS, 4)
        self.assertEqual((step.action, step.thinking_tokens), ("continue_answer", 2))

    def test_unclosed_thinking_is_force_closed(self):
        step = e1.plan_thinking_continuation([5, 6, 7, 8], END, EOS, 4)
        self.assertEqual((step.action, step.thinking_tokens), ("force_close", 4))


class ActionExtractionTests(unittest.TestCase):
    def test_takes_last_action_line(self):
        text = "Let me consider QUESTION: Q01 first.\nQUESTION: Q02\nQUESTION: Q07"
        self.assertEqual(e6.extract_action(text), "QUESTION: Q07")

    def test_strips_markdown_wrapping(self):
        self.assertEqual(e6.extract_action("**GUESS: chair**"), "GUESS: chair")

    def test_returns_none_without_action(self):
        self.assertIsNone(e6.extract_action("I am not sure."))


class ReasoningBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.all_targets = e1.load_targets()
        cls.targets = [target for target in cls.all_targets if target.e1]
        cls.questions = e5.load_question_bank(cls.targets)

    def test_schedule_is_matched_target_by_condition(self):
        schedule = e6.build_schedule(self.all_targets, 519)
        self.assertEqual(len(schedule), 40 * len(e6.CONDITIONS))
        self.assertTrue(all(plan.game.model.key == "qwen3-8b" for plan in schedule))
        self.assertEqual(len({plan.game.game_id for plan in schedule}), len(schedule))
        for start in range(0, len(schedule), len(e6.CONDITIONS)):
            triple = schedule[start:start + len(e6.CONDITIONS)]
            self.assertEqual(len({plan.game.target.id for plan in triple}), 1)
            self.assertEqual(len({plan.game.randomization_id for plan in triple}), 1)
            self.assertEqual([plan.condition for plan in triple], list(e6.CONDITIONS))

    def test_conditions_are_off_256_1024(self):
        self.assertEqual([c.thinking_budget for c in e6.CONDITIONS], [0, 256, 1024])
        self.assertFalse(e6.CONDITIONS[0].enable_thinking)

    def test_backend_routes_conditions_and_hides_reasoning(self):
        backend = e6.ReasoningBackend(e6.MockBackend())
        messages = [{"role": "user", "content": "Q03: Is it alive?"}]
        backend.start_game(e6.CONDITIONS[0])
        self.assertEqual(backend.generate(e6.MODEL, messages, 1).text, "QUESTION: Q03")
        self.assertEqual(backend.calls[-1]["thinking_tokens"], 0)
        backend.start_game(e6.CONDITIONS[1])
        backend.generate(e6.MODEL, messages, 1)
        self.assertEqual(backend.calls, [{
            **backend.calls[0],
            "thinking_tokens": 256,
            "budget_exhausted": True,
            "output_tokens": 262,
        }])
        backend.start_game(e6.CONDITIONS[2])
        backend.generate(e6.MODEL, messages, 1)
        self.assertFalse(backend.calls[-1]["budget_exhausted"])
        self.assertEqual(backend.calls[-1]["thinking_tokens"], 400)

    def test_mock_run_resumes_and_summarizes_matched_pairs(self):
        schedule = e6.build_schedule(self.all_targets, 519)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e6.jsonl"
            backend = e6.ReasoningBackend(e6.MockBackend())
            first = e6.run(schedule, self.targets, self.questions, path, backend, max_games=4)
            self.assertEqual(len(first), 4)
            records = e6.run(schedule, self.targets, self.questions, path, backend, max_games=2)
            self.assertEqual(len(records), 6)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 6)
            record = json.loads(lines[1])
            self.assertEqual(record["experiment"], e6.EXPERIMENT)
            self.assertEqual(record["reasoning_condition"]["thinking_budget_tokens"], 256)
            self.assertTrue(record["reasoning_calls"])

        summary = e6.summarize(records)
        self.assertEqual(summary["games"], 6)
        self.assertEqual([group["games"] for group in summary["groups"]], [2, 2, 2])
        self.assertEqual(
            [(p["reference_condition"], p["condition"]) for p in summary["paired_comparisons"]],
            [("thinking_off", "thinking_256"), ("thinking_off", "thinking_1024"),
             ("thinking_256", "thinking_1024")],
        )
        self.assertTrue(all(p["matched_targets"] == 2 for p in summary["paired_comparisons"]))
        off, small, large = summary["groups"]
        self.assertEqual(off["mean_thinking_tokens"], 0)
        self.assertGreater(large["mean_thinking_tokens"], small["mean_thinking_tokens"])
        self.assertEqual(small["mean_budget_exhausted_rate"], 1.0)

    def test_rejects_foreign_records_on_resume(self):
        schedule = e6.build_schedule(self.all_targets, 519)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e6.jsonl"
            path.write_text(json.dumps({"experiment": "E5", "game_id": "x"}) + "\n")
            with self.assertRaises(e6.ReasoningExperimentError):
                e6.read_existing(path, schedule)


if __name__ == "__main__":
    unittest.main()

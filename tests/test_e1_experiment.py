import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.experiments import e1


ROOT = Path(__file__).resolve().parents[1]


class SequenceBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def prepare_pair(self, answerer, guesser):
        return None

    def generate(self, model, messages, seed):
        del model, messages, seed
        return e1.Generation(next(self.outputs), 0.001)

    def close(self):
        return None


class SmokeBackend:
    def __init__(self, fail_model=None):
        self.fail_model = fail_model
        self.tested = []

    def prepare_pair(self, answerer, guesser):
        if answerer.key == self.fail_model or guesser.key == self.fail_model:
            raise e1.ExperimentError("simulated model access failure")
        self.tested.extend((answerer.key, guesser.key))

    def generate(self, model, messages, seed):
        del messages, seed
        return e1.Generation(f"OK {model.key}", 0.001)

    def close(self):
        return None


class E1ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool_id, cls.targets = e1.load_targets()
        cls.schedule = e1.build_schedule(cls.targets, 519)

    def test_target_manifest_is_stratified_and_has_40_e1_secrets(self):
        self.assertEqual(self.pool_id, "twenty-questions-targets-v1")
        self.assertEqual(len(self.targets), 128)
        self.assertEqual(sum(target.e1 for target in self.targets), 40)
        for category in e1.CATEGORIES:
            category_targets = [target for target in self.targets if target.category == category]
            self.assertEqual(len(category_targets), 32)
            self.assertEqual(sum(target.e1 for target in category_targets), 10)
            for frequency in ("common", "rare"):
                self.assertEqual(
                    sum(target.e1 and target.frequency == frequency for target in category_targets),
                    5,
                )

    def test_schedule_covers_all_ordered_pairs_with_same_40_targets(self):
        self.assertEqual(len(self.schedule), 640)
        pairings = {(spec.answerer.key, spec.guesser.key) for spec in self.schedule}
        self.assertEqual(len(pairings), 16)
        target_sets = [
            {
                spec.target.id for spec in self.schedule
                if spec.answerer.key == answerer.key and spec.guesser.key == guesser.key
            }
            for answerer in e1.MODELS for guesser in e1.MODELS
        ]
        self.assertTrue(all(len(targets) == 40 for targets in target_sets))
        self.assertTrue(all(targets == target_sets[0] for targets in target_sets))

    def test_schedule_can_be_split_by_gpu_requirements(self):
        self_play = e1.select_pairings(self.schedule, "self-play")
        cross_play = e1.select_pairings(self.schedule, "cross-play")
        self.assertEqual(len(self_play), 160)
        self.assertEqual(len(cross_play), 480)
        self.assertTrue(all(spec.answerer.key == spec.guesser.key for spec in self_play))
        self.assertTrue(all(spec.answerer.key != spec.guesser.key for spec in cross_play))

    def test_schedule_order_is_stable_for_a_seed(self):
        first = [spec.game_id for spec in e1.build_schedule(self.targets, 519, self.pool_id)]
        second = [spec.game_id for spec in e1.build_schedule(self.targets, 519, self.pool_id)]
        other_seed = [spec.game_id for spec in e1.build_schedule(self.targets, 520, self.pool_id)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)

    def test_game_retries_malformed_outputs_without_losing_turns(self):
        target = next(target for target in self.targets if target.id == "object-chair")
        spec = next(
            spec for spec in self.schedule
            if spec.target.id == target.id and spec.answerer.key != spec.guesser.key
        )
        backend = SequenceBackend([
            "I would like to know if it is alive.",
            "QUESTION: Is it alive?",
            "Maybe",
            "Yes.",
            "GUESS: chair",
        ])
        game = e1.play_game(spec, self.targets, backend)
        self.assertTrue(game["won"])
        self.assertEqual(game["turns_used"], 2)
        self.assertEqual(game["invalid_answers"], 0)
        self.assertEqual(game["invalid_moves"], 0)
        self.assertEqual(len(game["turns"][0]["raw_answers"]), 2)

    def test_wrong_listed_guess_spends_a_turn_instead_of_retrying(self):
        target = next(target for target in self.targets if target.id == "object-chair")
        other = next(t for t in self.targets if t.e1 and t.id != target.id)
        spec = next(
            spec for spec in self.schedule
            if spec.target.id == target.id and spec.answerer.key != spec.guesser.key
        )
        backend = SequenceBackend([f"GUESS: {other.name}", "GUESS: chair"])
        game = e1.play_game(spec, self.targets, backend)
        self.assertTrue(game["won"])
        self.assertEqual(game["turns_used"], 2)
        self.assertEqual(game["invalid_moves"], 0)
        self.assertEqual(game["turns"][0]["type"], "guess")
        self.assertFalse(game["turns"][0]["correct"])
        self.assertEqual(len(game["turns"][0]["raw_responses"]), 1)

    def test_unlisted_guess_is_a_format_error(self):
        target = next(target for target in self.targets if target.id == "object-chair")
        spec = next(
            spec for spec in self.schedule
            if spec.target.id == target.id and spec.answerer.key != spec.guesser.key
        )
        backend = SequenceBackend(["GUESS: spaceship", "GUESS: chair"])
        game = e1.play_game(spec, self.targets, backend)
        self.assertTrue(game["won"])
        self.assertEqual(game["turns_used"], 1)
        self.assertEqual(len(game["turns"][0]["raw_responses"]), 2)

    def test_unparseable_answer_is_explicit_and_spends_one_turn(self):
        target = next(target for target in self.targets if target.id == "object-chair")
        spec = next(
            spec for spec in self.schedule
            if spec.target.id == target.id and spec.answerer.key != spec.guesser.key
        )
        backend = SequenceBackend([
            "QUESTION: Is it alive?",
            "Sometimes",
            "Maybe",
            "I cannot answer",
            "GUESS: chair",
        ])
        game = e1.play_game(spec, self.targets, backend)
        self.assertTrue(game["won"])
        self.assertEqual(game["invalid_answers"], 1)
        self.assertIsNone(game["turns"][0]["answer"])
        self.assertEqual(game["turns"][0]["answer_status"], "invalid")

    def test_record_validation_rejects_mismatched_model_revision(self):
        spec = self.schedule[0]
        game = {
            "schema_version": 1,
            "experiment": "E1",
            "game_id": spec.game_id,
            "seed": spec.seed,
            "answerer": e1.asdict(spec.answerer),
            "guesser": e1.asdict(spec.guesser),
            "target_pool": spec.target_pool,
            "target": e1.target_record(spec.target),
            "won": False,
            "turns_used": 1,
            "turns": [{"turn": 1, "type": "invalid_move"}],
        }
        game["answerer"]["revision"] = "different"
        with self.assertRaisesRegex(e1.ExperimentError, "metadata mismatch"):
            e1.validate_records([game], self.schedule, require_complete=False)

    def test_cluster_bootstrap_reports_stable_interval_for_constant_outcome(self):
        spec = self.schedule[:40]
        records = [
            {
                "target": e1.target_record(game.target),
                "answerer": e1.asdict(game.answerer),
                "guesser": e1.asdict(game.guesser),
                "won": True,
            }
            for game in spec
        ]
        low, high = e1.cluster_bootstrap_ci(
            records, lambda record: True, lambda record: float(record["won"]), seed=5
        )
        self.assertEqual((low, high), (1.0, 1.0))

    def test_summary_has_all_pairings_and_role_intervals(self):
        records = [
            {
                "target": e1.target_record(spec.target),
                "answerer": e1.asdict(spec.answerer),
                "guesser": e1.asdict(spec.guesser),
                "inference_mode": "transformers",
                "won": index % 2 == 0,
                "turns_used": 7,
                "invalid_answers": 0,
                "invalid_moves": 0,
            }
            for index, spec in enumerate(self.schedule)
        ]
        summary = e1.summarize_results(records, seed=519)
        self.assertEqual(summary["games"], 640)
        self.assertEqual(len(summary["pairings"]), 16)
        self.assertEqual(len(summary["roles"]), 4)
        self.assertEqual(len(summary["paired_model_comparisons"]["guesser"]), 6)
        self.assertEqual(len(summary["paired_model_comparisons"]["answerer"]), 6)
        for role in ("guesser", "answerer"):
            for comparison in summary["paired_model_comparisons"][role]:
                self.assertGreaterEqual(comparison["holm_adjusted_p_value"], 0)
                self.assertLessEqual(comparison["holm_adjusted_p_value"], 1)
        for role in summary["roles"].values():
            self.assertEqual(len(role["guesser_target_clustered_95_ci"]), 2)
            self.assertEqual(len(role["answerer_target_clustered_95_ci"]), 2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = Path(temporary_directory) / "e1.jsonl"
            summary_file = e1.write_summary(
                summary, results_path, Path(temporary_directory) / "reports"
            )
            saved = json.loads(summary_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["games"], 640)

    def test_merge_requires_and_validates_complete_partition_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self_play = root / "self.jsonl"
            cross_play = root / "cross.jsonl"
            destination = root / "merged.jsonl"
            partition_records = {
                "self-play": [],
                "cross-play": [],
            }
            for spec in self.schedule:
                record = {
                    "schema_version": 1,
                    "experiment": "E1",
                    "inference_mode": "transformers",
                    "run_config": e1.RUN_CONFIG,
                    "game_id": spec.game_id,
                    "seed": spec.seed,
                    "target_pool": spec.target_pool,
                    "answerer": e1.asdict(spec.answerer),
                    "guesser": e1.asdict(spec.guesser),
                    "target": e1.target_record(spec.target),
                    "won": False,
                    "turns_used": 1,
                    "turns": [{"turn": 1, "type": "invalid_move"}],
                }
                group = "self-play" if spec.answerer.key == spec.guesser.key else "cross-play"
                partition_records[group].append(record)
            self_play.write_text(
                "".join(json.dumps(record) + "\n" for record in partition_records["self-play"]),
                encoding="utf-8",
            )
            cross_play.write_text(
                "".join(json.dumps(record) + "\n" for record in partition_records["cross-play"]),
                encoding="utf-8",
            )

            count = e1.merge_result_files([cross_play, self_play], destination, self.schedule)
            self.assertEqual(count, 640)
            merged = e1.read_records(destination)
            e1.validate_records(merged, self.schedule, require_complete=True)
            with self.assertRaisesRegex(e1.ExperimentError, "already exists"):
                e1.merge_result_files([cross_play, self_play], destination, self.schedule)

    def test_mock_partial_run_continues_without_duplicate_games(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "partial.jsonl"
            command = [
                sys.executable,
                "-m",
                "src.experiments.e1",
                "--mock",
                "--max-games",
                "2",
                "--results-file",
                str(output),
            ]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("2 already complete; 2 to run", second.stdout)
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual(len({json.loads(line)["game_id"] for line in lines}), 4)

    def test_mock_game_cap_is_rejected_for_real_inference(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = e1.main(["--max-games", "1"])
        self.assertEqual(result, 1)
        self.assertIn("--max-games requires --mock", stderr.getvalue())

    def test_mock_records_cannot_be_reported_as_model_results(self):
        with self.assertRaisesRegex(e1.ExperimentError, "cannot be summarized"):
            e1.summarize_results([{"inference_mode": "mock"}], seed=519)

    def test_smoke_continues_after_one_model_access_failure(self):
        backend = SmokeBackend(fail_model="gemma3-12b-it")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = e1.run_smoke(backend)
        self.assertEqual(result, 1)
        self.assertIn("FAILED Gemma 3 12B IT", stdout.getvalue())
        self.assertIn("OK Granite 3.3 8B Instruct", stdout.getvalue())

    def test_per_model_precision_is_recorded_in_run_config(self):
        expected = {
            "qwen3-8b": "float16",
            "gemma3-12b-it": "float32",
            "mistral-nemo-12b": "float16",
            "granite-3.3-8b": "float16",
        }
        configured = {
            model.key: model.inference_dtype for model in e1.MODELS
        }
        self.assertEqual(configured, expected)
        self.assertEqual(e1.RUN_CONFIG["model_dtypes"], expected)


if __name__ == "__main__":
    unittest.main()

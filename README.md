# How Good Are LLMs at 20 Questions?

This project studies how language models perform as both the answerer (who knows the secret) and the guesser (who asks questions and guesses) in 20 Questions. The Python runner supports local Hugging Face Transformers models and an Ollama-compatible local endpoint, and records turn-level JSONL logs. The checked-in pilot results are exploratory feasibility runs; they are not enough to rank models or establish general conclusions.

## Run a game experiment

Use Python with `torch`, `transformers`, `accelerate`, and `openai` installed. Install a PyTorch build suitable for your hardware, then install the remaining packages with:

```powershell
python -m pip install transformers accelerate openai
```

For a quick pipeline check that does not load language models or call an API:

```powershell
python twenty_questions_pilot.py --mock --results-file mock_games.jsonl
```

For real inference, make sure the Ollama server and `llama3.1:8b` are available locally. Set `OLLAMA_BASE_URL` if the server is not at `http://localhost:11434/v1`; the runner defaults `OLLAMA_API_KEY` to `ollama` for the local endpoint. The Hugging Face checkpoints are downloaded and run locally on first use. Then run:

```powershell
python twenty_questions_pilot.py --results-file hf_ollama_games.jsonl
```

The default cohort has four models, three fixed target concepts, and one game per ordered answerer/guesser pairing (16 pairings; 48 games total). Existing game IDs are skipped when resuming the same output file. Use `--repeats N` for more games per pairing and target, `--model-set additional` for the other configured cohort, or `--smoke-models` to test model loading/generation. Review the runner's `--help` output for all options. These runs consume compute and may download model weights; choose an appropriate GPU and output path before running a larger experiment.

The notebook, `Abstract.ipynb`, contains the two preflight workflows. On Stony Brook's AI3 cluster, follow its access and SLURM guidance and run scripts as scheduled jobs rather than running notebooks on the server.

## Preliminary pilot results

| Pilot output | Games | Wins | Invalid answer attempts / answer turns |
|---|---:|---:|---:|
| `free_hf_ollama_games.jsonl` (original cohort) | 48 | 2 (4.2%) | 91 / 729 (12.5%) |
| `additional_hf_ollama_games.jsonl` (additional cohort) | 48 | 10 (20.8%) | 64 / 754 (8.5%) |

These are one-repeat preflights across three fixed targets, not a powered comparison. The 50 picks in `picks.jsonl` are a separate sample: “desk lamp” appears 16 times and “toaster oven” 13 times. Neither the game pilots nor the pick sample establish which model is better or whether model selection is statistically non-uniform.

## Research plan

The planned study will compare all 16 ordered role pairings among four models, including self-play with isolated role histories. It will add a defined target-sampling experiment, information-gain scoring for questions, and independent answer-key evaluation for oracle accuracy. The final model cohort and sample size remain provisional pending compute access, pilot validation, and power analysis. See [the proposal](./project_proposal.md) for the study design and references.

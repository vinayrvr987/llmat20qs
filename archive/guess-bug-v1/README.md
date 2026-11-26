# Results recorded under the v1 guess rule (superseded)

These E1 and E2–E4 game logs were produced before the fix in
`GUESS_RULE_VERSION = "listed-candidate-guess-spends-turn-v2"`. Under v1, a
wrong `GUESS:` was treated as a format error and retried within the same turn,
so wrong guesses cost nothing and leaked "not this one" feedback. About half of
Qwen3 8B's wins (and some of Granite's) depend on such retries. Win rates here
are not valid; the files are kept only for auditing. E3 and E2 scouting are
partial (jobs cancelled on 2026-11-25). The E2 pick file was not affected and
remains in `data/raw/results/`.

# Raw experiment results

Append-only model picks and turn-level game logs written by the experiment
runners. Do not edit records in place.

The committed files are the final v2 runs (completed 2026-11-26): every game
record has `guess_rule = listed-candidate-guess-spends-turn-v2` and
`prompt_version = e1-prompts-v1` (E5/E6: `dialogue = current-state-only-v2`).

| File | Experiment | Records |
|---|---|---:|
| `e1-cross-play-r1.jsonl` | E1 cross-play (12 pairings × 40) | 480 |
| `e1-self-play-r1.jsonl` | E1 self-play (4 pairings × 40) | 160 |
| `e2-predictable-picks-s519.jsonl` | E2 model target picks (4 × 100) | 400 |
| `e2-predictable-games-s519.jsonl` | E2 predictable minds | 640 |
| `e2-scouting-s519.jsonl` | E2 self-secret/scouting | 640 |
| `e3-liar-s519.jsonl` | E3 liar | 500 |
| `e4-noisy-s519.jsonl` | E4 noisy oracle | 300 |
| `e5-exact-information-s519.jsonl` | E5 exact-information pilot | 160 |
| `e6-reasoning-budget-s519.jsonl` | E6 reasoning budget | 120 |

E2 and E3 started before guesser attempts were logged on question turns
(`raw_moves`); their question turns lack that field. Other output written here
(mock or test runs) is ignored by Git. The invalid v1 logs are kept locally
under `archive/guess-bug-v1/` and are not committed.

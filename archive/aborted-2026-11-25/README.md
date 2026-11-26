# Partial runs cancelled on 2026-11-25 (protocol changed mid-run)

* `e1-*`, `e4-noisy-*`: v2 guess rule **plus** a turns-left reminder in guesser
  feedback (`e1-prompts-v2`). The reminder made Gemma 3 delay guessing until the
  final turn (17 -> 13 wins on the same 22 E1 games), so it was dropped. The
  re-runs use the v2 guess rule with the original `e1-prompts-v1` prompts.
* `e5-*`, `e6-*`: each turn appended the full state prompt to the dialogue.
  E5 ran out of GPU memory on Gemma 3 (float32). The re-runs send only the
  system prompt and the current state prompt (`dialogue = current-state-only-v2`).

Not valid for analysis; kept for auditing.

# How Good Are LLMs at 20 Questions?

**Project proposal**

**Instructor:** Professor Steven Skiena

## Problem

20 Questions is a great test of many skills, such as choosing a secret, asking good questions, being truthful about the secret, and choosing when to guess. There are many ways a model can lose: being misled by a lie from the answerer, dealing with a particularly difficult secret, or asking uninformative questions. These causes are not easily separated from the win rate.

We will play all 16 directed pairings of 4 LLMs, using each model as both guesser and answerer. We will separate the individual factors that affect performance.

### Research questions

1. **Guesser, answerer, and combination:** How do each model's guesser ability, answerer ability, and pairing affect winning? Does a model play better against itself?
2. **Randomness:** To what extent are the model's "random" secret choices truly random? Is there a bias that an opponent can exploit?
3. **Information in questions:** How much information is gained by each question compared with the ideal 1 bit from a perfect yes/no split?
4. **Answer accuracy:** How often are answers wrong? How costly is a wrong answer to the guesser?

This game is useful because we know the optimal play. For *N* equally likely secrets, a perfect guesser needs approximately log₂ *N* questions. We also know the approximate cost of each lie from the answerer. If the answerer is wrong with fraction ε, it has at most 1 − h(ε) bits per answer. We can compare model behavior with these theoretical optima rather than only with other models.

## Approach and evaluation

### Models

The first implementation phase uses four instruction-tuned open-weight checkpoints from four developers. Each is pinned to a Hugging Face revision:

| Model | Checkpoint | Revision |
|---|---|---|
| Qwen3 8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| Gemma 3 12B IT | `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| Mistral Nemo 12B Instruct | `mistralai/Mistral-Nemo-Instruct-2407` | `04d8a90549d23fc6bd7f642064003592df51e9b3` |
| Granite 3.3 8B Instruct | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |

This 8B–12B cohort balances the professor's feedback about very small models against the available cluster memory. The models differ in size, architecture, and training, so this is a comparison of model systems, not a causal test of parameter count. Inference uses local Transformers on Quadro RTX 8000 GPUs. Gemma runs in FP32 because its FP16 logits were NaN on V100 and Quadro RTX 8000 hardware; the other three models run in FP16. This mixed precision is a practical compromise for the available GPUs and is a potential confound in comparisons involving Gemma. A pairing uses one GPU per distinct player model. The same role prompts and decoding configuration are used throughout E1. No judge model is included in this E1 implementation.

### Secrets

The versioned target manifest contains a project-curated fixed pool of 128 concepts:

- 32 objects
- 32 animals
- 32 places
- 32 famous people

There are 32 concepts per category and 16 common / 16 rare concepts within each category. The common/rare labels are operational strata, not measured corpus-frequency estimates. The manifest is informed by the proposal's WordNet/Wikidata source plan. E1 uses the same fixed, stratified subset of 40 concepts in every pairing: 10 per category, with five common and five rare in each. The shared targets allow matched comparisons across model pairings.

## E1: assigned-secret round robin (initial implementation)

E1 includes all 16 directed pairings of four models, including self-play, with 40 assigned-secret games per pairing (640 games total). The answerer and guesser have separate conversation histories. Games stop at a correct guess or after 20 turns. A guess naming any listed candidate spends a turn; a wrong guess is answered No. (A first run treated wrong guesses as formatting retries, which gave free guesses; it was discarded and re-run.) The runner records each turn, model revision, seed, formatting retry, and result in resumable JSONL logs. Cross-play jobs request two GPUs; self-play jobs request one GPU, avoiding an idle allocation.

The initial E1 report includes pairwise win rates, target-clustered 95% bootstrap intervals, paired sign-flip permutation tests of model-role differences with Holm correction, turns used, invalid answers, and invalid moves. E1 does not report factual answer accuracy or question information gain: free-form questions need a reviewed answer key before those metrics are valid. The new implementation and run instructions are in the repository README.

## Follow-up experiments (implemented; pending execution)

| Experiment | Proposed design | Listed games |
|---|---|---:|
| E2. Predictable minds | Each model selects 100 secrets from the fixed 128-target pool (25 picks per category); 16 ordered pairings × 40 games, sampling each answerer's picks once and matching targets across guessers | 640 |
| E2. Self-secret/scouting comparison | 4 self-play pairings × 80 sampled model picks × 2 conditions (no scouting report / top-five favorite-pick report) | 640 |
| E3. Liar's 20 questions | Qwen3 8B is the fixed answerer; 4 guessers × 0/1/2 scheduled flips × warned/unwarned; 500 games balanced across 24 cells, with 20 or 21 matched games per cell | 500 |
| E4. Noisy oracle | Qwen3 8B is the fixed answerer; 4 guessers × 5%/10%/17% independent answer-flip probability × 25 games | 300 |
| E5. Exact-information pilot | Four models each play 40 shared E1 targets against a deterministic oracle; choose from a fixed question bank with exact binary labels; compare expected/realized information gain with greedy and random baselines | 160 |
| E6. Reasoning-budget pilot ("more thinking, more bits?") | Qwen3 8B plays the E5 setup (same 40 targets, question bank, oracle) with thinking off, a 256-token, and a 1024-token thinking budget; conditions matched by target and sampling seed | 120 |

E2 uses model-generated picks constrained to the versioned 128-concept pool so the secret is known to the game and can be compared with a uniform-pick baseline. The pick distribution is descriptive because 100 picks per model provide limited power. Scouting conditions use matched secrets drawn from the same model's pick sample.

In E3 and E4, Qwen3 provides the baseline natural-language Yes/No answers and the game controller applies the assigned corruption. These tests therefore measure resilience to flips on top of Qwen3's own answer behavior, not a gold factual oracle. E3's warning says that some answers may be changed but does not reveal the count or turns. Flip positions are sampled without replacement among the first 20 answer turns; if a game ends early, fewer scheduled flips may be applied, which is recorded. E3's 500 games are allocated as 20 or 21 replicates for each warned/unwarned cell, paired on the same target and flip schedule.

The four original follow-up rows total 2,080 games; E5 adds a 160-game pilot. E5 measures question selection against a fixed, project-curated binary question bank rather than free-form questions; its feature labels need human review, and its results should be treated as a controlled pilot rather than a direct estimate of open-ended question information gain.

E6 implements the reasoning-budget idea from Future work as a controlled pilot. Unlike adding games or larger models to the round robin, it tests a new causal question: does extra test-time reasoning improve question quality, and at what compute cost? Everything except the thinking budget is held fixed (model, targets, question bank, oracle, sampling settings, and per-turn seeds), so conditions are compared target by target with paired bootstrap intervals rather than pooled with earlier totals. Budgets are enforced by budget forcing: at the cap, the controller closes the reasoning block and allows at most 32 answer tokens; reasoning traces are excluded from the dialogue history. Primary outcomes are expected bits per question and optimality ratio relative to the best available question; secondary outcomes are turns, win rate (expected to be near ceiling with an exact oracle), invalid actions, generated and thinking tokens, budget-exhaustion rate, and wall-clock generation time, summarized as bits gained per 1,000 extra tokens. With 40 targets, E6 can detect only moderate effects and is reported as a pilot. Together with E5, the plan totals 2,360 follow-up games and 3,000 including E1.

## Evaluation

### Information gain

If a fraction p of the remaining secrets would answer Yes, question information gain is:

H(p) = −p log₂(p) − (1 − p) log₂(1 − p).

The maximum is 1 bit for an even split. We will report mean information gain, wasted questions below 0.1 bits, and questions asked after the remaining answer was already determined. We will consider results against random-question and ideal-partition baselines.

### Answer accuracy

For a future answer-key phase, every question can be labeled Yes or No for the candidate concepts, with disagreements resolved by a person and a sample hand-checked. E1 preserves raw questions and answers for that annotation. The judge model and annotation protocol are not part of the current E1 implementation.

For each answerer, we will estimate:

- The error rate ε
- Information capacity 1 − h(ε)
- The turn on which a wrong answer made the secret unreachable

The proposed 70B Llama judge and independent human review remain future design choices.

### Winning

For E1 we report win rates with target-clustered 95% bootstrap intervals because the same targets are reused across pairings. Future phases can fit models including guesser, answerer, same-model pairing, secret type, and concept category. Unsolved games stop after 20 turns; survival analysis remains a planned follow-up.

### Randomness

We will measure:

- Distinct picks
- Shannon entropy
- Chi-squared deviation from a seeded uniform sampler

The gap log₂ N − H estimates how many questions an informed opponent could save. We will compare that value with the actual savings produced by a scouting report.

### Lies and noise

For deliberate lies and noisy answers, we will measure extra questions compared with the theoretical reference of approximately 2.8 extra questions for 128 secrets. We will also measure how often guessers recheck an answer after a lie.

## Future work

- **Larger open models:** Use additional cluster memory to compare players such as Llama 3.3 70B with the 4 main models on the same secrets.
- **Reasoning budgets:** Implemented as the E6 pilot (Qwen3 8B, three budgets). Natural extensions are more budget levels, other reasoning-capable models, and free-form rather than menu-based questions.
- **Strategic liars:** Let the answerer decide when to lie to cause the most damage instead of introducing random lies.
- **Committees of guessers:** Have several models discuss one question and test whether a mixed team outperforms the best individual model.
- **Model size:** Compare different sizes from the same model family.
- **Human baseline:** Let humans play as both guesser and answerer against the models.

## References

1. Bertolazzi, L., Mazzaccara, D., Merlo, F., & Bernardi, R. (2023). "ChatGPT's Information Seeking Strategy: Insights from the 20-Questions Game." *Proceedings of INLG 2023*, 153–162. ACL Anthology.
2. Hutson, D., Vennemeyer, D., Deshmukh, Zhan, & Jiang (2025). "GuessingGame: Measuring the Informativeness of Open-Ended Questions in Large Language Models." *Proceedings of EMNLP 2025*. ACL Anthology.
3. Dagan, Y., Filmus, Y., Kane, D., & Moran, S. (2021). "The Entropy of Lies: Playing Twenty Questions with a Liar." *ITCS 2021*. arXiv:1811.02177.
4. Choudhury, D., et al. (2026). "BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design." *ICLR 2026*. arXiv:2508.21184.
5. Miller, G. A. (1995). "WordNet: A Lexical Database for English." *Communications of the ACM*, 38(11), 39–41. DOI:10.1145/219717.219748.
6. Wikimedia Foundation. *Wikidata: Introduction*. https://www.wikidata.org/
7. Stony Brook University AI Institute. *Computing Resources*. https://ai.stonybrook.edu/resources/computingresources

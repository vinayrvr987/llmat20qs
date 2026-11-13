# How Good Are LLMs at 20 Questions?

**Project proposal**

**Instructor:** Professor Steven Skiena

## Problem

Twenty Questions combines several abilities that are often evaluated separately: choosing a target, asking informative questions, interpreting answers, maintaining a consistent hidden state, and making a correct guess under a strict turn budget. A model can lose because it selected an unusually difficult target, asked low-value questions, misunderstood the oracle, or gave an inaccurate answer. A win rate alone cannot distinguish these causes.

Our project brief asks for all possible pairings of at least four models in both roles. We will therefore distinguish **answerer effects**, **guesser effects**, and **answerer–guesser compatibility**, rather than treating a model’s self-play score as its overall ability. We will also measure the quality of target selection, question strategy, and factual answers directly.

In the ordinary rules, the hidden-word holder is the *answerer/oracle* and must answer the guesser truthfully. A false oracle answer is an answerer error or a deliberate rule violation; the questioner can influence the interaction but is not the player responsible for truthful answers. We will keep truthful play as the primary condition and label any deliberate-deception test as a separate stress test.

### Research questions

1. **Pairing and role effects:** How do answerer identity, guesser identity, and their interaction affect the probability of a correct guess within 20 turns?
2. **Self-play versus cross-play:** Does a model perform differently when paired with an independent instance of itself than when paired with another model? Are some models stronger questioners, answerers, or both?
3. **Target selection:** How closely do model-generated selections approximate uniform sampling from a fixed, documented pool of nouns, places, people, and general vocabulary, compared with a seeded pseudorandom sampler?
4. **Question quality:** Do models ask questions that efficiently partition the remaining candidate concepts and reduce uncertainty?
5. **Answer reliability:** How often does the answerer give the correct, consistent answer to an objectively scorable question about its assigned target, and how much do inaccurate answers affect game outcomes?

## Background

Prior work provides useful pieces of this evaluation, but uses different tasks and measurements:

- **A direct LLM 20 Questions preprint.** Kumar’s 2024 engrXiv preprint, *Large Language Models (LLM) for Automating 20 Questions Game*, directly concerns LLMs playing the game. It makes the topic clearly non-novel in the broad sense, so our proposal focuses on transparent all-pairs role comparisons and separate measurements of sampling, information gain, and answer accuracy. The repository metadata and abstract identify this as a preprint; its evidence should be assessed cautiously, not treated as an established benchmark.[1]
- **Twenty questions for model identification.** Le Merrer’s 2024 paper uses a sequence of binary questions to distinguish which LLM is responding. This is a useful information-efficient questioning analogy, but it is a *model fingerprinting* task, not two models playing hidden-object 20 Questions.[2]
- **Adaptive information gathering.** Choudhury et al.’s *BED-LLM* formulates adaptive question selection using Bayesian experimental design and expected information gain, and reports tests that include 20 Questions. It motivates measuring information gain rather than only counting turns. Our proposed comparison asks how unassisted model-generated questions perform across answerer–guesser pairs, alongside explicit random and information-oriented baselines.[3]
- **Interactive dialogue benchmark.** The GuessWhat?! dataset contains human-played visual object-discovery dialogues and separates questioner and oracle tasks. It demonstrates the value of evaluating both roles and dialogue trajectories, although its image-grounded objects and questions differ from our text-only, multi-model setting.[4]
- **Factual answer evaluation.** TruthfulQA measures whether models reproduce common misconceptions in answers to questions. It is not a game-specific oracle test, but motivates an independent truthfulness measure rather than assuming that a fluent answer is correct.[5]

This is a focused initial literature review, not a systematic review. Before submission or publication, we will check the full text and publication status of the direct 20 Questions preprint and search for additional recent benchmarks. In particular, the existence of BED-LLM means the contribution should not be framed as the first use of 20 Questions with LLMs. The contribution is the controlled, role-by-role model-pair evaluation and its linked measurements of target sampling, information gain, and oracle accuracy.

## Preliminary work

The repository already contains a runner ([`twenty_questions_pilot.py`](./twenty_questions_pilot.py)), an experiment notebook ([`Abstract.ipynb`](./Abstract.ipynb)), early planning notes ([`brainstorming.txt`](./brainstorming.txt)), and saved JSONL game logs. The runner implements 20-turn games, separate role prompts, four-model ordered pair schedules, retries for malformed answers, resumable game IDs, and local Hugging Face/Ollama backends. The notebook runs two distinct 48-game preflights.

The notebook reports **2 wins in 48 games (4.2%)** for the original cohort, with **91 invalid answers in 729 answer turns (12.5%)**. It reports **10 wins in 48 games (20.8%)** for the additional-model cohort, with **64 invalid answers in 754 answer turns (8.5%)**. These are exploratory feasibility results only: each pairing-target combination has one game, the two runs use different model sets, and there are only three fixed target words. They do not establish that one cohort or model is better.

The separate [`picks.jsonl`](./picks.jsonl) file contains 50 selections attributed to “Llama (local)”; “desk lamp” occurs 16 times and “toaster oven” 13 times. This illustrates why model-generated selection should be measured rather than assumed random, but the sample and candidate frame are inadequate for a statistical uniformity claim. The current pilot also lacks a reference attribute table and independent answer auditing, so its invalid-answer metric is not a substitute for factual truthfulness.

This work establishes that the local experiment pipeline can run and save games. It does not yet test a broad target set, estimate stable model effects, compare selections with a defined uniform distribution, measure question information gain, or independently score oracle truthfulness. The proposed study addresses those gaps.

## Approach and evaluation

### Models and game matrix

Following Professor Steven Skiena’s advice, model choice will be conditional on available compute: we will first investigate the Stony Brook AI3 cluster rather than assume that the project must use small models. The AI3 computing page describes an AI cluster with H100, A100, V100, and RTX 8000 GPUs, accessed through a submit node using SLURM. We will confirm student eligibility, allocation, supported environments, and queue/runtime limits with the AI Institute or advisor before committing to a cohort or run size. Cluster runs will use scripts and scheduled jobs, not interactive notebooks, in accordance with the posted guidance [8].

The following roster is the **current prototype configuration, not a finalized model-selection decision**:

| Model | Repository checkpoint/configuration |
|---|---|
| Qwen3 0.6B | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen2.5 1.5B Instruct | `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| SmolLM2 1.7B Instruct | `HuggingFaceTB/SmolLM2-1.7B-Instruct`, revision `31b70e2e869a7173562077fd711b654946d38674` |
| Llama 3.1 8B | Ollama tag `llama3.1:8b` by default; record the resolved model digest |

After a short cluster feasibility test, we will select four instruction-tuned checkpoints that the team can run reliably and whose comparison answers a clear research question. If resources permit, the preferred primary comparison is four models from distinct families at similar parameter scales; if the scientific question is model size, we will instead choose multiple sizes from one family and explicitly study scaling. We will not choose models solely by parameter count or combine unrelated pilot cohorts as if they were controlled. We will use the same frozen prompts and comparable generation settings, record exact revisions, software, hardware, precision/quantization and decoding parameters, and disclose any model-specific template differences. We will not mix the optional additional-model cohort into the primary comparison; it can be a separately analyzed extension.

If only small models or limited GPU time are available, the project will remain useful by studying *why* models succeed or fail rather than treating size or a leaderboard as the contribution. The four-model all-pairs design and core outcomes will be retained where feasible, while the study will emphasize role-specific performance, question information gain, oracle consistency, invalid outputs, and target-selection bias. A controlled comparison of several sizes within one accessible model family is another option. The pilot and power analysis will determine a feasible number of targets and repetitions; we will not promise a large cohort or fixed run count before confirming resources.

For four models, we will run all \(4 \times 4 = 16\) ordered answerer–guesser pairings. Each self-pair uses separate role-specific conversation histories, with no shared transcript or memory. We will randomize run order and use the same target set across pairings so comparisons are paired by target. Each game ends at a correct guess or after 20 turns. The primary condition requires truthful, concise yes/no answers; ambiguous questions will be flagged rather than silently treated as objectively answerable.

### Targets and sampling experiment

We will define and freeze a reproducible candidate frame divided into four target groups: common nouns/objects, places, people, and general vocabulary concepts. Each entry will have a canonical label, aliases, category, source, and the set of verified attributes needed to score game questions. Sampling will use a public, versioned resource (for example, a curated WordNet-derived vocabulary for common concepts [6] and a documented Wikidata query for places and public figures [7]). Human review will remove ambiguous entries and normalize aliases. People targets will be public figures only, and the question set will avoid sensitive personal attributes.

For the selection experiment, each model will independently choose one target from a presented, frozen candidate list for the specified category, without seeing the other models’ outputs. We will randomize list order on each draw and compare the model’s empirical distribution with a seeded pseudorandom generator sampling uniformly from the *same finite candidate list*. This makes “uniform” testable: a model cannot be expected to be uniform over an undefined or unbounded universe. We will report unique-item coverage, duplicate/top-item concentration, entropy, and distributional deviation from the uniform baseline, with uncertainty intervals. A separate free-generation sample may be retained as exploratory evidence, but will not be called a uniformity test unless its outputs can be mapped to a defined frame.

Where possible, game targets will be balanced or sampled independently of model selection. This prevents a model’s tendency to repeat easy or familiar concepts from confounding its questioner score. A random-candidate guesser and simple random-question strategy will provide performance floors; a balanced-partition/information-gain strategy will provide a stronger reference point.

### Question quality and answer truthfulness

For each turn, we will record the raw response, parsed move, any repair/retry, the remaining candidate set, and the target. For every candidate target, an independently curated attribute table will supply answers to objective yes/no properties. Questions will be scored as:

1. **Scorable and correct:** the question maps to an objective attribute and the answer matches the reference.
2. **Scorable and incorrect:** the answer conflicts with the reference.
3. **Ambiguous/unscorable:** the wording or property has no defensible binary answer.

The primary oracle-reliability measure will be the proportion of first-attempt, objectively scorable answers that are correct; we will report retries and invalid answers separately. A sample of question-to-attribute mappings will be independently reviewed by two human annotators, with disagreements adjudicated and agreement reported. The oracle’s true answer will not be inferred from its own explanation.

For each valid binary question \(q\), we will estimate its expected information gain over the remaining candidate distribution:

\[
IG(q) = H(C) - \sum_{a \in \{\text{yes},\text{no}\}} P(a)H(C \mid a),
\]

where \(C\) is the remaining candidate set and \(H\) is entropy. We will compare realized uncertainty reduction with random valid questions and a balanced-partition oracle baseline. A question with high theoretical information gain but an incorrect answer will not count as useful; we will report information gain conditional on correct answers as well as end-to-end realized gain. This separates question choice from oracle reliability.

### Scale and procedure

The existing pilot is too small to set a defensible confirmatory sample size. We propose two stages:

1. **Protocol pilot:** use a small, balanced set of targets across all 16 pairings to validate target aliases, objective answer keys, parsers, logging, runtime, and annotation guidelines. Pilot games will be excluded from confirmatory estimates.
2. **Confirmatory run:** a planning scenario is at least 30 distinct targets per category and three independent game repetitions per target and ordered model pair (a minimum of \(16 \times 4 \times 30 \times 3 = 5{,}760\) games). This is a provisional scenario, not a commitment. Use pilot-derived simulation/power analysis and the confirmed GPU allocation and queue limits to select the final target and repetition counts, testing for at least 80% power for a pre-registered practically meaningful difference in win rate (initial planning threshold: 10 percentage points). If the full scenario is infeasible, justify a smaller study and narrow the claims accordingly.

Independent repetitions will use separate fresh conversation histories and recorded random seeds where the backend permits. A single model instance may serve both roles in a self-pair, but its role histories must remain isolated. We will capture failures and incomplete games explicitly, preserve first-attempt outputs, and resume only by stable game IDs; failed runs will not be silently counted as losses or omitted without a documented reason.

### Outcomes and analysis

**Primary outcome:** correct guess within 20 turns (game-level binary win).

**Secondary outcomes:** turns to correct guess among wins; answerer factual accuracy; invalid-answer and malformed-move rates; proportion of turns spent asking questions versus guessing; average and cumulative information gain; target-selection coverage and deviation from the uniform sampler; latency and resource use.

We will analyze win probability with a mixed-effects logistic model containing guesser, answerer, their interaction, and target category as fixed effects, and target identity (and repetition where estimable) as random effects. Pre-registered contrasts will compare same-model pairings with cross-model pairings and estimate each model’s questioner and answerer effects. We will report effect sizes and confidence intervals, not only p-values, and adjust families of pairwise comparisons for multiple testing. Target-clustered bootstrap intervals will be a robustness check. Selection distributions will be compared with the seeded uniform baseline using permutation or multinomial methods appropriate to the observed counts; question quality and answer accuracy will be summarized at both game and turn levels.

We will keep the primary protocol cooperative and truthful. If resources allow, a separate, clearly labeled stress test will instruct the answerer to conceal or misrepresent the target. It will measure the effect of intentional deception on the guesser, but will not be mixed into estimates of ordinary answer reliability.

## Risks, limitations, and mitigations

- **Target-frame dependence:** uniformity depends on the chosen finite vocabulary. Publish the exact pool, query, filters, exclusions, aliases, and random seed; report category-specific results.
- **Objective truth is hard to annotate:** restrict primary scoring to verifiable binary properties, mark ambiguous questions as unscorable, and use independent review.
- **Model and prompt sensitivity:** pin weights and software, disclose prompts and decoding settings, and avoid changing settings after viewing confirmatory outcomes.
- **Compute and repeated-game cost:** run a small protocol pilot first, estimate runtime, then power the confirmatory study before launching it. Keep the four-model cohort fixed for the primary study.
- **Cluster access and model feasibility:** access to AI3 and a specific GPU allocation are not assumed. Confirm eligibility and test checkpoint loading and throughput first; if large models are unavailable, revise the design toward careful small-model behavioral comparisons or within-family scaling.
- **Limited generality:** results from four local models and a defined target pool do not represent all commercial models, languages, or human play. State this boundary explicitly.
- **Strategic deception versus error:** analyze intentional deception only in a separate condition; do not infer intent from a single incorrect answer.

## Future work

1. Confirm AI3 access, GPU allocation, cluster job requirements, and feasible model checkpoints; finalize the cohort only after the feasibility test. Then finalize the candidate vocabulary, category definitions, target metadata, objective attributes, and game protocol.
2. Update the prototype and notebook to log raw outputs/retries, score truthfulness and information gain, support random baselines, and validate data integrity.
3. Run and review the protocol pilot; preregister the final hypotheses, exclusions, analysis, and power-based sample size before confirmatory games.
4. Run all 16 ordered pairings, complete annotation and statistical analysis, and publish the prompts, model/version details, target frame, analysis scripts, and de-identified game logs where licensing permits.
5. Deliver a progress report with a literature summary, descriptive results, limitations, and next steps, followed by a final report/presentation.

The expected research output is a reproducible comparison of model-pair and role effects, together with evidence about selection bias, information-seeking behavior, and truthful answering—not simply a leaderboard of 20-question win rates.

## References

1. Kumar, C. R. S. (2024). “Large Language Models (LLM) for Automating 20 Questions Game.” *engrXiv* preprint. [https://doi.org/10.31224/3842](https://doi.org/10.31224/3842). Preprint; evaluate its full methods and evidence before relying on its claims.
2. Le Merrer, E. (2024). “The 20 questions game to distinguish large language models.” *arXiv:2409.10338*. [https://arxiv.org/abs/2409.10338](https://arxiv.org/abs/2409.10338).
3. Choudhury, D., Williamson, S., Goliński, A., Miao, N., Bickford Smith, F., Kirchhof, M., Zhang, Y., & Rainforth, T. (2026). “BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design.” *International Conference on Learning Representations (ICLR 2026)*. [https://arxiv.org/abs/2508.21184](https://arxiv.org/abs/2508.21184).
4. de Vries, H., Strub, F., Chandar, S., Pietquin, O., Larochelle, H., & Courville, A. (2017). “GuessWhat?! Visual Object Discovery Through Multi-Modal Dialogue.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 5503–5512. [CVF paper page](https://openaccess.thecvf.com/content_cvpr_2017/html/de_Vries_GuessWhat_Visual_Object_CVPR_2017_paper.html).
5. Lin, S., Hilton, J., & Evans, O. (2022). “TruthfulQA: Measuring How Models Mimic Human Falsehoods.” *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, 3214–3252. [ACL Anthology](https://aclanthology.org/2022.acl-long.229/).
6. Miller, G. A. (1995). “WordNet: A Lexical Database for English.” *Communications of the ACM*, 38(11), 39–41. [https://doi.org/10.1145/219717.219748](https://doi.org/10.1145/219717.219748).
7. Wikimedia Foundation. *Wikidata: Introduction*. [https://www.wikidata.org/wiki/Wikidata:Introduction](https://www.wikidata.org/wiki/Wikidata:Introduction).
8. Stony Brook University AI Institute. *Computing Resources*. [https://ai.stonybrook.edu/resources/computingresources](https://ai.stonybrook.edu/resources/computingresources).

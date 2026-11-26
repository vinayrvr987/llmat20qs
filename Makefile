PYTHON ?= python3

.PHONY: test check mock-e1 mock-e5 mock-e6 smoke-e1 run-e1 run-followups run-e5 run-e6

test:
	$(PYTHON) -m unittest discover -s tests -v

check:
	$(PYTHON) -m py_compile src/experiments/e1.py src/experiments/followups.py src/experiments/e5_information.py src/experiments/e6_reasoning_budget.py src/models/registry.py
	bash -n jobs/slurm/*.job

mock-e1:
	$(PYTHON) -m src.experiments.e1 --mock --max-games 8 --results-file /tmp/e1-mock.jsonl

mock-e5:
	$(PYTHON) -m src.experiments.e5_information --mock --max-games 8 --results-dir /tmp/e5-mock

mock-e6:
	$(PYTHON) -m src.experiments.e6_reasoning_budget --mock --max-games 9 --results-dir /tmp/e6-mock

smoke-e1:
	sbatch jobs/slurm/run_e1_smoke.job

run-e1:
	sbatch jobs/slurm/run_e1.job
	sbatch jobs/slurm/run_e1_self.job

run-followups:
	sbatch jobs/slurm/run_e2_predictable.job

run-e5:
	sbatch jobs/slurm/run_e5_information.job

run-e6:
	sbatch jobs/slurm/run_e6_reasoning_budget.job

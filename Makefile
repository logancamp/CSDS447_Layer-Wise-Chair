tqa_random:
	python src/baselines.py --baseline random --limit 200
tqa_first:
	python src/baselines.py --baseline first  --limit 200
tqa_zero:
	python src/zero_shot.py --limit 50
tqa_eval:
	python src/eval_mc1.py --limit 100

report:
	python src/report.py --preds outputs/my_test_run.tagged.jsonl

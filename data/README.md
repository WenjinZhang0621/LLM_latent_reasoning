# Data Directory

Place public benchmark prompts here before running the workflow.

Expected JSONL schema:

```json
{"sample_id": "gsm8k_train_000000", "dataset": "gsm8k", "question": "...", "ground_truth": "..."}
```

Large benchmark files, generated CoTs, activation features, and model-derived
artifacts should not be committed to GitHub.

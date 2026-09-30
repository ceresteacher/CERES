# data/ —— Sample Data Directory

This directory only keeps a small amount of sample data for verifying the data pipeline and training flow; the full datasets must be regenerated with the following steps.

## Processing Steps

The data pipeline lives in [`data_pipeline/`](../data_pipeline/README.md), and the processing flow is:

1. **Register raw data**: `prepare_datasets.py` — register and validate local raw medical datasets.
2. **Extract questions**: `extract_questions.py` — extract medical questions and ophthalmology-related content.
3. **Rewrite into student questions**: `rewrite_question.py` — rewrite the original questions into student-style questions.
4. **Synthesize multi-turn teaching dialogues**: `synthesize_dialogue.py` — generate ≤5-turn teaching demo dialogues.
5. **Quality filtering**: `quality_filter.py` — filter by grammar and medical rules, producing SFT warmup data.
6. **Build the GRPO query set**: `build_query_dataset.py` — produce multi-turn GRPO query data.

For the complete commands and how to obtain each dataset, see [`data_pipeline/README.md`](../data_pipeline/README.md).

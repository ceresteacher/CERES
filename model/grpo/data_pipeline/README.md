# data_pipeline —— Offline Data Construction Pipeline for CERES Ophthalmology Teaching GRPO

Implements design §6.2/§6.3: open-source medical data → "student question" rewriting → API-simulated ≤5-turn teaching dialogues → quality check. Fully API-driven (does not occupy training cards), two outputs:

| Output | Purpose | Reached at step |
|---|---|---|
| `data/ceres_oph_queries.jsonl` | GRPO query set (design §6.1 contract schema) | Step 0→1→2→build |
| `data/ceres_oph_sft_warmup.jsonl` | SFT warmup set (design §6.5) | Step 0→1→2→3→4 |
| `data/ceres_cmexam4k_queries.jsonl` | **CMExam 4k contract query set (with gt three columns)** | `scripts/run_cmexam4k.sh` |
| `data/cmexam4k_qa_grpo.jsonl` | **CMExam 4k single-turn QA GRPO set (student-style MCQ question + gt standard answer)** | same as above |

## Pipeline Overview (Step 0 → 4)

```
Open-source data (manually downloaded to RAW/)
        │  Step 0  prepare_datasets        validate + register → data/manifest.jsonl
        ▼
Step 1  extract_questions        manifest → data/raw_pool.jsonl
        (ophthalmology keyword filtering / reading-image question template / finding_text exam findings)
        ▼
Step 2  rewrite_question         raw_pool → data/queries_candidates.jsonl
        (exam question → classroom student question + profile/misconception/difficulty/persona_seed)
        ▼
        ├──────────────► build_query_dataset  candidates → data/ceres_oph_queries.jsonl (GRPO query set)
        ▼
Step 3  synthesize_dialogue      candidates → data/sft_raw.jsonl
        (teacher = llm_client strong model + student = ceres_plugin.student_sim same persona_seed, ≤5 turns forced wrap-up)
        ▼
Step 4  quality_filter           sft_raw → data/ceres_oph_sft_warmup.jsonl + data/filter_report.json
```

## Copy-and-use Complete Command Block

Run from the project root (`grpo/`); `RAW/` is the local directory for each dataset (download yourself, see the table below).

```bash
export CERES_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export CERES_API_KEY="<your key>"            # if unset, the whole pipeline auto-enters offline mock (smoke only)
export CERES_TEACHER_MODEL="qwen-max"        # teacher synthesis model
# export CERES_SYNTH_CONCURRENCY=8 CERES_SYNTH_TIMEOUT=60 CERES_SYNTH_CACHE_DIR=./.llm_cache

# ---- Step 0: register raw data (run once per source; re-registering the same source overwrites the old entry) ----
python -m data_pipeline.prepare_datasets --list-sources
python -m data_pipeline.prepare_datasets --type cmexam --input RAW/CMExam/data/train.csv
python -m data_pipeline.prepare_datasets --type cmb     --input RAW/CMB/CMB-Exam/CMB-exam/CMB-exam.json
python -m data_pipeline.prepare_datasets --type idrid   --input RAW/IDRiD/a.\ IDR\ grading\ labels/IDRiD_grading.csv --image-dir RAW/IDRiD/images
python -m data_pipeline.prepare_datasets --type odir    --input RAW/ODIR-5K/full_df.csv --image-dir RAW/ODIR-5K/ODIR-5K --eyes both
python -m data_pipeline.prepare_datasets --type generic_images --source-name octid --image-dir RAW/OCTID --labels RAW/OCTID/labels.txt
# on column-name drift, override: --col-map question=题干 --col-map answer=答案 (repeatable)

# ---- Step 1: question/case extraction (ophthalmology subset + reading template + exam findings) ----
python -m data_pipeline.extract_questions --input data/manifest.jsonl --output data/raw_pool.jsonl \
    --max-per-source 2000            # add --lang {auto,zh,en} to override the trajectory language (default auto)

# ---- Step 2: first-turn "student question" rewrite (the GRPO query set only needs Step 1-2) ----
python -m data_pipeline.rewrite_question --input data/raw_pool.jsonl --output data/queries_candidates.jsonl

# ---- GRPO query set (contract schema) ----
python -m data_pipeline.build_query_dataset --input data/queries_candidates.jsonl \
    --output data/ceres_oph_queries.jsonl            # add --balance --balance-cap 800 for optional within-group downsampling

# ---- Step 3: API-simulate ≤5-turn teaching dialogues (needed by the SFT warmup set) ----
python -m data_pipeline.synthesize_dialogue --input data/queries_candidates.jsonl \
    --output data/sft_raw.jsonl --concurrency 4

# ---- Step 4: quality filtering (grammar G1~G5 real validation + medical key-point check) ----
python -m data_pipeline.quality_filter --input data/sft_raw.jsonl \
    --output data/ceres_oph_sft_warmup.jsonl --report data/filter_report.json

# single-shot LLM self-check (mock smoke):
python -m data_pipeline.llm_client --prompt "Rewrite this question into a student question" --mock
```

All CLIs support `--help`. Common parameter conventions (not all modules have all parameters): the 6 pipeline modules (prepare/extract/rewrite/synthesize/quality_filter/build_query) all have `--input/--output`; the data-processing steps (extract/rewrite/synthesize) additionally have `--mock` (automatic by default when the key is unset), `--limit`, `--seed` (prepare's `--limit`/`--seed`/`--mock` are non-effective reserved slots); `llm_client` is a **single-shot self-check CLI** (only `--prompt/--system/--seed/--json-mode/--mock/--no-cache/--model` etc., no `--limit/--input/--output`); `prompts` is the template layer (no CLI).

## Language Mechanism (trajectory language follows the open-source dataset language)

**Source data is English → sampled/synthesized trajectories (student questions, teacher actions, student replies) are all English; Chinese source → all Chinese.** The language is determined row-by-row in Step 1 and written to the `lang` column (`'zh' | 'en'`), and every subsequent step only passes it through without re-judging:

* **auto detection rules** (default, `--lang auto`):
  * Text question-bank sources (CMExam/CMB/MedQA/generic_jsonl) → by stem `llm_client.detect_lang`: CJK characters (including Chinese punctuation) proportion of non-whitespace characters **≥ 0.25 → zh**, otherwise en; empty/non-string → **zh** (Chinese question banks are the main force; missing text falls back to the existing Chinese pipeline by project default);
  * Image sources `idrid/odir/refuge/palm/gamma/octid/kermany/octdl` are **English sources → directly en** (their gt_label has been translated to Chinese by the adapter, so label-based detection would misjudge zh); self-provided image source `generic_images` detects by gt_label;
  * The ophthalmology-subset keyword table and the curriculum-node/difficulty rule table are bilingual (English MCQ/question banks hit the same table).
* **`--lang zh|en` explicit override**: force all rows to the specified language (regardless of source and detection result).
* **Assets selected by language**: the reading-image stem template and the `finding_text` fallback template (Chinese/English versions; the English version also follows A6 de-conclusion—only gives annotation-level/symptom-level info), all Step 2/3 system prompts and user-assembly labels, the degraded student-question template (English classroom phrasing "Dr., while reviewing my notes I got stuck on … Could you walk me through it step by step?"), profile/misconception seed templates, the `courseware_context` assembly label (Chinese「检查所见：」/ English "Findings: "), the teacher forced wrap-up instruction, student-side observation copy, and the **mock output language** of `llm_client`/`student_sim` (offline mock is likewise all English, determinism unchanged).
* **GRPO-side consumption**: `ceres_oph_queries.jsonl` each row carries a `lang` column → swift passes it into the `teacher_env` plugin as an extra dataset column → `plugin._student_messages` produces the corresponding-language observation per `dd['lang']`, `student_chat(..., lang='en')` switches the student simulator language (the real path appends an English-output instruction in the system prompt, JSON schema unchanged). **When a dataset row has no lang column, it defaults to zh, fully backward compatible**; the teacher model's output language is naturally determined by the dialogue history and prompts, with no forced output-layer control.
* **Wrap-up instruction keywords**: the mock teacher identifies the forced-wrap-up branch by keywords—Chinese「收束 + `<end/>`」, English "wrap up + `<end/>`"; the gt extraction anchor is「正确要点：」/ "Reference answer:". Before changing these two places, `llm_client._mock_chat` must be synced.

## CMExam 4k Dual-format Dataset (added 2026-09-01)

Offline-generates **4,000** GRPO records from `/hy-tmp/dataset/CMExam/data/train.csv` (54,497 Chinese medical-licensing multiple-choice questions), with the same sample producing two formats. Ophthalmology questions are fewer than 4k (only ~1,065 rows in train hit ophthalmology keywords and pass all filtering), so it uses **ophthalmology-first + general-medicine fill-up** (`Random(seed=42)` deterministic); filters out multi-select/empty-explanation/bad-option/duplicate-stem questions; only uses train (val/test reserved for benchmark evaluation).

```bash
MOCK=1 bash scripts/run_cmexam4k.sh   # offline smoke (no API calls; rewrite 100% degradation is expected)
bash scripts/run_cmexam4k.sh          # formal (default source scripts/env_deepseek.sh, deepseek-v4-flash
                                      #   + thinking mode low; ENV_SCRIPT=scripts/env_glm.sh can switch back to BigModel;
                                      #   tunable CMEXAM_TRAIN/TOTAL/SEED/CONCURRENCY/PYTHON)
# training hookup:
CERES_DATASET=data/ceres_cmexam4k_queries.jsonl bash scripts/run_grpo_oph_lora.sh
```

8 steps: `sample_cmexam` (sampling → `data/cmexam_4k_sample.csv`, idempotent byte-identical) → `prepare_datasets --source-name cmexam4k` (independent manifest) → `extract_questions --no-keyword` (lets general-medicine rows through; ophthalmology/general-medicine node splitting, general-medicine rows `curriculum_node=general_medicine/qa`, profile/courseware use the "clinical" version) → `rewrite_question` ×2 (open = classroom question with option traces removed; `--mode mcq` = student question keeping A–E options, the two modes have different prompts and naturally isolated cache keys) → `build_query_dataset` (contract schema + **gt three columns**) → `build_qa_dataset` (QA schema: question self-contains options, `_ensure_options` fallback supplements the option block) → `verify_cmexam4k` (hard validation: row count/uid uniqueness/gt completeness/queries without option traces/qa options self-contained with the answer letter among them; report `data/cmexam4k_report.json`).

**gt field semantics**: raw_pool/candidates gain `gt_letter` (answer letter) and `options` (letter→full text); the query set's gt three columns = `gt_label` (multiple-choice source = letter; image/legacy rows = original label text) / `gt_answer` (full text of the correct option) / `gt_explanation` (original question explanation). The current reward does not consume gt (columns passed through safely), reserved for evaluation and future outcome reward. **Bundled fix**: `parse_options` now supports the CMExam "A space text" format (the old version parsed 99.9% as empty, gt_label degraded to a bare letter); the answer letter is no longer truncated with `[:1]` (multi-select safe).

## PubMedQA 1k English Dual-format Dataset (added 2026-09-01)

Generates **all-English** dual-format GRPO data from **PubMedQA pqa_labeled** (1,000 human-annotated English biomedical research Q&A, MIT license). The yes/no/maybe three-class maps to options `{A: Yes, B: No, C: Maybe}`, gt_label=A/B/C, gt_answer=Yes/No/Maybe, gt_explanation=long_answer; stem = research question + abstract (keeping section labels BACKGROUND/METHODS/…, the whole block truncated to 2400 characters). The few English questions hitting ophthalmology are routed to ophthalmology nodes, the rest `general_medicine/qa`; all rows `lang='en'` (the English prompt/template/student-simulator chain already exists).

```bash
bash scripts/run_pubmedqa.sh           # step 1 auto-downloads (network only needed the first time, via hf-mirror;
                                       #   parquet→jsonl uses venv python, pyarrow isolated in that step)
MOCK=1 bash scripts/run_pubmedqa.sh    # offline smoke (fully offline after jsonl is downloaded)
PYTHON=/hy-tmp/my-env/swift/bin/python bash scripts/run_pubmedqa.sh   # formal (~2.2k flash calls)
# training hookup:
CERES_DATASET=data/ceres_pubmedqa_queries.jsonl bash scripts/run_grpo_oph_lora.sh
```

Artifacts: `data/ceres_pubmedqa_queries.jsonl` (contract schema + gt three columns, multi-turn teaching, English student simulator) and `data/pubmedqa_qa_grpo.jsonl` (single-turn QA: student-style question + `A. Yes/B. No/C. Maybe` options + gt); validation `verify_cmexam4k --lang en` (English rows anchor option-trace regex at line start—biomedical abbreviations like "vitamin D." don't count as option traces). Adapter filtering (invalid final_decision/empty question/empty long_answer/duplicate question) warns per-row on stderr, dropped-row count = 1000 − manifest n_items is auditable (measured 1,000 all pass).

## mock mode (offline deterministic)

* Trigger: `--mock` explicitly specified, or `CERES_API_KEY` unset (`synthesize --mock` also forces `CERES_STUDENT_MOCK=1`).
* Behavior: same style as `ceres_plugin.student_sim`—seeded with `random.Random(f"{seed}:{sha1(message)}")`, same input same seed reproducible across processes, disk cache still takes effect (`--no-cache` disables it).
* The mock teacher produces **legal action-label** turns (hint level monotonically 1→2→3 across turns) and does not give `<end/>` in regular turns, which exactly stably triggers the "5th-turn forced wrap-up" branch; the mock's JSON output contains no business keys, and downstream automatically uses the degraded template—so **mock is only for getting the pipeline through and offline smoke runs**, and the produced data is not stored.
* The student side directly reuses `ceres_plugin.student_sim.student_chat` (same `persona_seed`, observation format byte-identical to the design §7.3 training-time plugin), ensuring the SFT synthesis distribution ≈ the training-time environment distribution.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CERES_API_BASE` | DashScope compatible-mode | OpenAI-compatible gateway |
| `CERES_API_KEY` | empty | **unset → whole pipeline mock** |
| `CERES_TEACHER_MODEL` | `qwen-max` | teacher synthesis/rewrite model |
| `CERES_SYNTH_CONCURRENCY` | 8 | concurrency cap (semaphore + thread pool) |
| `CERES_SYNTH_TIMEOUT` | 60 | per-call timeout (seconds) |
| `CERES_SYNTH_MAX_TOKENS` | 1024 | per-call output cap `max_tokens` (<=0 not passed, uses server default); thinking mode includes thinking tokens, needs enlarging |
| `CERES_API_THINKING` | empty | `enabled`\|`disabled` → `extra_body {"thinking":{"type":…}}` (DeepSeek thinking mode; empty = not passed) |
| `CERES_API_REASONING_EFFORT` | empty | `high`\|`medium`\|`low` → `reasoning_effort` (empty = not passed) |
| `CERES_SYNTH_CACHE_DIR` | `./.llm_cache` | LLM disk cache (sha1(message+seed+model)) |
| `CERES_STUDENT_*` / `CERES_STUDENT_MOCK` | see ceres_plugin | student simulator (reused by Step 3) |
| `CERES_MAX_TURNS` | 5 | synthesize teacher-turn cap default value |

## Per-step Schema

* **Step 0** `data/manifest.jsonl`: `source/type/path/image_dir/labels_path/license/n_items/col_map/eyes/warnings/ts`
* **Step 1** `data/raw_pool.jsonl`: contract `qid/source/question_type(text|image)/image/gt_label/gt_letter/gt_explanation/options/wrong_options/curriculum_node/difficulty_hint(low|mid|high)/finding_text` + extension columns `stem` (original stem/reading template, Step 2 input), `label_extra` (image annotation layout), `lang` ('zh'|'en', trajectory language, passed through row-by-row to downstream)
* **Step 2** `data/queries_candidates.jsonl`: the above columns + `student_question/learner_profile/misconception_seed/difficulty(three levels)/persona_seed(int)/courseware_context/rewrote/lang`
* **Step 3** `data/sft_raw.jsonl`: `{"messages":[u,a,u,a,…], "images":[optional], "uid":qid}` + audit columns `turns/unresolved/forced_close/curriculum_node/difficulty/gt_label/lang/...`
* **Step 4** `data/ceres_oph_sft_warmup.jsonl`: strict contract `{"messages", "images"(optional), "uid"}` (without lang); `data/filter_report.json`: `total/kept/dropped/drop_reasons(grouped by violation reason)/dropped_rows(first 50)/label_unverified/kept_by_difficulty/kept_by_curriculum_node/kept_by_lang(default counted as zh)`
* **GRPO** `data/ceres_oph_queries.jsonl`: contract `uid/messages(first-turn user, image questions start with `<image>`)/images(optional)/courseware_context/curriculum_node/learner_profile/misconception_seed/difficulty/persona_seed` + **`lang` ('zh'|'en', missing column/illegal value defaults to 'zh')—the GRPO-side `teacher_env` plugin decides the student simulator language by this column** + **gt three columns `gt_label/gt_answer/gt_explanation` (constant output; semantics in the "CMExam 4k" section above)**
* **QA GRPO** `data/cmexam4k_qa_grpo.jsonl`: `uid/messages(first-turn user, student-style MCQ question with options listed line-by-line at the end)/gt_label(answer letter)/gt_answer(option full text)/gt_explanation`—for answer-scoring reward

Filtering convention (Step 4, report grouping keys): `G1~G5`/`message structure` (wording verbatim from `ceres_plugin.grammar.validate_sft_trajectory`: at least 1 action label per turn, `<end/>` self-closing and only in the last turn, the six content-label types must not self-close, G5 reading first `<check>` then explain/correct) + this module's own rules `turn count out of bounds`(2~5)/`not wrapped up`(last turn must have `<end/>`)/`single turn too long`(>1200 chars)/`medical key points mismatch`/`structurally illegal`. Medical check is keyword heuristic (60% character overlap for long labels), **physician spot-check of 5~10% must still be performed**.

## Open-source Dataset Acquisition and License (design §6.2)

| Dataset | Content | Corresponding node | Acquisition | License |
|---|---|---|---|---|
| IDRiD | 516 fundus photos, DR 5-level + DME 3-level + pixel-level lesion annotations | `retina/DR_staging` | [idrid.grand-challenge.org](https://idrid.grand-challenge.org/) (also IEEE DataPort/Zenodo/Kaggle mirrors) | **CC BY 4.0** (cleanest, reading-question main force) |
| ODIR-5K | 5,000 patients × both eyes fundus photos + 8-class labels | multi-disease differential diagnosis | [Kaggle andrewmvd/ocular-disease-recognition-odir5k](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) | Kaggle research use |
| REFUGE | 1,200 fundus photos, glaucoma + optic disc/cup segmentation | `glaucoma/*`, `neuro_ophth/*` | [refuge.grand-challenge.org](https://refuge.grand-challenge.org/) + figshare | CC family (**note non-commercial terms**) |
| PALM | 1,200 pathological-myopia fundus photos + lesion annotations | retina (high myopia) | [palm.grand-challenge.org](https://palm.grand-challenge.org/) + figshare/Springer Nature (Scientific Data) | open |
| GAMMA | 300 cases fundus + 3D OCT paired, glaucoma grading | `glaucoma/*` | [gamma.grand-challenge.org](https://gamma.grand-challenge.org/) | **CC BY** |
| OCTID | 500+ OCT images, 5 classes | retina (OCT reading) | Kaggle `octid-dataset` / Borealis Dataverse | open |
| Kermany 2018 | 84,495 OCT images, 4 classes | retina (OCT scale-up) | Kaggle / Mendeley | research use (sampling suffices) |
| OCTDL | 2,000+ OCT images with multi-disease group annotations | retina (OCT scale-up) | Nature Scientific Data open | open |
| CMExam | 60K+ Chinese medical-licensing exam questions (stem/options/answer/**explanation**) | all text-question nodes | [github.com/williamliujl/CMExam](https://github.com/williamliujl/CMExam) | follows the repo LICENSE (research use); **text-question main force** |
| PubMedQA pqa_labeled | 1k English biomedical research Q&A (yes/no/maybe + long answers) | `general_medicine/qa` + ophthalmology keyword subset | `hf-mirror.com/datasets/qiaojin/PubMedQA` (pqa_labeled parquet; `scripts/download_pubmedqa.sh` auto-downloads via mirror, huggingface.co unreachable on this machine) | **MIT** |
| CMB | CMB-Exam 11,200 questions (including ophthalmology) | text-question expansion | [github.com/FreedomIntelligence/CMB](https://github.com/FreedomIntelligence/CMB) | follows the repo LICENSE (research use, deduplicated against CMExam) |
| MedQA-MCMLE | domestic medical-licensing multiple-choice questions | text-question supplement | [github.com/jind11/MedQA](https://github.com/jind11/MedQA) | research use |
| English ophthalmology question bank (MedQA-USMLE·English MCQ·any English question bank) | English multiple-choice questions | text questions (**English trajectory**) | self-provided jsonl, adapted via `--type generic_jsonl` (column names `question/answer/explanation/options`, overridable via `--col-map`) | self-provided data (`--license` overrides) |
| Tianchi Chinese medical dialogue | 792k doctor-patient Q&A | — | Alibaba Tianchi dataset/90163 | only as colloquial style reference, **not as a medical knowledge source**; parsed via `generic_jsonl` adapter (`--type tianchi_dialog`, column-name drift overridable via `--col-map`) |

> **License note**: APTOS 2019 (~3.7k) and EyePACS (~35k) are restricted by Kaggle competition terms (must accept terms, research/non-commercial only), **not used in the first batch**; re-evaluate when scaling up is needed. When ingesting images, register source and license in the manifest in sync. First-batch ratio suggestion (design §6.4): image questions 40% / text questions 60%, total 3k~5k.

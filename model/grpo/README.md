# CERES Ophthalmology Teaching GRPO —— Multi-turn Teaching Agent Training for Qwen2.5-VL-32B (ms-swift 3.4.1 / 4×A800-80G)

Turn the "virtual ophthalmology resident simulator" into the environment for GRPO multi-turn rollout (the `teacher_env` plugin), and use a two-layer reward of **sequence reward (format + teaching-rule − medical safety red lines) + anchor credit (learner-state-aware group-relative credit)** to teach the 32B VL model to produce a structured teaching action sequence (7 action DSL types) of "activate first → graded hints → let the student speak first → delayed correction". Authoritative design: [`../design.md`](../design.md); interface contract: [`.sdd/contracts.md`](.sdd/contracts.md) (in case of conflict, the contract prevails).

## 1. Architecture One-page Diagram (design §2.1 simplified)

```
 Ophthalmology query set data/ceres_oph_queries.jsonl (first-turn user question + course/profile/misconception extra columns)
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                ms-swift GRPO Trainer (4×A800, ZeRO-3)         │
 │   Sampling engine (default pt infer; vLLM colocate optional CERES_USE_VLLM=1) │
 │        ▲   Each turn: teacher action → student simulator API (cloud/or mock)    │
 │        │   State transition + incrementally record TrajectoryStore (traj_dump/*.jsonl) │
 │   Reward: ceres_sequence (fmt+rule−risk)                      │
 │           ceres_anchor (anchor group-relative credit)  --reward_weights 1.0 0.5 │
 │        ▼                                                      │
 │   GRPO group advantage + clip + KL (G=8, loss covers all teacher turns)          │
 └─────────────────────────────────────────────────────────────┘
        │ trajectory dump                            │ LoRA adapter
        ▼                                            ▼
 eval/analyze_traj.py (design §10 offline metrics)   output/ceres-oph-grpo-lora-v1
```

## 2. Directory Structure

| Path | Contents | Owner |
|---|---|---|
| `ceres_plugin/grammar.py` | Action DSL parsing + G1~G5 teaching grammar validation (pure stdlib, shared by plugin/quality-check/evaluation) | Task 1 |
| `ceres_plugin/student_sim.py` | Virtual student simulator (OpenAI-compatible API + cache/rate-limit/degradation/mock) | Task 2 |
| `ceres_plugin/store.py` | TrajectoryStore trajectory store + anchor key + step_reward + final_eval | Task 2 |
| `ceres_plugin/plugin.py` | Registration entry: `multi_turns['teacher_env']`, `orms['ceres_sequence'/'ceres_anchor']`, `RISK_PATTERNS` (`--external_plugins` points to this file) | Task 3 |
| `data_pipeline/` | Open-source medical data → student questions → ≤5-turn demo dialogues → quality check (8 CLI modules, usage in its README) | Task 4 |
| `scripts/run_sft_warmup.sh` | SFT format warmup (LoRA r=64 + ZeRO-3, run first) | Task 5 |
| `scripts/run_grpo_oph_lora.sh` | GRPO main training (LoRA r=64 + ZeRO-3 + vLLM colocate) | Task 5 |
| `scripts/run_demo_dialogue.sh` | Training-effect demo wrapper (BigModel env + single GPU + params forwarded to eval/demo_dialogue.py) | Work package B |
| `eval/analyze_traj.py` | Trajectory offline evaluation → markdown report (design §10 metrics + time-evolution collapse monitoring) | Task 5 |
| `eval/check_train_log.py` | Training log health check: clipped_ratio→1 / reward decreasing / loss≡β·kl etc. collapse alarms (exit code 0/1) | Fix round 2 |
| `eval/demo_dialogue.py` | Training-effect viewer: N questions × ≤4-turn teaching dialogue demo (baseline vs `--adapters` post-training comparison) | Work package B |
| `ceres_plugin/save_on_exit.py` | Crash-protection callback: save LoRA to `<output_dir>/emergency-adapter` before exiting on any error/kill (auto-registered at the end of `plugin.py`) | Work package C |
| `data/` | Pipeline artifacts (schema in `data/README.md`, generation commands in `data_pipeline/README.md`) | Task 4 |
| `.sdd/` | Contract / task brief / report / ledger (process docs) | Coordinator |

## 3. Environment Variables Overview (contract full)

### 3.1 Virtual Student Simulator (ceres_plugin/student_sim.py)
| Variable | Default | Description |
|---|---|---|
| `CERES_STUDENT_API_BASE` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible endpoint (local degradation can point to a self-hosted vLLM server) |
| `CERES_STUDENT_API_KEY` | — (unset) | When unset, **automatically enters mock mode** (no network) |
| `CERES_STUDENT_MODEL` | `qwen-plus` | Switch to `qwen-vl-plus` when the student needs to "see images" |
| `CERES_STUDENT_CONCURRENCY` | `64` | Concurrency cap (semaphore rate limiting) |
| `CERES_STUDENT_TIMEOUT` | `30` | Per-call timeout (seconds) |
| `CERES_STUDENT_MOCK` | — | `=1` forces mock (deterministic offline replies, for offline runs) |

### 3.2 Plugin and Scripts (ceres_plugin/plugin.py + scripts/)
| Variable | Default | Description |
|---|---|---|
| `CERES_MAX_TURNS` | `5` | Max dialogue turns per trajectory (swift 3.4.1 **has no `--max_turns`**, executed by the plugin) |
| `CERES_NUM_GENERATIONS` | `8` | Plugin-side group size G, must match `--num_generations` |
| `CERES_END_PENALTY` | `0.2` | Fix round 2: fixed penalty of `ceres_sequence` for non-`<end/>` endings (length truncation / max_turns / error); `0` = ablate back to old behavior (the 2026-09-01 first full training collapsed because of this: `<end/>` termination rate 55%→0%, see train_oph_lora-0901.log postmortem) |
| `CERES_PLUGIN_STUB_SWIFT` | — (unset) | `=1` forces the plugin to skip real swift imports and degrade to pure-logic mode (for environments without ms-swift: `orms`/`multi_turns` register into a local empty table; **do not set in real training**) |
| `CERES_TRAJ_DIR` | `./traj_dump` | Trajectory dump directory (`traj_YYYYMMDD.jsonl` append-write; consumed by eval) |
| `CERES_MODEL` | `/hy-tmp/model/Qwen2.5-VL-32B-Instruct` | Base model path for training scripts (can be overridden to 7B for smoke tests to speed up) |
| `CERES_ADAPTERS` | — (empty) | SFT→GRPO handoff: SFT warmup adapter directory (scripts convert it to `--adapters`) |
| `CERES_USE_VLLM` | `0` (pt infer) | Only `=1` appends the vLLM colocate params (including `--num_infer_workers`==card count). **Default off** reasons and enablement prerequisites in §7 known-limitation item 2 and the run_grpo_oph_lora.sh header comment |
| `CERES_OUTPUT_DIR` / `CERES_DATASET` / `CERES_SFT_DATASET` | see each script | Output directory / dataset path overrides |
| `NPROC_PER_NODE` / `CUDA_VISIBLE_DEVICES` | `4` / `0,1,2,3` | Card count and card ids (in vLLM mode `--num_infer_workers` must equal card count to enter colocate; scripts already pass this accordingly) |
| `PER_DEV_BS` / `GRAD_ACCUM` | `1` / `16` | Batch params (scripts do generation_batch divisibility checks) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (exported by scripts) | 32B memory-fragmentation OOM insurance (Appendix A) |

### 3.3 Data Pipeline (data_pipeline/, see its README)
| Variable | Default | Description |
|---|---|---|
| `CERES_API_BASE` / `CERES_API_KEY` | DashScope-compatible endpoint / — (unset→mock) | Teacher-side LLM for offline synthesis |
| `CERES_TEACHER_MODEL` | `qwen-max` | Strong model for synthesizing demo dialogues |
| `CERES_SYNTH_CONCURRENCY` / `CERES_SYNTH_TIMEOUT` | `8` / `60` | Synthesis concurrency and timeout |
| `CERES_SYNTH_CACHE_DIR` | `./.llm_cache` | Disk cache (sha1(message+seed+model)) |

## 4. Data Preparation (four-step pipeline)

For the data schema and artifact list, see [`data/README.md`](data/README.md); for **acquisition methods, license notes (APTOS/EyePACS not used in the first batch) and complete parameters, see [`data_pipeline/README.md`](data_pipeline/README.md)**:

```bash
# Step 0  register local raw datasets (IDRiD/ODIR-5K/REFUGE/PALM/OCTID/CMExam/CMB…; run once per source)
python -m data_pipeline.prepare_datasets --type cmexam --input RAW/CMExam/data/train.csv
python -m data_pipeline.prepare_datasets --type idrid --input RAW/IDRiD/IDRiD_grading.csv --image-dir RAW/IDRiD/images
#        (image-directory sources use --image-dir; see data_pipeline/README.md for all source types and real parameter forms)
# Step 1  extract questions → raw_pool.jsonl
python -m data_pipeline.extract_questions
# Step 2  rewrite into "classroom student questions" → queries_candidates.jsonl
python -m data_pipeline.rewrite_question
# Step 3  API-simulate ≤5-turn teaching dialogues → sft_raw.jsonl
python -m data_pipeline.synthesize_dialogue
# Step 4  quality-filter → ceres_oph_sft_warmup.jsonl (for SFT) + filter_report.json
python -m data_pipeline.quality_filter
#        build query dataset → ceres_oph_queries.jsonl (for GRPO)
python -m data_pipeline.build_query_dataset

# CMExam 4k dual-format dataset (8 steps one-shot; MOCK=1 offline smoke, two rewrite rounds ~8k flash calls):
#   → data/ceres_cmexam4k_queries.jsonl (contract schema + gt three columns, directly feeds GRPO)
#   → data/cmexam4k_qa_grpo.jsonl (single-turn QA GRPO: student-style MCQ question + gt standard answer)
bash scripts/run_cmexam4k.sh
CERES_DATASET=data/ceres_cmexam4k_queries.jsonl bash scripts/run_grpo_oph_lora.sh

# PubMedQA 1k English dual-format (step 1 auto-downloads via hf-mirror; MOCK=1 offline smoke; ~2.2k flash calls):
#   → data/ceres_pubmedqa_queries.jsonl (contract schema + gt three columns, all rows lang=en, directly feeds GRPO)
#   → data/pubmedqa_qa_grpo.jsonl (single-turn QA GRPO: A. Yes/B. No/C. Maybe options + gt)
bash scripts/run_pubmedqa.sh
CERES_DATASET=data/ceres_pubmedqa_queries.jsonl bash scripts/run_grpo_oph_lora.sh
```

**Trajectory language follows the source data language**: if the source data is English → sampled/synthesized trajectories (student questions, teacher actions, student replies) are all English; Chinese source → all Chinese. The language is determined row-by-row in Step 1 and written to the `lang` column (`--lang auto` default: text sources judge zh by stem CJK ratio ≥0.25, English image sources idrid/odir/refuge/palm/gamma/octid/kermany/octdl are directly judged en; `--lang zh|en` can force-override), the Chinese/English two sets of prompts/templates/mock are chosen row-by-row, and `ceres_oph_queries.jsonl` output rows carry the `lang` column—during GRPO training swift passes it to the `teacher_env` plugin as an extra dataset column, `plugin._student_messages` produces the corresponding-language observation according to `dd['lang']`, and `student_chat(..., lang='en')` switches the student simulator language (the real path appends an English-output instruction in the system prompt, JSON schema unchanged). **When a dataset row has no lang column, it defaults to zh, fully backward compatible**; the teacher model's output language is naturally determined by the dialogue history and prompts, with no forced output-layer control. See the "Language mechanism" section of [`data_pipeline/README.md`](data_pipeline/README.md) for details.

## 5. Training Order

```bash
# ① SFT format warmup (1 epoch; raises action-label success rate from ~40% to ~95%)
bash scripts/run_sft_warmup.sh             # → output/ceres-oph-sft-warmup

# ② GRPO main training (requires export CERES_STUDENT_API_KEY=sk-xxx; the script validates it)
export CERES_ADAPTERS=output/ceres-oph-sft-warmup   # hand off from SFT (3.4.1 mounts via --adapters)
bash scripts/run_grpo_oph_lora.sh                    # → output/ceres-oph-grpo-lora-v1
#    default pt infer (rollout and scoring in the same process, trajectory store naturally consistent); vLLM requires explicit
#    CERES_USE_VLLM=1 (enablement prerequisites in §6 step 5 and §7 known-limitation item 2)

# ③ offline evaluation (can run anytime during/after training, reads traj_dump)
python eval/analyze_traj.py --traj-dir traj_dump --dataset data/ceres_oph_queries.jsonl --out report.md

# ④ training-effect viewing (human-readable dialogue demo, inference only, no training; see §9)
bash scripts/run_demo_dialogue.sh                                     # baseline (no adapter mounted)
bash scripts/run_demo_dialogue.sh --adapters output/ceres-oph-grpo-lora-v1/checkpoint-xxx
```

Key convention: `generation_batch = 1 (per-card micro-batch) × 4 (cards) × 16 (gradient accumulation) = 64`, `64 / 8 (G) = 8` independent prompt/step groups (both training scripts do divisibility checks, and grpo_trainer.py:232-247 also hard-validates).

## 6. Runtime Checks

| # | Command (run from project root) | Expected behavior |
|---|---|---|
| 1 | `python eval/analyze_traj.py --traj-dir traj_dump --out report.md` | Generates report; "observation-only, not in reward" annotation in the "learning effect" section; safety red-line hits should be 0; "time evolution" section judged ✅ |
| 2 | Before formal training if vLLM is needed: `CERES_USE_VLLM=1 bash scripts/run_grpo_oph_lora.sh` (must install vllm in venv first; a few steps then can interrupt) | **Compare whether rewards are zeroed**: log shows "You are using colocate mode..."; completions.jsonl rewards not all 0, no uid-not-found warning flooding → proves rollout rank and scoring rank are consistent, vLLM can be enabled; if rewards are all 0, vLLM risk is confirmed (§7-2), keep default pt infer |
| 3 | During/after GRPO training: `python eval/check_train_log.py <train.log>` (during training can `tail -f train.log \| python eval/check_train_log.py -`) | Exit code 0 = healthy; exit code 1 = collapse alarm (C1 clipped_ratio≥0.95 / C2 reward decreases over training / C3 consecutive declines), stop training and fall back to SFT baseline, and use `eval/analyze_traj.py` to re-check trajectory-side end_tag/recall·hint evolution |

## 7. Known Limitations (ledger rulings, kept as-is)

1. **Anchor credit only groups by uid within the same reward call on the same rank**: under GRPO multi-card, the G trajectories of the same uid may be split across different ranks/batches, so `ceres_anchor`'s within-group relative advantage degrades to "grouping within rank"; GRPO's own within-group normalization is done globally after gather, so the whole still approximates the paper's two-layer credit (Pre-flight #8, accepted as a known limitation; when there is only one uid, credit is naturally 0, and the plugin has defensively handled this).
2. **vLLM off by default (fix round 1 ruling), default pt infer**: under colocate, `multi_turn_func` only runs in the infer rank process (rollout side effects write that process's TrajectoryStore), while `_score_completions` is each rank scoring its **local batch** (grpo_trainer.py:887-907), and `_fast_infer` first gathers all inputs then round_robin redistributes (:760-772, infer_rank determination :510-521)—rollout and scoring rank/batch are not guaranteed to align → this rank's STORE cannot find the trajectory → **reward silently zeroed**. Under pt infer (without `--use_vllm`), both are in the same process and same batch (:866-871), the same path as the already-tested run_4card_32b.sh (that venv also has no vllm installed). The only entry point to enable vLLM is `CERES_USE_VLLM=1` (auto-appends the colocate-required `--num_infer_workers`=card count), and the **enablement prerequisite** is the runtime check "completions.jsonl rewards not all 0, no uid-not-found warning flooding" (§6-2). Also: swift 3.4.1 has no `--max_turns` / `--vllm_mode` / `--steps_per_generation` / `--vllm_tensor_parallel_size` (grep-verified, disabled; the last's equivalent param `tensor_parallel_size` defaults to 1, omitted); the turn cap goes through `CERES_MAX_TURNS`.
3. **Loss coverage in multi-turn mode**: when `multi_turn_func` is active, grpo_trainer forces the template loss_scale to `default` (all assistant turns count toward loss, user turns do not, grpo_trainer.py:558-562); GRPO scripts still pass `--loss_scale all` per contract, and actual gradient coverage should be confirmed by inspecting the loss_mask in the training logs (design §9.4). **The SFT warmup script uses `--loss_scale default`** (fix round 2 ruling): swift's `'all'`=TrainAllLossScale would also count the student user turns toward loss (loss_scale.py:125-128), while SFT should only teach the teacher-side actions—the `all` in the original design §8.3 was a flaw of wrongly carrying the §8.1 GRPO convention into SFT, and it has been corrected.
4. **learner-outcome not in reward**: `W = dict(fmt=0.4, rule=0.6, out=0.0, risk=0.1)` (§7.5 reserved slot)—the student's self-assessed mastery is noisy and easily induced by rhetoric; mastery gain/misconception resolution are only used as offline observation metrics.
5. **Reward lookup depends on the `(uid, last-turn assistant text)` hash**: completions and trajectories have no stable index correspondence; on miss, reward is 0 with a warning (never raises an exception). If rewards in completions.jsonl are constantly 0, check here first. On traj_key collision (G rollouts of the same uid with byte-identical last-turn text, a common degeneration of max_turns repetition), the reward side already selects the best by full-text comparison of `kwargs['messages']` (store main index multi-valued + `plugin._pick_by_messages`, the messages column passed through as-is by swift rows_to_batched); **offline eval's jsonl dedup still overwrites by traj_key in write order**, so colliding trajectories merge into one in the offline report (n_traj slightly underestimated).
6. **Two multi-card must-break points**: `--split_dataset_ratio 0` (otherwise post-training eval crashes with `KeyError: eval_reward`) and `--gradient_checkpointing_kwargs '{"use_reentrant": false}'` (otherwise `lora_B... marked as ready twice`); both training scripts have these built in (Appendix A, SFT multi-card also applies).
7. **Medical safety red lines are a substring blacklist** (`RISK_PATTERNS`, same source for reward and eval): for production deployment it is recommended to add a medical LLM reviewer + guideline key-point checklist + physician spot-check (design §11).
8. **`READING_NODES` is an implementation-side convention**: `grammar.READING_NODES` = design §5's reading-image prefixes (`ophthalmology/retina` / `ophthalmology/neuro_ophth`) + the implementation-side extension `ophthalmology/fundus` (the data pipeline's self-made `fundus/normal_reading`, `fundus/multilabel_reading` nodes happen to need to hit G5). **When adding new reading-type course nodes, `grammar.READING_NODES` and design §5 must be updated in sync**, otherwise the node's G5 "let the student describe what they see first" hard validation is silently skipped.
9. **Crash save is "best effort" rather than a transactional guarantee** (§10): `save_on_exit` calls `PeftModel.save_pretrained` at the last moment of a process crash—under ZeRO-3 the weight shards are on each rank, and at extreme crash moments (e.g., NCCL collective communication already broken) gather may fail, in which case it only warns and gives up, original exception/exit-code semantics unchanged; and it tries at most once. **It does not replace the regular `--save_steps` checkpoint**, it only lowers the probability of "total loss".
10. **Demo script defaults to single-card 7B**: `run_demo_dialogue.sh` fixes `CUDA_VISIBLE_DEVICES=0` (overridable), `--model` defaults to the 7B local path; the demo's PtEngine and the training sampling engine are mutually exclusive in the same process—**finish training (or open another card) before running the demo**, don't share cards and fight over memory with the training process. The teacher-side PtEngine path cannot be validated offline (no swift/torch outside the training environment); the real-machine path must be validated by actually running the demo.

## 8. Version and Dependencies

Training framework ms-swift **3.4.1.post1** (local `/hy-tmp/my-env/swift`, Python 3.10); all training flags have been grep-verified one by one in the installed source, and the `flag → source file:line` mapping table is in [`.sdd/task-5-report.md`](.sdd/task-5-report.md). Direct dependencies in [`requirements.txt`](requirements.txt).

## 9. Training-effect Viewing: N Questions × ≤4-turn Dialogue Demo (`eval/demo_dialogue.py`, work package B)

Give the training effect a "human-visible" comparison: the same batch of questions, the same student simulator (BigModel glm, lang follows row data), only the teacher changes—`--adapters` is the core switch: **not mounted = baseline, mount a LoRA checkpoint = post-training**. The teacher runs an in-process swift PtEngine (loaded once and reused in a loop, temperature 0.7 / max_tokens 1024), each question at most `--rounds 4` turns, and the teacher output `<end` ends early.

```bash
bash scripts/run_demo_dialogue.sh                                          # baseline (default data/ceres_oph_queries.jsonl + 7B)
bash scripts/run_demo_dialogue.sh --adapters output/ceres-oph-sft-warmup \
                                  --out-dir output/demo/sft                # after SFT
bash scripts/run_demo_dialogue.sh --adapters output/ceres-oph-grpo-lora-v1/checkpoint-150 \
                                  --out-dir output/demo/grpo               # after GRPO
```

- **Three input forms** (`--type` defaults to auto: `.csv`→cmexam, rows containing `messages`→queries, otherwise pool): ① GRPO query-set jsonl used directly; ② raw_pool/candidates jsonl—raw_pool without `student_question` calls the BigModel API on the spot to generate using data_pipeline rewrite prompts (candidates pass through with zero API), `--pool-limit` (default 48) controls the rewrite cost; ③ CMExam csv (`--type cmexam`) reuses `extract_questions.extract_records` rules to extract questions (ophthalmology keyword filtering/node mapping) then rewrites.
- **Question-selection determinism**: `--num 8 --seed 0` samples by curriculum_node grouping in round-robin (covering different nodes), same input same seed gives identical results; prints the question-selection list (uid/node/language) at startup.
- **Artifacts** (`--out-dir`, default output/demo): `demo_dialogues.jsonl` (full dialogue + per-turn details + statistics) and `demo_dialogues.md` (Chinese human-readable version: teacher-turn DSL labels rendered with `` `recall` → `hint(L2)` `` badges + code blocks, per-question grammar statistics at the end—G1 turn pass rate/G2~G5/label coverage/action_stats/red-line hits; summary table at the end).
- **Interpretation convention**: action-label usage rate (question-level appearance rate) is the most intuitive metric—baseline should be near 0%, and should rise significantly after SFT/GRPO; also check the proportion of "ending = teacher actively closing with `<end/>`" and red-line hits (target 0).
- Student simulator API failures have built-in degradation (`degraded` replies) and don't interrupt the whole batch; teacher engine exceptions seal that question as an error record and the whole batch continues.
- Requires an interpreter that can import swift (local venv: `PY_BIN=/hy-tmp/my-env/swift/bin/python bash scripts/run_demo_dialogue.sh ...`), defaults to single card (`CUDA_VISIBLE_DEVICES` overridable).

## 10. Crash Protection: Save LoRA Before Exiting on Any Error (`ceres_plugin/save_on_exit.py`, work package C)

Auto-registered at the end of `plugin.py` (takes effect when loaded via `--external_plugins` in a real swift environment; `extra_callbacks` is an in-process shared list, **appending takes effect immediately with no CLI switch needed**—defined in swift 3.4.1 `swift/plugin/callback.py:30`, consumed by `swift/llm/train/sft.py:231`, and the SFT/PT/GRPO pipelines all inherit this path). Test-stub mode (`CERES_PLUGIN_STUB_SWIFT=1`) does not register, behavior unchanged.

**Behavior**: at training start (`on_train_begin`/`on_step_end`) capture the model and output_dir references and install `sys.excepthook` + SIGTERM/SIGINT handlers; when any uncaught exception (including cascading exceptions after OOM/NCCL/API) or kill/Ctrl-C is triggered, the local_rank 0 process saves the current LoRA delta via `model.save_pretrained(<output_dir>/emergency-adapter)` (PeftModel only saves the delta, small and fast), then **restores the original excepthook/signal default behavior before exiting** (the original exception is still raised, original exit-code semantics preserved: SIGINT→130, SIGTERM→143). Idempotent: each process tries at most once; if the save itself fails again, only warns, never masks the original exception; `on_train_end` does not uninstall the hook (crashes in post-training eval/dump phase also need saving).

Note the division of labor: student_sim's API failures **already degrade without raising** (training won't exit due to API failure); this mechanism covers all other uncaught errors and external kills.

**Recovery** (emergency-adapter is a standard LoRA adapter directory):

```bash
# resume GRPO from the crash point:
export CERES_ADAPTERS=output/ceres-oph-grpo-lora-v1/emergency-adapter
bash scripts/run_grpo_oph_lora.sh
# or directly mount to continue training / view the effect:
bash scripts/run_demo_dialogue.sh --adapters output/ceres-oph-grpo-lora-v1/emergency-adapter
```

Known boundaries in §7 item 9 (ZeRO-3 gather may fail at crash time, best effort, does not replace regular checkpoints).

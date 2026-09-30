#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CERES GRPO training log checker (from the 2026-09-02 train_oph_lora.log incident).

Background: the first full run looked stable (smooth loss, reward 0.7+, no NaN/OOM, 250/250)
but the policy had collapsed -- <end/> termination 55% -> 0%, the second half 100% truncated at
768 tokens, sequence reward 0.986 -> 0.73 (RL trained the reward down). This script turns those
"only-visible-in-hindsight" alarms into checks that run during/after training:

    python eval/check_train_log.py train_oph_lora.log          # post-training check
    tail -f train_oph_lora.log | python eval/check_train_log.py -   # streaming

Checks (any hit -> exit code 1):
    C1  completions/clipped_ratio >= 0.95 -- nearly all truncated (strongest collapse signal);
    C2  rewards/<sequence ORM>/mean: last-5 mean is >= 0.05 below first-5 mean -- reward trained down;
    C3  reward decline streak -- >= 10 consecutive strictly-decreasing points (tolerance 1e-6);
    C4  loss ≡ β·kl -- |loss - beta*kl| < 1e-6 throughout means loss has no PG signal (warn only,
        not an exit code: this is normal swift GRPO shape, but loss can't be read as a progress metric).

Metric extraction: dict lines like {'loss': ..., 'reward': ..., 'kl': ...} in swift/trl GRPO logs
(regex per dict; tqdm bars don't interfere). stdlib only.

Note: this script only looks at training-side metrics; use eval/analyze_traj.py's time-evolution
buckets for trajectory-side collapse metrics (end_tag rate / recall·hint disappearance).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys

__all__ = [
    "parse_metric_dicts", "extract_series", "check_series", "render_report", "main",
]

#: One full metric dict in the log (contains 'loss'; swift emits one every logging_steps)
_METRIC_DICT_RE = re.compile(r"\{[^{}]*'loss'[^{}]*\}")

#: Extracted metrics -> (name, key regex, description). Sequence reward keys carry an ORM class
#: prefix, so regex is used instead of exact match.
_SERIES_SPECS = [
    ("loss", r"loss", "策略损失（注意：swift GRPO 里 ≈ β·KL，见 C4）"),
    ("reward", r"reward", "总奖励（加权合成）"),
    ("seq_reward", r"rewards/[^/]+/mean", "序列奖励（rewards/<ORM>/mean）"),
    ("reward_std", r"reward_std", "组内奖励标准差（优势信号强度）"),
    ("kl", r"kl", "与参考模型的 KL"),
    ("clipped_ratio", r"completions/clipped_ratio", "截断收场轨迹占比"),
    ("comp_len", r"completions/mean_length", "完成长度（token，多轮累加）"),
]

#: Thresholds (empirical, from the incident; overridable via CLI)
CLIP_CRIT = 0.95          # C1: clipped_ratio critical
CLIP_WARN = 0.80          # C1: warning line
REWARD_DROP = 0.05        # C2: last-5 vs first-5 mean drop
DECLINE_STREAK = 10       # C3: consecutive declining log points
BETA_DEFAULT = 0.04       # C4: beta of that run (only for the loss≈β·kl note; --beta overrides)


def parse_metric_dicts(text):
    """Extract all metric dicts from the log text -> list[dict] (numbers as float, rest as-is)."""
    out = []
    for m in _METRIC_DICT_RE.finditer(text):
        raw = m.group(0)
        rec = {}
        for kv in re.finditer(r"'([^']+)': ([^,}]+)", raw):
            key, val = kv.group(1), kv.group(2).strip()
            if len(val) >= 2 and val[0] == "'" and val[-1] == "'":
                val = val[1:-1]     # strip quotes from string values (e.g. '1/250' -> 1/250)
            try:
                rec[key] = float(val)
            except ValueError:
                rec[key] = val
        if rec:
            out.append(rec)
    return out


def extract_series(rows):
    """metric dicts -> {metric name: [float, ...]} (keys matched by _SERIES_SPECS regex).

    If one pattern matches multiple keys (e.g. rewards/<ORM>/mean for both ORMs), deterministically
    prefer the key containing "Sequence", then lexicographic -- independent of dict order.
    """
    series = {name: [] for name, _pat, _desc in _SERIES_SPECS}
    for row in rows:
        for name, pat, _desc in _SERIES_SPECS:
            matches = [(key, val) for key, val in row.items()
                       if re.fullmatch(pat, key) and isinstance(val, float) and math.isfinite(val)]
            if matches:
                matches.sort(key=lambda kv: ("Sequence" not in kv[0], kv[0]))
                series[name].append(matches[0][1])
    return series


def _streak(values, tol=1e-6):
    """Length of the trailing strictly-decreasing streak (equal points pause but don't reset it)."""
    best = cur = 0
    for i in range(1, len(values)):
        if values[i] < values[i - 1] - tol:
            cur += 1
            best = max(best, cur)
        elif values[i] > values[i - 1] + tol:
            cur = 0
    return best


def check_series(series, beta=BETA_DEFAULT, clip_crit=CLIP_CRIT, clip_warn=CLIP_WARN,
                 reward_drop=REWARD_DROP, decline_streak=DECLINE_STREAK):
    """Run C1~C4 checks on the extracted series -> (alarms, notes); non-empty alarms = problem.

    alarms: list[dict(code, level, detail)], level in {"crit","warn"};
    notes:  list[str] (informational, not counted in the exit code).
    """
    alarms, notes = [], []
    clip = series.get("clipped_ratio") or []
    if any(v >= clip_crit for v in clip):
        n_crit = sum(1 for v in clip if v >= clip_crit)
        alarms.append({"code": "C1", "level": "crit", "detail": (
            "completions/clipped_ratio 有 %d/%d 个日志点 ≥ %.2f（轨迹几乎全部截断收场，"
            "<end/> 终止已消失——策略坍缩最强信号）" % (n_crit, len(clip), clip_crit))})
    elif any(v >= clip_warn for v in clip):
        alarms.append({"code": "C1", "level": "warn", "detail": (
            "completions/clipped_ratio 出现 ≥ %.2f 的日志点（截断率偏高，盯紧是否趋 1）"
            % clip_warn)})
    seq = series.get("seq_reward") or []
    if len(seq) >= 10:
        head = sum(seq[:5]) / 5.0
        tail = sum(seq[-5:]) / 5.0
        if head - tail >= reward_drop:
            alarms.append({"code": "C2", "level": "crit", "detail": (
                "序列奖励首 5 点均值 %.4f → 末 5 点均值 %.4f（降 %.4f ≥ %.2f）——"
                "RL 把 reward 练降，训练方向错误" % (head, tail, head - tail, reward_drop))})
    reward = series.get("reward") or []
    streak = _streak(reward)
    if streak >= decline_streak:
        alarms.append({"code": "C3", "level": "warn", "detail": (
            "总奖励出现 %d 个日志点的连续下降（阈值 %d）——建议人工确认或早停"
            % (streak, decline_streak))})
    loss, kl = series.get("loss") or [], series.get("kl") or []
    if loss and kl and len(loss) == len(kl):
        if max(abs(l - beta * k) for l, k in zip(loss, kl)) < 1e-6:
            notes.append("C4：|loss − β·kl| 全程 < 1e-6 —— loss 只是 KL 惩罚项，"
                         "不含 PG 信号，不能当训练进展读（swift GRPO 正常形态，仅提示）")
    if not alarms and not notes:
        notes.append("未检出已知坍缩模式（C1~C3）；loss/reward 数值形态见上表。")
    return alarms, notes


def render_report(series, alarms, notes, meta=None):
    """Check results -> human-readable text (for stdout)."""
    meta = meta or {}
    lines = []
    ap = lines.append
    ap("=" * 72)
    ap("CERES GRPO 训练日志体检（%s 个日志点%s）"
       % (len(series.get("loss") or []), meta.get("source", "")))
    ap("=" * 72)
    ap("")
    ap("指标  首 → 末（min ~ max）")
    ap("-" * 72)
    for name, _pat, desc in _SERIES_SPECS:
        vals = series.get(name) or []
        if not vals:
            continue
        ap("%-14s %+10.4f → %+10.4f（%+.4f ~ %+.4f）  # %s"
           % (name, vals[0], vals[-1], min(vals), max(vals), desc))
    ap("")
    if alarms:
        ap("!! 警报（%d 条）" % len(alarms))
        for a in alarms:
            ap("  [%s/%s] %s" % (a["code"], a["level"], a["detail"]))
        ap("")
        ap("结论：⚠ 训练异常——按事故复盘流程处理：停训、回退 SFT warmup 基线、")
        ap("      用 eval/analyze_traj.py --traj-dir traj_dump 复核轨迹侧坍缩指标。")
    else:
        ap("结论：✅ 未触发警报阈值（C1~C3）。")
    for note in notes:
        ap("提示：%s" % note)
    ap("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="CERES GRPO 训练日志体检（坍缩警报 C1~C4）")
    p.add_argument("log", nargs="?", default="-",
                   help="swift GRPO 训练日志路径（缺省 '-' 读 stdin，可 tail -f 管道）")
    p.add_argument("--beta", type=float, default=BETA_DEFAULT, help="C4 的 β（默认 %(default)s）")
    p.add_argument("--clip-crit", type=float, default=CLIP_CRIT)
    p.add_argument("--clip-warn", type=float, default=CLIP_WARN)
    p.add_argument("--reward-drop", type=float, default=REWARD_DROP)
    p.add_argument("--decline-streak", type=int, default=DECLINE_STREAK)
    p.add_argument("--json", action="store_true", help="以 JSON 输出（供脚本消费）")
    args = p.parse_args(argv)

    if args.log == "-":
        text = sys.stdin.read()
        source = "（stdin）"
    else:
        try:
            with open(args.log, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            source = "（来源：%s）" % args.log
        except OSError as e:
            print("[ceres-check] 错误：无法读取 %s（%s）" % (args.log, e), file=sys.stderr)
            return 2
    rows = parse_metric_dicts(text)
    if not rows:
        print("[ceres-check] 错误：日志里未找到任何 metric dict（{'loss': ...} 形态）。", file=sys.stderr)
        return 2
    series = extract_series(rows)
    alarms, notes = check_series(series, beta=args.beta, clip_crit=args.clip_crit,
                                 clip_warn=args.clip_warn, reward_drop=args.reward_drop,
                                 decline_streak=args.decline_streak)
    if args.json:
        print(json.dumps({"n_points": len(rows), "series": series,
                          "alarms": alarms, "notes": notes}, ensure_ascii=False, indent=2))
    else:
        print(render_report(series, alarms, notes, meta={"source": source}))
    return 1 if alarms else 0


if __name__ == "__main__":
    sys.exit(main())

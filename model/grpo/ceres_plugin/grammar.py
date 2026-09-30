# -*- coding: utf-8 -*-
"""Teaching action DSL parser and grammar checks (G1-G5) for the CERES ophthalmology GRPO project.

Zero-dependency (stdlib `re` only): consumed by the plugin reward, SFT quality checks, and eval
behavior stats.

The teacher emits text containing up to 7 action tags per turn:
    <recall>...</recall>              activate prior knowledge
    <hint level="1~3">...</hint>      graded hint (scaffold)
    <check>...</check>                let the student speak / describe first
    <explain>...</explain>            explain
    <correct>...</correct>            explicit correction
    <encourage>...</encourage>        encourage
    <end/>                            self-closing closing tag (no body)

Grammar rules (semantics per contracts.md):
    G1  only the 7 tags; paired, no nesting; <end/> self-closing; >=1 action tag per turn
    G2  <correct> requires a prior recall/hint (scaffold) and a prior check (participation)
    G3  hint level monotonically non-decreasing; same tag >=3 times in one turn is a violation
    G4  <end/> only in the final turn (no teacher turns after, no actions after in the same turn)
    G5  reading tasks (curriculum_node prefix in READING_NODES, difficulty != routine_clarification):
        the first explain/correct must be preceded by a check; missing info is conservatively
        treated as enforced (dd None/non-dict, curriculum_node missing/non-str/blank)

Known limit: design §5 G4 also requires <correct> or a correct student answer before <end/>,
which needs the simulator state (state_after) that the grammar layer can't see; per contracts.md
only the position is checked here, and the state condition is overlaid by the plugin at rollout.

Defensive: every public function is safe for None / empty / malformed input -- never raises,
degrades to 0.0 / False / empty.
"""
import re

__all__ = [
    "TAG_RE",
    "READING_NODES",
    "VALID_TAGS",
    "parse_actions",
    "check_grammar_turn",
    "check_grammar_global",
    "hint_level",
    "validate_sft_trajectory",
    "action_stats",
]

# ---------------------------------------------------------------- Constants and regex

#: The 7 legal action tags (end is the self-closing closing tag).
VALID_TAGS = ("recall", "hint", "check", "explain", "correct", "encourage", "end")

#: Reading-task curriculum node prefixes (a match can trigger the G5 reading-order check).
READING_NODES = ("ophthalmology/retina", "ophthalmology/neuro_ophth", "ophthalmology/fundus")

#: Action tag regex (7 tags, incl. self-closing <end/>):
#:   group "tag"   tag name
#:   group "attrs" attribute fragment (e.g. ' level="1"', no angle brackets)
#:   group "body"  body text (None for self-closing <end/>)
#: (?![\\w-]) word boundary prevents misreading "<endx/>" as <end/>;
#: attrs/body may contain any text (angle brackets, newlines; "< 10" / "> 21" in body is fine).
TAG_RE = re.compile(
    r"<(?P<tag>recall|hint|check|explain|correct|encourage|end)"
    r"(?![\w-])"
    r"(?P<attrs>[^<>]*?)"
    r"(?:/>|>(?P<body>.*?)</(?P=tag)>)",
    re.S,
)

#: Tag-token scan regex for G1: groups are closing slash / tag name / inner text / self-closing slash.
#: Only `<` followed by (optionally `/` plus) letters counts as a tag token; "a < b" / "5<6" in body are safe.
_TAG_TOKEN_RE = re.compile(r"<(/?)([A-Za-z_][A-Za-z0-9_\-]*)([^<>]*?)(/?)>")

#: Hint level extraction regex (tolerates level=2 / level="2" / level = '2').
_LEVEL_RE = re.compile(r"level\s*=\s*[\"']?(\d+)")


# ---------------------------------------------------------------- Public API

def parse_actions(text):
    """Parse all action tags in one teacher turn.

    :param text: teacher single-turn output (tolerates None / non-str / angle brackets / newlines)
    :return: [(tag, attrs, body), ...] in order; body stripped; "" for self-closing <end/>; [] if none.
    Never raises; malformed fragments are skipped and flagged by check_grammar_turn.
    """
    if not isinstance(text, str):
        return []
    acts = []
    for m in TAG_RE.finditer(text):
        tag = m.group("tag")
        attrs = m.group("attrs") or ""
        body = m.group("body")
        acts.append((tag, attrs, (body if body is not None else "").strip()))
    return acts


def hint_level(attrs):
    """Extract the hint level int from attrs; missing/malformed -> 1."""
    if not isinstance(attrs, str):
        return 1
    m = _LEVEL_RE.search(attrs)
    if not m:
        return 1
    try:
        return int(m.group(1))
    except (TypeError, ValueError):  # unreachable in practice, defensive
        return 1


def check_grammar_turn(text):
    """G1: single-turn structural validity -> 1.0 / 0.0.

    Only the 7 legal tags; content tags paired, no nesting, no self-closing; <end/> must be
    self-closing and not nested; at least one action tag per turn. None/empty/free text -> 0.0.
    """
    return 1.0 if not _turn_defects(text) else 0.0


def check_grammar_global(steps, dd=None):
    """G2/G3/G4/G5: whole-trajectory grammar check -> bool.

    :param steps: list of step dicts (only st['teacher'] is consumed).
    :param dd: optional dataset row; G5 uses curriculum_node / difficulty. Missing info (dd
        None / non-dict / curriculum_node missing, non-str, or blank) is conservatively enforced
        so G5 is never silently skipped.
    :return: True if no violation; False for None/empty/malformed steps.
    """
    return not _global_defects(steps, dd)


def validate_sft_trajectory(messages, dd=None):
    """SFT multi-turn quality check: build steps then full validation (structure + G1 + G2~G5).

    :param messages: [{"role": "user"|"assistant", "content": str}, ...] (user=student,
        assistant=teacher; strict alternation ending with assistant; content tolerates multimodal
        list form).
    :param dd: optional dataset row (passed through for G5).
    :return: (ok, reasons); ok=True iff no violations; reasons is a list of violation strings.
    Never raises.
    """
    if messages is None:
        return False, ["messages 为 None"]
    if isinstance(messages, (str, bytes)) or not isinstance(messages, (list, tuple)):
        return False, ["messages 非法：应为消息 dict 列表"]
    msgs = list(messages)
    if not msgs:
        return False, ["messages 为空列表"]

    reasons = []

    # 1) Structure check: dict, legal role, strict user/assistant alternation (starting with user)
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            reasons.append("第%d条消息不是 dict" % (i + 1))
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            reasons.append("第%d条消息角色非法: %r（仅允许 user/assistant）" % (i + 1, role))
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        role = m.get("role") if isinstance(m, dict) else None
        if role != want:
            reasons.append("第%d条消息应为 %s，实际为 %r（要求 user/assistant 交替，"
                           "且以 assistant 收尾）" % (i + 1, want, role))

    # Also require assistant last: the i%2 alternation check alone passes truncated
    # [user, assistant, user], so it must be blocked separately.
    last = msgs[-1]
    last_role = last.get("role") if isinstance(last, dict) else None
    if last_role != "assistant":
        reasons.append("最后一条消息应为 assistant（要求以 assistant 收尾）")

    # 2) Extract assistant (teacher) turns -> steps; per-turn G1
    steps = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            txt = _as_text(m.get("content"))
            steps.append({"turn": len(steps), "teacher": txt})
            for defect in _turn_defects(txt):
                reasons.append("G1: 第%d轮 %s" % (len(steps), defect))
    if not steps:
        reasons.append("轨迹为空：未找到 assistant（教师）消息")
        return False, _dedupe(reasons)

    # 3) Global G2~G5
    reasons.extend(_global_defects(steps, dd))
    reasons = _dedupe(reasons)
    return (not reasons), reasons


def action_stats(texts):
    """Eval behavior stats: teacher texts -> action distribution dict.

    :param texts: iterable of per-turn teacher texts (a single str is treated as one turn)
    :return: dict with n_turns / tag_counts / tag_turns / max_hint_level / hint_levels / has_end.
    Never raises.
    """
    if texts is None:
        turns = []
    elif isinstance(texts, str):
        turns = [texts]
    else:
        try:
            turns = list(texts)
        except Exception:  # non-iterable etc., defensive
            turns = []

    tag_counts = dict.fromkeys(VALID_TAGS, 0)
    tag_turns = dict.fromkeys(VALID_TAGS, 0)
    hint_levels = []
    has_end = False
    for item in turns:
        text = item if isinstance(item, str) else ""
        seen = set()
        for tag, attrs, _body in parse_actions(text):
            tag_counts[tag] += 1
            seen.add(tag)
            if tag == "hint":
                hint_levels.append(hint_level(attrs))
            elif tag == "end":
                has_end = True
        for tag in seen:
            tag_turns[tag] += 1
    return {
        "n_turns": len(turns),
        "tag_counts": tag_counts,
        "tag_turns": tag_turns,
        "max_hint_level": max(hint_levels) if hint_levels else 0,
        "hint_levels": hint_levels,
        "has_end": bool(has_end),
    }


# ---------------------------------------------------------------- Internal helpers

def _turn_defects(text):
    """G1 internal: return structural defect strings for one turn ([] <=> valid turn, 1.0)."""
    if not isinstance(text, str):
        return ["教师文本非字符串"]
    if not text.strip():
        return ["教师文本为空"]
    tokens = _TAG_TOKEN_RE.findall(text)
    if not tokens:
        return ["无任何动作标签（要求只用 7 类动作标签的结构化输出）"]

    defects = []
    stack = []          # stack of unclosed content tags (no nesting => depth <= 1)
    n_actions = 0
    for closing, name, _mid, selfclose in tokens:
        if name not in VALID_TAGS:
            defects.append("未知标签 <%s>（仅允许 7 类动作标签）" % name)
            continue
        if name == "end":
            if closing:
                defects.append("出现配对形式的 </end>（<end/> 必须自闭合）")
            elif not selfclose:
                defects.append("<end> 必须写成自闭合 <end/>")
            elif stack:
                defects.append("<end/> 不得嵌套在其它动作标签内部")
            else:
                n_actions += 1
            continue
        if closing:
            if not stack:
                defects.append("多余的闭合标签 </%s>" % name)
            elif stack[-1] != name:
                defects.append("标签配对错误：<%s> 被 </%s> 闭合" % (stack[-1], name))
                stack = []
            else:
                stack.pop()
        elif selfclose:
            defects.append("<%s/> 不允许自闭合（%s 必须携带正文）" % (name, name))
        else:
            if stack:
                defects.append("标签嵌套：<%s> 内出现 <%s>（禁止嵌套）" % (stack[-1], name))
            else:
                stack.append(name)
                n_actions += 1
    if stack:
        defects.append("标签未闭合: " + "、".join("<%s>" % t for t in stack))
    if n_actions == 0 and not defects:
        defects.append("无任何动作标签")
    return defects


def _g5_enforced(dd):
    """Whether G5 is enforced: reading-node prefix + difficulty != routine_clarification.

    Missing info is conservatively enforced (dd None / non-dict / curriculum_node missing,
    non-str, or blank) so G5 is never silently skipped when rollout data lacks the column.
    """
    if dd is None or not isinstance(dd, dict):
        return True
    node = dd.get("curriculum_node")
    if not isinstance(node, str) or not node.strip():
        return True  # missing/malformed/blank node -> conservative enforce (same as dd=None)
    if not node.startswith(READING_NODES):
        return False
    difficulty = dd.get("difficulty")
    if not isinstance(difficulty, str):
        difficulty = ""
    return difficulty != "routine_clarification"


def _global_defects(steps, dd=None):
    """G2~G5 internal: return violation strings for the whole trajectory ([] <=> pass).

    Shared by check_grammar_global and validate_sft_trajectory so bool and reasons stay in sync.
    """
    if steps is None:
        return ["steps 为 None"]
    if isinstance(steps, (str, bytes)) or not isinstance(steps, (list, tuple)):
        return ["steps 非法：应为 step dict 列表"]
    step_list = list(steps)
    if not step_list:
        return ["轨迹为空：没有任何教师回合"]

    defects = []
    g5 = _g5_enforced(dd)
    has_scaffold = False   # G2: recall/hint seen
    has_check = False      # G2/G5: student participated (check)
    last_level = 0         # G3: hint level monotonic baseline
    end_turn = -1          # G4: index of first <end/> turn

    for i, st in enumerate(step_list):
        teacher = st.get("teacher") if isinstance(st, dict) else None
        if isinstance(teacher, str):
            if not teacher.strip():
                defects.append("第%d轮 teacher 文本为空" % (i + 1))
        else:
            defects.append("第%d轮 teacher 文本缺失或类型非法" % (i + 1))
            teacher = ""

        acts = parse_actions(teacher)

        # G3-b: same tag repeated >=3 times in one turn (redundancy penalty)
        counts = {}
        for tag, _attrs, _body in acts:
            counts[tag] = counts.get(tag, 0) + 1
        for tag, c in counts.items():
            if c >= 3:
                defects.append("G3: 第%d轮 <%s> 重复 %d 次（同轮同标签 ≥3 次违规）"
                               % (i + 1, tag, c))

        ended_here = False
        for tag, attrs, _body in acts:
            if ended_here:
                # G4: another action after <end/> in the same turn
                defects.append("G4: 第%d轮 <end/> 之后仍出现 <%s>" % (i + 1, tag))
                ended_here = False
            if tag == "end":
                if end_turn < 0:
                    end_turn = i
                ended_here = True
                continue
            # G2: correct must come after scaffold and student participation (check before set => order-sensitive in turn)
            if tag == "correct" and not (has_scaffold and has_check):
                if not has_scaffold:
                    defects.append("G2: 第%d轮 <correct> 之前没有任何 <recall>/<hint> 脚手架"
                                   % (i + 1))
                if not has_check:
                    defects.append("G2: 第%d轮 <correct> 之前没有 <check>（学生未参与）"
                                   % (i + 1))
            if tag in ("recall", "hint"):
                has_scaffold = True
            if tag == "check":
                has_check = True
            if tag == "hint":
                level = hint_level(attrs)
                if level < last_level:
                    defects.append("G3: 第%d轮 hint level 由 %d 回退到 %d（应单调不减）"
                                   % (i + 1, last_level, level))
                last_level = level
            # G5: reading task: explain/correct must be preceded by a check
            if tag in ("explain", "correct") and g5 and not has_check:
                defects.append("G5: 第%d轮 <%s> 之前未出现 <check>"
                               "（阅片任务必须先让学生描述所见）" % (i + 1, tag))

    # G4: <end/> only in the final turn
    if end_turn >= 0 and end_turn < len(step_list) - 1:
        defects.append("G4: <end/> 出现在第%d轮，但其后仍有教师回合（只能出现在最后一轮）"
                       % (end_turn + 1))
    return defects


def _as_text(content):
    """Safely coerce message content to str: tolerates None / str / list (multimodal chunks) / dict."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content"):
            val = content.get(key)
            if isinstance(val, str):
                return val
        return ""
    return ""


def _dedupe(reasons):
    """Dedupe preserving order (structure and G1 checks may report the same issue)."""
    return list(dict.fromkeys(reasons))

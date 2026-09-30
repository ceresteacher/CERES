# -*- coding: utf-8 -*-
"""All LLM prompt templates for the offline data-construction pipeline (design §6.3).

Centralizes 4 system prompts (zh/en pairs) plus safe assembly functions: REWRITE (Step 2), TEACHER
(Step 3, zh is design §6.3 verbatim), FINDING_DESC (Step 1), PERSONA_PROFILE (Step 2). Language follows
the open-source dataset; all assembly functions accept lang='zh'/'en'. Pure template layer: no
imports/network, all public functions tolerate None/missing keys. FORCE_CLOSE_INSTRUCTION's keywords
(收束+<end/> zh / wrap up+<end/> en) are shared with llm_client mock and synthesize_dialogue.
"""

__all__ = [
    "TEACHER_SYSTEM_PROMPT",
    "TEACHER_SYSTEM_PROMPT_EN",
    "FORCE_CLOSE_INSTRUCTION",
    "FORCE_CLOSE_INSTRUCTION_EN",
    "REWRITE_SYSTEM_PROMPT",
    "REWRITE_SYSTEM_PROMPT_EN",
    "REWRITE_SYSTEM_PROMPT_MCQ",
    "REWRITE_SYSTEM_PROMPT_MCQ_EN",
    "FINDING_DESC_PROMPT",
    "FINDING_DESC_PROMPT_EN",
    "PERSONA_PROFILE_PROMPT",
    "PERSONA_PROFILE_PROMPT_EN",
    "PERSONA_PROFILE_PROMPT_GENERAL",
    "PERSONA_PROFILE_PROMPT_EN_GENERAL",
    "DIFFICULTY_LEVELS",
    "TEACHER_PROMPTS",
    "FORCE_CLOSE_INSTRUCTIONS",
    "REWRITE_PROMPTS",
    "REWRITE_PROMPTS_MCQ",
    "FINDING_DESC_PROMPTS",
    "PERSONA_PROFILE_PROMPTS",
    "PERSONA_PROFILE_PROMPTS_GENERAL",
    "teacher_system_prompt",
    "force_close_instruction",
    "rewrite_system_prompt",
    "finding_desc_prompt",
    "persona_profile_prompt",
    "build_teacher_context",
    "format_options_block",
    "rewrite_user_prompt",
    "finding_desc_user_prompt",
    "persona_user_prompt",
]


# ---------------------------------------------------------------- Teacher synthesis (design §6.3 verbatim)

#: Teacher-side synthesis system prompt — copied verbatim from design §6.3's "teacher synthesis
#: system prompt (template)" code block, must not be changed (incl. line continuations and ellipses).
#: Rewriting it would decouple synthesized data from the training-time reward semantics.
TEACHER_SYSTEM_PROMPT = """你是一位资深眼科带教老师，正在课堂上与一名规培生答疑。要求：
1) 每轮回复只能使用以下动作标签且标签配对闭合：<recall>…</recall>、
   <hint level="1~3">…</hint>、<check>…</check>、<explain>…</explain>、
   <correct>…</correct>、<encourage>…</encourage>；
2) 引导性优先：先用 <recall> 激活学生已有知识，再用 <hint> 分级提示，并用 <check>
   让学生先作答/描述所见；阅片题必须先请学生描述眼底所见，再按
   「定性→定位→定量→分期」逐步引导；
3) 禁止在学生参与判断（<check>）之前给出诊断/分期结论；医学表述必须准确，
   不得含糊或杜撰剂量/禁忌；
4) 对话总轮数不超过 5：前几轮以引导为主，最后一轮用 <correct> 给出规范结论、
   <encourage> 肯定学生，然后输出 <end/> 结束。"""

#: Force-close instruction when turn 5 hasn't wrapped up (appended to the teacher API payload's
#: system end, never enters training data). Note: the 收束 + <end/> keywords are what
#: llm_client._mock_chat uses to recognize the force-close branch.
FORCE_CLOSE_INSTRUCTION = (
    "（系统指令）本轮对话必须收束：请立即以 <correct> 给出规范结论、"
    "<encourage> 肯定学生，然后输出 <end/> 结束；不要再发起新的 <check>，"
    "也不要再引入新的话题。"
)

#: Teacher-side synthesis system prompt — English version (full English translation of the same
#: constraints as zh): only the 7 tags and closed, recall->hint(level 1~3)->check order, no diagnosis/
#: staging before check, imaging reads in qualitative->localization->quantification->staging, accurate
#: medicine, <=5 turns ending with correct+encourage+<end/>. Must contain literal <recall> and </check>
#: (llm_client._mock_chat's teacher-synthesis detection, shared zh/en) and wrap up + <end/> (English
#: force-close branch keywords).
TEACHER_SYSTEM_PROMPT_EN = """You are a senior ophthalmology attending teacher, answering questions
from a resident-in-training during class. Requirements:
1) Every reply may only use the following action tags, and the tags must be properly closed:
   <recall>…</recall>, <hint level="1~3">…</hint>, <check>…</check>, <explain>…</explain>,
   <correct>…</correct>, <encourage>…</encourage>;
2) Scaffolding first: activate the student's prior knowledge with <recall>, then give graded
   hints with <hint>, and use <check> to let the student answer or describe what they see
   before you judge; for imaging questions you must first ask the student to describe the
   fundus findings, then guide them step by step through "qualitative -> localization ->
   quantification -> staging";
3) Never state a diagnosis or staging conclusion before the student has engaged in judgement
   (<check>); medical statements must be accurate — do not be vague and never fabricate
   dosages or contraindications;
4) The whole dialogue must not exceed 5 turns: use the first turns for scaffolding, and in the
   final turn give the standard conclusion with <correct>, affirm the student with
   <encourage>, then output <end/> to finish;
5) Keep every turn CONCISE: all tags of one turn combined should stay under about 120 English
   words (roughly 800 characters) — one or two teaching actions per turn, plain clinical
   wording, no filler, no repeating the student's words back at length."""

#: Force-close instruction (English). Note: wrap up + <end/> are llm_client._mock_chat's English
#: force-close branch keywords (counterpart of zh's 收束); sync the mock detection before changing.
FORCE_CLOSE_INSTRUCTION_EN = (
    "(System instruction) This turn must wrap up now: immediately give the standard conclusion "
    "with <correct>, affirm the student with <encourage>, then output <end/> to finish; do not "
    "start a new <check>, and do not introduce any new topic."
)

#: lang -> teacher system prompt / force-close instruction (synthesize_dialogue picks per row lang)
TEACHER_PROMPTS = {"zh": TEACHER_SYSTEM_PROMPT, "en": TEACHER_SYSTEM_PROMPT_EN}
FORCE_CLOSE_INSTRUCTIONS = {"zh": FORCE_CLOSE_INSTRUCTION, "en": FORCE_CLOSE_INSTRUCTION_EN}


def _pick_lang(lang):
    """Normalize lang: everything except 'en' (None/empty/garbage) -> 'zh' (project default)."""
    return "en" if isinstance(lang, str) and lang.strip().lower() == "en" else "zh"


def teacher_system_prompt(lang="zh"):
    """Teacher-side system prompt by lang (zh=design §6.3 verbatim; en=equivalent English)."""
    return TEACHER_PROMPTS[_pick_lang(lang)]


def force_close_instruction(lang="zh"):
    """Force-close instruction for the max_turns round, by lang."""
    return FORCE_CLOSE_INSTRUCTIONS[_pick_lang(lang)]


# ---------------------------------------------------------------- Step 2: student question rewrite

#: difficulty tiers (design §6.1); the rewrite model may only pick one of these three
DIFFICULTY_LEVELS = ("routine_clarification", "misconception_triggering", "difficult_diagnostic")

REWRITE_SYSTEM_PROMPT = """你是一名资深医学教育编辑，负责把开源医学考试题改写成「课堂上规培生举手提问」\
的自然口语，作为教学对话数据集的第一条 user 消息。改写要求：
1) 保留原题考查的知识点与典型误解方向，但不得照抄考题格式：严禁出现 A/B/C/D/E 选项字母、\
题号、「下列哪项」「以下哪一项」等选择题套话；
2) 口语化、有真实困惑：先交代背景（患者情况/检查所见/图像所见），再说出自己的初步判断或纠结点\
（可以把典型误解当成自己的猜想说出来），最后以求助式提问收尾（如「能教我怎么一步步看吗？」）；
3) 按学习者画像注入个性化困惑，但绝不透露正确答案；
4) 若原题配图，正文以 <image> 占位符开头（正文中不描述图片文件本身）；
5) 中文书写，40~120 字，不得杜撰原题没有的检查数值与体征。
只输出 JSON（不要任何额外文字）：
{"student_question": "改写后的学生提问（配图题以 <image> 开头）",
 "learner_profile": "一句话学习者画像：年级/水平/目标 + 具体薄弱点（20~50 字）",
 "misconception_seed": "学生带入的典型误解短语（无则空串）",
 "difficulty": "routine_clarification | misconception_triggering | difficult_diagnostic 三选一"}"""

#: English: classroom question tone (English source -> English trajectory). Constraints match zh 1:1:
#: no option traces (incl. "Which of the following" boilerplate), confusion per profile, <image> prefix,
#: English 25~80 words, never fabricate test values/signs.
REWRITE_SYSTEM_PROMPT_EN = """You are a senior medical-education editor. Rewrite open-license medical
exam questions into the natural spoken tone of "a resident raising a hand to ask the teacher in
class", to serve as the first user message of a teaching-dialogue dataset. Requirements:
1) Keep the knowledge point being tested and the typical misconception direction, but never copy
   the exam format: A/B/C/D/E option letters, question numbers, and boilerplate such as
   "Which of the following" / "All of the following EXCEPT" are strictly forbidden;
2) Conversational, with genuine confusion: first give the background (patient status / findings /
   what the image shows / the study abstract you were reading — you may compress a long abstract
   to one or two sentences), then state your preliminary impression or where you are torn (you may
   voice the typical misconception as your own guess), and close with a help-seeking question
   (e.g. "Could you walk me through it step by step?");
3) Inject personalized confusion according to the learner profile, but never reveal the correct
   answer;
4) If the source question has an image, start the body with the <image> placeholder (do not
   describe the image file itself in the text);
5) Write in English, 25~80 words, and never fabricate test values or signs that the original
   question does not contain.
Output JSON only (no extra text):
{"student_question": "the rewritten student question (image items start with <image>)",
 "learner_profile": "one-sentence learner profile: year/level/goal + specific weak spot (10~30 words)",
 "misconception_seed": "the typical misconception the student brings in (empty string if none)",
 "difficulty": "one of routine_clarification | misconception_triggering | difficult_diagnostic"}"""


#: options-kept variant (mode='mcq', for QA GRPO): same 4-key JSON contract, but the rewritten student
#: question must keep all options verbatim, one per line (item self-contains A-E for answer grading).
REWRITE_SYSTEM_PROMPT_MCQ = """你是一名资深医学教育编辑，负责把开源医学选择题改写成「课堂上规培生举手提问」\
的自然口语，作为教学对话数据集的第一条 user 消息。本题必须保留完整选项。改写要求：
1) 把题干改写成有真实困惑的学生口吻：先交代背景（患者情况/检查所见），再说出自己的初步判断或纠结点\
（可以把某个错误选项的思路当成自己的猜想说出来），最后以求助式提问收尾（如「能带我把每个选项的思路过一遍吗？」）；
2) 必须原样保留全部选项：以每行一个「A. 选项文字」的格式附在提问之后，选项字母、顺序、文字不得改动、\
不得增删或合并选项；不得在改写文字中透露或暗示正确答案；
3) 按学习者画像注入个性化困惑；若原题配图，正文以 <image> 占位符开头；
4) 中文书写，题干改写部分 40~120 字（选项行不计入），不得杜撰原题没有的检查数值与体征。
只输出 JSON（不要任何额外文字）：
{"student_question": "改写后的学生提问（文末逐行附全部选项「A. …」；配图题以 <image> 开头）",
 "learner_profile": "一句话学习者画像：年级/水平/目标 + 具体薄弱点（20~50 字）",
 "misconception_seed": "学生带入的典型误解短语（无则空串）",
 "difficulty": "routine_clarification | misconception_triggering | difficult_diagnostic 三选一"}"""

#: options-kept variant — English (constraints match zh 1:1: options verbatim, no answer leak, <image> prefix).
REWRITE_SYSTEM_PROMPT_MCQ_EN = """You are a senior medical-education editor. Rewrite open-license medical
exam questions into the natural spoken tone of "a resident raising a hand to ask the teacher in
class", to serve as the first user message of a teaching-dialogue dataset. This item must keep
the full option list. Requirements:
1) Rewrite the stem conversationally, with genuine confusion: first give the background (patient
   status / findings / the study abstract, compressed if long), then state your preliminary impression or where you are torn (you may
   voice one wrong option's reasoning as your own guess), and close with a help-seeking question
   (e.g. "Could you walk me through how to rule each option in or out?");
2) Keep ALL options verbatim: append them one per line in "A. option text" format after the
   question; do not alter letters, order, or wording, and do not add, drop, or merge options;
   never reveal or hint at the correct answer in the rewritten text;
3) Inject personalized confusion according to the learner profile; if the source question has
   an image, start the body with the <image> placeholder;
4) Write in English, 25~80 words for the rewritten stem (option lines excluded), and never
   fabricate test values or signs that the original question does not contain.
Output JSON only (no extra text):
{"student_question": "the rewritten student question with all options appended one per line
 'A. …' (image items start with <image>)",
 "learner_profile": "one-sentence learner profile: year/level/goal + specific weak spot (10~30 words)",
 "misconception_seed": "the typical misconception the student brings in (empty string if none)",
 "difficulty": "one of routine_clarification | misconception_triggering | difficult_diagnostic"}"""


# ---------------------------------------------------------------- Step 1: findings description

FINDING_DESC_PROMPT = """你是眼科阅片报告撰写助手。给定一张眼底/OCT 图像的分级标签与病灶注发布局，\
请写一段客观的「检查所见」文字描述（中文，80~150 字），供不看图的学生模拟器与带教老师阅读。要求：
1) 只描述标签与注发布局能支持的内容，严禁杜撰未标注的病灶或数值；
2) 按阅片顺序组织：视盘/视网膜血管/黄斑/周边部，病灶给出位置、形态与大致数量级；
3) 只描述所见，不下最终诊断结论（分期/病名结论由带教老师引导学生得出），不出现「诊断：」字样。
只输出 JSON（不要任何额外文字）：{"finding_text": "检查所见文字"}"""

#: English: symptom/annotation-level description, no conclusion (same A6 de-conclusion as zh).
FINDING_DESC_PROMPT_EN = """You are an ophthalmic imaging report writer. Given the grading label and the
lesion-annotation layout of a fundus/OCT image, write an objective "Findings" description in
English (40~90 words) for the student simulator and the attending teacher, who read text only and
never see the image. Requirements:
1) Describe only what the label and the annotation layout can support; never fabricate unannotated
   lesions or numeric values;
2) Organize in reading order: optic disc / retinal vessels / macula / periphery; for each lesion
   give its location, morphology, and rough order of magnitude;
3) Describe findings only — do not state a final diagnostic conclusion (staging or disease-name
   conclusions are to be guided out by the teacher), and never use the word "Diagnosis:".
Output JSON only (no extra text): {"finding_text": "the findings text"}"""


# ---------------------------------------------------------------- Step 2: learner profile

PERSONA_PROFILE_PROMPT = """你是医学教育培训师。根据题目知识点、难度与误解方向，为举手提问的规培生\
生成一句学习者画像（用于个性化教学与学生模拟）。要求：画像水平与学生提问口吻一致；不包含正确答案；\
中文 20~50 字。只输出 JSON（不要任何额外文字）：
{"learner_profile": "眼科规培N年级，已掌握…，混淆…，目标是…"}"""

#: English: one-sentence learner profile (same constraints: no correct answer, tone matches question).
PERSONA_PROFILE_PROMPT_EN = """You are a medical-education trainer. Based on the knowledge point,
difficulty, and misconception direction of the question, generate a one-sentence learner profile
in English for the resident who raises the question (used for personalized teaching and student
simulation). Requirements: the profile level must match the student's questioning tone; it must
not contain the correct answer; 10~25 words in English. Output JSON only (no extra text):
{"learner_profile": "Ophthalmology resident in year N, has mastered ..., confuses ..., aims to ..."}"""

#: general-department profile prompt (general=True, CMExam 4k general rows): department follows the
#: question (never pretend to be ophthalmology)
PERSONA_PROFILE_PROMPT_GENERAL = """你是医学教育培训师。根据题目知识点、难度与误解方向，为举手提问的规培生\
生成一句学习者画像（用于个性化教学与学生模拟）。要求：画像科室与题目科室一致（非眼科题不要写眼科规培）；\
画像水平与学生提问口吻一致；不包含正确答案；中文 20~50 字。只输出 JSON（不要任何额外文字）：
{"learner_profile": "临床规培N年级（轮转科室与题目一致），已掌握…，混淆…，目标是…"}"""

#: general-department profile prompt — English
PERSONA_PROFILE_PROMPT_EN_GENERAL = """You are a medical-education trainer. Based on the knowledge point,
difficulty, and misconception direction of the question, generate a one-sentence learner profile
in English for the resident who raises the question (used for personalized teaching and student
simulation). Requirements: the profile's department must match the question's department (do not
call non-ophthalmology items ophthalmology); the profile level must match the student's
questioning tone; it must not contain the correct answer; 10~25 words in English. Output JSON
only (no extra text):
{"learner_profile": "Clinical resident in year N (rotation matches the question), has mastered ..., confuses ..., aims to ..."}"""


# ---------------------------------------------------------------- Safe assembly helpers

#: lang -> Step 2 rewrite system prompt (rewrite_question picks per row lang);
#: _MCQ tables are the options-kept variants (mode='mcq', QA GRPO) — same JSON contract, only the
#: option constraint differs
REWRITE_PROMPTS = {"zh": REWRITE_SYSTEM_PROMPT, "en": REWRITE_SYSTEM_PROMPT_EN}
REWRITE_PROMPTS_MCQ = {"zh": REWRITE_SYSTEM_PROMPT_MCQ, "en": REWRITE_SYSTEM_PROMPT_MCQ_EN}
FINDING_DESC_PROMPTS = {"zh": FINDING_DESC_PROMPT, "en": FINDING_DESC_PROMPT_EN}
PERSONA_PROFILE_PROMPTS = {"zh": PERSONA_PROFILE_PROMPT, "en": PERSONA_PROFILE_PROMPT_EN}
PERSONA_PROFILE_PROMPTS_GENERAL = {"zh": PERSONA_PROFILE_PROMPT_GENERAL,
                                   "en": PERSONA_PROFILE_PROMPT_EN_GENERAL}


def rewrite_system_prompt(lang="zh", mode="open"):
    """Step 2 rewrite system prompt by lang; mode='mcq' -> options-kept variant."""
    table = REWRITE_PROMPTS_MCQ if mode == "mcq" else REWRITE_PROMPTS
    return table[_pick_lang(lang)]


def finding_desc_prompt(lang="zh"):
    """Step 1 findings-generation system prompt by lang."""
    return FINDING_DESC_PROMPTS[_pick_lang(lang)]


def persona_profile_prompt(lang="zh", general=False):
    """Step 2 learner-profile system prompt by lang; general=True -> general-department variant."""
    table = PERSONA_PROFILE_PROMPTS_GENERAL if general else PERSONA_PROFILE_PROMPTS
    return table[_pick_lang(lang)]


def format_options_block(options):
    """options dict -> "A. text\nB. text" (sorted by letter, empties skipped; non-dict -> "").

    Shared by mode='mcq' rewrite and the degrade template: options kept verbatim per line for QA GRPO
    grading.
    """
    if not isinstance(options, dict):
        return ""
    lines = []
    for letter in sorted(options.keys()):
        text = _s(options.get(letter))
        if text:
            lines.append("%s. %s" % (str(letter).upper()[:1], text))
    return "\n".join(lines)


def build_teacher_context(courseware_context=None, gt_label=None, gt_explanation=None,
                          misconception_seed=None, difficulty=None, lang="zh"):
    """Assemble the "case briefing" note for the teacher system prompt (design §6.3 Step 3).

    The briefing only anchors the final conclusion, appended after the system prompt (not into training
    messages, avoiding SFT/GRPO first-turn distribution mismatch). All args tolerate None/non-str;
    returns non-empty str. lang='en' -> English labels (the "Reference answer:" prefix is also the anchor
    llm_client._mock_chat uses to extract gt).
    """
    if _pick_lang(lang) == "en":
        lines = ["[Case briefing (for you to anchor the final conclusion only; "
                 "never state it before turn 5)]"]
        if isinstance(courseware_context, str) and courseware_context.strip():
            lines.append("Courseware key points / findings: " + courseware_context.strip())
        if isinstance(gt_label, str) and gt_label.strip():
            lines.append("Reference answer: " + gt_label.strip())
        if isinstance(gt_explanation, str) and gt_explanation.strip():
            lines.append("Explanation key points: " + gt_explanation.strip())
        if isinstance(misconception_seed, str) and misconception_seed.strip():
            lines.append("Possible student misconception: " + misconception_seed.strip())
        if isinstance(difficulty, str) and difficulty.strip():
            lines.append("Difficulty tier: " + difficulty.strip())
        return "\n".join(lines)
    lines = ["【本次答疑背景（仅供你把握最终结论，不得在第 5 轮之前直接给出）】"]
    if isinstance(courseware_context, str) and courseware_context.strip():
        lines.append("课件要点/检查所见：" + courseware_context.strip())
    if isinstance(gt_label, str) and gt_label.strip():
        lines.append("正确要点：" + gt_label.strip())
    if isinstance(gt_explanation, str) and gt_explanation.strip():
        lines.append("解析要点：" + gt_explanation.strip())
    if isinstance(misconception_seed, str) and misconception_seed.strip():
        lines.append("学生可能的误解：" + misconception_seed.strip())
    if isinstance(difficulty, str) and difficulty.strip():
        lines.append("难度档：" + difficulty.strip())
    return "\n".join(lines)


def _s(v):
    """Safe str: None/non-str -> "" (uniform fallback for prompt assembly)."""
    if isinstance(v, str):
        return v.strip()
    return ""


def rewrite_user_prompt(row, lang="zh", mode="open"):
    """Step 2 rewrite user message body (source question + persona seed + misconception material/options).

    :param row:  raw_pool row dict (missing keys safe)
    :param lang: prompt language ('en' -> English labels; else zh, decoupled from row lang for direct calls)
    :param mode: 'open' (default, no option traces) / 'mcq' (keep options: option block + answer letter
                 (text); no wrong_options list — the options are already in the question)
    """
    row = row if isinstance(row, dict) else {}
    en = _pick_lang(lang) == "en"
    mcq = mode == "mcq"
    stem = _s(row.get("stem")) or _s(row.get("gt_label")) or ("(source question missing)" if en else "（原题缺失）")
    wrong = row.get("wrong_options")
    wrong_txt = ""
    if isinstance(wrong, (list, tuple)) and wrong:
        wrong_txt = ("; " if en else "；").join([_s(w) for w in wrong[:4] if _s(w)])

    # mcq: option block + "letter (text)" answer line; rows without options fall back to the open answer line
    opts_block = format_options_block(row.get("options")) if mcq else ""
    gt_letter = _s(row.get("gt_letter"))
    if mcq and opts_block and gt_letter:
        opts = row.get("options") if isinstance(row.get("options"), dict) else {}
        ans_line_en = "[Correct answer] %s (%s)" % (gt_letter, _s(opts.get(gt_letter[:1])) or "(missing)")
        ans_line_zh = "【正确答案】%s（%s）" % (gt_letter, _s(opts.get(gt_letter[:1])) or "（缺失）")
    else:
        ans_line_en = "[Correct answer] " + (_s(row.get("gt_label")) or "(not provided)")
        ans_line_zh = "【正确答案】" + (_s(row.get("gt_label")) or "（未提供）")

    if en:
        parts = [
            "[Source question] " + stem,
        ]
        if opts_block:
            parts.append("[Options (the rewritten question must keep them VERBATIM, one per line "
                         "'A. text')] \n" + opts_block)
        parts.append(ans_line_en)
        if _s(row.get("gt_explanation")):
            parts.append("[Explanation] " + _s(row.get("gt_explanation")))
        if wrong_txt and not opts_block:
            parts.append("[Common wrong options (only for deriving the misconception direction; "
                         "no option traces may appear in the rewrite)] " + wrong_txt)
        if _s(row.get("finding_text")):
            parts.append("[Image findings] " + _s(row.get("finding_text")))
        if row.get("image"):
            parts.append("[Note] This item has an image; student_question must start with <image>.")
        parts.append("[Difficulty reference] " + (_s(row.get("difficulty_hint")) or "mid"))
        parts.append("[Persona seed] persona_seed=%s (generate a profile consistent with this seed's "
                     "temperament; do not restate the seed itself)" % row.get("persona_seed", 0))
        parts.append("Output JSON as required by the system prompt.")
        return "\n".join(parts)
    parts = [
        "【原题】" + stem,
    ]
    if opts_block:
        parts.append("【选项（改写后的提问必须逐字保留，每行一个「A. 文字」）】\n" + opts_block)
    parts.append(ans_line_zh)
    if _s(row.get("gt_explanation")):
        parts.append("【题目解析】" + _s(row.get("gt_explanation")))
    if wrong_txt and not opts_block:
        parts.append("【常见错误选项（仅供推导学生误解方向，改写中不得出现选项痕迹）】" + wrong_txt)
    if _s(row.get("finding_text")):
        parts.append("【图像检查所见】" + _s(row.get("finding_text")))
    if row.get("image"):
        parts.append("【提示】本题配图，student_question 必须以 <image> 开头。")
    parts.append("【难度档参考】" + (_s(row.get("difficulty_hint")) or "mid"))
    parts.append("【画像种子】persona_seed=%s（请生成与该种子气质一致的画像，无需复述种子本身）"
                 % row.get("persona_seed", 0))
    parts.append("请按 system 要求输出 JSON。")
    return "\n".join(parts)


def finding_desc_user_prompt(row, lang="zh"):
    """Step 1 finding_text user message body (input = dataset label / lesion annotation layout)."""
    row = row if isinstance(row, dict) else {}
    en = _pick_lang(lang) == "en"
    extra = row.get("label_extra")
    if en:
        parts = ["[Image label] " + (_s(row.get("gt_label")) or "(no label)")]
        if isinstance(extra, dict) and extra:
            frags = ["%s=%s" % (k, extra.get(k)) for k in sorted(extra.keys())]
            parts.append("[Lesion annotation layout / extra annotations] " + "; ".join(frags))
        parts.append("[Image type] " + (_s(row.get("image_kind"))
                                        or ("OCT" if "OCT" in _s(row.get("curriculum_node")).upper()
                                            else "fundus photograph")))
        parts.append("Output JSON as required by the system prompt.")
        return "\n".join(parts)
    parts = ["【图像标签】" + (_s(row.get("gt_label")) or "（无标签）")]
    if isinstance(extra, dict) and extra:
        frags = ["%s=%s" % (k, extra.get(k)) for k in sorted(extra.keys())]
        parts.append("【病灶注发布局/附加标注】" + "；".join(frags))
    parts.append("【图像类型】" + (_s(row.get("image_kind")) or ("OCT" if "OCT" in _s(row.get("curriculum_node")).upper() else "眼底照片")))
    parts.append("请按 system 要求输出 JSON。")
    return "\n".join(parts)


def persona_user_prompt(row, lang="zh"):
    """Step 2 degrade path: user message body for a standalone learner_profile generation."""
    row = row if isinstance(row, dict) else {}
    en = _pick_lang(lang) == "en"
    if en:
        parts = [
            "[Knowledge point] " + (_s(row.get("stem")) or _s(row.get("gt_label")) or "(missing)"),
            "[Difficulty tier] " + (_s(row.get("difficulty_hint")) or "mid"),
            "[Misconception direction] " + (_s(row.get("misconception_seed"))
                                            or "(unknown; write from the weak-concept angle)"),
        ]
        parts.append("Output JSON as required by the system prompt.")
        return "\n".join(parts)
    parts = [
        "【题目知识点】" + (_s(row.get("stem")) or _s(row.get("gt_label")) or "（缺失）"),
        "【难度档】" + (_s(row.get("difficulty_hint")) or "mid"),
        "【误解方向】" + (_s(row.get("misconception_seed")) or "（未知，请从薄弱概念角度写）"),
    ]
    parts.append("请按 system 要求输出 JSON。")
    return "\n".join(parts)

# -*- coding: utf-8 -*-
"""Step 1: manifest -> data/raw_pool.jsonl (design §6.3 Step1).

Question-bank sources are filtered by --keyword ophthalmology keywords; image sources get reading-stem
templates and a finding_text "findings" description via FINDING_DESC_PROMPT (deterministic fallback).
difficulty_hint and curriculum_node follow keyword rule tables; qid is deterministically numbered per
source. Each row carries a `lang` column ('zh'|'en') that downstream steps pass through.
"""
import argparse
import os
import sys

# Ensure project root is on sys.path (same as prepare_datasets)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_pipeline import llm_client as L  # noqa: E402
from data_pipeline import prepare_datasets as P  # noqa: E402
from data_pipeline.prompts import (  # noqa: E402
    finding_desc_prompt,
    finding_desc_user_prompt,
)

__all__ = ["DEFAULT_OPH_KEYWORDS", "IMAGE_LANG_DEFAULT_SOURCES", "node_for_text",
           "node_for_image_source", "row_lang", "difficulty_hint_for",
           "image_stem_template", "fallback_finding_text", "extract_records", "main",
           "GENERAL_NODE"]

# ---------------------------------------------------------------- Ophthalmology keyword table (--keyword overridable)
#: bilingual (English sources like MedQA-en also need subset filtering; Chinese keywords never match
#: English stems and vice versa)
DEFAULT_OPH_KEYWORDS = (
    "眼,青光眼,白内障,视网膜,屈光,角膜,葡萄膜,视神经,眼压,瞳孔,晶状体,玻璃体,黄斑,"
    "房水,结膜,虹膜,巩膜,泪器,眼眶,眼睑,糖网,近视,远视,散光,视野,眼底,OCT,眼底照相,"
    "视乳头,视盘,视路,瞳仁,睑缘,泪囊,前房,玻璃体混浊,视网膜脱离,"
    "retina,retinopathy,retinal,glaucoma,cataract,cornea,corneal,pupil,macula,macular,"
    "fundus,optic disc,optic nerve,intraocular,myopia,hyperopia,astigmatism,visual field,"
    "timolol,papilledema,conjunctiva,iris,sclera,uveitis,optometrist,ophthalmolog"
)

DEFAULT_NODE = "ophthalmology/general/ophthalmology_qa"

#: generic medical node for non-ophthalmology rows (with --no-keyword): never pretends to be an
#: ophthalmology node; grammar G5 only applies to ophthalmology reading nodes, so this is auto-skipped
GENERAL_NODE = "general_medicine/qa"

#: pre-split default keyword list (used in no_keyword mode to route ophthalmology vs general; not
#: affected by --keyword override)
_DEFAULT_KW_LIST = tuple(k.strip() for k in DEFAULT_OPH_KEYWORDS.split(",") if k.strip())

#: curriculum-node keyword rules (order-sensitive: emergency/medication-safety first so broad words
#: can't steal; English keywords map 1:1 to Chinese for the same design §6.4 nodes)
NODE_RULES = (
    ("ophthalmology/emergency/acute_angle_closure",
     ("闭角", "急性发作", "眼压升高", "高眼压", "急性充血", "抢救", "缩瞳", "甘露醇", "房角关闭",
      "angle closure", "acute attack", "raised intraocular", "intraocular pressure rise",
      "miotic", "mannitol")),
    ("ophthalmology/glaucoma/medication_safety",
     ("噻吗洛尔", "毛果芸香碱", "缩瞳剂", "降眼压药", "前列腺素", "碳酸酐酶", "β受体阻滞",
      "滴眼液", "用药", "禁忌", "药物", "剂量",
      "timolol", "pilocarpine", "beta-blocker", "beta blocker", "prostaglandin",
      "carbonic anhydrase", "eye drops", "medication", "contraindication", "dosage")),
    ("ophthalmology/retina/DR_staging",
     ("糖尿病视网膜病变", "糖网", "NPDR", "PDR", "微血管瘤", "新生血管", "视网膜病变", "眼底出血",
      "diabetic retinopathy", "microaneurysm", "neovascular", "neovascularization",
      "vitreous hemorrhage")),
    ("ophthalmology/neuro_ophth/papilledema_vs_papillitis",
     ("视盘", "视乳头", "视神经", "视盘水肿", "视野缺损", "传入性瞳孔", "RAPD",
      "optic disc", "optic nerve", "papilledema", "papillitis", "visual field defect",
      "relative afferent pupillary", "RAPD")),
    ("ophthalmology/cataract/preop_assessment",
     ("白内障", "晶状体混浊", "人工晶体", "IOL", "超声乳化", "术前评估",
      "cataract", "lens opacity", "intraocular lens", "IOL", "phacoemulsification",
      "preoperative")),
    ("ophthalmology/refraction/astigmatism_axis",
     ("散光", "轴位", "验光", "交叉柱镜", "屈光", "近视", "远视", "检影",
      "astigmatism", "axis", "refraction", "cross cylinder", "myopia", "hyperopia",
      "retinoscopy", "spectacle")),
    ("ophthalmology/retina/OCT_reading",
     ("OCT", "黄斑水肿", "玻璃膜疣", "CNV", "视网膜厚度",
      "optical coherence tomography", "macular edema", "drusen",
      "choroidal neovascularization", "retinal thickness")),
    ("ophthalmology/retina/pathological_myopia",
     ("病理性近视", "高度近视", "近视性", "后巩膜葡萄肿",
      "pathological myopia", "high myopia", "myopic", "posterior staphyloma")),
    ("ophthalmology/basic/anatomy_physics",
     ("解剖", "房水循环", "生理", "屈光介质", "光学", "泪膜",
      "anatomy", "aqueous humor", "circulation", "physiology", "optics", "tear film")),
)

#: image source -> curriculum node (design §6.4; refuge maps to neuro_ophth to hit the G5 reading check)
IMAGE_SOURCE_NODE = {
    "idrid": "ophthalmology/retina/DR_staging",
    "octid": "ophthalmology/retina/OCT_reading",
    "kermany": "ophthalmology/retina/OCT_reading",
    "octdl": "ophthalmology/retina/OCT_reading",
    "palm": "ophthalmology/retina/pathological_myopia",
    "refuge": "ophthalmology/neuro_ophth/papilledema_vs_papillitis",
    "gamma": "ophthalmology/glaucoma/disc_evaluation",
}

#: ODIR label letter -> curriculum node
ODIR_NODE = {
    "D": "ophthalmology/retina/DR_staging",
    "G": "ophthalmology/glaucoma/disc_evaluation",
    "C": "ophthalmology/cataract/preop_assessment",
    "A": "ophthalmology/retina/AMD_reading",
    "H": "ophthalmology/retina/hypertensive_retinopathy",
    "M": "ophthalmology/retina/pathological_myopia",
    "N": "ophthalmology/fundus/normal_reading",
    "O": "ophthalmology/fundus/multilabel_reading",
}

#: image reading stem templates (per node; never leak the label, only task instruction) — Chinese
IMAGE_STEM_TEMPLATES = {
    "ophthalmology/retina/DR_staging":
        "请阅这张眼底照片：判断糖尿病视网膜病变的分期，并按「定性→定位→定量→分期」说明判读依据。",
    "ophthalmology/retina/OCT_reading":
        "请阅这张 OCT 图像：描述层间结构与异常所见，并给出可能的诊断方向与依据。",
    "ophthalmology/retina/pathological_myopia":
        "这张眼底照片可见哪些与病理性近视相关的改变？请描述所见并说明判读顺序。",
    "ophthalmology/neuro_ophth/papilledema_vs_papillitis":
        "请评估这张眼底照片的视盘形态：杯盘比与青光眼/视盘病变的判读要点是什么？",
    "ophthalmology/glaucoma/disc_evaluation":
        "请评估这张眼底照片的视盘与视杯形态：如何判读青光眼性改变？依据是什么？",
    "ophthalmology/cataract/preop_assessment":
        "这张眼底照片的屈光介质清晰度提示什么？白内障术前评估还需注意什么？",
    "ophthalmology/retina/AMD_reading":
        "请阅这张眼底照片：黄斑区所见提示什么？年龄相关性黄斑变性的判读要点是什么？",
    "ophthalmology/retina/hypertensive_retinopathy":
        "请阅这张眼底照片：视网膜血管所见提示什么？高血压视网膜病变如何分级判读？",
    "ophthalmology/fundus/normal_reading":
        "请阅这张眼底照片：按视盘/血管/黄斑/周边顺序描述所见，并判断是否正常。",
    "ophthalmology/fundus/multilabel_reading":
        "请阅这张眼底照片：描述异常所见，并给出可能的诊断方向与鉴别思路。",
}
DEFAULT_IMAGE_STEM = "这张眼底照片可见哪些异常？请描述所见并给出鉴别思路。"

#: image reading stem templates — English (English image sources -> English trajectories; same no-label rule)
IMAGE_STEM_TEMPLATES_EN = {
    "ophthalmology/retina/DR_staging":
        "Review this fundus photograph: what is the DR severity stage, and what findings support it? "
        "Walk through your reading in the order qualitative, localization, quantification, staging.",
    "ophthalmology/retina/OCT_reading":
        "Review this OCT scan: describe the retinal layers and any abnormalities, then give your "
        "possible diagnostic direction and the evidence for it.",
    "ophthalmology/retina/pathological_myopia":
        "Which changes related to pathological myopia can be seen in this fundus photograph? "
        "Describe the findings and explain your reading order.",
    "ophthalmology/neuro_ophth/papilledema_vs_papillitis":
        "Assess the optic disc in this fundus photograph: what are the key points for reading the "
        "cup-to-disc ratio and glaucomatous or disc pathology?",
    "ophthalmology/glaucoma/disc_evaluation":
        "Assess the optic disc and cup in this fundus photograph: how do you interpret glaucomatous "
        "change, and on what evidence?",
    "ophthalmology/cataract/preop_assessment":
        "What does the clarity of the optical media in this fundus photograph suggest? What else "
        "matters in the preoperative cataract assessment?",
    "ophthalmology/retina/AMD_reading":
        "Review this fundus photograph: what do the macular findings suggest, and what are the key "
        "points for reading age-related macular degeneration?",
    "ophthalmology/retina/hypertensive_retinopathy":
        "Review this fundus photograph: what do the retinal vessels suggest, and how do you grade "
        "hypertensive retinopathy?",
    "ophthalmology/fundus/normal_reading":
        "Review this fundus photograph: describe the findings in disc / vessels / macula / periphery "
        "order, and judge whether it is normal.",
    "ophthalmology/fundus/multilabel_reading":
        "Review this fundus photograph: describe any abnormalities and give your possible "
        "diagnostic direction and differential approach.",
}
DEFAULT_IMAGE_STEM_EN = ("What abnormalities can be seen in this fundus photograph? Describe the "
                         "findings and outline your differential approach.")

#: image-source lang lookup (--lang auto; same check picks the template language per node)
IMAGE_STEM_TEMPLATES_BY_LANG = {"zh": IMAGE_STEM_TEMPLATES, "en": IMAGE_STEM_TEMPLATES_EN}
DEFAULT_IMAGE_STEM_BY_LANG = {"zh": DEFAULT_IMAGE_STEM, "en": DEFAULT_IMAGE_STEM_EN}

#: English-origin image datasets (hardcoded 'en' under --lang auto; their gt_label is already
#: translated to Chinese by the adapter, so label-based detect_lang would misjudge zh)
IMAGE_LANG_DEFAULT_SOURCES = frozenset(
    {"idrid", "odir", "refuge", "palm", "gamma", "octid", "kermany", "octdl"})

# difficulty keywords (difficulty_hint; English keywords map 1:1 to Chinese)
_HIGH_KEYWORDS = ("急", "急性", "发作", "闭角", "抢救", "急救", "禁忌", "过敏", "中毒",
                  "不良反应", "噻吗洛尔", "阿托品", "毛果芸香碱", "散瞳", "24小时", "48小时",
                  "全身", "感染", "穿孔", "破裂",
                  "acute", "attack", "emergency", "contraindica", "contraindication",
                  "allerg", "toxic", "adverse", "timolol", "atropine", "pilocarpine",
                  "dilation", "infection", "perforation", "rupture")
_MID_KEYWORDS = ("手术", "术前", "计算", "鉴别", "分期", "剂量", "儿童", "孕妇",
                 "并发症", "机制", "顺序", "原则",
                 "surgery", "surgical", "preoperative", "calculat", "differential",
                 "staging", "dosage", "child", "pregnan", "complication", "mechanism",
                 "order", "principle")


# ---------------------------------------------------------------- Rule functions (unit-testable)
def node_for_text(text):
    """Text-item curriculum node: first NODE_RULES keyword hit in order; DEFAULT_NODE when none.

    Matching is case-insensitive (English keywords IOL/NPDR etc. share the table with Chinese ones).
    """
    t = (text if isinstance(text, str) else "").lower()
    for node, kws in NODE_RULES:
        for kw in kws:
            if kw and kw.lower() in t:
                return node
    return DEFAULT_NODE


def node_for_image_source(source, gt_label="", label_extra=None):
    """Image-item curriculum node: source-level mapping first (odir subdivides by label codes),
    otherwise keyword-based fallback into a reading node."""
    src = str(source or "").lower()
    if src == "odir":
        codes = label_extra.get("codes") if isinstance(label_extra, dict) else None
        if isinstance(codes, dict):
            hits = [ODIR_NODE[c] for c, on in codes.items() if on and c in ODIR_NODE]
            # N (normal) always sorted last (A5 fix: codes keep insertion order with N first, and the
            # old note claimed "N last" — the opposite): N+D patients should map to DR_staging etc.,
            # not be stolen by normal_reading
            hits.sort(key=lambda n: n == ODIR_NODE["N"])
            if hits:
                return hits[0]
        return ODIR_NODE["O"]
    if src in IMAGE_SOURCE_NODE:
        return IMAGE_SOURCE_NODE[src]
    node = node_for_text(gt_label or "")
    if node == DEFAULT_NODE:
        return "ophthalmology/fundus/multilabel_reading"   # reading fallback node (hits G5)
    return node


def difficulty_hint_for(question_type, stem="", gt_label="", wrong_options=None):
    """Difficulty heuristic: emergency/medication-safety -> high; differential/surgery/mechanism or
    has wrong options -> mid; else low. Keywords are bilingual and case-insensitive."""
    text = "%s %s" % (stem if isinstance(stem, str) else "", gt_label if isinstance(gt_label, str) else "")
    if question_type == "image":
        low_labels = ("正常", "无糖尿病视网膜病变", "normal", "no dr")
        lbl = (gt_label or "").lower()
        return "mid" if any(k in lbl for k in low_labels) else "high"
    low_text = text.lower()
    for kw in _HIGH_KEYWORDS:
        if kw and kw in low_text:
            return "high"
    if isinstance(wrong_options, (list, tuple)) and wrong_options:
        return "mid"
    for kw in _MID_KEYWORDS:
        if kw and kw in low_text:
            return "mid"
    return "low"


def _norm_lang(lang):
    """Normalize lang: only 'en' stays 'en'; 'zh'/None/garbage -> 'zh' (project default)."""
    return "en" if isinstance(lang, str) and lang.strip().lower() == "en" else "zh"


def row_lang(question_type, source, stem="", gt_label="", lang="auto"):
    """Per-row lang decision (core of --lang auto).

    * explicit 'zh'/'en' -> forced (overrides all detection);
    * text source -> detect_lang(stem);
    * image source: idrid/odir/refuge/palm/gamma/octid/kermany/octdl are English-origin -> 'en';
      others (generic_images etc.) -> detect_lang(gt_label).
    """
    forced = str(lang or "").strip().lower()
    if forced in ("zh", "en"):
        return forced
    if question_type == "text":
        return L.detect_lang(stem)
    src = str(source or "").lower()
    if src in IMAGE_LANG_DEFAULT_SOURCES:
        return "en"
    return L.detect_lang(gt_label)


def image_stem_template(curriculum_node, lang="zh"):
    """image stem template (per node and lang; unknown node -> generic reading template)."""
    table = IMAGE_STEM_TEMPLATES_BY_LANG[_norm_lang(lang)]
    default = DEFAULT_IMAGE_STEM_BY_LANG[_norm_lang(lang)]
    return table.get(curriculum_node, default)


def fallback_finding_text(gt_label, label_extra=None, lang="zh"):
    """Deterministic finding_text fallback (LLM unavailable / parse failure; offline mock path).

    A6 de-conclusion: no longer states "this image shows gt_label" (finding_text feeds
    courseware_context -> student observation, i.e. feeding the answer); gives only annotation-level
    info (grade code / risk level / eye) + reading-order instruction; the conclusion stays on the
    teacher-only channel build_teacher_context. lang='en' -> English version.
    """
    del gt_label  # conclusion label never enters student-side text (param kept to stabilize signature)
    extra = label_extra if isinstance(label_extra, dict) else {}
    if _norm_lang(lang) == "en":
        frags = []
        if str(extra.get("grade_code", "")).strip():
            frags.append("DR grade code %s" % extra.get("grade_code"))
        if str(extra.get("dme_risk", "")).strip():
            frags.append("DME risk level %s" % extra.get("dme_risk"))
        if extra.get("side"):
            frags.append("%s eye" % str(extra.get("side")))
        ann = (" (dataset annotations: %s)" % "; ".join(frags)) if frags \
            else " (no structured annotations from the dataset)"
        return ("Findings (offline template): this image comes from a public fundus dataset%s. "
                "Student, please first describe what you see in disc -> vessels -> macula order; "
                "the teacher then guides qualitative -> localization -> quantification -> staging. "
                "The conclusion is reached through the teacher's step-by-step guidance." % ann)
    frags = []
    if str(extra.get("grade_code", "")).strip():
        frags.append("DR 分级码 %s" % extra.get("grade_code"))
    if str(extra.get("dme_risk", "")).strip():
        frags.append("DME 风险等级 %s" % extra.get("dme_risk"))
    if extra.get("side"):
        frags.append("%s眼" % {"left": "左", "right": "右"}.get(extra.get("side"), extra.get("side")))
    ann = ("（数据集标注：%s）" % "；".join(frags)) if frags else "（数据集未附结构化标注）"
    return "检查所见（离线模板）：本图来自公开眼底数据集%s，学生请先按视盘→血管→黄斑顺序描述" \
           "所见，教师再按定性→定位→定量→分期引导判读。结论以带教老师逐步引导得出。" % ann


# ---------------------------------------------------------------- Core extraction
def _keyword_hit(stem, keywords):
    """stem keyword hit on any ophthalmology keyword (no normalization beyond lowercasing)."""
    if not isinstance(stem, str):
        return False
    low = stem.lower()
    for kw in keywords:
        k = kw.strip().lower()
        if k and k in low:
            return True
    return False


def extract_records(manifest_rows, keywords=None, max_per_source=0, mock=False,
                    use_cache=True, limit=0, concurrency=None, seed=0, lang="auto",
                    no_keyword=False):
    """manifest rows -> raw_pool rows (core logic, decoupled from CLI for testing).

    finding_text goes through chat_many (one concurrent batch), falling back to the deterministic
    template per row. lang: 'auto' (per-row detection, default) / 'zh' / 'en' (forced). no_keyword=True
    disables text-source keyword filtering (all rows pass; node routing falls back to GENERAL_NODE for
    rows missing the default keyword table, so generic words like "medication" don't mislabel rows).
    """
    keywords = keywords if keywords else [k.strip() for k in DEFAULT_OPH_KEYWORDS.split(",") if k.strip()]
    max_per_source = int(max_per_source or 0)
    limit = int(limit or 0)
    rows, finding_jobs = [], []

    for entry in manifest_rows:
        if not isinstance(entry, dict):
            continue
        mtype = str(entry.get("type") or entry.get("source") or "").lower()
        src = str(entry.get("source") or mtype)
        records = P.load_records(mtype, entry.get("path") or None,
                                 entry.get("image_dir") or None,
                                 entry.get("labels_path") or None,
                                 entry.get("col_map") if isinstance(entry.get("col_map"), dict) else {},
                                 entry.get("eyes") or "both", src)
        kept = 0
        for rec in records:
            if max_per_source and kept >= max_per_source:
                break
            if limit and len(rows) >= limit:
                break
            qtype = rec.get("question_type")
            if qtype == "text":
                stem = rec.get("stem") or ""
                if no_keyword:
                    node = node_for_text(stem) if _keyword_hit(stem, _DEFAULT_KW_LIST) else GENERAL_NODE
                else:
                    if not _keyword_hit(stem, keywords):
                        continue          # non-ophthalmology subset: skip whole row
                    node = node_for_text(stem)
                opts = rec.get("options") if isinstance(rec.get("options"), dict) else {}
                ans_letter = str(rec.get("answer_letter") or "").upper()
                wrong = [opts[l] for l in sorted(opts.keys()) if l not in ans_letter]
                image, label_extra, finding = None, {}, ""
                gt_letter = ans_letter
                gt_label = rec.get("answer") or rec.get("gt_label") or ""
                row_lang_v = row_lang("text", src, stem=stem, gt_label=str(gt_label or ""), lang=lang)
            else:                      # image
                stem = ""
                node = node_for_image_source(src, rec.get("gt_label") or "",
                                             rec.get("label_extra"))
                gt_label = rec.get("gt_label") or rec.get("answer") or ""
                row_lang_v = row_lang("image", src, gt_label=str(gt_label or ""), lang=lang)
                stem = image_stem_template(node, row_lang_v)
                wrong = []
                opts = {}
                gt_letter = ""          # image sources have no option letter (gt_label is label text)
                image = rec.get("image") or None
                label_extra = rec.get("label_extra") if isinstance(rec.get("label_extra"), dict) else {}
                finding = ""           # batch-generated later
            rows.append({
                "qid": "",            # assigned after per-source ordering
                "source": src,
                "question_type": "text" if qtype == "text" else "image",
                "image": image,
                "gt_label": str(gt_label or ""),
                "gt_letter": str(gt_letter or ""),
                "gt_explanation": str(rec.get("explanation") or ""),
                "options": dict(opts) if isinstance(opts, dict) else {},
                "wrong_options": wrong,
                "curriculum_node": node,
                "difficulty_hint": difficulty_hint_for(
                    "text" if qtype == "text" else "image", stem, gt_label, wrong),
                "finding_text": finding,
                "stem": stem,
                "label_extra": label_extra,
                "lang": row_lang_v,
            })
            kept += 1
            if qtype != "text":
                finding_jobs.append(len(rows) - 1)
        if limit and len(rows) >= limit:
            break

    # ---- qid deterministic per-source ordering
    counters = {}
    for r in rows:
        counters[r["source"]] = counters.get(r["source"], 0) + 1
        r["qid"] = "%s-%06d" % (r["source"], counters[r["source"]])

    # ---- batch-generate finding_text (image items; LLM failure -> fallback; prompt lang per row)
    if finding_jobs:
        batch = []
        for idx in finding_jobs:
            r = rows[idx]
            batch.append([{"role": "system", "content": finding_desc_prompt(r.get("lang"))},
                          {"role": "user", "content": finding_desc_user_prompt(
                              {"gt_label": r["gt_label"], "label_extra": r["label_extra"],
                               "curriculum_node": r["curriculum_node"]}, lang=r.get("lang"))}])
        outs = L.chat_many(batch, seeds=[seed + i for i in range(len(batch))], json_mode=True,
                           mock=mock, use_cache=use_cache, concurrency=concurrency)
        for i, idx in enumerate(finding_jobs):
            r = rows[idx]
            obj = L.parse_json_dict(outs[i] if i < len(outs) else "")
            text = obj.get("finding_text") if isinstance(obj, dict) else None
            if isinstance(text, str) and text.strip() and not (isinstance(obj, dict) and obj.get("mock")):
                r["finding_text"] = text.strip()
            else:
                r["finding_text"] = fallback_finding_text(r["gt_label"], r.get("label_extra"),
                                                          lang=r.get("lang"))
    return rows


# ---------------------------------------------------------------- CLI
def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.extract_questions",
        description="Step 1：manifest → data/raw_pool.jsonl（眼科子集抽取 + 阅片题干模板 + finding_text）")
    p.add_argument("--input", default="data/manifest.jsonl", help="manifest 路径（prepare_datasets 产出）")
    p.add_argument("--output", default="data/raw_pool.jsonl", help="raw_pool 输出路径")
    p.add_argument("--keyword", default=DEFAULT_OPH_KEYWORDS,
                   help="眼科子集关键词（逗号分隔，覆写默认表）")
    p.add_argument("--max-per-source", type=int, default=0,
                   help="每个源最多抽取条数（0=不限）")
    p.add_argument("--limit", type=int, default=0, help="总条数上限（0=不限）")
    p.add_argument("--mock", action="store_true", help="强制离线 mock（默认：key 未设时自动）")
    p.add_argument("--no-cache", action="store_true", help="关闭 LLM 缓存")
    p.add_argument("--seed", type=int, default=0, help="finding_text 生成的基准 seed")
    p.add_argument("--concurrency", type=int, default=None, help="覆写并发上限")
    p.add_argument("--lang", choices=["auto", "zh", "en"], default="auto",
                   help="轨迹语种（默认 auto：text 源逐行检测、英文图像源默认 en；"
                        "显式 zh/en 则强制覆写全部行）")
    p.add_argument("--no-keyword", action="store_true",
                   help="关闭 text 源眼科关键词过滤（放行全部行；非眼科行 "
                        "curriculum_node=general_medicine/qa）")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    manifest = L.read_jsonl(args.input)
    if not manifest:
        sys.stderr.write("[extract] manifest 为空：%s（先运行 prepare_datasets）\n" % args.input)
        return 2
    keywords = [k.strip() for k in str(args.keyword).split(",") if k.strip()]
    rows = extract_records(manifest, keywords=keywords,
                           max_per_source=args.max_per_source, mock=args.mock,
                           use_cache=not args.no_cache, limit=args.limit,
                           concurrency=args.concurrency, seed=args.seed, lang=args.lang,
                           no_keyword=args.no_keyword)
    L.write_jsonl(args.output, rows)
    n_text = sum(1 for r in rows if r["question_type"] == "text")
    n_img = len(rows) - n_text
    nodes, langs = {}, {}
    for r in rows:
        nodes[r["curriculum_node"]] = nodes.get(r["curriculum_node"], 0) + 1
        lg = str(r.get("lang") or "zh")
        langs[lg] = langs.get(lg, 0) + 1
    print("[extract] raw_pool=%s total=%d text=%d image=%d lang=%s"
          % (args.output, len(rows), n_text, n_img,
             " ".join("%s:%d" % kv for kv in sorted(langs.items()))))
    for node in sorted(nodes, key=lambda k: -nodes[k]):
        print("    %-58s %d" % (node, nodes[node]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

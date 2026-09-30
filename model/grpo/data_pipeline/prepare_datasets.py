# -*- coding: utf-8 -*-
"""Step 0: register raw datasets and normalize formats (design §6.2). No real download happens here.

Validates the local raw dir/file structure, then writes data/manifest.jsonl (one entry per source with
n_items/col_map/warnings); built-in per-source adapters (load_records) parse cmexam/cmb/pubmedqa/idrid/
odir/generic_images/generic_jsonl defensively. Column names can be overridden with --col-map k=v.
"""
import argparse
import csv
import json
import os
import re
import sys
import time

# Support both `python -m data_pipeline.prepare_datasets` and `python data_pipeline/prepare_datasets.py`:
# ensure project root is on sys.path (data_pipeline is a namespace package)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_pipeline import llm_client as L  # noqa: E402

__all__ = ["SOURCE_INFO", "IDRID_DR_MAP", "ODIR_LABEL_CODES", "parse_options",
           "norm_answer_letters", "load_records", "main",
           "PUBMEDQA_OPTIONS", "PUBMEDQA_DECISION_LETTER", "PUBMEDQA_CONTEXT_CAP",
           "load_pubmedqa"]

# ---------------------------------------------------------------- Source registry (design §6.2)
SOURCE_INFO = {
    "cmexam": {"license": "研究用途，遵循 github.com/williamliujl/CMExam 仓库 LICENSE",
               "desc": "60K+ 中国国家医师资格考试题（题干/选项/答案/解析）"},
    "cmb": {"license": "研究用途，遵循 github.com/FreedomIntelligence/CMB 仓库 LICENSE",
            "desc": "CMB-Exam 11,200 道中文医学选择题（含眼科学）"},
    "pubmedqa": {"license": "MIT（huggingface.co/datasets/qiaojin/PubMedQA）",
                 "desc": "PubMedQA pqa_labeled 1k 英文生物医学研究问答（yes/no/maybe + 长答案）"},
    "medqa": {"license": "研究用途，遵循 github.com/jind11/MedQA 仓库 LICENSE",
              "desc": "MedQA-MCMLE 国内医师资格考试选择题"},
    "idrid": {"license": "CC BY 4.0（idrid.grand-challenge.org）",
              "desc": "516 张眼底照，DR 5 级分级 + DME 风险 + 像素级病灶标注"},
    "odir": {"license": "Kaggle andrewmvd/ocular-disease-recognition-odir5k（研究用途）",
             "desc": "5,000 患者 × 左右眼眼底照 + 8 类诊断标签"},
    "refuge": {"license": "CC 系（注意非商用条款；refuge.grand-challenge.org + figshare）",
               "desc": "1,200 张眼底照，青光眼标签 + 视盘/视杯分割"},
    "palm": {"license": "开放（figshare / Springer Nature Scientific Data；palm.grand-challenge.org）",
             "desc": "1,200 张病理性近视眼底照 + 病变标注"},
    "gamma": {"license": "CC BY（gamma.grand-challenge.org）",
              "desc": "300 例 fundus + 3D OCT 配对，青光眼分级"},
    "octid": {"license": "开放（Kaggle octid-dataset / Borealis Dataverse）",
              "desc": "500+ 张 OCT，5 类（normal/drusen/CNV/DME/ARMD）"},
    "kermany": {"license": "研究用途（Kaggle / Mendeley，84,495 张 OCT 4 类）",
                "desc": "OCT 阅片扩量池（抽样即可）"},
    "octdl": {"license": "开放（Nature Scientific Data 2024）",
              "desc": "2,000+ 张 OCT 多病种组标注"},
    "tianchi_dialog": {"license": "阿里天池 dataset/90163（研究用途，仅作口语化风格参考）",
                       "desc": "79.2 万条医患问答（非眼科专用，不作医学知识源）"},
    "generic_jsonl": {"license": "自备数据（--license 覆写）", "desc": "兜底 jsonl 格式"},
    "generic_images": {"license": "自备数据（--license 覆写）", "desc": "图片目录 + 标签文件"},
}

#: IDRiD DR grade code -> Chinese label (gt_label; shared with extract/quality_filter keyword checks)
IDRID_DR_MAP = {
    "0": "无糖尿病视网膜病变(No DR)",
    "1": "轻度非增殖期糖尿病视网膜病变(Mild NPDR)",
    "2": "中度非增殖期糖尿病视网膜病变(Moderate NPDR)",
    "3": "重度非增殖期糖尿病视网膜病变(Severe NPDR)",
    "4": "增殖期糖尿病视网膜病变(Proliferative DR, PDR)",
}

#: ODIR-5K eight-label letter -> meaning (curriculum_node mapping lives in extract_questions)
ODIR_LABEL_CODES = {"N": "normal", "D": "diabetes", "G": "glaucoma", "C": "cataract",
                    "A": "amd", "H": "hypertension", "M": "myopia", "O": "other"}

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


# ---------------------------------------------------------------- Low-level readers
def _read_csv_rows(path):
    """csv -> list[dict] (DictReader). Tries utf-8-sig / utf-8 / gbk in order; returns [] on failure."""
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        sys.stderr.write("[prepare] csv 不存在：%r\n" % (path,))
        return []
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return [dict(r) for r in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
        except Exception as e:
            sys.stderr.write("[prepare] 解析 csv 失败 %s：%s\n" % (path, e))
            return []
    sys.stderr.write("[prepare] csv 编码无法识别：%s\n" % path)
    return []


def _read_json_any(path):
    """json / jsonl -> list[dict] (jsonl per line, whole json as list); returns [] on failure."""
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        sys.stderr.write("[prepare] json 不存在：%r\n" % (path,))
        return []
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            sys.stderr.write("[prepare] 读 json 失败 %s：%s\n" % (path, e))
            return []
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return [o for o in obj if isinstance(o, dict)]
            if isinstance(obj, dict):
                for v in obj.values():          # wrapper like {"data": [...]}
                    if isinstance(v, list):
                        return [o for o in v if isinstance(o, dict)]
                return [obj]
        except Exception:
            rows = []
            # Split on "\n" rather than splitlines(): json.dumps doesn't escape U+2028/U+2029 (legal
            # JSON chars) while splitlines() does split on them — one record would be split in two
            # (observed pitfall with PubMedQA)
            for ln in text.split("\n"):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    o = json.loads(ln)
                    if isinstance(o, dict):
                        rows.append(o)
                except Exception:
                    continue
            if rows:
                return rows
    sys.stderr.write("[prepare] json 解析失败：%s\n" % path)
    return []


def norm_answer_letters(raw):
    """answer column -> canonical letter string: keep only A-J, uppercase, dedup, ascending.

    CMExam has multi-answer items (e.g. 'ACE'); the old `[:1]` truncation silently dropped answers.
    """
    s = str(raw or "").upper()
    return "".join(sorted({ch for ch in s if "A" <= ch <= "J"}))


def parse_options(raw):
    """option text -> {"A": "...", "B": "..."}. Tolerates multiline / single-line-run / dict / list."""
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, (str, int, float)):
                out[k.strip().upper()[:1] or "?"] = str(v).strip()
        return out
    if isinstance(raw, (list, tuple)):
        out = {}
        for i, v in enumerate(raw):
            if v is None:
                continue
            letter = chr(ord("A") + i) if i < 26 else str(i)
            s = str(v).strip()
            m = re.match(r"^([A-Ja-j])\s*[.、:：)）]?\s*(.+)$", s)
            if m:
                out[m.group(1).upper()] = m.group(2).strip()
            else:
                out[letter] = s
        return out
    if not isinstance(raw, str):
        return {}
    opts = {}
    for ln in raw.splitlines():  # multiline: A.xxx\nB.yyy (CMExam is "A <space> text", no punct)
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"^([A-Ja-j])(?:\s*[.、:：)）]\s*|\s+)(.+)$", ln)
        if m:
            opts[m.group(1).upper()] = m.group(2).strip()
    if not opts:                 # single-line run: A.xxx B.yyy
        found = re.findall(r"([A-Ja-j])\s*[.、:：)）]\s*([^A-Ja-j]*)", raw)
        if len(found) >= 2:
            for letter, body in found:
                body = body.strip()
                if body:
                    opts[letter.upper()] = body
    return opts


def _pick(row, col_map, key, default):
    """Get a value by column map (col_map can override default column names); tolerates None/empty."""
    col = col_map.get(key, default)
    v = row.get(col)
    if v is None or (isinstance(v, str) and not v.strip()):
        return ""
    return str(v).strip()


def _join_image(image_dir, name):
    """Join image path: return name unchanged when image_dir is absent (extract logs a warning)."""
    name = str(name).strip()
    if image_dir:
        return os.path.join(str(image_dir), name)
    return name


# ---------------------------------------------------------------- Per-source adapters (unified record schema)
def load_cmexam(path, col_map=None):
    """CMExam csv -> text record (stem/options/answer/explanation)."""
    cm = col_map or {}
    records = []
    for row in _read_csv_rows(path):
        stem = _pick(row, cm, "question", "Question")
        if not stem:
            continue
        options = parse_options(row.get(cm.get("options", "Options")))
        ans_letter = norm_answer_letters(_pick(row, cm, "answer", "Answer"))
        answer = "; ".join([options[l] for l in ans_letter if l in options]) \
            or _pick(row, cm, "answer", "Answer")
        explanation = _pick(row, cm, "explanation", "Explanation")
        records.append({"source": "cmexam", "question_type": "text", "stem": stem,
                        "options": options, "answer_letter": ans_letter, "answer": answer,
                        "explanation": explanation, "image": None, "gt_label": answer,
                        "label_extra": {}, "meta": {}})
    return records


def load_cmb(path, col_map=None):
    """CMB-Exam json -> text record (question/option/answer/detail)."""
    cm = col_map or {}
    records = []
    for item in _read_json_any(path):
        stem = _pick(item, cm, "question", "question")
        if not stem:
            continue
        options = parse_options(item.get(cm.get("options", "option")))
        ans_letter = norm_answer_letters(_pick(item, cm, "answer", "answer"))
        answer = "; ".join([options[l] for l in ans_letter if l in options]) \
            or _pick(item, cm, "answer", "answer")
        explanation = _pick(item, cm, "explanation", "detail")
        records.append({"source": "cmb", "question_type": "text", "stem": stem,
                        "options": options, "answer_letter": ans_letter, "answer": answer,
                        "explanation": explanation, "image": None, "gt_label": answer,
                        "label_extra": {}, "meta": {
                            "exam_subject": _pick(item, cm, "subject", "exam_subject")}})
    return records


# ---------------------------------------------------------------- PubMedQA (yes/no/maybe -> A/B/C single-choice)
#: three-class decision -> option letter (gt_letter semantics match CMExam MCQ source; downstream zero-change)
PUBMEDQA_OPTIONS = {"A": "Yes", "B": "No", "C": "Maybe"}
PUBMEDQA_DECISION_LETTER = {"yes": "A", "no": "B", "maybe": "C"}
#: whole-paragraph cumulative truncation of abstract (typical 1.3k-1.8k chars, max ~2.8k)
PUBMEDQA_CONTEXT_CAP = 2400
#: long_answer truncation (typical 60-150 words; guards only rare multi-paragraph answers)
PUBMEDQA_EXPLANATION_CAP = 1200


def _pubmedqa_context(context_text, cap=PUBMEDQA_CONTEXT_CAP):
    """Cumulatively accumulate context_text paragraphs up to cap, append " [...]" when truncated.

    Keeps section labels (BACKGROUND/METHODS/...) for the rewrite model, and uppercase ASCII won't
    match the English option-trace regex anchored at line starts.
    """
    text = context_text if isinstance(context_text, str) else ""
    if len(text) <= cap:
        return text
    kept, used = [], 0
    for seg in text.split("\n\n"):
        if used + len(seg) + 2 > cap and kept:
            break
        kept.append(seg)
        used += len(seg) + 2
    out = "\n\n".join(kept)
    if len(out) > cap:                       # first paragraph alone over cap: hard-cut once
        out = out[:cap].rstrip()
    return out + " [...]"


def load_pubmedqa(path, col_map=None):
    """PubMedQA pqa_labeled jsonl (convert_pubmedqa output) -> text record.

    Fields question/context_text/long_answer/final_decision (--col-map overrides). Yes/No/Maybe ->
    A/B/C; stem = question + blank line + abstract (section labels kept, truncated per-paragraph).
    Filtering (per-row stderr warning; dropped count = 1000 - manifest n_items is auditable): invalid
    final_decision / empty question / empty long_answer / duplicate question (first kept).
    """
    cm = col_map or {}
    records, seen = [], set()
    for item in _read_json_any(path):
        pubid = item.get("pubid")
        question = _pick(item, cm, "question", "question")
        decision = _pick(item, cm, "decision", "final_decision").lower()
        letter = PUBMEDQA_DECISION_LETTER.get(decision)
        long_answer = _pick(item, cm, "explanation", "long_answer")
        if not letter:
            sys.stderr.write("[prepare] pubmedqa 跳过非法行（pubid=%s decision=%r）\n"
                             % (pubid, decision[:20]))
            continue
        if not question:
            sys.stderr.write("[prepare] pubmedqa 跳过空 question（pubid=%s）\n" % pubid)
            continue
        if not long_answer:
            sys.stderr.write("[prepare] pubmedqa 跳过空 long_answer（pubid=%s）\n" % pubid)
            continue
        if question in seen:
            sys.stderr.write("[prepare] pubmedqa 跳过重复 question（pubid=%s）\n" % pubid)
            continue
        seen.add(question)
        context = _pick(item, cm, "context", "context_text")
        context = _pubmedqa_context(context) if context else ""
        stem = question + ("\n\n" + context if context else "")
        records.append({"source": "pubmedqa", "question_type": "text", "stem": stem,
                        "options": dict(PUBMEDQA_OPTIONS), "answer_letter": letter,
                        "answer": PUBMEDQA_OPTIONS[letter],
                        "explanation": long_answer[:PUBMEDQA_EXPLANATION_CAP],
                        "image": None, "gt_label": PUBMEDQA_OPTIONS[letter],
                        "label_extra": {"pubid": pubid, "final_decision": decision},
                        "meta": {}})
    return records


def load_generic_jsonl(path, col_map=None):
    """Fallback jsonl: {"question","answer","explanation","options"} (column names overridable)."""
    cm = col_map or {}
    records = []
    for item in _read_json_any(path):
        stem = _pick(item, cm, "question", "question")
        if not stem:
            continue
        options = parse_options(item.get(cm.get("options", "options")))
        answer = _pick(item, cm, "answer", "answer")
        ans_letter = ""
        for k, v in options.items():
            if v == answer:
                ans_letter = k
                break
        records.append({"source": "generic_jsonl", "question_type": "text", "stem": stem,
                        "options": options, "answer_letter": ans_letter, "answer": answer,
                        "explanation": _pick(item, cm, "explanation", "explanation"),
                        "image": None, "gt_label": answer, "label_extra": {}, "meta": {}})
    return records


def load_idrid(path, image_dir=None, col_map=None):
    """IDRiD grade csv -> image record (DR grade + DME risk; lesion layout lives in *.csv annotations,
    supplementable later via generic_images; this adapter registers the grade label first)."""
    cm = col_map or {}
    records = []
    for row in _read_csv_rows(path):
        pid = _pick(row, cm, "id", "Image Name")
        if not pid:
            continue
        grade = _pick(row, cm, "grade", "Retinopathy grade")
        dme = _pick(row, cm, "dme", "Risk of macular edema")
        label = IDRID_DR_MAP.get(grade, ("糖尿病视网膜病变分级 %s" % grade) if grade else "")
        if not label:
            continue
        records.append({"source": "idrid", "question_type": "image", "stem": "",
                        "options": {}, "answer_letter": "", "answer": label,
                        "explanation": "", "image": _join_image(image_dir, pid + ".jpg"),
                        "gt_label": label,
                        "label_extra": {"grade_code": grade, "dme_risk": dme},
                        "meta": {"image_kind": "fundus"}})
    return records


def _odir_flag(row, col):
    v = row.get(col)
    s = str(v).strip().lower() if v is not None else ""
    return s in ("1", "1.0", "true", "yes", "y", "是")


def load_odir(path, image_dir=None, col_map=None, eyes="both"):
    """ODIR-5K csv -> image record (one per patient eye; label = diagnostic keywords + label codes)."""
    cm = col_map or {}
    eyes = (eyes or "both").lower()
    if eyes not in ("left", "right", "both"):
        eyes = "both"
    records = []
    for row in _read_csv_rows(path):
        pid = _pick(row, cm, "id", "ID")
        codes = {}
        for letter in ODIR_LABEL_CODES:
            codes[letter] = _odir_flag(row, cm.get("code_" + letter, letter))
        for side in (("left",) if eyes == "left" else ("right",) if eyes == "right"
                     else ("left", "right")):
            fname = _pick(row, cm, side + "_img", {"left": "Left-Fundus",
                                                   "right": "Right-Fundus"}[side])
            kw = _pick(row, cm, side + "_kw", {"left": "Left-Diagnostic Keywords",
                                               "right": "Right-Diagnostic Keywords"}[side])
            if not fname and not kw:
                continue
            label = kw if kw else "/".join([ODIR_LABEL_CODES[c] for c, on in codes.items() if on])
            if not label:
                label = "normal"
            records.append({
                "source": "odir", "question_type": "image", "stem": "",
                "options": {}, "answer_letter": "", "answer": label, "explanation": "",
                "image": _join_image(image_dir, fname),
                "gt_label": label,
                "label_extra": {"codes": codes, "age": _pick(row, cm, "age", "Patient Age"),
                                "sex": _pick(row, cm, "sex", "Patient Sex"), "side": side},
                "meta": {"image_kind": "fundus", "pid": pid}})
    return records


def _list_images(image_dir):
    """Recursively list images (paths relative to image_dir); returns [] on failure."""
    out = []
    if not image_dir or not isinstance(image_dir, str) or not os.path.isdir(image_dir):
        return out
    for root, _dirs, files in os.walk(image_dir):
        for fn in sorted(files):
            if fn.lower().endswith(_IMAGE_EXTS):
                out.append(os.path.relpath(os.path.join(root, fn), image_dir))
    return sorted(out)


def load_generic_images(image_dir=None, labels_path=None, col_map=None, source="generic_images"):
    """image dir + labels -> image record.

    labels file (label.txt/csv, encoding-adaptive): one "relative_path,label[,extra]" per line; when
    labels are absent, first-level subdir name is used as the class label (REFUGE/PALM/OCTID layout).
    """
    del col_map  # columns fixed here; keep the param to unify the signature
    records = []
    labels = {}
    if labels_path and os.path.isfile(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8-sig") as f:
                for ln in f:
                    parts = [p.strip() for p in ln.strip().split(",") if p.strip()]
                    if len(parts) >= 2:
                        labels[parts[0]] = parts[1]
        except Exception as e:
            sys.stderr.write("[prepare] 读标签文件失败 %s：%s\n" % (labels_path, e))
    images = _list_images(image_dir)
    if not images:
        sys.stderr.write("[prepare] 图片目录为空或不存在：%r\n" % (image_dir,))
        return records
    for rel in images:
        label = labels.get(rel)
        if not label:
            base = os.path.basename(rel)
            label = labels.get(base)
        if not label:
            parent = os.path.basename(os.path.dirname(rel))
            label = parent if parent and parent != os.path.basename(image_dir) else ""
        if not label:
            label = "unlabeled"
        records.append({"source": source, "question_type": "image", "stem": "",
                        "options": {}, "answer_letter": "", "answer": label,
                        "explanation": "", "image": _join_image(image_dir, rel),
                        "gt_label": label,
                        "label_extra": {"rel_path": rel},
                        "meta": {"image_kind": "oct" if "oct" in source.lower() else "fundus"}})
    return records


# ---------------------------------------------------------------- Dispatch
def load_records(mtype, path=None, image_dir=None, labels_path=None, col_map=None,
                 eyes="both", source=None):
    """Read one batch of raw records per manifest entry (or equivalent args) -> list[record] (unified schema).

    Any adapter exception is caught and returns [] (the pipeline never crashes on a single source).
    """
    mtype = str(mtype or "").lower()
    cm = dict(col_map) if isinstance(col_map, dict) else {}
    src = source or mtype
    try:
        if mtype == "cmexam":
            recs = load_cmexam(path, cm)
        elif mtype in ("cmb", "medqa"):
            recs = load_cmb(path, cm)
        elif mtype == "pubmedqa":
            recs = load_pubmedqa(path, cm)
        elif mtype in ("generic_jsonl", "tianchi_dialog"):
            # tianchi_dialog (A5/N2): no dedicated adapter, folded into generic_jsonl fallback
            # ({"question","answer","explanation","options"}; column drift via --col-map)
            recs = load_generic_jsonl(path, cm)
        elif mtype == "idrid":
            recs = load_idrid(path, image_dir, cm)
        elif mtype == "odir":
            recs = load_odir(path, image_dir, cm, eyes)
        elif mtype in ("generic_images", "refuge", "palm", "gamma", "octid",
                       "kermany", "octdl"):
            recs = load_generic_images(image_dir, labels_path, cm, source=src)
        else:
            sys.stderr.write("[prepare] 未知源类型：%r（可用 --list-sources 查看）\n" % mtype)
            return []
        for r in recs:
            r["source"] = str(src) if src else r["source"]
        return recs
    except Exception as e:
        sys.stderr.write("[prepare] 适配器 %s 解析失败：%s\n" % (mtype, e))
        return []


# ---------------------------------------------------------------- Manifest construction
def build_manifest_entry(mtype, path=None, image_dir=None, labels_path=None,
                         col_map=None, eyes="both", source=None, license_=None):
    """Parse one source -> manifest entry dict (with n_items and warnings; n_items=0 on parse failure)."""
    source = source or mtype
    warnings = []
    if path and not os.path.isfile(path):
        warnings.append("input 文件不存在：%s" % path)
    if image_dir and not os.path.isdir(image_dir):
        warnings.append("image_dir 不存在：%s" % image_dir)
    if labels_path and not os.path.isfile(labels_path):
        warnings.append("labels 文件不存在：%s" % labels_path)
    records = load_records(mtype, path, image_dir, labels_path, col_map, eyes, source)
    if not records:
        warnings.append("未解析到任何记录（检查路径/列名/编码，或用 --col-map 覆写）")
    for w in warnings:
        sys.stderr.write("[prepare] %s：%s\n" % (source, w))
    return {"source": source, "type": mtype, "path": path or "",
            "image_dir": image_dir or "", "labels_path": labels_path or "",
            "license": license_ or SOURCE_INFO.get(mtype, {}).get("license", "未知（请人工登记）"),
            "n_items": len(records), "col_map": col_map or {}, "eyes": eyes,
            "warnings": warnings, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}


def _parse_col_map(pairs):
    out = {}
    for p in pairs or []:
        if not isinstance(p, str) or "=" not in p:
            continue
        k, _, v = p.partition("=")
        if k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.prepare_datasets",
        description="Step 0：登记本地原始数据集 → data/manifest.jsonl（不做真实下载）。"
                    "每个源执行一次本命令；同 source 重复登记覆盖旧条目。")
    p.add_argument("--type", default="", help="源类型：cmexam/cmb/medqa/pubmedqa/idrid/odir/generic_images"
                    "/generic_jsonl/tianchi_dialog/refuge/palm/gamma/octid/kermany/octdl")
    p.add_argument("--input", default="", help="题目/标签文件路径（csv 或 json/jsonl）")
    p.add_argument("--image-dir", default="", help="图片目录（图像类源必填）")
    p.add_argument("--labels", default="", help="图片标签文件（generic_images 可选，每行『路径,标签』）")
    p.add_argument("--col-map", action="append", default=[], metavar="k=v",
                   help="列名覆写，如 --col-map question=题干（可多次）")
    p.add_argument("--source-name", default="", help="登记名（默认=--type；同源去重）")
    p.add_argument("--license", default="", help="覆写默认许可登记")
    p.add_argument("--eyes", choices=["left", "right", "both"], default="both",
                   help="ODIR 取哪侧眼（默认 both）")
    p.add_argument("--output", default="data/manifest.jsonl", help="manifest 输出路径")
    p.add_argument("--replace", action="store_true", help="清空已有 manifest 后重写（默认追加去重）")
    p.add_argument("--limit", type=int, default=0, help="登记时最多解析的记录数（0=不限）")
    p.add_argument("--seed", type=int, default=0, help="保留位（本步骤无随机性）")
    p.add_argument("--mock", action="store_true", help="保留位（本步骤不调 LLM）")
    p.add_argument("--list-sources", action="store_true", help="打印支持的数据源与许可后退出")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.list_sources:
        for name in sorted(SOURCE_INFO):
            info = SOURCE_INFO[name]
            print("%-16s %-46s %s" % (name, info["desc"], info["license"]))
        return 0
    if not args.type:
        sys.stderr.write("必须提供 --type（--list-sources 查看全部支持类型）\n")
        return 2
    needs_input = args.type in ("cmexam", "cmb", "medqa", "pubmedqa", "idrid", "odir", "generic_jsonl",
                                "tianchi_dialog")  # tianchi_dialog uses the generic_jsonl adapter (A5)
    needs_image = args.type in ("generic_images", "refuge", "palm", "gamma", "octid",
                                "kermany", "octdl")
    if needs_input and not args.input:
        sys.stderr.write("--type %s 需要 --input（题目/标签文件）\n" % args.type)
        return 2
    if needs_image and not args.image_dir:
        sys.stderr.write("--type %s 需要 --image-dir\n" % args.type)
        return 2

    entry = build_manifest_entry(args.type, args.input or None, args.image_dir or None,
                                 args.labels or None, _parse_col_map(args.col_map),
                                 args.eyes, args.source_name or None,
                                 args.license or None)
    rows = L.read_jsonl(args.output)
    if args.replace:
        rows = []
    # note: no sampling at registration (--limit is a shared reserved arg; limits are enforced by
    # extract --max-per-source)
    if args.limit and args.limit > 0:
        sys.stderr.write("[prepare] --limit 在登记阶段不生效（保留位）：限量请改用 "
                         "extract_questions --max-per-source\n")
    rows = [r for r in rows if not (isinstance(r, dict) and r.get("source") == entry["source"])]
    rows.append(entry)
    L.write_jsonl(args.output, rows)
    print("[prepare] manifest=%s sources=%d 本次：%s n_items=%d warnings=%d"
          % (args.output, len(rows), entry["source"], entry["n_items"],
             len(entry.get("warnings", []))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

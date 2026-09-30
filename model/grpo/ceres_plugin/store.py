# ceres_plugin/store.py -- trajectory store and anchor-credit module (design §7.4 + make_traj_key/uid index).
#
# Contract signatures (do not change):
#   make_traj_key(uid: str, last_text: str) -> str   # = sha1(f"{uid}||{last_text.strip()}").hexdigest()
#   anchor_key(dd, state, teacher_text) -> str       # curriculum node x mastery bucket x misconception x phase
#   step_reward(state_after, delta, teacher_text) -> float
#   final_eval(rec) -> dict                          # offline observation only, not used in reward
#   class TrajectoryStore: put / get / put_by_uid / get_by_uid / flush
# Added later (A2 / final review I1, additive only):
#   TrajectoryStore.get_all(key) -> list   # all candidates for one traj_key (collision disambiguation)
#   TrajectoryStore.clear()                # clear both indexes under lock (batch-start entry point)
#
# Deps: stdlib only (numpy allowed by contract but unused). Disk: append traj_YYYYMMDD.jsonl under
# CERES_TRAJ_DIR (default ./traj_dump); write failures only warn.
import datetime
import hashlib
import json
import math
import os
import re
import threading
import warnings

__all__ = ["make_traj_key", "anchor_key", "step_reward", "final_eval", "TrajectoryStore"]

DEFAULT_FLUSH_DIR = "./traj_dump"
_DEFAULT_MASTERY = 0.3
_DEFAULT_ENGAGEMENT = 0.7
_DEFAULT_SEVERITY = 1.0

# Teaching-phase detection (design §7.4; first match in order wins)
PHASE = [("recall", r"<recall"), ("hint", r"<hint"), ("check", r"<check"),
         ("explain", r"<explain"), ("correct", r"<correct"), ("encourage", r"<encourage")]


def _to_float(v, default=0.0):
    """Defensive float conversion: invalid / NaN / inf -> default."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def make_traj_key(uid: str, last_text: str) -> str:
    """Shared derivation of the trajectory primary key (plugin write side + reward lookup side).

    Always sha1(f"{uid}||{last_text.strip()}").hexdigest(); both sides must call this function,
    never hand-roll the hash.
    """
    uid_s = "" if uid is None else str(uid)
    text_s = "" if last_text is None else str(last_text)
    return hashlib.sha1(f"{uid_s}||{text_s.strip()}".encode("utf-8", "ignore")).hexdigest()


def _phase_of(text: str) -> str:
    """Identify the teaching phase from teacher action text (design §7.4 + type fallback)."""
    if not isinstance(text, str):
        return "other"
    for name, pat in PHASE:
        if re.search(pat, text):
            return name
    return "other"


def anchor_key(dd: dict, state: dict, teacher_text: str) -> str:
    """The paper's pedagogical anchor φ(s_t, a_t): node x mastery bucket x misconception x phase.

    Mastery bucket: <0.34 -> L, <0.67 -> M, else H; all inputs have default/type fallbacks.
    """
    dd = dd if isinstance(dd, dict) else {}
    state = state if isinstance(state, dict) else {}
    m = _to_float(state.get("mastery"), _DEFAULT_MASTERY)
    m_bucket = "L" if m < 0.34 else ("M" if m < 0.67 else "H")
    node = dd.get("curriculum_node", "na")
    node = "na" if node is None or str(node) == "" else str(node)
    mis = state.get("misconception", "none")
    mis = "none" if mis is None or str(mis) == "" else str(mis)
    return "|".join([node, m_bucket, mis, _phase_of(teacher_text)])


def step_reward(state_after: dict, delta: dict, teacher_text: str) -> float:
    """Immediate teaching reward (design §7.4 formula, used for suffix return):

        r = 0.6*Δmastery + 0.3*Δengagement + 0.1*(Δm>0) - 0.1*max(0, fatigue-0.6) - 0.05*(hint count>2)
    """
    state_after = state_after if isinstance(state_after, dict) else {}
    delta = delta if isinstance(delta, dict) else {}
    dm = _to_float(delta.get("mastery"), 0.0)
    de = _to_float(delta.get("engagement"), 0.0)
    fatigue = _to_float(state_after.get("fatigue"), 0.0)
    fatigue_pen = 0.1 * max(0.0, fatigue - 0.6)
    text = teacher_text if isinstance(teacher_text, str) else ""
    redundant = -0.05 if len(re.findall(r"<hint", text)) > 2 else 0.0
    return 0.6 * dm + 0.3 * de + 0.1 * (1 if dm > 0 else 0) - fatigue_pen + redundant


def final_eval(rec) -> dict:
    """Final learner-outcome evaluation.

    Observation-only, not used in reward: written to traj_dump as offline metrics; the deployment
    f_outcome reward term reads from here. Falls back to defaults (mastery 0.3 / engagement 0.7 /
    severity 1.0) for empty steps or missing state_after.
    """
    steps = []
    if isinstance(rec, dict):
        maybe = rec.get("steps")
        if isinstance(maybe, list):
            steps = maybe
    last = {}
    for st in reversed(steps):  # take the last step with state_after (guard against missing middle steps)
        if isinstance(st, dict) and isinstance(st.get("state_after"), dict):
            last = st["state_after"]
            break
    return {"mastery": _to_float(last.get("mastery"), _DEFAULT_MASTERY),
            "engagement": _to_float(last.get("engagement"), _DEFAULT_ENGAGEMENT),
            "misconception_severity": _to_float(last.get("misconception_severity"), _DEFAULT_SEVERITY)}


class TrajectoryStore:
    """Thread-safe trajectory store: traj_key main index (put/get/get_all) + uid secondary index.

    - put() also maintains the uid index; the main index is multi-valued (same key appends, no
      overwrite): traj_key=(uid, last text) is not injective (G rollouts can repeat the same last
      text), and overwriting would leave only the survivor visible to reward (I1). get() returns
      the last write (single-candidate semantics unchanged); get_all() returns all candidates for
      kwargs['messages'] disambiguation. Old records of a colliding key stay in the uid index.
    - put_by_uid() doesn't require traj_key; if rec has one it is written to the main index too.
    - clear() empties both indexes under lock (the plugin's batch-start entry point).
    - flush() appends to disk; any IO failure only warns (training safety first).
    """

    def __init__(self, flush_dir=None):
        self._d = {}          # traj_key -> list[rec] (multiple per key, in write order)
        self._uid = {}        # uid -> list[rec]
        self._lock = threading.RLock()
        self._flush_dir_override = flush_dir  # test-injectable; None -> read CERES_TRAJ_DIR each flush
        self._ensure_dir()

    # ---- dir / disk ----
    def _flush_dir(self) -> str:
        if self._flush_dir_override is not None:
            return str(self._flush_dir_override)
        d = os.environ.get("CERES_TRAJ_DIR", DEFAULT_FLUSH_DIR) or DEFAULT_FLUSH_DIR
        return str(d)

    def _ensure_dir(self):
        try:
            os.makedirs(self._flush_dir(), exist_ok=True)
        except Exception as e:  # dir creation failure is non-fatal: flush retries and degrades to warning
            warnings.warn(f"[ceres store] 轨迹目录创建失败（忽略，flush 时重试）: {e}")

    def flush(self, rec):
        """Append to {CERES_TRAJ_DIR}/traj_YYYYMMDD.jsonl; returns the path, or None on failure (warn only)."""
        try:
            d = self._flush_dir()
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"traj_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            return path
        except Exception as e:
            warnings.warn(f"[ceres store] 轨迹落盘失败（忽略）: {e}")
            return None

    # ---- Main index (multi-valued: same key appends, no overwrite) ----
    def put(self, key, rec):
        """traj_key -> rec (appends per key) and maintain the uid index.

        None key falls to the empty-string key (defensive). Old records of a colliding key are
        neither overwritten nor removed from the uid index -- both must stay retrievable.
        """
        key_s = "" if key is None else str(key)
        with self._lock:
            self._main_append(key_s, rec)
            self._uid_append(rec)

    def get(self, key):
        """Get a record by traj_key (last write if multiple; single-candidate semantics unchanged).
        Returns None if missing (reward gives 0 and warns).
        """
        key_s = "" if key is None else str(key)
        with self._lock:
            lst = self._d.get(key_s)
            return lst[-1] if lst else None

    def get_all(self, key) -> list:
        """Get all candidate records for a traj_key (shallow-copied list, write order; used for
        collision disambiguation); [] if none.
        """
        key_s = "" if key is None else str(key)
        with self._lock:
            lst = self._d.get(key_s)
            return list(lst) if lst else []

    # ---- uid index ----
    def put_by_uid(self, uid, rec):
        """Write directly to the uid index (multiple trajectories per uid, in write order).

        Empty uid falls back to rec['uid']; if rec has traj_key it is also written to the main
        index (append, same entry as put) for key lookup.
        """
        uid_s = "" if uid is None else str(uid)
        if uid_s == "" and isinstance(rec, dict):
            uid_s = str(rec.get("uid", "") or "")
        with self._lock:
            self._uid_append(rec, uid_s)
            tk = rec.get("traj_key") if isinstance(rec, dict) else None
            if tk:
                self._main_append(str(tk), rec)

    def get_by_uid(self, uid) -> list:
        """Get all trajectories for a uid (shallow-copied list); [] if none."""
        uid_s = "" if uid is None else str(uid)
        with self._lock:
            lst = self._uid.get(uid_s)
            return list(lst) if lst else []

    def clear(self):
        """Clear both indexes under lock (the plugin's batch-start entry point; replaces direct
        access to _d/_uid). Flushed files are unaffected, so eval can still consume them offline.
        """
        with self._lock:
            self._d.clear()
            self._uid.clear()

    # ---- Internal: index maintenance (dedupe by object identity to avoid dict== O(n) and wrong deletion) ----
    def _main_append(self, key_s, rec):
        """Append to the main index; re-writing the same object replaces it in place (idempotent)."""
        lst = self._d.get(key_s)
        if lst is None:
            self._d[key_s] = [rec]
            return
        for i, item in enumerate(lst):
            if item is rec:
                lst[i] = rec
                return
        lst.append(rec)

    def _uid_append(self, rec, uid=None):
        if not isinstance(rec, dict):
            return
        if uid is None:
            uid = str(rec.get("uid", "") or "")
        if uid == "":
            return
        lst = self._uid.setdefault(uid, [])
        for item in lst:
            if item is rec:
                return
        lst.append(rec)

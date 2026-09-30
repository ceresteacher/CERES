# -*- coding: utf-8 -*-
"""ceres_plugin/save_on_exit.py -- save LoRA weights before any error-driven exit (work package C).

Motivation: GRPO/SFT runs take tens of hours; OOM / NCCL timeout / cascading API failures / a
mistaken kill all evaporate progress since the last --save_steps. This module adds crash
protection: on any process-exit event, save the LoRA deltas to <output_dir>/emergency-adapter.

Scope: it does NOT cover failures that already degrade internally (student_sim retries then
returns a conservative reply; llm_client returns empty/degraded). It covers everything else:
GPU OOM / NCCL watchdog / deepspeed collective errors, cascading failures after the student API
(checkpoint save, eval, log threads), and external kill (SIGTERM) / Ctrl-C (SIGINT). Normal
checkpointing stays with swift/transformers.

Registration (verified from swift 3.4.1 source, 2026-08-31): swift/plugin/callback.py:30 defines
a module-level shared extra_callbacks list; sft.py:231 `callbacks += extra_callbacks` takes the
same list object, so append (in-place mutation) takes effect with no CLI flag. SwiftPt/SwiftRLHF
(GRPO) inherit SwiftSft, so SFT/PT/GRPO all apply. --external_plugins is imported during arg
parsing (before the callback table is collected). swift 3.4.1 / transformers 4.51.3 TrainerCallback
has NO on_exception hook, so sys.excepthook + SIGTERM/SIGINT are used; on_exception is still
implemented duck-typed (auto-catches if a future version adds it). callback_event passes
model=self.model, and output_dir comes from args.output_dir.

Behavior: only local_rank 0 saves; idempotent (at most one save attempt per process);
model.save_pretrained(<output_dir>/emergency-adapter) saves only LoRA deltas (PeftModel); under
ZeRO-3 the gather may fail, so the save is fully try/except'd (never masks the original
exception). After saving, restore the original excepthook / default signal behavior (re-raise the
original exception / exit code: SIGINT->130, SIGTERM->143). Hooks are NOT uninstalled on
on_train_end (post-training eval/flush crashes are worth saving too).

Recovery: emergency-adapter is a standard LoRA adapter dir; resume with
    GRPO: export CERES_ADAPTERS=<output_dir>/emergency-adapter && bash scripts/run_grpo_oph_lora.sh
    demo: bash scripts/run_demo_dialogue.sh --adapters <output_dir>/emergency-adapter
    (or swift --adapters <output_dir>/emergency-adapter).
"""
import logging
import os
import signal
import sys
import threading

__all__ = ["SaveOnExit", "register_save_on_exit", "emergency_save", "install_exit_hooks",
           "uninstall_exit_hooks", "EMERGENCY_DIRNAME", "TRANSFORMERS_AVAILABLE"]

_LOG = logging.getLogger(__name__)

#: Crash-save directory name (under the training output_dir)
EMERGENCY_DIRNAME = "emergency-adapter"

# Lazy import of the transformers base class: swift's extra_callbacks collects
# TrainerCallback subclasses; test environments (no swift/torch/transformers) use a placeholder.
try:
    from transformers.trainer_callback import TrainerCallback as _TrainerCallback
    TRANSFORMERS_AVAILABLE = True
except Exception:  # noqa: BLE001  transformers missing/broken: still usable via duck typing
    class _TrainerCallback:  # type: ignore[no-redef]
        """Placeholder base (tests without transformers; real training always has it)."""

    TRANSFORMERS_AVAILABLE = False

#: Module-level state (process singleton; lock-protected -- signals/excepthook may re-enter).
_STATE = {
    "model": None,            # training model reference (PeftModel)
    "output_dir": None,       # args.output_dir reference
    "saved": False,           # idempotence flag: at most one attempt per process
    "save_path": None,        # saved path on success
    "hooks_installed": False,
    "orig_excepthook": None,
    "orig_handlers": {},      # signum -> original signal handler
}
_LOCK = threading.RLock()

_HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGINT)


# ---------------------------------------------------------------- helpers
def _local_rank():
    """Current process local rank (env LOCAL_RANK; 0 without accelerate/torchrun)."""
    try:
        return int(str(os.environ.get("LOCAL_RANK", "0") or "0").strip())
    except (TypeError, ValueError):
        return 0


def _output_dir_of(args):
    """args -> output_dir (str/None; malformed input -> None)."""
    try:
        v = getattr(args, "output_dir", None)
        return str(v).strip() if v else None
    except Exception:
        return None


def _capture_reference(model, output_dir):
    """Store the trainer's model/output_dir references into module state (don't overwrite with empty)."""
    with _LOCK:
        if model is not None:
            _STATE["model"] = model
        if output_dir:
            _STATE["output_dir"] = output_dir


# ---------------------------------------------------------------- crash save
def emergency_save(trigger=""):
    """Emergency-save LoRA deltas to <output_dir>/emergency-adapter -> bool (never raises).

    * only local_rank 0; skip if model/output_dir not yet captured;
    * idempotent: at most one attempt per process;
    * the save itself is try/except'd: under ZeRO-3 the gather may fail, but that must never
      mask/replace the original exception.
    """
    with _LOCK:
        if _STATE["saved"]:
            return False
        _STATE["saved"] = True                 # set first (at most one attempt regardless of outcome)
        model = _STATE.get("model")
        output_dir = _STATE.get("output_dir")
    if model is None or not output_dir:
        _LOG.warning("[save_on_exit] 触发（%s）但 model/output_dir 尚未捕获，跳过保存", trigger)
        return False
    rank = _local_rank()
    if rank != 0:
        _LOG.info("[save_on_exit] 触发（%s）但 local_rank=%d ≠ 0，跳过（rank0 负责保存）",
                  trigger, rank)
        return False
    path = os.path.join(output_dir, EMERGENCY_DIRNAME)
    _LOG.warning("=" * 72)
    _LOG.warning("[save_on_exit] ⚠ 进程即将退出（触发：%s）——正在紧急保存 LoRA 增量到 %s",
                 trigger or "unknown", path)
    _LOG.warning("=" * 72)
    try:
        model.save_pretrained(path)            # PeftModel: saves only LoRA deltas, small and fast
        with _LOCK:
            _STATE["save_path"] = path
        _LOG.warning("[save_on_exit] ✔ 紧急保存完成：%s（可用 --adapters %s 挂载续训）", path, path)
        return True
    except Exception as e:                     # noqa: BLE001  a crash inside the crash handler must not mask the original exception
        _LOG.error("[save_on_exit] ✘ 紧急保存失败（%s）：%s——放弃保存，原异常/退出语义不变",
                   type(e).__name__, e)
        return False


# ---------------------------------------------------------------- three trigger surfaces
def _excepthook(exc_type, exc_value, exc_tb):
    """sys.excepthook: uncaught exception -> save first, then exit as before."""
    try:
        emergency_save("sys.excepthook:%s" % getattr(exc_type, "__name__", exc_type))
    finally:
        # restore the original excepthook and default semantics (print traceback, exit code 1)
        hook = _STATE.get("orig_excepthook") or sys.__excepthook__
        try:
            sys.excepthook = hook
        except Exception:                      # noqa: BLE001
            pass
        hook(exc_type, exc_value, exc_tb)


def _signal_handler(signum, frame):  # noqa: ARG001  frame is required by the signal protocol
    """SIGTERM/SIGINT handler: save first, then restore default behavior and re-raise the signal."""
    try:
        emergency_save("signal:%d" % int(signum))
    finally:
        try:
            signal.signal(signum, signal.SIG_DFL)
        except Exception:                      # noqa: BLE001  e.g. non-main thread
            pass
        os.kill(os.getpid(), signum)           # SIG_DFL takes over: SIGINT->130 / SIGTERM->143


def install_exit_hooks():
    """Install sys.excepthook + SIGTERM/SIGINT handlers (idempotent; signal install on non-main thread fails -> warn only)."""
    with _LOCK:
        if _STATE["hooks_installed"]:
            return True
        _STATE["hooks_installed"] = True
        _STATE["orig_excepthook"] = sys.excepthook
        for sig in _HANDLED_SIGNALS:
            try:
                _STATE["orig_handlers"][int(sig)] = signal.getsignal(sig)
            except Exception:                  # noqa: BLE001
                pass
    sys.excepthook = _excepthook
    ok = True
    for sig in _HANDLED_SIGNALS:
        try:
            signal.signal(sig, _signal_handler)
        except Exception as e:                 # noqa: BLE001  ValueError: not the main thread
            ok = False
            _LOG.warning("[save_on_exit] 无法安装信号处理器 sig=%s（%s）", sig, e)
    _LOG.info("[save_on_exit] 崩溃保护已就绪（sys.excepthook + %s）——异常/kill 退出前将保存 "
              "LoRA 到 <output_dir>/%s", "/".join(str(int(s)) for s in _HANDLED_SIGNALS),
              EMERGENCY_DIRNAME)
    return ok


def uninstall_exit_hooks():
    """Restore the original excepthook and signal handlers (test isolation; not called in normal training)."""
    with _LOCK:
        if not _STATE["hooks_installed"]:
            return
        _STATE["hooks_installed"] = False
        orig_hook = _STATE.get("orig_excepthook")
        handlers = dict(_STATE.get("orig_handlers") or {})
        _STATE["orig_excepthook"] = None
        _STATE["orig_handlers"] = {}
    try:
        if orig_hook is not None:
            sys.excepthook = orig_hook
    except Exception:                          # noqa: BLE001
        pass
    for sig, handler in handlers.items():
        try:
            signal.signal(sig, handler if handler is not None else signal.SIG_DFL)
        except Exception:                      # noqa: BLE001
            pass


def _reset_for_tests():
    """Reset module state and uninstall hooks (test isolation only; never call in training)."""
    uninstall_exit_hooks()
    with _LOCK:
        _STATE.update({"model": None, "output_dir": None, "saved": False,
                       "save_path": None})


# ---------------------------------------------------------------- callback
class SaveOnExit(_TrainerCallback):
    """swift/transformers training callback: capture model/output_dir references + install hooks.

    Registered by register_save_on_exit() (called at the end of plugin.py), which appends an
    instance to swift.plugin.extra_callbacks (shared list, no CLI flag; see module docstring).
    """

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        """Training start: capture model/output_dir and install hooks."""
        _capture_reference(kwargs.get("model"), _output_dir_of(args))
        install_exit_hooks()

    def on_step_end(self, args=None, state=None, control=None, model=None, **kwargs):
        """Step end: refresh references (fallback if on_train_begin ran before model was ready) and ensure hooks are installed."""
        _capture_reference(model if model is not None else kwargs.get("model"),
                           _output_dir_of(args))
        install_exit_hooks()

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        """Training end: do NOT uninstall hooks -- post-training (eval/final flush) crashes are worth saving too."""
        _capture_reference(kwargs.get("model"), _output_dir_of(args))

    def on_exception(self, args=None, state=None, control=None, exception=None, **kwargs):
        """If swift/transformers provide an on_exception hook, catch it here (3.4.1/4.51.3 don't,
        so this is never called; auto-takes effect if a future version adds it).
        """
        trigger = ("on_exception:%s" % type(exception).__name__
                   if exception is not None else "on_exception")
        return emergency_save(trigger)


# ---------------------------------------------------------------- registration
def register_save_on_exit(callbacks_list=None):
    """Register a SaveOnExit instance into swift's extra_callbacks table -> bool.

    :param callbacks_list: explicit callback list (test injection); defaults to
        swift.plugin.extra_callbacks (append takes effect, no CLI flag).
    Returns False with a warning if swift is unavailable; never affects other plugin features.
    """
    targets = []
    if callbacks_list is not None:
        targets.append(callbacks_list)
    else:
        try:
            from swift.plugin import extra_callbacks as _swift_extra_callbacks
            targets.append(_swift_extra_callbacks)
        except Exception as e:                 # noqa: BLE001  swift missing/broken
            _LOG.warning("[save_on_exit] swift 不可用（%s）——无法注册进 extra_callbacks，"
                         "本次进程无崩溃保护", type(e).__name__)
            return False
    cb = SaveOnExit()
    for lst in targets:
        if not any(isinstance(c, SaveOnExit) for c in lst):
            lst.append(cb)
    _LOG.info("[save_on_exit] SaveOnExit 已注册（%d 处回调表）——异常/kill 退出前将保存 LoRA "
              "增量到 <output_dir>/%s", len(targets), EMERGENCY_DIRNAME)
    return True

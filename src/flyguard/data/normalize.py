"""Text normalisation and language detection (ТЗ 1.2, docs/design.md §2).

Why a separate module: every source (deepset, BIPIA, trace steps, paraphrases, NotInject) must be normalised by
the *same* function so that injection spans located by exact match in one module (agentdojo_io) line up with
window coordinates computed here. The normalised text is what detectors see; the original is kept for the audit.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

_WS = re.compile(r"\s+")
UNK = "unk"


def normalize_text(text: str, form: str = "NFKC", collapse_whitespace: bool = True) -> str:
    """ТЗ 1.2: Unicode normalisation (NFKC by default) + runs of whitespace -> one space, stripped.

    NFKC folds full-width/compatibility characters that attackers use to dodge string matching; the whitespace
    collapse makes window coordinates independent of line wrapping. ``None`` is treated as the empty string.
    """
    if text is None:
        return ""
    t = unicodedata.normalize(form, str(text))
    if collapse_whitespace:
        t = _WS.sub(" ", t).strip()
    return t


def normalizer_from_cfg(cfg: Any):
    """Return ``normalize_text`` bound to ``cfg.default['normalize']`` (unicode form, collapse flag)."""
    section = cfg.default.get("normalize", {})
    form = section.get("unicode", "NFKC")
    collapse = bool(section.get("collapse_whitespace", True))

    def _norm(text: str) -> str:
        return normalize_text(text, form=form, collapse_whitespace=collapse)

    return _norm


# --------------------------------------------------------------------------------------------- language

_LANG_SEED_SET: int | None = None
_LANG_CACHE: dict[str, str] = {}


def _ensure_langdetect_seed(seed: int) -> None:
    """langdetect is randomised internally; ТЗ 1.2 + ``language.seed`` in configs make it deterministic."""
    global _LANG_SEED_SET
    from langdetect import DetectorFactory

    if _LANG_SEED_SET != seed:
        DetectorFactory.seed = int(seed)
        _LANG_SEED_SET = int(seed)
        _LANG_CACHE.clear()


def detect_language(text: str, seed: int = 0) -> str:
    """ТЗ 1.2: langdetect code of ``text``; ``'unk'`` when detection fails (empty, digits only, too short).

    Results are cached by text because BIPIA variants and trace steps repeat the same texts thousands of times.
    """
    _ensure_langdetect_seed(seed)
    if not text or not text.strip():
        return UNK
    cached = _LANG_CACHE.get(text)
    if cached is not None:
        return cached
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException

    try:
        code = str(detect(text))
    except LangDetectException:
        code = UNK
    except Exception:  # pragma: no cover - defensive: never let one document kill the build
        code = UNK
    _LANG_CACHE[text] = code
    return code


def lang_stratum(lang: str, main: str = "en") -> str:
    """ТЗ 1.2: two strata, the main language (``en``) and everything else (``non-en``, including ``unk``)."""
    return main if lang == main else f"non-{main}"


def language_columns(texts: Iterable[str], cfg: Any) -> tuple[list[str], list[str]]:
    """Language code and stratum for each text, using ``cfg.default['language']`` (seed, main language)."""
    section = cfg.default.get("language", {})
    seed = int(section.get("seed", 0))
    main = str(section.get("main", "en"))
    langs = [detect_language(t, seed) for t in texts]
    return langs, [lang_stratum(code, main) for code in langs]

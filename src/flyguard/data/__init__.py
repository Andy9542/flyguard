"""flyguard.data — documents, windows, dedup, splits and pools (docs/design.md §2, §4; ТЗ 1.2–1.4, 1.7–1.10).

Submodules are imported explicitly (``from flyguard.data import windows``) so that importing the package stays
cheap for the modules that only need ``windows.make_windows`` or ``dedup.char_shingles``.
"""
__all__ = ["normalize", "windows", "dedup", "loaders", "splits", "pools", "audit", "build"]

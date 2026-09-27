"""flyguard.experiments — the per-seed engine behind E0–E6 and the contract run (docs/design.md §9,
docs/design_experiments.md §1–§2, §5).

Layout: :mod:`context` loads the tables and manifests and is the only place that opens test windows
(``Context.load_test_windows``, journaled through ``flyguard.netlog.log_data_access``); :mod:`engine` builds the
per-seed feature context (counts, noses, matrices, codes), the detector registry of ТЗ Этап 2–3, scoring,
document-level evaluation helpers and the ``Runner``; :mod:`results` writes ``results/<E>/<seed>.json`` and the
per-experiment ``summary.json``. Experiment scripts (``e1.py`` … ``contract_run.py``) import from here and never
read data files themselves.
"""
from flyguard.experiments.context import Context, load_test_windows
from flyguard.experiments.engine import (DETECTORS, DetectorSpec, Evaluator, FeatureContext, FittedDetector,
                                         ResultBuilder, Runner, WindowSet, by_source, detector_spec, fly_spec,
                                         standard_evaluation)
from flyguard.experiments.results import (is_current, number, power_path, read_result, result_path, results_dir,
                                          summarize, write_result)

__all__ = [
    "Context", "load_test_windows",
    "DETECTORS", "DetectorSpec", "Evaluator", "FeatureContext", "FittedDetector", "ResultBuilder", "Runner",
    "WindowSet", "by_source", "detector_spec", "fly_spec", "standard_evaluation",
    "is_current", "number", "power_path", "read_result", "result_path", "results_dir", "summarize", "write_result",
]

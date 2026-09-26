"""flyguard.eval — metrics, thresholds, bootstrap, TOST/Holm/randomisation, E0 power and verdicts (design §8)."""
from flyguard.eval.bootstrap import (CI, WeightedAUC, as_ci, auc_stat, bootstrap_defaults, cluster_bootstrap,
                                     macro_auc_bootstrap, macro_auc_draws, paired_cluster_bootstrap,
                                     two_stage_bootstrap_h3)
from flyguard.eval.metrics import (auc, contract_frame, contract_metrics, contract_metrics_by_variant,
                                   contract_point_metrics, doc_scores, fpr_at_threshold, macro_auc, tau_for_tpr,
                                   threshold_for_fpr, tpr_at_fpr, tpr_at_threshold)
from flyguard.eval.power import carrier_rule, notinject_table, power_cell, power_table, spread
from flyguard.eval.thresholds import (contract_threshold, default_threshold_record, fpr_target_for_pool,
                                      tau_fpr_record, tau_tpr_record, threshold_record)
from flyguard.eval.tost import (bootstrap_p, equivalence_margin, holm, holm_reject, p_from_ci, randomization_p,
                                tost, tost_equivalent)
from flyguard.eval.verdicts import (CONFIRMED, INSUFFICIENT, PRECONDITION, REFUTED, Verdict, carriers_from_power,
                                    verdict_h1a, verdict_h1b, verdict_h2, verdict_h3, verdicts_h1b)

__all__ = [
    "CI", "WeightedAUC", "as_ci", "auc_stat", "bootstrap_defaults", "cluster_bootstrap", "macro_auc_bootstrap",
    "macro_auc_draws", "paired_cluster_bootstrap", "two_stage_bootstrap_h3",
    "auc", "contract_frame", "contract_metrics", "contract_metrics_by_variant", "contract_point_metrics",
    "doc_scores", "fpr_at_threshold",
    "macro_auc", "tau_for_tpr", "threshold_for_fpr", "tpr_at_fpr", "tpr_at_threshold",
    "carrier_rule", "notinject_table", "power_cell", "power_table", "spread",
    "contract_threshold", "default_threshold_record", "fpr_target_for_pool", "tau_fpr_record", "tau_tpr_record",
    "threshold_record",
    "bootstrap_p", "equivalence_margin", "holm", "holm_reject", "p_from_ci", "randomization_p", "tost",
    "tost_equivalent",
    "CONFIRMED", "INSUFFICIENT", "PRECONDITION", "REFUTED", "Verdict", "carriers_from_power", "verdict_h1a",
    "verdict_h1b", "verdict_h2", "verdict_h3", "verdicts_h1b",
]

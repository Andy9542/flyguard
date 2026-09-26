"""Trace logs of AgentDojo / AgentDyn -> step documents, episode classes and the comparison-contract files
(ТЗ 1.5 "Извлечение", 1.8, 1.10, Этап 5; Контракт_сравнения_v3 §2–§8; design §3).

Modules: `parse` (log -> steps), `labels` (spans, classes, first harmful call, splits), `contract` (16-column CSV,
validator, split manifest), `extract` (episodes/documents frames + `traces_extraction.json`).
"""
from flyguard.agentdojo_io.contract import (COLUMNS, DETECTOR, VARIANTS, ContractRow, split_manifest, validate_csv,
                                            validate_split_manifest, write_csv, write_split_manifest)
from flyguard.agentdojo_io.extract import DOCUMENT_COLUMNS, EPISODE_COLUMNS, build_episode_documents
from flyguard.agentdojo_io.labels import (contract_split, episode_class, first_harmful_step, injection_spans,
                                          injection_step, is_e1_val_task, is_test_task, normalize_text, val_task_ids)
from flyguard.agentdojo_io.parse import Step, TraceLog, episode_id, iter_log_paths, read_log, tool_steps

__all__ = [
    "COLUMNS", "DETECTOR", "VARIANTS", "ContractRow", "split_manifest", "validate_csv", "validate_split_manifest",
    "write_csv", "write_split_manifest", "DOCUMENT_COLUMNS", "EPISODE_COLUMNS", "build_episode_documents",
    "contract_split", "episode_class", "first_harmful_step", "injection_spans", "injection_step", "is_e1_val_task",
    "is_test_task", "normalize_text", "val_task_ids", "Step", "TraceLog", "episode_id", "iter_log_paths", "read_log",
    "tool_steps",
]

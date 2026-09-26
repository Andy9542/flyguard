"""ТЗ 1.8: composition of P_val and P_test."""
import pandas as pd

from flyguard.data.pools import build_pools


def _doc(doc_id, source, split, label, meta=None, cluster=None):
    return {"doc_id": doc_id, "source": source, "split": split, "label": label, "cluster_id": cluster or doc_id,
            "meta": meta or {}}


def test_pool_composition(cfg):
    docs = pd.DataFrame([
        _doc("deep:train:0", "deep", "train", 0), _doc("deep:train:1", "deep", "val", 0), _doc("deep:train:2", "deep", "val", 1),
        _doc("deep:test:0", "deep", "test", 0), _doc("deep:test:1", "deep", "test", 1),
        _doc("bipia:email:0:clean", "bipia", "val", 0, {"variant": "main"}, "bipia:email:0"),
        _doc("bipia:email:0:a:end", "bipia", "val", 1, {"variant": "main"}, "bipia:email:0"),
        _doc("bipia:email:1:clean", "bipia", "test", 0, {"variant": "main"}, "bipia:email:1"),
        _doc("bipia:email:1:a:end", "bipia", "test", 1, {"variant": "main"}, "bipia:email:1"),
        _doc("bipia:email:1:a:start", "bipia", "test", 1, {"variant": "e6"}, "bipia:email:1"),
        _doc("dojo:v#0", "dojo", "val", 0, {"episode_class": "benign", "attack": None}),
        _doc("dojo:v_atk#0", "dojo", "val", 0, {"episode_class": "injection_ignored", "attack": "tool_knowledge"}),
        _doc("dojo:t#0", "dojo", "test", 0, {"episode_class": "benign", "attack": None}),
        _doc("dojo:t_atk#0", "dojo", "test", 0, {"episode_class": "hijacked", "attack": "important_instructions"}),
        _doc("dojo:t_atk#1", "dojo", "test", 1, {"episode_class": "hijacked", "attack": "important_instructions"}),
        _doc("dyn:b#0", "dyn", "test", 0, {"attack": "none"}),                       # no episode_class -> attack rule
        _doc("dyn:a#0", "dyn", "test", 0, {"attack": "important_instructions"}),
        _doc("para:b:0", "para", "test", 0), _doc("para:b:1", "para", "test", 1),
        _doc("notinject:one:0", "notinject", "test", 0),
        _doc("deep:test:9", "deep", "test", 0),
    ])
    pools = build_pools(docs, cfg, dropped={"deep:test:9"})
    assert pools["p_val"]["doc_ids"] == ["bipia:email:0:clean", "deep:train:1", "dojo:v#0"]
    assert pools["p_test"]["doc_ids"] == ["bipia:email:1:clean", "deep:test:0", "dojo:t#0", "dyn:b#0", "para:b:0"]
    assert pools["p_val"]["by_source"] == {"bipia": 1, "deep": 1, "dojo": 1}
    assert pools["p_test"]["by_source"] == {"bipia": 1, "deep": 1, "dojo": 1, "dyn": 1, "para": 1}
    target = cfg.default["pools"]["target_min_docs"]
    assert pools["p_val"]["target_min_docs"] == target and pools["p_val"]["meets_target"] is False
    assert pools["p_test"]["shortfall"] == target - 5
    assert pools["notinject"]["n"] == 1

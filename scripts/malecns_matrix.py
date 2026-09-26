#!/usr/bin/env python
"""Extract the measured glomerulus x Kenyon-cell projection from the `flypath build` graph (ТЗ 2.2).

Runs in .venv-flypath from inside data/ext/FlyHash-Connectome (imports flypath). Uses FlyHash-Connectome's own
selection (`flypath.flyhash.mushroom_body`: ALPN and Kenyon_Cell of one hemisphere, PNs merged by glomerulus,
non-olfactory VP*/MZ glomeruli dropped) so the matrix is the reference's 51 x 1886. Writes:
  <out>/malecns_R.npz   m (synapse counts, float64 [n_glomeruli, n_kc]), pn_counts, kc_body_ids
  <out>/malecns_R.json  glomeruli, side, n_pn, n_kc, mean in-degree (distinct glomerular inputs), stats
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from flypath import config, data, flyhash  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../processed/connectome")
side = sys.argv[2] if len(sys.argv) > 2 else "R"
out.mkdir(parents=True, exist_ok=True)
cfg = config.load()
graph = data.load_graph(cfg)
p = flyhash.mushroom_body(graph, side=side)
m = np.asarray(p.matrix if hasattr(p, "matrix") else p.M if hasattr(p, "M") else p[1], dtype=np.float64)
binary = m > 0
indeg = binary.sum(axis=0)
kc_idx = None
nodes = graph.nodes
kc = nodes[(nodes["class"] == "Kenyon_Cell") & (nodes["somaSide"] == side)]
np.savez_compressed(out / f"malecns_{side}.npz", m=m, pn_counts=np.asarray(p.meta["pn_counts"]),
                    pn_partners=np.asarray(p.meta["pn_partners"]))
meta = {"source": "flypath build (ssenge/FlyHash-Connectome) + flypath.flyhash.mushroom_body", "side": side,
        "min_weight": int(cfg["build"]["min_weight"]), "glomeruli": list(p.glomeruli if hasattr(p, "glomeruli") else p[2]),
        "n_glomeruli": int(m.shape[0]), "n_kc": int(m.shape[1]), "n_kc_candidates_side": int(len(kc)),
        "n_pn": int(p.meta["n_pn"]), "nnz": int(binary.sum()), "n_synapses": int(m.sum()),
        "indegree_mean": float(indeg.mean()), "indegree_min": int(indeg.min()), "indegree_max": int(indeg.max()),
        "graph_stats": graph.stats}
(out / f"malecns_{side}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in meta.items() if k not in ("glomeruli", "graph_stats")}, indent=1))

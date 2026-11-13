"""Re-bin never-found GT by the best box IoU any sample in the group achieved: unseen (<0.1), mis-scaled (0.1-0.5), lost-in-matching (>0.5).
For mis-scaled: is the best box 'merged' (also covers another GT with IoU>0.1 / contains >=50% of another GT box), 'oversized' (area>2x GT), 'partial' (area<0.5x GT) or 'offset'."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.dont_write_bytecode = True
import never_found as NF

def area(b): return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

RUNS = {"G7": "runs/grpo_20261020/g7-27b448c128-pq/samples.jsonl",
        "G2": "runs/trustmix_20261104/g2-27b448-trustmix-pq-from-s2c256/samples.jsonl",
        "G5": "runs/trustmix_20261104/g5-27b448-trustmix-wpq-fp03-fn07-from-s2c256/samples.jsonl",
        "G6": "runs/trustmix_20261104/g6-27b448-trustmix-pq-g16x1-from-s2c256/samples.jsonl"}
res = {}
for name, rel in RUNS.items():
    rows = [json.loads(l) for l in (NF.ROOT / rel).read_text().splitlines()]
    grp = defaultdict(list)
    for r in rows:
        if r.get("gt_member"): continue  # ground-truth answers are not policy samples
        grp[(int(r["step"]), r["id"], int(r["transform"]))].append(r)
    C = defaultdict(Counter)
    for (step, cid, t), g in grp.items():
        G, size = NF.gt(cid, t)
        cnt = [0] * len(G); allb = []
        for s in g:
            b = NF.parse(s["text"], size) if s.get("valid") else None
            if not b: continue
            allb += b
            for j in NF.match(b, G): cnt[j] += 1
        ds = NF.MAN[cid].get("dataset", "monuseg")
        for j, x in enumerate(G):
            if cnt[j] or x["sliver"]: continue
            best, bb = 0.0, None
            for b in allb:
                v = NF.biou(b, x["box"])
                if v > best: best, bb = v, b
            if best > 0.5: k = "lost_in_matching(>0.5)"
            elif best > 0.1:
                others = [y for i, y in enumerate(G) if i != j]
                merged = any(inter(bb, y["box"]) >= 0.5 * area(y["box"]) for y in others)
                ra = area(bb) / max(1, area(x["box"]))
                k = "misscaled:merged_with_neighbour" if merged else ("misscaled:oversized" if ra > 2 else ("misscaled:partial" if ra < 0.5 else "misscaled:offset"))
            else: k = "unseen(<0.1)"
            for key in (ds, "ALL"): C[key][k] += 1
    res[name] = {k: dict(v) for k, v in C.items()}
    print("\n==", name, "(never-found, edge slivers excluded)")
    for ds in ("ALL", "monuseg", "consep", "puma"):
        if ds not in C: continue
        n = sum(C[ds].values())
        un = C[ds]["unseen(<0.1)"]; ms = sum(v for k, v in C[ds].items() if k.startswith("misscaled")); lm = C[ds]["lost_in_matching(>0.5)"]
        sub = {k.split(":")[1]: v for k, v in C[ds].items() if k.startswith("misscaled")}
        print(f"  {ds:8s} n={n:4d}  unseen {un/n:.0%}  mis-scaled {ms/n:.0%} {sub}  lost-in-matching {lm/n:.0%}")
json.dump(res, open("rebin.json", "w"), indent=1)

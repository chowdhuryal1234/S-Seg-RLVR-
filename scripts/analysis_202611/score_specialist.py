"""Read-only: score InstanSeg label maps on the 170-tile validation suite with OUR metrics, compare to G5-final,
check which G5 / G7 misses the specialist finds, and compute oracle-selection ceilings.

Inputs (all validation only):
  specialist maps  <SPEC>/context_maps.npz, crop_maps.npz          (run_instanseg.py)
  G5-final maps    <G5PRED>/000001..000170.npy (copied read-only from FASRC e-g5final-val170-cap8/after)
  G5 boxes         runs/trustmix_20261104/e-g5final-val170-cap8/after/completions.jsonl (converted_objects, pixel frame)
  G7 boxes         runs/grpo_20261020/g7-27b448c128-pq/after/completions.jsonl  (set A definition)
Metrics: nucleus_rl.rewards.instance_metrics (strict IoU > 0.5) imported read-only from the main worktree.
Usage: <python> score_specialist.py <SPEC dir> <G5PRED dir> <out json>
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/conny/Desktop/RLVR/grpo-worktree-20261020")
from nucleus_rl.rewards import instance_metrics  # noqa: E402

SPEC, G5P, OUTJ = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ROOT = Path("/Users/conny/Desktop/RLVR/S-Seg-RLVR-")
TM = ROOT / "data/trustmix_v2"
GAP = Path("/Users/conny/Desktop/RLVR/local-notes/next-phase-20261104/gap-analysis-1105")
rows = [json.loads(l) for l in (TM / "validation.jsonl").read_text().splitlines()]
IDS = [r["id"] for r in rows]; MAN = {r["id"]: r for r in rows}
DS = {r["id"]: r["dataset"] for r in rows}


def load_maps(p):
    z = np.load(p); assert list(z["ids"]) == IDS; return dict(zip(IDS, z["maps"].astype(np.int32)))


SP = {"context": load_maps(SPEC / "context_maps.npz"), "crop": load_maps(SPEC / "crop_maps.npz")}
G5C = [json.loads(l) for l in (ROOT / "runs/trustmix_20261104/e-g5final-val170-cap8/after/completions.jsonl").read_text().splitlines()]
assert [c["id"] for c in G5C] == IDS
G5 = {c["id"]: np.load(G5P / Path(c["prediction_path"]).name).astype(np.int32) for c in G5C}
G5BOX = {c["id"]: [o["box"] for o in c["converted_objects"]] for c in G5C}
GT = {i: np.asarray(Image.open((TM / MAN[i]["mask_path"]).resolve())).astype(np.int32) for i in IDS}


def biou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
def inter(a, b): return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def gt_objs(cid):
    lab = GT[cid]; trunc = set(MAN[cid].get("truncated_instance_ids", []))
    out = {}
    for g in np.unique(lab[lab > 0]):
        m = lab == g; yx = np.argwhere(m)
        out[int(g)] = dict(mask=m, area=int(m.sum()), edge=int(g) in trunc,
                           box=[int(yx[:, 1].min()), int(yx[:, 0].min()), int(yx[:, 1].max()) + 1, int(yx[:, 0].max()) + 1])
    return out


GTO = {i: gt_objs(i) for i in IDS}


def inst_list(lab):
    return {int(k): lab == k for k in np.unique(lab[lab > 0])}


def best_match(gm, insts):
    """best mask IoU of one GT mask against a dict of candidate masks -> (iou, key)."""
    best, bk = 0.0, None
    for k, m in insts.items():
        i = np.count_nonzero(m & gm)
        if not i: continue
        v = i / np.count_nonzero(m | gm)
        if v > best: best, bk = v, k
    return best, bk


def summarise(per_tile):
    """macro PQ (= mean over tiles, as the eval runner reports) + pooled counts, overall and per dataset."""
    out = {}
    for key in ("ALL", "monuseg", "consep", "puma"):
        ts = [t for t in per_tile if key == "ALL" or t["ds"] == key]
        tp = sum(t["tp"] for t in ts); fp = sum(t["fp"] for t in ts); fn = sum(t["fn"] for t in ts)
        out[key] = dict(tiles=len(ts), pq=round(float(np.mean([t["pq"] for t in ts])), 4),
                        tp=tp, fp=fp, fn=fn, gt=tp + fn, pred=tp + fp,
                        recall_pooled=round(tp / max(1, tp + fn), 4), precision_pooled=round(tp / max(1, tp + fp), 4),
                        recall_macro=round(float(np.mean([t["recall"] for t in ts])), 4),
                        precision_macro=round(float(np.mean([t["precision"] for t in ts])), 4),
                        sq_matched=round(float(sum(t["pq"] * (t["tp"] + .5 * t["fp"] + .5 * t["fn"]) for t in ts) / max(1, tp)), 4))
    return out


def score(maps):
    pt = []
    for i in IDS:
        m = instance_metrics(maps[i], GT[i]); m["ds"] = DS[i]; m["id"] = i; pt.append(m)
    return pt


res = {"method": {}, "scores": {}}
per_tile = {"G5-final": score(G5), "InstanSeg-context": score(SP["context"]), "InstanSeg-crop": score(SP["crop"])}
for k, v in per_tile.items(): res["scores"][k] = summarise(v)

# ---------------- per-GT table: G5 status/category + specialist IoU
SPI = {mode: {i: inst_list(SP[mode][i]) for i in IDS} for mode in SP}
G5I = {i: inst_list(G5[i]) for i in IDS}
gtrows = []
for i in IDS:
    G = GTO[i]; boxes = G5BOX[i]
    for g, x in G.items():
        g5iou, g5k = best_match(x["mask"], G5I[i])
        bi = [biou(b, x["box"]) for b in boxes]; best = max(bi) if bi else 0.0
        if g5iou > 0.5: cat = "TP"
        elif best > 0.5: cat = "good_box_mask_fail"
        elif best > 0.1:
            bb = boxes[int(np.argmax(bi))]
            others = [y for gg, y in G.items() if gg != g]
            merged = any(inter(bb, y["box"]) >= 0.5 * area(y["box"]) for y in others)
            ra = area(bb) / max(1, area(x["box"]))
            cat = "box_mislocated:" + ("merged_with_neighbour" if merged else ("oversized" if ra > 2 else ("partial" if ra < 0.5 else "offset")))
        else:
            cat = "no_box:" + ("edge_sliver<100px" if (x["edge"] and x["area"] < 100) else ("small<186px" if x["area"] < 186 else "other"))
        row = dict(id=i, gid=g, ds=DS[i], area=x["area"], edge=x["edge"], g5_cat=cat, g5_iou=round(g5iou, 4), g5_best_box_iou=round(best, 4))
        for mode in SP:
            v, _ = best_match(x["mask"], SPI[mode][i]); row[f"spec_{mode}_iou"] = round(v, 4)
        gtrows.append(row)
assert sum(r["g5_cat"] != "TP" for r in gtrows) == res["scores"]["G5-final"]["ALL"]["fn"]


def frac_table(rs, mode):
    out = {}
    groups = defaultdict(list)
    for r in rs:
        for key in ("ALL", r["ds"]):
            groups[(key, "ALL")].append(r); groups[(key, r["g5_cat"])].append(r)
            groups[(key, r["g5_cat"].split(":")[0] + ":*")].append(r)
    for (ds, cat), v in sorted(groups.items()):
        n = len(v); h = sum(r[f"spec_{mode}_iou"] > 0.5 for r in v)
        out.setdefault(ds, {})[cat] = dict(n=n, specialist_found=h, frac=round(h / n, 3) if n else None)
    return out


fn = [r for r in gtrows if r["g5_cat"] != "TP"]
tp = [r for r in gtrows if r["g5_cat"] == "TP"]
res["g5_fn_found_by_specialist"] = {m: frac_table(fn, m) for m in SP}
res["g5_tp_also_found_by_specialist"] = {m: {ds: dict(n=len(v), found=sum(r[f"spec_{m}_iou"] > 0.5 for r in v))
                                             for ds in ("ALL", "monuseg", "consep", "puma")
                                             for v in [[r for r in tp if ds == "ALL" or r["ds"] == ds]]} for m in SP}

# ---------------- set A (G7 MoNuSeg-val no_box|other, 41 nuclei)
A = json.load(open(GAP / "gallery/index.json"))["A"]
lookup = {(r["id"], r["gid"]): r for r in gtrows}
arows = []
for a in A:
    r = lookup[(a["id"], a["gid"])]  # KeyError would mean set A is not inside the 170 suite
    assert r["area"] == a["area"]
    arows.append(dict(idx=a["idx"], id=a["id"], gid=a["gid"], area=a["area"], edge=a["edge"], g5_cat=r["g5_cat"],
                      spec_context_iou=r["spec_context_iou"], spec_crop_iou=r["spec_crop_iou"]))
res["setA_g7_nobox_other"] = dict(
    n=len(arows),
    specialist_context_found=sum(r["spec_context_iou"] > 0.5 for r in arows),
    specialist_crop_found=sum(r["spec_crop_iou"] > 0.5 for r in arows),
    g5_final_found=sum(r["g5_cat"] == "TP" for r in arows),
    specialist_context_found_among_g5_still_missed=[sum(r["spec_context_iou"] > 0.5 for r in arows if r["g5_cat"] != "TP"),
                                                    sum(r["g5_cat"] != "TP" for r in arows)],
    spec_context_iou_hist={b: sum(lo < r["spec_context_iou"] <= hi for r in arows) for b, (lo, hi) in
                           {"0": (-1, 0.0), "(0,0.3]": (0.0, 0.3), "(0.3,0.5]": (0.3, 0.5), "(0.5,0.7]": (0.5, 0.7), ">0.7": (0.7, 2)}.items()},
    rows=arows)


# ---------------- oracle-selection ceilings and a non-oracle fusion
def paint(masks, shape):
    lab = np.zeros(shape, np.int32)
    for k, m in enumerate(masks, 1): lab[m & (lab == 0)] = k  # earlier masks win overlaps
    return lab


def oracle(mode, keep_g5_fp, sources=("g5", "spec")):
    pt = []
    for i in IDS:
        G = GTO[i]; cands = []
        if "g5" in sources: cands += [("g5", k, m) for k, m in G5I[i].items()]
        if "spec" in sources: cands += [("spec", k, m) for k, m in SPI[mode][i].items()]
        # IoU matrix candidates x GT; pick per GT the best candidate with IoU > 0.5 (one-to-one, greedy by IoU)
        pairs = []
        for ci, (_, _, m) in enumerate(cands):
            for g, x in G.items():
                it = np.count_nonzero(m & x["mask"])
                if it: pairs.append((it / np.count_nonzero(m | x["mask"]), ci, g))
        used_c, used_g, chosen = set(), set(), []
        for v, ci, g in sorted(pairs, reverse=True):
            if v <= 0.5: break
            if ci in used_c or g in used_g: continue
            used_c.add(ci); used_g.add(g); chosen.append(ci)
        masks = [cands[c][2] for c in sorted(chosen, key=lambda c: -max(p[0] for p in pairs if p[1] == c))]
        if keep_g5_fp:  # G5 instances that match no GT stay in the output (selector cannot drop G5 false positives)
            g5_matched = {cands[c][1] for c in chosen if cands[c][0] == "g5"}
            g5_unmatched = [m for k, m in G5I[i].items() if k not in g5_matched and
                            best_match(m, {g: x["mask"] for g, x in G.items()})[0] <= 0.5]
            masks = masks + g5_unmatched  # painted after the chosen masks, so they never overwrite them
        m = instance_metrics(paint(masks, GT[i].shape), GT[i]); m["ds"] = DS[i]; m["id"] = i; pt.append(m)
    return pt


def fill_in(mode):
    """G5 output unchanged; add only specialist instances that match a G5-missed GT (IoU>0.5), painted under G5."""
    pt = []
    for i in IDS:
        G = GTO[i]; masks = [m for _, m in sorted(G5I[i].items())]
        g5hit = {g for g, x in G.items() if best_match(x["mask"], G5I[i])[0] > 0.5}
        for g, x in G.items():
            if g in g5hit: continue
            v, k = best_match(x["mask"], SPI[mode][i])
            if v > 0.5: masks.append(SPI[mode][i][k])
        m = instance_metrics(paint(masks, GT[i].shape), GT[i]); m["ds"] = DS[i]; m["id"] = i; pt.append(m)
    return pt


def naive_union(mode, max_overlap=0.2):
    """Non-oracle fusion: G5 output + every specialist instance whose overlap with G5 foreground is <= 20% of its area."""
    pt = []
    for i in IDS:
        fg = G5[i] > 0; masks = [m for _, m in sorted(G5I[i].items())]
        for k, m in SPI[mode][i].items():
            if np.count_nonzero(m & fg) <= max_overlap * np.count_nonzero(m): masks.append(m)
        m = instance_metrics(paint(masks, GT[i].shape), GT[i]); m["ds"] = DS[i]; m["id"] = i; pt.append(m)
    return pt


for mode in ("context", "crop"):
    res["scores"][f"oracle_union[{mode}] keepG5FP"] = summarise(oracle(mode, True))
    res["scores"][f"oracle_union[{mode}] FP=0"] = summarise(oracle(mode, False))
    res["scores"][f"G5+specialist_fill_FN_only[{mode}]"] = summarise(fill_in(mode))
    res["scores"][f"naive_union_nonoracle[{mode}]"] = summarise(naive_union(mode))
res["scores"]["oracle_G5_only FP=0"] = summarise(oracle("context", False, sources=("g5",)))
res["scores"]["oracle_specialist_only[context] FP=0"] = summarise(oracle("context", False, sources=("spec",)))

res["method"] = {
    "pq": "instance_metrics per tile (strict mask IoU>0.5, PQ = sum matched IoU / (TP + FP/2 + FN/2)); PQ reported = mean over tiles (same as eval runner mean_pq); TP/FP/FN pooled.",
    "g5_categories": "per GT not matched by G5 mask: best G5 box IoU (any box vs GT bbox) >0.5 good_box_mask_fail; 0.1-0.5 box_mislocated (merged/oversized/partial/offset as rebin.py); <=0.1 no_box (edge_sliver = truncated & <100 px; small <186 px; other) -- same thresholds as gap-analysis/val_gap.py and gallery.py set A.",
    "specialist_found": "max mask IoU of any specialist instance (crop-relabelled) with the GT nucleus > 0.5.",
    "oracle_union": "candidates = G5-final instances (resolved label map) U specialist instances; per tile greedy one-to-one assignment of candidates to GT by descending mask IoU, only IoU>0.5; chosen masks painted into one label map (higher-IoU first wins any overlap pixel) and rescored with instance_metrics. 'FP=0' drops every unmatched candidate (perfect selector). 'keepG5FP' additionally keeps G5's unmatched instances (painted underneath), i.e. the selector can choose per nucleus but cannot remove G5's own FPs.",
    "fill_FN_only": "G5 map kept as is; for each GT G5 missed, add the specialist instance with IoU>0.5 (painted under G5 pixels).",
    "naive_union": "non-oracle: G5 + every specialist instance overlapping G5 foreground by <=20% of its own area (no GT used).",
}
res["gt_rows"] = gtrows
json.dump(res, open(OUTJ, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for k, v in res["scores"].items():
    a = v["ALL"]
    print(f"{k:42s} PQ {a['pq']:.4f}  TP/FP/FN {a['tp']}/{a['fp']}/{a['fn']}  R {a['recall_pooled']:.3f} P {a['precision_pooled']:.3f} | " +
          "  ".join(f"{d[:3]} {v[d]['pq']:.3f} ({v[d]['tp']}/{v[d]['fp']}/{v[d]['fn']})" for d in ("monuseg", "consep", "puma")))
print("setA", {k: v for k, v in res["setA_g7_nobox_other"].items() if k != "rows"})
for mode in SP:
    print("G5 FN found by specialist", mode)
    for ds, t in res["g5_fn_found_by_specialist"][mode].items():
        print("  ", ds, {c: f"{x['specialist_found']}/{x['n']}" for c, x in t.items()})

"""Read-only: never-found GT nuclei across GRPO samples (box IoU>0.5), per run/dataset/phase.
For G>8 runs also reports the expected never-found rate of a random 8-sample subset (exact hypergeometric)."""
import json, sys
from collections import defaultdict
from math import comb
from pathlib import Path
import numpy as np
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/conny/Desktop/RLVR/grpo-worktree-20261020")
from nucleus_rl.sft_data import apply_d4  # noqa

ROOT = Path("/Users/conny/Desktop/RLVR/S-Seg-RLVR-")
TM = ROOT / "data/trustmix_v2"
MAN = {}
for f in ("train.jsonl",):
    for l in (TM / f).read_text().splitlines():
        r = json.loads(l); MAN[r["id"]] = r
for D in ("monuseg_official30_grpo_v1",):
    for l in (ROOT / "data" / D / "train.jsonl").read_text().splitlines():
        r = json.loads(l); r.setdefault("dataset", "monuseg"); r["_base"] = ROOT / "data" / D; MAN.setdefault(r["id"], r)
CACHE = {}

def gt(cid, t):
    k = (cid, t)
    if k not in CACHE:
        m = MAN[cid]; base = m.get("_base", TM)
        mask = np.asarray(Image.open((base / m["mask_path"]).resolve())).astype(np.int32)
        _, lab = apply_d4(np.zeros((*mask.shape, 3), np.uint8), mask, int(t))
        trunc = set(m.get("truncated_instance_ids", []))
        bs = []
        for i in np.unique(lab[lab > 0]):
            yx = np.argwhere(lab == i)
            bs.append(dict(box=[yx[:, 1].min(), yx[:, 0].min(), yx[:, 1].max() + 1, yx[:, 0].max() + 1], area=len(yx),
                           sliver=(int(i) in trunc and len(yx) < 100), gid=int(i)))
        CACHE[k] = (bs, mask.shape[0])
    return CACHE[k]

def biou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

def match(pred, g, thr=0.5):
    pairs = sorted(((biou(p, x["box"]), i, j) for i, p in enumerate(pred) for j, x in enumerate(g)), reverse=True)
    ua, ub = set(), set()
    for v, i, j in pairs:
        if v <= thr: break
        if i not in ua and j not in ub: ua.add(i); ub.add(j)
    return ub

def parse(text, size):
    try:
        body = text.split("<answer>", 1)[1].rsplit("</answer>", 1)[0]
        return [[v * size / 1000 for v in o["box"]] for o in json.loads(body)["objects"]]
    except Exception:
        return None

def analyse(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    groups = defaultdict(list)
    for r in rows:
        if r.get("gt_member"): continue  # ground-truth answers are not policy samples
        groups[(int(r["step"]), r["id"], int(r["transform"]))].append(r)
    nsteps = max(k[0] for k in groups)
    acc = defaultdict(lambda: defaultdict(float)); never_list = []
    for (step, cid, t), g in sorted(groups.items()):
        G, size = gt(cid, t); n = len(g)
        cnt = np.zeros(len(G)); tpm = 0
        for s in g:
            b = parse(s["text"], size) if s.get("valid") else None
            if not b: continue
            ms = match(b, G); tpm += len(ms)
            for j in ms: cnt[j] += 1
        ph = "early" if step <= nsteps / 3 else ("late" if step > 2 * nsteps / 3 else "mid")
        ds = MAN[cid].get("dataset", "monuseg")
        p8 = [comb(n - int(c), 8) / comb(n, 8) if n > 8 else float(c == 0) for c in cnt]
        for key in ((ds, ph), (ds, "all"), ("ALL", ph), ("ALL", "all")):
            a = acc[key]
            a["gt"] += len(G); a["groups"] += 1; a["G"] = n
            a["never"] += float((cnt == 0).sum()); a["never_nosliver"] += float(sum(c == 0 and not x["sliver"] for c, x in zip(cnt, G)))
            a["gt_nosliver"] += float(sum(not x["sliver"] for x in G))
            a["never_at8_expected"] += float(sum(p8)); a["all_found"] += float((cnt == n).sum())
            a["mean_sample_recall_num"] += tpm / n
        for c, x in zip(cnt, G):
            if c == 0: never_list.append(dict(step=step, id=cid, t=t, gid=x["gid"], area=x["area"], sliver=x["sliver"], ds=ds, phase=ph))
    out = {}
    for (ds, ph), a in acc.items():
        out[f"{ds}|{ph}"] = dict(groups=int(a["groups"]), G=int(a["G"]), gt=int(a["gt"]),
            never=round(a["never"] / a["gt"], 3), never_at8_expected=round(a["never_at8_expected"] / a["gt"], 3),
            never_excl_slivers=round(a["never_nosliver"] / max(1, a["gt_nosliver"]), 3),
            all_found=round(a["all_found"] / a["gt"], 3), mean_sample_recall=round(a["mean_sample_recall_num"] / a["gt"], 3))
    return out, never_list

if __name__ == "__main__":
    RUNS = {
        "G7 (H2 init, pq, G8x2)": ROOT / "runs/grpo_20261020/g7-27b448c128-pq/samples.jsonl",
        "G7r (seed43)": ROOT / "runs/grpo_20261020/g7r-27b448c128-pq-seed43/samples.jsonl",
        "G2 (S2@256, pq, G8x2)": ROOT / "runs/trustmix_20261104/g2-27b448-trustmix-pq-from-s2c256/samples.jsonl",
        "G4 (S4@256, pq, G8x2, s43)": ROOT / "runs/trustmix_20261104/g4-27b448-trustmix-pq-from-s4c256-seed43/samples.jsonl",
        "G5 (S2@256, wpq .3/.7, G8x2)": ROOT / "runs/trustmix_20261104/g5-27b448-trustmix-wpq-fp03-fn07-from-s2c256/samples.jsonl",
        "G6 (S2@256, pq, G16x1)": ROOT / "runs/trustmix_20261104/g6-27b448-trustmix-pq-g16x1-from-s2c256/samples.jsonl",
    }
    extra = dict(a.split("=", 1) for a in sys.argv[1:])
    RUNS.update({k: Path(v) for k, v in extra.items()})
    res, nev = {}, {}
    for k, p in RUNS.items():
        if not p.exists(): print("missing", k, p); continue
        res[k], nev[k] = analyse(p); print("done", k, file=sys.stderr)
    json.dump(res, open("never_found.json", "w"), indent=1)
    json.dump(nev, open("never_found_list.json", "w"))
    for k, r in res.items():
        print("\n==", k)
        for kk in sorted(r):
            print(f"  {kk:18s}", r[kk])

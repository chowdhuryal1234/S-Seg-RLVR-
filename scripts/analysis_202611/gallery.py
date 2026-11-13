"""Read-only: contact sheets of missed nuclei. Sets:
 A  G7 final greedy on MoNuSeg val 64: every 'no_box|other' FN (not sliver, area>=186)
 B  G7 training never-found (non-sliver), sampled across tiles
 C  G5 (trustmix wpq) training never-found (non-sliver), stratified CoNSeP/PUMA/MoNuSeg
Panel: left = whole 128px tile x3 with target GT outline (yellow), other GT (green thin), predicted boxes (cyan; for B/C the union of all group samples);
right = 4x zoom around the target, raw (no overlay) with a small corner tick."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/conny/Desktop/RLVR/grpo-worktree-20261020")
from nucleus_rl.sft_data import apply_d4  # noqa
import never_found as NF

ROOT = Path("/Users/conny/Desktop/RLVR/S-Seg-RLVR-")
OUT = Path("gallery"); OUT.mkdir(exist_ok=True)
MV = ROOT / "data/monuseg_official30_grpo_v1"
VAL = {json.loads(l)["id"]: json.loads(l) for l in (MV / "validation.jsonl").read_text().splitlines()}
FONT = ImageFont.load_default()

def load(rec, base, t=0):
    img = Image.open((base / rec["image_path"]).resolve()).convert("RGB")
    mask = np.asarray(Image.open((base / rec["mask_path"]).resolve())).astype(np.int32)
    img, lab = apply_d4(np.asarray(img), mask, int(t))
    return np.asarray(img), lab

def outline(lab, gid):
    m = lab == gid
    e = m & ~(np.pad(m, 1)[2:, 1:-1] & np.pad(m, 1)[:-2, 1:-1] & np.pad(m, 1)[1:-1, 2:] & np.pad(m, 1)[1:-1, :-2])
    return e

def panel(img, lab, gid, boxes, title):
    S = 3; H = img.shape[0]
    big = np.kron(img, np.ones((S, S, 1), np.uint8)).copy()
    for g in np.unique(lab[lab > 0]):
        e = np.kron(outline(lab, g), np.ones((S, S), bool))
        if g == gid:
            # thicken
            e2 = e | np.roll(e, 1, 0) | np.roll(e, 1, 1)
            big[e2] = [255, 230, 0]
        else:
            big[e] = [0, 200, 0]
    pi = Image.fromarray(big); d = ImageDraw.Draw(pi)
    for b in boxes:
        d.rectangle([b[0] * S, b[1] * S, b[2] * S, b[3] * S], outline=(0, 220, 255), width=1)
    yx = np.argwhere(lab == gid); cy, cx = yx.mean(0).astype(int)
    r = 20; y0, x0 = max(0, cy - r), max(0, cx - r); y1, x1 = min(H, cy + r), min(H, cx + r)
    z = Image.fromarray(img[y0:y1, x0:x1]).resize(((x1 - x0) * 8, (y1 - y0) * 8), Image.NEAREST)
    zz = Image.new("RGB", (r * 2 * 8, r * 2 * 8), (40, 40, 40)); zz.paste(z, (0, 0))
    zd = ImageDraw.Draw(zz)
    # tick marks at the nucleus centroid position (edges only, no overlay on nucleus)
    tx, ty = (cx - x0) * 8, (cy - y0) * 8
    zd.line([tx, 0, tx, 12], fill=(255, 230, 0), width=3); zd.line([0, ty, 12, ty], fill=(255, 230, 0), width=3)
    W = pi.width + zz.width + 10
    canvas = Image.new("RGB", (W, max(pi.height, zz.height) + 18), (255, 255, 255))
    canvas.paste(pi, (0, 18)); canvas.paste(zz.resize((pi.height, pi.height)), (pi.width + 10, 18))
    ImageDraw.Draw(canvas).text((2, 2), title, fill=(0, 0, 0), font=FONT)
    return canvas

def sheets(name, panels, per=8, cols=2):
    meta = []
    for k in range(0, len(panels), per):
        chunk = panels[k:k + per]
        w, h = chunk[0][0].size
        rows = (len(chunk) + cols - 1) // cols
        sh = Image.new("RGB", (cols * (w + 12), rows * (h + 12)), (255, 255, 255))
        for i, (p, m) in enumerate(chunk):
            sh.paste(p, ((i % cols) * (w + 12), (i // cols) * (h + 12)))
        fn = OUT / f"{name}_{k // per + 1:02d}.png"; sh.save(fn)
        for i, (_, m) in enumerate(chunk):
            m["sheet"] = fn.name; m["slot"] = i + 1; meta.append(m)
    return meta

random.seed(0)
allmeta = {}
# ---- Set A
src = ROOT / "runs/grpo_20261020/g7-27b448c128-pq/after/completions.jsonl"
recs = {r["id"]: r for r in map(json.loads, src.read_text().splitlines())}
pa = []
for cid, r in sorted(recs.items()):
    m = VAL[cid]; img, lab = load(m, MV)
    trunc = set(m.get("truncated_instance_ids", []))
    boxes = [o["box"] for o in r["converted_objects"]]
    for gid in np.unique(lab[lab > 0]):
        yx = np.argwhere(lab == gid); area = len(yx)
        gb = [yx[:, 1].min(), yx[:, 0].min(), yx[:, 1].max() + 1, yx[:, 0].max() + 1]
        best = max([NF.biou(b, gb) for b in boxes] or [0])
        sliver = int(gid) in trunc and area < 100
        if best <= 0.1 and not sliver and area >= 186:
            title = f"A{len(pa)+1} {cid[:23]} g{gid} a={area} edge={int(int(gid) in trunc)} n={len(np.unique(lab[lab>0]))}"
            pa.append((panel(img, lab, gid, boxes, title), dict(set="A", idx=len(pa) + 1, id=cid, gid=int(gid), area=area,
                        edge=int(gid) in trunc, n_gt=int(len(np.unique(lab[lab > 0]))), n_pred=len(boxes))))
print("set A panels", len(pa))
allmeta["A"] = sheets("A_g7val_nobox", pa)

# ---- Sets B, C from training never-found
def train_set(label, path, strata, n_per, prefix):
    _, nev = NF.analyse(path)
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    grp = defaultdict(list)
    for r in rows:
        if r.get("gt_member"): continue  # ground-truth answers are not policy samples
        grp[(int(r["step"]), r["id"], int(r["transform"]))].append(r)
    cand = [x for x in nev if not x["sliver"]]
    out = []
    seen_tiles = set()
    for ds in strata:
        c = [x for x in cand if x["ds"] == ds]; random.shuffle(c)
        picked = []
        for x in c:
            if (x["id"], x["gid"]) in seen_tiles: continue
            seen_tiles.add((x["id"], x["gid"])); picked.append(x)
            if len(picked) >= n_per: break
        for x in picked:
            m = NF.MAN[x["id"]]; base = m.get("_base", NF.TM)
            img, lab = load(m, base, x["t"])
            G, size = NF.gt(x["id"], x["t"])
            bx = []
            for s in grp[(x["step"], x["id"], x["t"])]:
                b = NF.parse(s["text"], size) if s.get("valid") else None
                bx += b or []
            k = len(out) + 1
            title = f"{prefix}{k} {ds} {x['id'][:20]} g{x['gid']} a={x['area']} step{x['step']}"
            out.append((panel(img, lab, x["gid"], bx, title), dict(set=prefix, idx=k, ds=ds, id=x["id"], gid=x["gid"], t=x["t"],
                        step=x["step"], area=x["area"], n_gt=len(G), n_pred_boxes_union=len(bx))))
    print(label, "panels", len(out))
    return out

pb = train_set("B G7 train", NF.ROOT / "runs/grpo_20261020/g7-27b448c128-pq/samples.jsonl", ["monuseg"], 40, "B")
allmeta["B"] = sheets("B_g7train_never", pb)
pc = train_set("C G5 train", NF.ROOT / "runs/trustmix_20261104/g5-27b448-trustmix-wpq-fp03-fn07-from-s2c256/samples.jsonl",
               ["consep", "puma", "monuseg"], 16, "C")
allmeta["C"] = sheets("C_g5train_never", pc)
json.dump(allmeta, open(OUT / "index.json", "w"), indent=1)

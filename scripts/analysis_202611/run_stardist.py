"""[StarDist variant of run_instanseg.py] Run StarDist 2D_versatile_he (stardist-models v0.1) on the 170-tile trustmix_v2 validation suite.

Two modes, both written as int32 label maps (128x128, one per tile, same order as validation.jsonl):
  context: run on the whole validation source image (MoNuSeg/CoNSeP 1000x1000, PUMA 1024x1024), then crop to the tile.
           Instances cut by the crop keep their visible part (same convention as our GT masks).
  crop:    run on the 128x128 tile alone.
StarDist 2D_versatile_he was trained at native MoNuSeg 40x; we pass scale = um_per_px / 0.25:
  MoNuSeg 0.25 (40x, manifest says 'not_verified'), CoNSeP 0.25 (manifest), PUMA 0.2269 (manifest).
Before running, every crop of the source image is checked pixel-exact against the tile PNG used by our pipeline.
Validation source images only (official test archives/members are never opened).
Usage: <venv python> -I run_stardist.py <out_dir> <models_dir>
"""
import io, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path(sys.argv[1]); MODELS = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
import tifffile  # noqa: E402
from stardist.models import StarDist2D  # noqa: E402
from csbdeep.utils import normalize  # noqa: E402

ROOT = Path("/Users/conny/Desktop/RLVR/S-Seg-RLVR-/data/trustmix_v2")
MONUSEG_ZIP = "/Users/conny/Desktop/RLVR/RLVR/data/pathology/monuseg/MoNuSeg_2018_Training_data.zip"
CONSEP_ZIP = "/Users/conny/Desktop/RLVR/RLVR/data/pathology/consep/CoNSeP/data/consep.zip"
PUMA_DIR = Path("/Users/conny/Desktop/RLVR/dataset-survey/data/puma/x/01_training_dataset_tif_ROIs")
PX = {"monuseg": 0.25, "consep": 0.25, "puma": 0.2269}
PX_SCALE = float(os.environ.get("PX_SCALE", "1"))  # sensitivity check only; headline run uses 1
PX = {k: v * PX_SCALE for k, v in PX.items()}

rows = [json.loads(l) for l in (ROOT / "validation.jsonl").read_text().splitlines()]
assert len(rows) == 170 and all(r["split"] == "validation" for r in rows)


def source_image(r):
    ds = r["dataset"]
    if ds == "monuseg":
        mem = r["source_image_member"]; assert mem.startswith("MoNuSeg 2018 Training Data/")
        with zipfile.ZipFile(MONUSEG_ZIP) as z:
            a = np.asarray(Image.open(io.BytesIO(z.read(mem))).convert("RGB"))
    elif ds == "consep":
        mem = r["source_image_member"]; assert mem.startswith("CoNSeP/Train/Images/")
        with zipfile.ZipFile(CONSEP_ZIP) as z:
            a = np.asarray(Image.open(io.BytesIO(z.read(mem))))
    else:
        a = tifffile.imread(PUMA_DIR / r["source_image_member"])
    return np.ascontiguousarray(a[..., :3]).astype(np.uint8)


def tile_image(r):
    return np.asarray(Image.open((ROOT / r["image_path"]).resolve()).convert("RGB"))


def to_labels(out):
    lab = out[0] if isinstance(out, tuple) else out
    lab = lab.detach().cpu().numpy()
    while lab.ndim > 2: lab = lab[0]
    return lab.astype(np.int64)


def relabel(lab):
    ids = np.unique(lab[lab > 0])
    res = np.zeros(lab.shape, np.int32)
    for k, i in enumerate(ids, 1): res[lab == i] = k
    return res



model = StarDist2D(None, name="2D_versatile_he", basedir=str(MODELS))
meta = dict(model="StarDist 2D_versatile_he (stardist-models v0.1, default thresholds)", pixel_sizes=PX, tiles=[])
srcs = {}
for r in rows: srcs.setdefault(r["source_image_id"], []).append(r)
ctx_maps, crop_maps = {}, {}
t0 = time.time()
for sid, rs in sorted(srcs.items()):
    img = source_image(rs[0]); ds = rs[0]["dataset"]; sc = PX[ds] / 0.25
    for r in rs:
        x0, y0, x1, y1 = r["bbox_xyxy"]
        if not np.array_equal(img[y0:y1, x0:x1], tile_image(r)):
            raise SystemExit(f"crop mismatch {r['id']}")
    full, _ = model.predict_instances(normalize(img, 1, 99.8, axis=(0, 1, 2)), scale=sc if abs(sc - 1) > 1e-3 else None)
    full = np.asarray(full).astype(np.int64)
    assert full.shape == img.shape[:2], (full.shape, img.shape)
    np.save(OUT / f"full_{sid}.npy", full.astype(np.int32))
    for r in rs:
        x0, y0, x1, y1 = r["bbox_xyxy"]
        ctx_maps[r["id"]] = relabel(full[y0:y1, x0:x1])
        cm, _ = model.predict_instances(normalize(tile_image(r), 1, 99.8, axis=(0, 1, 2)), scale=sc if abs(sc - 1) > 1e-3 else None)
        crop_maps[r["id"]] = relabel(np.asarray(cm))
    print(f"{sid} ({ds}, {len(rs)} tiles) full {img.shape} -> {len(np.unique(full)) - 1} nuclei  [{time.time() - t0:.0f}s]", flush=True)

ids = [r["id"] for r in rows]
np.savez_compressed(OUT / "context_maps.npz", ids=np.array(ids), maps=np.stack([ctx_maps[i] for i in ids]))
np.savez_compressed(OUT / "crop_maps.npz", ids=np.array(ids), maps=np.stack([crop_maps[i] for i in ids]))
meta["seconds"] = round(time.time() - t0, 1)
json.dump(meta, open(OUT / "run_meta.json", "w"), indent=1)
print("done", meta["seconds"], "s")

# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "numpy==2.2.6",
#   "Pillow==11.3.0",
# ]
# ///
# MAGIC %pip install numpy==2.2.6 Pillow==11.3.0

# COMMAND ----------

spark.sql("CREATE VOLUME IF NOT EXISTS aistudio_dev_catalog.aios_segrlvr_dev.processed")

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ---------- Configuration ----------
REPO_DIR = "/Workspace/Users/xukan_zhang@dfci.harvard.edu//S-Seg-RLVR-"
TRAIN_DIR = "/Volumes/aistudio_dev_catalog/aios_segrlvr_dev/raw/monuseg/train"
OUT_DIR = "/Volumes/aistudio_dev_catalog/aios_segrlvr_dev/processed/nuclei/monuseg"

# COMMAND ----------

import sys, json, hashlib
from pathlib import Path
import numpy, PIL

scripts_dir = str(Path(REPO_DIR) / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
import audit_monuseg, prepare_monuseg

pinned = dict(line.strip().split("==") for line in
              (Path(REPO_DIR) / "requirements-cpu.txt").read_text().splitlines() if "==" in line)
installed = {"numpy": numpy.__version__, "Pillow": PIL.__version__}
print("pinned:", pinned, "| installed:", installed)
assert all(installed[k] == v for k, v in pinned.items()), "different version"

# COMMAND ----------

class FolderArchive:
    def __init__(self, root, mode="r"):
        if mode != "r":
            raise ValueError("FolderArchive is read-only")
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Not a directory: {self.root}")
        self._names = sorted(p.relative_to(self.root).as_posix()
                             for p in self.root.rglob("*") if p.is_file())
        self._name_set = set(self._names)

    def namelist(self):
        return list(self._names)

    def read(self, name):
        if name not in self._name_set:
            raise KeyError(name)
        return (self.root / name).read_bytes()

    def testzip(self):          
        return None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

_ORIGINAL_ZIPFILE = prepare_monuseg.ZipFile

def use_zip_reader(reader):
    audit_monuseg.ZipFile = reader
    prepare_monuseg.ZipFile = reader

# COMMAND ----------

import tempfile, zipfile
from PIL import Image

def _make_synthetic(root):
    rng = numpy.random.default_rng(0)
    d = root / "Training"
    (d / "Tissue Images").mkdir(parents=True); (d / "Annotations").mkdir(parents=True)
    for pid in ["TCGA-AA-0001", "TCGA-BB-0002", "TCGA-CC-0003"]:
        stem = f"{pid}-01Z-00-DX1"
        Image.fromarray((rng.random((256, 256, 3)) * 255).astype(numpy.uint8)).save(d / "Tissue Images" / f"{stem}.tif")
        regions = []
        for k in range(12):
            x, y = rng.uniform(5, 240, 2); r = rng.uniform(3, 9)
            verts = "".join(f'<Vertex X="{x + r*numpy.cos(t):.3f}" Y="{y + r*numpy.sin(t):.3f}"/>'
                            for t in numpy.linspace(0, 2 * numpy.pi, 9)[:-1])
            regions.append(f'<Region Id="{k+1}" NegativeROA="0"><Vertices>{verts}</Vertices></Region>')
        (d / "Annotations" / f"{stem}.xml").write_text(
            f"<Annotations><Annotation><Regions>{''.join(regions)}</Regions></Annotation></Annotations>")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp); folder = tmp / "extracted"; _make_synthetic(folder)
    zpath = tmp / "synthetic_train.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder).as_posix())
    args = ["--border-policy", "visible", "--train-patient-ids", "TCGA-AA-0001,TCGA-BB-0002",
            "--val-patient-ids", "TCGA-CC-0003", "--train-crops", "3", "--val-crops", "1"]
    use_zip_reader(_ORIGINAL_ZIPFILE); prepare_monuseg.main(["--train-archive", str(zpath), "--out", str(tmp / "a")] + args)
    use_zip_reader(FolderArchive);     prepare_monuseg.main(["--train-archive", str(folder), "--out", str(tmp / "b")] + args)
    a, b = tmp / "a", tmp / "b"
    files = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    for f in files:
        if f.endswith(".png"):
            assert (a / f).read_bytes() == (b / f).read_bytes(), f
        if f.endswith(".jsonl"):
            assert ((a / f).read_text().replace(str(zpath.resolve()), "<SRC>")
                    == (b / f).read_text().replace(str(folder.resolve()), "<SRC>")), f
print("check passed")

# COMMAND ----------

use_zip_reader(FolderArchive)
audit = audit_monuseg.audit_archive(TRAIN_DIR)
print("Paired count:", audit["pair_count"], "| Patient Count:", audit["patient_count"])
print("Pair validation passed:", audit["pair_validation_passed"])
print("Issue:", {k: v for k, v in audit["issues"].items() if v})
print("Images with negative ROA / degenerate regions:",
      [p["id"] for p in audit["pairs"] if p.get("negative_region_entries") or p.get("degenerate_region_entries")])

# COMMAND ----------

protocol = json.loads((Path(REPO_DIR) / "reports/2026-10-09/data_protocol.json").read_text())
argv = list(protocol["preparation_argv"][1:])          
argv[argv.index("--train-archive") + 1] = TRAIN_DIR      
argv[argv.index("--out") + 1] = OUT_DIR                  

use_zip_reader(FolderArchive)
audit = audit_monuseg.audit_archive(TRAIN_DIR, expected_images=30)
eligible = {p["patient_id"] for p in audit["pairs"]
            if not p.get("negative_region_entries") and not p.get("degenerate_region_entries")}

deviation = {"dataset_version": f"{audit['pair_count']}-image MoNuSeg training set (protocol was built on 37)",
             "kept": {}, "dropped": {}}
for flag, split in (("--train-patient-ids", "train"), ("--val-patient-ids", "validation")):
    ids = argv[argv.index(flag) + 1].split(",")
    kept = [i for i in ids if i in eligible]            
    deviation["kept"][split] = kept
    deviation["dropped"][split] = [i for i in ids if i not in eligible]
    argv[argv.index(flag) + 1] = ",".join(kept)

print(f"This data：{audit['pair_count']} image pairs, {len(eligible)} valid patients")
for split in ("train", "validation"):
    print(f"{split}: keep {len(deviation['kept'][split])} patients，remove {deviation['dropped'][split]}")
unassigned = sorted(eligible - set(deviation["kept"]["train"]) - set(deviation["kept"]["validation"]))
print("Unassigned patients:", unassigned)
deviation["unassigned_eligible_patients"] = unassigned
assert deviation["kept"]["train"] and deviation["kept"]["validation"], "no remaning patients in train or validation"

print("argv:", argv)
prepare_monuseg.main(argv)

# COMMAND ----------

src = FolderArchive(TRAIN_DIR)
manifest = []
for name in src.namelist():
    data = src.read(name)
    manifest.append({"member": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
(Path(OUT_DIR) / "source_folder_manifest.json").write_text(json.dumps(
    {"source_dir": TRAIN_DIR, "total_bytes": sum(m["bytes"] for m in manifest), "files": manifest}, indent=2) + "\n")

report_path = Path(OUT_DIR) / "preparation_report.json"
report = json.loads(report_path.read_text())
report["source_format"] = "extracted_folder read via FolderArchive (zipfile.ZipFile stand-in); see source_folder_manifest.json"
report["training_archive_audit"]["archive_bytes"] = sum(m["bytes"] for m in manifest)   
report["protocol_deviation"] = {**deviation, "final_argv": argv,
                                "note": "Patient assignments follow data_protocol.json intersected with this dataset."}
report_path.write_text(json.dumps(report, indent=2) + "\n")
print("Written to source_folder_manifest.json，format annotated in preparation_report.json.")

# COMMAND ----------

produced = {split: [json.loads(line) for line in (Path(OUT_DIR) / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "validation")}
for split in ("train", "validation"):
    expected = [c["id"] for c in protocol["selected_crops"][split]]
    got = [r["id"] for r in produced[split]]
    status = "same" if got == expected else ("same content but different sequence" if sorted(got) == sorted(expected) else "different")
    print(f"{split}: generate {len(got)} crop，previous {len(expected)} 个 → {status}")
    if status == "different":
        print("  generate only:", sorted(set(got) - set(expected)))
        print("  record only:", sorted(set(expected) - set(got)))

# COMMAND ----------

import matplotlib.pyplot as plt

rows = produced["train"][:4]
fig, axes = plt.subplots(2, len(rows), figsize=(4 * len(rows), 8))
for j, r in enumerate(rows):
    img = numpy.asarray(Image.open(Path(OUT_DIR) / r["image_path"]))
    mask = numpy.asarray(Image.open(Path(OUT_DIR) / r["mask_path"]))
    axes[0, j].imshow(img); axes[0, j].set_title(r["id"][-12:], fontsize=9)
    axes[1, j].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
    axes[1, j].set_title(f"{r['instance_count']} instances", fontsize=9)
    for ax in axes[:, j]:
        ax.axis("off")
plt.tight_layout(); plt.show()

# COMMAND ----------


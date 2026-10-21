# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Check the Serverless GPU environment
# MAGIC
# MAGIC Run this first on **Serverless GPU, 1×H100, environment Standard v6**. It installs nothing and changes nothing
# MAGIC except a small probe file on the Volume. It answers:
# MAGIC
# MAGIC 1. Which GPU, driver, Python, and how much local disk?
# MAGIC 2. **Can this compute reach Hugging Face, PyPI, GitHub, and the SAM2 checkpoint host?**
# MAGIC 3. Can it list, write and read the Unity Catalog Volume through the Files API (which all notebooks use)?
# MAGIC
# MAGIC If every network row is `OK`, notebooks 01–04 download what they need. If any row fails, see the
# MAGIC "offline" notes printed at the end and in `README.md`.

# COMMAND ----------

import os, platform, shutil, socket, ssl, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

sys.path.insert(0, os.getcwd())  # Databricks runs notebooks with their own folder as the working directory
import dbx_common as dbx

print("Python", platform.python_version(), "|", sys.executable)
print("Repository:", dbx.REPO)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                     capture_output=True, text=True).stdout or "nvidia-smi not found: attach Serverless GPU compute")
for path in ("/local_disk0", "/tmp", str(dbx.local_root())):
    if Path(path).exists():
        total, used, free = shutil.disk_usage(path)
        print(f"disk {path}: {free / 2**30:.0f} GiB free of {total / 2**30:.0f} GiB")
print("CPU cores:", os.cpu_count(), "| git available:", bool(shutil.which("git")))
print("Proxy variables:", {k: v for k, v in os.environ.items() if "proxy" in k.lower()} or "none")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Network reachability
# MAGIC
# MAGIC Each probe downloads at most a few bytes. Hugging Face model files are served from a CDN host after a redirect,
# MAGIC so the second Hugging Face row matters more than the first.

# COMMAND ----------

QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REV = dbx.sft_config()["models"]["7b"]["revision"]
PROBES = [
    ("Hugging Face API (model metadata)", f"https://huggingface.co/api/models/{QWEN}/revision/{QWEN_REV}"),
    ("Hugging Face file download (CDN)", f"https://huggingface.co/{QWEN}/resolve/{QWEN_REV}/model-00001-of-00005.safetensors"),
    ("PyPI index", "https://pypi.org/simple/transformers/"),
    ("PyPI files (torch wheel host)", "https://files.pythonhosted.org/"),
    ("GitHub source archive (SAM-2 package)", "https://github.com/facebookresearch/sam2/archive/2b90b9f5ceec907a1c18123530e92e794ad901a4.zip"),
    ("SAM2 checkpoint host", dbx.SAM_URL),
]


def probe(url, timeout=20):
    request = urllib.request.Request(url, headers={"Range": "bytes=0-15", "User-Agent": "segrlvr-env-check"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(16)
            final_host = urllib.parse.urlparse(response.geturl()).hostname
            return "OK", f"HTTP {response.status} via {final_host} in {time.time() - started:.1f}s"
    except urllib.error.HTTPError as error:
        # 401/403 from the real server still proves the network path works.
        status = "OK (reachable)" if error.code in (401, 403, 404, 416) else "FAIL"
        return status, f"HTTP {error.code} {error.reason}"
    except (urllib.error.URLError, socket.timeout, ssl.SSLError, ConnectionError) as error:
        return "FAIL", f"{type(error).__name__}: {getattr(error, 'reason', error)}"


results = {}
for name, url in PROBES:
    status, detail = probe(url)
    results[name] = status
    print(f"{status:15} {name:42} {detail}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volume access
# MAGIC
# MAGIC The notebooks use the Files API, which must pass. The `/Volumes` mount row is informational only: on Serverless
# MAGIC GPU it can fail to reach the storage account (`Input/output error`, `Resource temporarily unavailable`).

# COMMAND ----------

store, checks = dbx.volume(), {}
try:
    images = [rel for rel, _ in store.walk(dbx.PATHS["train_dir"])]
    checks["list training folder"] = (f"{sum(r.lower().endswith(('.tif', '.tiff')) for r in images)} images, "
                                      f"{sum(r.lower().endswith('.xml') for r in images)} annotations")
except Exception as error:
    checks["list training folder"] = f"FAIL {type(error).__name__}: {error}"
probe = dbx.remote("_env_probe", time.strftime("%Y%m%d-%H%M%S") + ".txt")
try:
    store.write_text(probe, "ok")
    checks["write+read artifacts"] = "OK" if store.read_text(probe) == "ok" else "FAIL content differs"
    store.files.delete(probe)
except Exception as error:
    checks["write+read artifacts"] = f"FAIL {type(error).__name__}: {error}"
try:
    os.listdir(dbx.PATHS["train_dir"])
    checks["/Volumes mount (info)"] = "OK"
except OSError as error:
    checks["/Volumes mount (info)"] = f"unavailable ({error.strerror}); not needed"
for name, value in checks.items():
    print(f"{name:24} {value}")
print("Volume artifacts root:", dbx.PATHS["artifacts"])

# COMMAND ----------

failed = [name for name, status in results.items() if status == "FAIL"]
if not failed:
    print("All network probes passed: run 01_setup_assets next; it installs packages and downloads Qwen and SAM2.")
else:
    print("Blocked:", failed)
    print("""
Offline fallback (do this from any machine with internet, then upload to the Volume, e.g. with the Catalog UI):
  * Hugging Face blocked -> on a connected machine run
        hf download Qwen/Qwen2.5-VL-7B-Instruct --revision <sha> --cache-dir hub
    and upload the folder  hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<sha>  (files, no symlinks)
    to  <artifacts>/cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<sha>.
  * PyPI or GitHub blocked -> build a wheelhouse on Linux x86_64 / Python 3.12:
        pip download -r requirements-databricks.txt -d wheels   (and: pip wheel <SAM-2 url> -w wheels)
    upload to <artifacts>/wheels, download it to local disk with dbx.volume().download_tree(...), and install with
    --no-index --find-links <local wheels folder>.
    Ask your workspace admin whether a PyPI mirror/proxy is configured first.
  * SAM2 host blocked -> upload sam2.1_hiera_tiny.pt to <artifacts>/checkpoints/.
See scripts/dbx/README.md.""")

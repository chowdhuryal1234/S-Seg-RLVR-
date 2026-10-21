# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Check the Serverless GPU environment
# MAGIC
# MAGIC Run this first on **Serverless GPU, 1×H100, environment Standard v6**. It installs nothing and changes nothing
# MAGIC except a small probe folder on the Volume. It answers:
# MAGIC
# MAGIC 1. Which GPU, driver, Python, and how much local disk?
# MAGIC 2. **Can this compute reach Hugging Face, PyPI, GitHub, and the SAM2 checkpoint host?**
# MAGIC 3. Can it write, rename, and symlink on the Unity Catalog Volume?
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
# MAGIC ## Volume behavior
# MAGIC
# MAGIC Runs write to local disk and are mirrored to the Volume, so a failed rename or symlink here is expected and harmless.
# MAGIC Write and read must succeed.

# COMMAND ----------

probe_dir = dbx.artifact("_env_probe", time.strftime("%Y%m%d-%H%M%S"))
checks = {}
try:
    probe_dir.mkdir(parents=True)
    (probe_dir / "a.txt").write_text("ok")
    checks["write+read"] = (probe_dir / "a.txt").read_text() == "ok"
    for name, action in (("file rename", lambda: (probe_dir / "a.txt").rename(probe_dir / "b.txt")),
                         ("directory rename", lambda: ((probe_dir / "d1").mkdir(), (probe_dir / "d1").rename(probe_dir / "d2"))),
                         ("symlink", lambda: os.symlink(probe_dir / "b.txt", probe_dir / "link"))):
        try:
            action(); checks[name] = True
        except OSError as error:
            checks[name] = f"not supported ({error.strerror})"
finally:
    shutil.rmtree(probe_dir, ignore_errors=True)
for name, value in checks.items():
    print(f"{name:18} {value}")
print("Volume artifacts root:", dbx.PATHS["artifacts"])
print("Training images folder exists:", dbx.PATHS["train_dir"].is_dir())

# COMMAND ----------

failed = [name for name, status in results.items() if status == "FAIL"]
if not failed:
    print("All network probes passed: run 01_setup_assets next; it downloads Qwen and SAM2 and mirrors them to the Volume.")
else:
    print("Blocked:", failed)
    print("""
Offline fallback (do this from any machine with internet, then upload to the Volume):
  * Hugging Face blocked -> on a connected machine run
        hf download Qwen/Qwen2.5-VL-7B-Instruct --revision <sha> --cache-dir hub
    and upload the folder  hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<sha>  (files, no symlinks)
    to  <artifacts>/cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<sha>.
  * PyPI or GitHub blocked -> build a wheelhouse on Linux x86_64 / Python 3.12:
        pip download -r requirements-databricks.txt -d wheels   (and: pip wheel <SAM-2 url> -w wheels)
    upload to <artifacts>/wheels and install with  --no-index --find-links <artifacts>/wheels.
    Ask your workspace admin whether a PyPI mirror/proxy is configured first.
  * SAM2 host blocked -> upload sam2.1_hiera_tiny.pt to <artifacts>/checkpoints/.
See scripts/dbx/README.md.""")

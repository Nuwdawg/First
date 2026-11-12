print("CODE_READY", len(list(root.rglob("*.py"))), flush=True)

def run(command, **kwargs):
    print("RUN", " ".join(str(c) for c in command), flush=True)
    return subprocess.run([str(c) for c in command], check=True, **kwargs)

T0 = time.time()
def hours():
    return round((time.time() - T0) / 3600, 3)

source = Path("/kaggle/input")
work = Path("/kaggle/working")
annotations = next(source.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
train_images = annotations.parent / "train_images"
test_images = next(p for p in source.rglob("test_images") if p.is_dir())
init = next(source.rglob("model_e040.pt"))
env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision", "torchaudio"])
run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "kernels", "pycocotools", "scipy"])
gpus = len([line for line in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if line.strip()])
print("GPUS", gpus, "INIT", init, flush=True)
if gpus < 1:
    raise SystemExit("no GPU in this session; nothing was trained")

TRAIN = "experiments.exp_173_quality_ema.src.train"
CACHE = "experiments.exp_202_target_ft.src.cache"
SELECT = "experiments.exp_202_target_ft.src.select"
FUSED = root / "experiments" / "exp_203_fused" / "fused_records.json"
common = ["--annotations", annotations, "--images", train_images, "--fp16", "--batch", "8", "--micro-batch", "2",
          "--checkpoint", "whole", "--msda", "hub", "--workers", "3", "--no-final-d4", "--sweep-processes", "2",
          "--probe-steps", "20", "--epoch-policy", "fixed", "--price-per-hour", "0", "--budget-usd", "0",
          "--fold", "train", "--quality-target", "none", "--ema", "0", "--init", init, "--records-json", FUSED,
          "--warmup", "30", "--seed", "2026"]
RUNS = [("ftC", ["--fused-repeat", "0", "--lr", "5e-5", "--epochs", "12", "--eval-every", "4"]),
        ("ftD", ["--fused-repeat", "3", "--lr", "5e-5", "--epochs", "3", "--eval-every", "1"])]

def gpu_env(gpu):
    return dict(env, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

def events(folder):
    path = Path(folder) / "events.log"
    found = []
    if path.exists():
        for line in path.read_text().splitlines():
            tag, _, payload = line.partition(" ")
            try:
                item = json.loads(payload) if payload else {}
            except ValueError:
                continue
            item["event"] = tag
            found.append(item)
    return found

def cache_command(checkpoint, output, val_modes, test_modes, extra=()):
    return [sys.executable, "-u", "-m", CACHE, "--weights", checkpoint, "--annotations", annotations,
            "--train-images", train_images, "--test-images", test_images, "--output", output,
            "--val-modes", val_modes, "--test-modes", test_modes, "--skip-parity", *extra]

# ---- smoke test of every stage, with the real initial weights, before any long run ----
smoke = Path("/tmp/smoke"); shutil.rmtree(smoke, ignore_errors=True); smoke.mkdir(parents=True)
tiny = ["--batch", "2", "--micro-batch", "1", "--max-steps", "3", "--eval-every", "1", "--eval-limit", "2",
        "--no-preload", "--workers", "0", "--sweep-processes", "1"]
name0, extra0 = RUNS[0]
code = subprocess.run([str(c) for c in [sys.executable, "-u", "-m", TRAIN, "--output", smoke / "ft"] + common + extra0 + tiny],
                      cwd=root, env=gpu_env(0)).returncode
reports = sorted(p.name for p in (smoke / "ft").glob("*_sweep.json"))
print("SMOKE_TRAIN", code, reports, sorted(p.name for p in (smoke / "ft").glob("*")), flush=True)
if code != 0 or not reports:
    raise SystemExit("smoke training failed; the long runs were not started")
smoke_caches = sorted((smoke / "ft").glob("e[0-9][0-9][0-9]_cache.json"))
code = subprocess.run([str(c) for c in [sys.executable, "-u", "-m", SELECT, "--annotations", annotations, "--caches", *smoke_caches,
                       "--output", smoke / "selection.json", "--processes", "1"]], cwd=root, env=env).returncode
print("SMOKE_SELECT", code, flush=True)
if code != 0:
    raise SystemExit("smoke selection failed; the long runs were not started")
smoke_ckpt = sorted((smoke / "ft").glob("model_e[0-9][0-9][0-9].pt"))[-1]
code = subprocess.run([str(c) for c in cache_command(smoke_ckpt, smoke / "cache", "full,d4", "full,d4", ("--limit", "2"))],
                      cwd=root, env=env).returncode
made = sorted(p.name for p in (smoke / "cache").glob("*_cache.json"))
print("SMOKE_CACHE", code, made, flush=True)
if code != 0 or made != ["test_cache.json", "val_cache.json"]:
    raise SystemExit("smoke caching failed; the long runs were not started")
shutil.rmtree(smoke, ignore_errors=True)
print("SMOKE_OK hours", hours(), flush=True)

# ---- the fine-tunes, one per GPU ----
def start(index, name, extra):
    out = work / name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "stdout.log", "w")
    process = subprocess.Popen([str(c) for c in [sys.executable, "-u", "-m", TRAIN, "--output", out] + common + extra],
                               cwd=root, env=gpu_env(index % gpus), stdout=log, stderr=subprocess.STDOUT)
    print("STARTED", name, "gpu", index % gpus, extra, flush=True)
    return process

def wait(processes):
    while any(p.poll() is None for p in processes.values()):
        time.sleep(300)
        for name in processes:
            found = events(work / name)
            if found:
                print("LAST", name, hours(), json.dumps(found[-1])[:300], flush=True)

if gpus >= 2:
    processes = {name: start(i, name, extra) for i, (name, extra) in enumerate(RUNS)}
    wait(processes)
else:
    processes = {}
    for i, (name, extra) in enumerate(RUNS):
        processes[name] = start(i, name, extra)
        wait({name: processes[name]})
for name, process in processes.items():
    print("EXIT", name, process.returncode, "hours", hours(), flush=True)
    for e in events(work / name):
        if e.get("event") in ("GROUPS", "FUSED", "INIT", "DATA", "SCHEDULE", "EVAL", "BEST", "SUMMARY", "ERROR", "EPOCH"):
            print("RESULT", name, json.dumps(e)[:500], flush=True)
    tail = (work / name / "stdout.log").read_text()[-1500:] if process.returncode else ""
    if tail:
        print("STDOUT_TAIL", name, tail, flush=True)
    for pattern in ("last.pt", "*.partial"):
        for path in (work / name).glob(pattern):
            path.unlink()

# ---- rank the checkpoints on the target annotators' validation records ----
caches = sorted(work.glob("ft*/e[0-9][0-9][0-9]_cache.json"))
best_checkpoint = None
if caches:
    code = subprocess.run([str(c) for c in [sys.executable, "-u", "-m", SELECT, "--annotations", annotations, "--caches", *caches,
                           "--output", work / "selection.json", "--processes", "4"]], cwd=root, env=env).returncode
    print("SELECT_EXIT", code, "hours", hours(), flush=True)
    if code == 0:
        best_cache = Path(json.loads((work / "selection.json").read_text())["best"])
        best_checkpoint = best_cache.parent / f"model_{best_cache.name[:4]}.pt"
checkpoints = sorted(work.glob("ft*/model_e[0-9][0-9][0-9].pt"))
if best_checkpoint is None or not best_checkpoint.is_file():
    best_checkpoint = checkpoints[-1] if checkpoints else None
print("BEST_CHECKPOINT", best_checkpoint, flush=True)

# ---- candidate caches: eight views for the selected checkpoint, single-view test for the others ----
if best_checkpoint is not None and hours() < 4.0:
    code = subprocess.run([str(c) for c in cache_command(best_checkpoint, work / "d4_best", "full,d4", "full,d4")],
                          cwd=root, env=env).returncode
    (work / "d4_best").mkdir(parents=True, exist_ok=True)
    (work / "d4_best" / "checkpoint.txt").write_text(str(best_checkpoint.relative_to(work)))
    print("D4_EXIT", code, "hours", hours(), flush=True)
for checkpoint in checkpoints:
    if checkpoint == best_checkpoint:
        continue
    if hours() > 4.6:
        print("SKIPPED_TEST_CACHE", checkpoint, flush=True)
        continue
    tag = f"test_{checkpoint.parent.name}_{checkpoint.stem[6:]}"
    code = subprocess.run([str(c) for c in cache_command(checkpoint, work / tag, "none", "full")], cwd=root, env=env).returncode
    print("TEST_CACHE_EXIT", tag, code, "hours", hours(), flush=True)
for path in list(work.rglob("jobs.json")) + list(work.rglob("cache_shard*.json")):
    path.unlink()
print("OUTPUTS", sorted(f"{p.relative_to(work)}:{p.stat().st_size}" for p in work.rglob("*") if p.is_file()), flush=True)
print("KERNEL_COMPLETE hours", hours(), flush=True)

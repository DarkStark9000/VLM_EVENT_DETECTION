#!/usr/bin/env python3
import os, sys, signal, time, subprocess, threading, http.client

# ---------- Config (edit if needed) ----------
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")  # REQUIRED for gated models
CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") \
            or ("/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/mnt/models")

GPU_MEM_UTIL = os.environ.get("VLLM_GPU_MEM_UTIL", "0.92")  # safer default
DTYPE = os.environ.get("VLLM_DTYPE", "auto")
MAX_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "2048")
TP_SIZE = os.environ.get("VLLM_TP_SIZE", "1")
TRUST_REMOTE_CODE = True

# Two services mirroring your docker-compose
MODELS = [
    {
        "name": "qwen32b_vl",
        "repo_or_path": os.environ.get("QWEN_REPO", "Qwen/Qwen2.5-VL-32B-Instruct"),
        "port": int(os.environ.get("QWEN_PORT", "8000")),
        "cuda_index": os.environ.get("QWEN_GPU", "0"),
    },
    {
        "name": "llama8b",
        "repo_or_path": os.environ.get("LLAMA_REPO", "Meta-Llama/Meta-Llama-3.1-8B-Instruct"),
        "port": int(os.environ.get("LLAMA_PORT", "8001")),
        "cuda_index": os.environ.get("LLAMA_GPU", "1"),
    },
]
# ---------------------------------------------

def ensure_dirs():
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("/runpod-volume/logs", exist_ok=True) if os.path.isdir("/runpod-volume") else None

def build_cmd(repo_or_path: str, port: int) -> list:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        f"--model={repo_or_path}",
        f"--dtype={DTYPE}",
        f"--max-model-len={MAX_LEN}",
        f"--tensor-parallel-size={TP_SIZE}",
        f"--host=0.0.0.0",
        f"--port={port}",
        f"--gpu-memory-utilization={GPU_MEM_UTIL}",
    ]
    if TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
    # Force downloads/caching into CACHE_DIR (works for both repo IDs and local paths)
    if CACHE_DIR:
        cmd.extend(["--download-dir", CACHE_DIR])
    return cmd

def tee_stream(stream, prefix, logfile_path):
    with open(logfile_path, "ab", buffering=0) as f:
        for line in iter(stream.readline, b""):
            f.write(line)
            try:
                sys.stdout.buffer.write(prefix + line)
                sys.stdout.flush()
            except Exception:
                pass

def healthcheck(port: int, retries: int = 60, delay: float = 2.0) -> bool:
    for _ in range(retries):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def launch_service(svc):
    env = os.environ.copy()
    if HF_TOKEN:
        # Inherit token so vLLM can pull gated models
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    if CACHE_DIR:
        env["HF_HOME"] = CACHE_DIR
        env["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
    env["CUDA_VISIBLE_DEVICES"] = svc["cuda_index"]

    cmd = build_cmd(svc["repo_or_path"], svc["port"])
    log_dir = "/runpod-volume/logs" if os.path.isdir("/runpod-volume") else "."
    log_path = os.path.join(log_dir, f"{svc['name']}.log")

    print(f"[{svc['name']}] GPU={svc['cuda_index']} PORT={svc['port']} MODEL={svc['repo_or_path']}")
    print(f"[{svc['name']}] Cache/Download dir: {CACHE_DIR}")
    print(f"[{svc['name']}] Logs: {log_path}")
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1
    )

    # Tee stdout/stderr to file + console
    t1 = threading.Thread(target=tee_stream, args=(proc.stdout, f"[{svc['name']}] ".encode(), log_path), daemon=True)
    t2 = threading.Thread(target=tee_stream, args=(proc.stderr, f"[{svc['name']}:ERR] ".encode(), log_path), daemon=True)
    t1.start(); t2.start()

    return proc

def main():
    ensure_dirs()

    # Sanity checks
    if not HF_TOKEN:
        print("WARN: HUGGING_FACE_HUB_TOKEN not set. Gated models (e.g., Llama) will fail to download.", flush=True)

    # Launch both services
    procs = []
    for svc in MODELS:
        procs.append((svc, launch_service(svc)))

    # Health checks
    for svc, _ in procs:
        ok = healthcheck(svc["port"])
        if ok:
            print(f"[{svc['name']}] READY at http://0.0.0.0:{svc['port']}/v1", flush=True)
        else:
            print(f"[{svc['name']}] WARNING: health check failed; check logs.", flush=True)

    # Graceful shutdown on signals
    def handle_sig(sig, frame):
        print(f"\nSignal {sig} received, terminating children...", flush=True)
        for _, p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        deadline = time.time() + 15
        while time.time() < deadline and any(p.poll() is None for _, p in procs):
            time.sleep(0.2)
        for _, p in procs:
            if p.poll() is None:
                try: p.kill()
                except Exception: pass
        sys.exit(0)

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, handle_sig)

    # Wait on children; exit if any crashes
    try:
        while True:
            all_running = True
            for svc, p in procs:
                ret = p.poll()
                if ret is not None:
                    print(f"[{svc['name']}] EXITED with code {ret}. See logs.", flush=True)
                    handle_sig(signal.SIGTERM, None)
                    return
            time.sleep(1.0)
    except KeyboardInterrupt:
        handle_sig(signal.SIGINT, None)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os, sys, signal, time, subprocess, threading, http.client

# ---------- Config (edit if needed) ----------
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")  # REQUIRED for gated models
CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") \
            or ("/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/mnt/models")

GPU_MEM_UTIL = os.environ.get("VLLM_GPU_MEM_UTIL", "0.92")  # safer default
DTYPE = os.environ.get("VLLM_DTYPE", "auto")
MAX_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "2048")
TP_SIZE = os.environ.get("VLLM_TP_SIZE", "1")
TRUST_REMOTE_CODE = True

# Two services mirroring your docker-compose
MODELS = [
    {
        "name": "qwen32b_vl",
        "repo_or_path": os.environ.get("QWEN_REPO", "Qwen/Qwen2.5-VL-32B-Instruct"),
        "port": int(os.environ.get("QWEN_PORT", "8000")),
        "cuda_index": os.environ.get("QWEN_GPU", "0"),
    },
    {
        "name": "llama8b",
        "repo_or_path": os.environ.get("LLAMA_REPO", "Meta-Llama/Meta-Llama-3.1-8B-Instruct"),
        "port": int(os.environ.get("LLAMA_PORT", "8001")),
        "cuda_index": os.environ.get("LLAMA_GPU", "1"),
    },
]
# ---------------------------------------------

def ensure_dirs():
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("/runpod-volume/logs", exist_ok=True) if os.path.isdir("/runpod-volume") else None

def build_cmd(repo_or_path: str, port: int) -> list:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        f"--model={repo_or_path}",
        f"--dtype={DTYPE}",
        f"--max-model-len={MAX_LEN}",
        f"--tensor-parallel-size={TP_SIZE}",
        f"--host=0.0.0.0",
        f"--port={port}",
        f"--gpu-memory-utilization={GPU_MEM_UTIL}",
    ]
    if TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
    # Force downloads/caching into CACHE_DIR (works for both repo IDs and local paths)
    if CACHE_DIR:
        cmd.extend(["--download-dir", CACHE_DIR])
    return cmd

def tee_stream(stream, prefix, logfile_path):
    with open(logfile_path, "ab", buffering=0) as f:
        for line in iter(stream.readline, b""):
            f.write(line)
            try:
                sys.stdout.buffer.write(prefix + line)
                sys.stdout.flush()
            except Exception:
                pass

def healthcheck(port: int, retries: int = 60, delay: float = 2.0) -> bool:
    for _ in range(retries):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def launch_service(svc):
    env = os.environ.copy()
    if HF_TOKEN:
        # Inherit token so vLLM can pull gated models
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    if CACHE_DIR:
        env["HF_HOME"] = CACHE_DIR
        env["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
    env["CUDA_VISIBLE_DEVICES"] = svc["cuda_index"]

    cmd = build_cmd(svc["repo_or_path"], svc["port"])
    log_dir = "/runpod-volume/logs" if os.path.isdir("/runpod-volume") else "."
    log_path = os.path.join(log_dir, f"{svc['name']}.log")

    print(f"[{svc['name']}] GPU={svc['cuda_index']} PORT={svc['port']} MODEL={svc['repo_or_path']}")
    print(f"[{svc['name']}] Cache/Download dir: {CACHE_DIR}")
    print(f"[{svc['name']}] Logs: {log_path}")
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1
    )

    # Tee stdout/stderr to file + console
    t1 = threading.Thread(target=tee_stream, args=(proc.stdout, f"[{svc['name']}] ".encode(), log_path), daemon=True)
    t2 = threading.Thread(target=tee_stream, args=(proc.stderr, f"[{svc['name']}:ERR] ".encode(), log_path), daemon=True)
    t1.start(); t2.start()

    return proc

def main():
    ensure_dirs()

    # Sanity checks
    if not HF_TOKEN:
        print("WARN: HUGGING_FACE_HUB_TOKEN not set. Gated models (e.g., Llama) will fail to download.", flush=True)

    # Launch both services
    procs = []
    for svc in MODELS:
        procs.append((svc, launch_service(svc)))

    # Health checks
    for svc, _ in procs:
        ok = healthcheck(svc["port"])
        if ok:
            print(f"[{svc['name']}] READY at http://0.0.0.0:{svc['port']}/v1", flush=True)
        else:
            print(f"[{svc['name']}] WARNING: health check failed; check logs.", flush=True)

    # Graceful shutdown on signals
    def handle_sig(sig, frame):
        print(f"\nSignal {sig} received, terminating children...", flush=True)
        for _, p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        deadline = time.time() + 15
        while time.time() < deadline and any(p.poll() is None for _, p in procs):
            time.sleep(0.2)
        for _, p in procs:
            if p.poll() is None:
                try: p.kill()
                except Exception: pass
        sys.exit(0)

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, handle_sig)

    # Wait on children; exit if any crashes
    try:
        while True:
            all_running = True
            for svc, p in procs:
                ret = p.poll()
                if ret is not None:
                    print(f"[{svc['name']}] EXITED with code {ret}. See logs.", flush=True)
                    handle_sig(signal.SIGTERM, None)
                    return
            time.sleep(1.0)
    except KeyboardInterrupt:
        handle_sig(signal.SIGINT, None)

if __name__ == "__main__":
    main()


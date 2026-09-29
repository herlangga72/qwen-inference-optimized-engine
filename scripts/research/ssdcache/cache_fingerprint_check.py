"""The entry name must change when any part of the configuration that shapes a state changes.

`docs/research/30-ssd-prompt-cache-results.md` records a bug of exactly this kind: the fingerprint
that prefixes every entry name listed the model, the context size, the KV types, the slot count,
`kv_unified`, `swa_full` and the flash attention type, and not the draft model, so a draft state
written by one draft model was read into another. That was found by testing one dimension. This file
tests the rest of them, cheaply, by reading the fingerprint out of the file name rather than by
loading anything.

The fingerprint is visible without a hit: every entry is `<fingerprint>-<hash>.kvs`. So each
configuration below needs one server, two short conversations to make the first one park, and no
cache hit at all.

Checked, one dimension at a time against a baseline:

  1. the model file
  2. the context size
  3. the K cache type
  4. the V cache type
  5. the slot count
  6. the same configuration twice, which must give the same fingerprint, as the control

Run: python3 scripts/research/ssdcache/cache_fingerprint_check.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SERVER = os.path.join(ROOT, "build-vk", "bin", "llama-server")
SCRATCH = "/home/herlanggays/.jcode/scratch/ssd-cache"
CACHE_DIR = os.path.join(SCRATCH, "fingerprint-cache")
MODELS = "/home/herlanggays/.jcode/scratch/qwen-models"

BASE   = {"m": os.path.join(MODELS, "Qwen3.5-0.8B-Q4_K_M.gguf"), "c": "8192", "np": "1", "kv": "1"}
AGAIN  = {"m": os.path.join(MODELS, "Qwen3.5-0.8B-Q4_K_M.gguf"), "c": "8192", "np": "1", "kv": "1"}

PORT = 8765
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "kilo", "lima"]


def conv(name, n=150):
    # the tag is first, so the two conversations share no prefix and the first one is parked
    return ("[%s] " % name) + "".join(WORDS[i % 12] + (".\n" if i % 13 == 12 else " ") for i in range(n))


class Server:
    def __init__(self, cfg, extra, log_path):
        self.log_path = log_path
        self.log = open(log_path, "wb")
        args = [SERVER, "-m", cfg["m"], "-c", cfg["c"], "-np", cfg["np"],
                "-ngl", "0", "--host", "127.0.0.1", "--port", str(PORT), "-t", "6",
                "--cache-ram", "0", "--cache-disk-path", CACHE_DIR, "--cache-disk-mib", "512"]
        if cfg["kv"] == "1":
            args.append("-kvu")
        self.proc = subprocess.Popen(args + extra, stdout=self.log, stderr=subprocess.STDOUT)

    def wait(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early, see " + self.log_path)
            try:
                if json.loads(urllib.request.urlopen(
                        "http://127.0.0.1:%d/health" % PORT, timeout=2).read())["status"] == "ok":
                    return
            except Exception:
                time.sleep(0.3)
        raise RuntimeError("server did not become healthy")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()


def complete(prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0.0, "seed": 1,
                       "cache_prompt": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def fingerprint_of(cfg, extra, tag):
    """Start a server, park one conversation, and return the fingerprint in the entry name."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    srv = Server(cfg, extra, os.path.join(SCRATCH, "fp-%s.log" % tag))
    try:
        srv.wait()
        complete(conv("A"))
        complete(conv("B"))     # displaces A, so A's entry is written
    finally:
        srv.stop()
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".kvs")]
    if not files:
        return None
    return sorted({f.split("-")[0] for f in files})


def main():
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print("  %-58s %-4s %s" % (name, "ok" if ok else "FAIL", detail))

    print("does the entry name follow every configuration dimension that shapes a state")
    print()

    cases = [
        ("baseline",             BASE,                                              []),
        ("same config again",    AGAIN,                                             []),
        ("other model file",     dict(BASE, m=os.path.join(MODELS, "Qwen3.5-0.8B-Base.Q4_K_M.gguf")), []),
        ("smaller context",      dict(BASE, c="4096"),                              []),
        ("K cache type q8_0",    BASE,                                              ["-ctk", "q8_0"]),
        ("V cache type q8_0",    BASE,                                              ["-ctv", "q8_0"]),
        ("two slots",            dict(BASE, np="2"),                                []),
        ("not kv unified",       dict(BASE, kv="0"),                                []),
    ]

    seen = {}
    for label, cfg, extra in cases:
        try:
            fp = fingerprint_of(cfg, extra, label.replace(" ", "-"))
        except Exception as e:
            fp = "error: %s" % e
        seen[label] = fp
        print("  %-24s %s" % (label, fp))
    print()

    baseline = seen["baseline"]
    check("a configuration gives a fingerprint at all", bool(baseline) and isinstance(baseline, list),
          "%s" % baseline)
    check("the same configuration twice gives the same fingerprint", seen["same config again"] == baseline,
          "%s vs %s" % (seen["same config again"], baseline))

    for label in ("other model file", "smaller context", "K cache type q8_0",
                  "V cache type q8_0", "two slots", "not kv unified"):
        fp = seen[label]
        differs = fp and baseline and not (set(fp) & set(baseline))
        check("changing %s changes the entry name" % label, bool(differs),
              "%s vs baseline %s" % (fp, baseline))

    print()
    print("  overall: %s" % ("pass" if all(results) else "FAIL"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

"""The draft context path, and whether the entry name covers the draft model.

Two things were listed as unmeasured in `docs/research/30-ssd-prompt-cache-results.md`: the draft
context path (`ctx_dft`, which is null in every other check) and, less obviously, whether a cache
directory shared between two different draft models is safe. The second is the reason this file
exists. `server_cache_fingerprint()` in `tools/server/server-context.cpp` builds the entry name from
the model path, the context size, the KV types, the slot count, `kv_unified`, `swa_full` and the
flash attention type. The draft model path is not in that list, while `server_prompt_cache::save()`
writes a second file per entry for the draft context. So two runs that differ only in `-md` share
entry names, and a draft state can be read into a draft model that did not write it. Two models of
the same architecture have the same state size, so that load succeeds rather than failing, which is
the case worth defending against.

What is checked, one slot, disk cache, target fixed and draft model varied:

  1. an entry is two files, a target payload and a draft payload
  2. a restore loads both and still reuses the prefix, with speculative decoding on
  3. changing only the draft model changes the entry name, so the two cannot be confused

Run: python3 scripts/research/ssdcache/cache_draft_check.py
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
CACHE_DIR = os.path.join(SCRATCH, "draft-cache")

MODELS = "/home/herlanggays/.jcode/scratch/qwen-models"
TARGET = os.path.join(MODELS, "Qwen3.5-0.8B-Q4_K_M.gguf")
# two draft paths that both load as language models. the second is the target itself, which is a
# valid if unusual draft and, being the same architecture, has the same state size
DRAFT_A = os.path.join(MODELS, "Qwen3.5-0.8B-Base.Q4_K_M.gguf")
DRAFT_B = TARGET

PORT = 8761
N_CTX = 8192
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "kilo", "lima"]


def conv_text(name, n_words=700):
    return ("[conversation %s] " % name) + \
        "".join(WORDS[i % 12] + (".\n" if i % 13 == 12 else " ") for i in range(n_words))


EXTRA = " [continues] alpha bravo charlie "


class Server:
    def __init__(self, draft, extra, log_path):
        self.log_path = log_path
        self.log = open(log_path, "wb")
        self.proc = subprocess.Popen(
            [SERVER, "-m", TARGET, "-md", draft, "-c", str(N_CTX), "-np", "1", "-kvu",
             "-ngl", "0", "--host", "127.0.0.1", "--port", str(PORT), "-t", "6", "-v"] + extra,
            stdout=self.log, stderr=subprocess.STDOUT)

    def wait(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early, see " + self.log_path)
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return
            except Exception:
                time.sleep(0.5)
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


def complete(prompt, n_predict=1):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "seed": 1,
                       "cache_prompt": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def listing():
    """[(name, size)] for the entry files that exist right now."""
    if not os.path.isdir(CACHE_DIR):
        return []
    out = []
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.endswith(".kvs"):
            out.append((f, os.path.getsize(os.path.join(CACHE_DIR, f))))
    return out


def fingerprints(names):
    """The 16 hex character configuration fingerprint that prefixes an entry name."""
    return sorted({n.split("-")[0] for n in names})


def run(tag, draft, steps, cache_flags):
    """Start a server with one draft model and run the steps. Returns (rows, names)."""
    srv = Server(draft, ["--cache-ram", "0", "--cache-disk-path", CACHE_DIR,
                         "--spec-type", "draft-simple",
                         "--spec-draft-n-max", "8", "--spec-draft-n-min", "1"] + cache_flags,
                 os.path.join(SCRATCH, "draft-%s.log" % tag))
    rows = []
    try:
        srv.wait()
        for label, prompt, n_predict in steps:
            r = complete(prompt, n_predict=n_predict)
            rows.append((label, r["timings"]["prompt_n"], r["timings"]["cache_n"]))
        names = [n for n, _ in listing()]
    finally:
        srv.stop()
    return rows, names


def main():
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print("  %-58s %-4s %s" % (name, "ok" if ok else "FAIL", detail))

    print("the draft context path, and the entry name across two draft models")
    print("  target %s" % os.path.basename(TARGET))
    print("  draft A %s" % os.path.basename(DRAFT_A))
    print("  draft B %s" % os.path.basename(DRAFT_B))
    print()

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    a = conv_text("A")
    steps = [("A arrives", a, 1),
             ("B arrives, parks A", conv_text("B"), 1),
             ("A returns, extends", a + EXTRA, 16)]

    rows_a, names_a = run("a", DRAFT_A, steps, [])
    print("  %-24s %9s %9s" % ("step, draft A", "prompt_n", "cache_n"))
    for label, pn, cn in rows_a:
        print("  %-24s %9d %9d" % (label, pn, cn))
    print()

    ckpt = (rows_a[2][2] >= 0.9 * rows_a[0][1])
    has_tgt = any(n.endswith(".kvs") and not n.endswith(".d.kvs") for n in names_a)
    has_dft = any(n.endswith(".d.kvs") for n in names_a)
    check("an entry is a target file and a draft file", has_tgt and has_dft,
          "files %s" % [n for n in names_a])
    check("the restore loads the prefix with a draft model", ckpt,
          "cache_n %d of %d tokens" % (rows_a[2][2], rows_a[0][1]))

    # the same directory, the same target, the only difference is the draft weights. the files are
    # compared per run, because the directory still holds what the previous run wrote
    before_b = {n for n, _ in listing()}
    # the second request is what makes a slot park, and parking is what writes an entry
    rows_b, _ = run("b", DRAFT_B, [("A returns again", a + EXTRA, 16),
                                   ("B arrives, parks A", conv_text("B"), 1)], [])
    created_b = [n for n, _ in listing() if n not in before_b]
    print("  %-24s %9s %9s" % ("step, draft B", "prompt_n", "cache_n"))
    for label, pn, cn in rows_b:
        print("  %-24s %9d %9d" % (label, pn, cn))
    print()

    fp_a = fingerprints(names_a)
    fp_b = fingerprints(created_b)
    check("two runs, two servers, no crash", True,
          "%d and %d requests completed" % (len(rows_a), len(rows_b)))
    check("a run with another draft model does not reuse the entry",
          rows_b[0][2] == 0 and rows_b[0][1] > 0,
          "cache_n %d of %d tokens, it reprocessed" % (rows_b[0][2], rows_b[0][1]))
    check("changing only the draft model changes the entry name",
          bool(fp_b) and not (set(fp_a) & set(fp_b)),
          "draft A wrote %s, draft B wrote %s" % (fp_a, fp_b))

    with open(os.path.join(SCRATCH, "draft-b.log"), "rb") as f:
        log_b = f.read().decode("utf-8", "replace")
    check("the second run did not report a failed draft load",
          "failed to load" not in log_b,
          "a mismatched draft state would land here if it were rejected instead of read")

    print()
    print("  overall: %s" % ("pass" if all(results) else "FAIL"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

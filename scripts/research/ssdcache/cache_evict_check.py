"""Eviction under the disk budget, which document 30 lists as untested.

`server_prompt_cache::update()` enforces `--cache-disk-mib` by walking the entry list oldest
first and calling `unlink_state` on each entry until the total is back under the limit
(`tools/server/server-task.cpp`). For the disk cache `unlink_state` is a `remove()` of the entry
file, so this is the one path where the cache deletes something a user owns. A cache that leaks
files, or that deletes the wrong one, or that keeps serving an entry whose file it removed, is a
bug with consequences outside the process, and until now it had not been run.

What is checked, one slot, disk cache only, budget set from a measured entry size:

  1. on-disk bytes after every request are within the budget
  2. the file count stays at what the budget can hold, so old entries are deleted and not leaked
  3. the oldest conversation misses after it was evicted, and is reprocessed from scratch
  4. a conversation whose entry survived still restores, so eviction took the right files
  5. the server logged that it removed the oldest entry, so the path really ran. Note that
     `alloc()` makes room before it registers the new entry, so the budget is never exceeded even
     momentarily, and the `update()` message is not expected here
  6. no entry failed to load, which is what a wrong-file deletion would look like

The conversations deliberately share no prefix, because a shared preamble would be served from the
slot's own arena and would hide whether the disk entry was needed at all.

Run: python3 scripts/research/ssdcache/cache_evict_check.py [model]
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
CACHE_DIR = os.path.join(SCRATCH, "evict-cache")

PORT = 8757
N_CTX = 8192

# big enough that one entry is tens of MiB, so a budget of a couple of entries is a small
# number of requests away from being exceeded
PROMPT_WORDS = 900

# the returning requests extend their conversation by a few words. A restore only applies when
# there is something left to evaluate, which is also the shape the feature is built for
EXTRA = " [continues] " + " ".join(["alpha", "bravo", "charlie"]) + " "

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "kilo", "lima"]


def conv_text(name, n_words=PROMPT_WORDS):
    # the conversation tag is first, so two conversations share no prefix at all
    return ("[conversation %s] " % name) + \
        "".join(WORDS[i % 12] + (".\n" if i % 13 == 12 else " ") for i in range(n_words))


class Server:
    def __init__(self, extra, log_path):
        self.log_path = log_path
        self.log = open(log_path, "wb")
        self.proc = subprocess.Popen(
            [SERVER, "-m", MODEL, "-c", str(N_CTX), "-np", "1", "-kvu", "-ngl", "0",
             "--host", "127.0.0.1", "--port", str(PORT), "-t", "6", "-v"] + extra,
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


def complete(prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0.0, "seed": 1,
                       "cache_prompt": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def entries():
    """(file count, bytes on disk) for the entry files that exist right now."""
    if not os.path.isdir(CACHE_DIR):
        return 0, 0
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".kvs")]
    return len(files), sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)


def clear_cache():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def main():
    global MODEL

    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print("  %-58s %-4s %s" % (name, "ok" if ok else "FAIL", detail))

    MODEL = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf"

    print("eviction under the disk budget")
    print("  model %s" % MODEL)

    # 1. two entries' worth of traffic with an effectively unlimited budget, because a file is
    #    only written when a request displaces the slot: the first request leaves nothing behind
    clear_cache()
    rows = []
    srv = Server(["--cache-ram", "0", "--cache-disk-path", CACHE_DIR, "--cache-disk-mib", "8192"],
                 os.path.join(SCRATCH, "evict-measure.log"))
    try:
        srv.wait()
        complete(conv_text("A"))
        r = complete(conv_text("B"))
        n, b = entries()
        rows.append(("B arrives", r["timings"]["prompt_n"], r["timings"]["cache_n"], n, b))
    finally:
        srv.stop()

    if rows[0][3] != 1 or rows[0][4] == 0:
        print("  could not measure an entry: %d files, %d bytes" % (rows[0][3], rows[0][4]))
        return 1

    entry_bytes = rows[0][4]
    # a budget of 2.2 entries: it holds two, so the third write has to evict and the oldest goes
    budget_mib = max(8, int(2.2 * entry_bytes / 1048576.0))
    hold = max(1, int(budget_mib * 1048576.0 / entry_bytes))

    print("  one entry is %.1f MiB, budget %d MiB, so at most %d entries fit"
          % (entry_bytes / 1048576.0, budget_mib, hold))
    print()

    # 2. the sequence. four conversations over one slot, then the oldest and the newest return
    clear_cache()
    rows = []
    log2 = os.path.join(SCRATCH, "evict-run.log")
    names = ["A", "B", "C", "D"]
    srv = Server(["--cache-ram", "0", "--cache-disk-path", CACHE_DIR,
                  "--cache-disk-mib", str(budget_mib)], log2)
    try:
        srv.wait()
        for name in names:
            r = complete(conv_text(name))
            n, b = entries()
            rows.append(("%s arrives" % name, r["timings"]["prompt_n"],
                         r["timings"]["cache_n"], n, b))
        r = complete(conv_text("A") + EXTRA)
        rows.append(("A returns, evicted", r["timings"]["prompt_n"], r["timings"]["cache_n"],
                     *entries()))
        hit_a = r["timings"]["cache_n"]
        r = complete(conv_text("D") + EXTRA)
        rows.append(("D returns, retained", r["timings"]["prompt_n"], r["timings"]["cache_n"],
                     *entries()))
        hit_d = r["timings"]["cache_n"]
        len_d = rows[3][1]
    finally:
        srv.stop()

    print("  %-22s %9s %9s %8s %10s" % ("step", "prompt_n", "cache_n", "entries", "MiB"))
    for label, pn, cn, n, b in rows:
        print("  %-22s %9d %9d %8d %10.2f" % (label, pn, cn, n, b / 1048576.0))
    print()

    with open(log2, "rb") as f:
        log = f.read().decode("utf-8", "replace")

    # 3. the checks
    over = [(label, b) for label, _, _, _, b in rows if b > budget_mib * 1048576]
    check("on-disk bytes stay within the budget after every request",
          not over,
          "budget %d MiB, worst %s" % (budget_mib,
                                       ", ".join("%s %.1f MiB" % (l, b / 1048576.0) for l, b in over) or "none over"))
    check("old entries are deleted rather than leaked",
          all(n <= hold for _, _, _, n, _ in rows),
          "files %s, budget holds %d" % ([n for _, _, _, n, _ in rows], hold))
    check("something is still cached at the end",
          rows[-1][3] > 0, "entries %d, %.1f MiB" % (rows[-1][3], rows[-1][4] / 1048576.0))
    check("the evicted conversation misses and is reprocessed",
          hit_a < 0.1 * rows[0][1],
          "cache_n %d of %d tokens" % (hit_a, rows[0][1]))
    check("the retained conversation still restores",
          hit_d > 0.9 * len_d,
          "cache_n %d of %d tokens" % (hit_d, len_d))
    check("the server reported removing the oldest entry",
          "removing oldest entry" in log,
          "the eviction path ran" if "removing oldest entry" in log else "no such line in the log")
    check("no entry failed to load",
          "failed to load prompt cache entry" not in log,
          "which is what deleting the wrong file would look like")

    print()
    print("  overall: %s" % ("pass" if all(results) else "FAIL"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

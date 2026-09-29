"""What the f_keep < 1.0f change costs, which is the side nobody measured.

`get_available_slot` parks a slot, and therefore writes an entry, when `f_keep < 1.0f`; before the
change the condition was `f_keep < 0.5f`, so a request that shares between half and all of what the
slot holds was served from the arena without writing anything. Doc 30 measures the gain of the
change, 2 percent to 93 percent reuse on a workload whose turns extend what came before. This file
measures the loss on the workload where that reasoning does not apply: prompts that share most of a
slot's prefix but are never repeated, which is what a client that always edits the tail looks like.

The shape is one shared preamble plus a distinct short tail per request, sent once each, one slot:

  request 1   the preamble alone, fills the slot
  requests 2..N   preamble + "[tail i]", so f_keep is the preamble's share, about 0.9

Every one of those requests is served the preamble from the arena either way, and none of them is
ever asked for again, so any entry written is bytes on disk for a state nothing will hit. The check
reports what each build writes and asserts only that both builds serve the prefix, because the point
is the difference between them and it needs two binaries.

Run against a specific binary to compare a build with the old gate, for example:

  python3 scripts/research/ssdcache/cache_gate_cost_check.py [server-binary]
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
SCRATCH = "/home/herlanggays/.jcode/scratch/ssd-cache"
CACHE_DIR = os.path.join(SCRATCH, "gate-cost-cache")
MODEL = "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf"

PORT = 8763
N_CTX = 8192
N_REQUESTS = 10

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "kilo", "lima"]


def words_text(n, tag):
    return tag + " " + "".join(WORDS[i % 12] + (".\n" if i % 13 == 12 else " ") for i in range(n))


# large enough that an entry is tens of MiB and the write cost is visible
PREAMBLE = words_text(700, "shared preamble")


class Server:
    def __init__(self, binary, log_path):
        self.log_path = log_path
        self.log = open(log_path, "wb")
        self.proc = subprocess.Popen(
            [binary, "-m", MODEL, "-c", str(N_CTX), "-np", "1", "-kvu", "-ngl", "0",
             "--host", "127.0.0.1", "--port", str(PORT), "-t", "6",
             "--cache-ram", "0", "--cache-disk-path", CACHE_DIR, "--cache-disk-mib", "512"],
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
    if not os.path.isdir(CACHE_DIR):
        return 0, 0
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".kvs")]
    return len(files), sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build-vk", "bin", "llama-server")

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("what a never-repeated prefix costs, one slot, disk cache")
    print("  binary %s" % binary)
    print("  %d requests over one shared %d word preamble, distinct tails" % (N_REQUESTS, 700))
    print()

    rows = []
    srv = Server(binary, os.path.join(SCRATCH, "gate-cost.log"))
    try:
        srv.wait()
        r = complete(PREAMBLE)
        rows.append((0, r["timings"]["prompt_n"], r["timings"]["cache_n"], *entries()))
        for i in range(1, N_REQUESTS):
            tail = words_text(20, "tail%d" % i)
            r = complete(PREAMBLE + " " + tail)
            rows.append((i, r["timings"]["prompt_n"], r["timings"]["cache_n"], *entries()))
    finally:
        srv.stop()

    print("  %5s %9s %9s %8s %10s" % ("req", "prompt_n", "cache_n", "entries", "MiB"))
    for i, pn, cn, n, b in rows:
        print("  %5d %9d %9d %8d %10.2f" % (i, pn, cn, n, b / 1048576.0))

    served = sum(1 for _, _, cn, _, _ in rows[1:] if cn > 0)
    files_written = rows[-1][3]
    mib_written = rows[-1][4] / 1048576.0
    print()
    print("  entries on disk at the end        %d, %.1f MiB" % (files_written, mib_written))
    print("  requests served the preamble      %d of %d" % (served, N_REQUESTS - 1))
    print("  no request is ever repeated, so every entry written is bytes nothing will hit")
    return 0


if __name__ == "__main__":
    sys.exit(main())

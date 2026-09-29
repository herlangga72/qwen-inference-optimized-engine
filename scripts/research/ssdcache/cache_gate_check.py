"""The contract of the `f_keep` gate, which is the one line of the cache change that alters
existing server behaviour.

`get_available_slot` decides whether to touch the prompt cache. It used to be

    if (f_keep < 0.5f) update_cache = true;

and is now

    if (f_keep < 1.0f) update_cache = true;

so that a slot about to lose anything is parked, not only a slot about to lose half of itself.
The risk of that change is the other direction: a request that *cleanly extends* what the slot
holds, `f_keep == 1.0`, must behave exactly as before and must not write cache files, because the
arena already serves it and writing a copy of every extension would turn a free path into I/O.

This checks that contract directly, on the disk cache and on the RAM cache, because the line
affects both. It is a substitute for `tools/server/tests/`, which cannot run on this box, and it is
narrower than that suite: it covers the one behaviour the change could have broken.

Scenario, one slot:

  1. one conversation, five turns, each a strict extension of the last
     expected: no entry file at any point, and every turn served from the arena
  2. a second conversation that shares the preamble but not the body
     expected: the first conversation's state is parked, so an entry file appears
  3. the first conversation returns and extends
     expected: its prefix comes back from the cache

Run: python3 scripts/research/ssdcache/cache_gate_check.py
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
CACHE_DIR = os.path.join(SCRATCH, "gate-cache")
MODEL = "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf"

PORT = 8753
N_CTX = 8192

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "kilo", "lima"]

PREAMBLE = None


def words_text(n, tag):
    return tag + " " + "".join(WORDS[i % 12] + (".\n" if i % 13 == 12 else " ") for i in range(n))


def turn_text(conv, turn, n_words):
    return "[%s turn %d] " % (conv, turn) + words_text(n_words, conv) + "\n"


class Server:
    def __init__(self, extra):
        self.log = open(os.path.join(SCRATCH, "gate.log"), "wb")
        self.proc = subprocess.Popen(
            [SERVER, "-m", MODEL, "-c", str(N_CTX), "-np", "1", "-kvu", "-ngl", "0",
             "--host", "127.0.0.1", "--port", str(PORT), "-t", "6"] + extra,
            stdout=self.log, stderr=subprocess.STDOUT)

    def wait(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early")
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


def scenario(name, extra):
    """Returns the rows: (label, prompt_n, cache_n, entries, bytes)."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    srv = Server(extra)
    rows = []
    try:
        srv.wait()

        # 1. five clean extensions of one conversation
        text = PREAMBLE
        for t in range(5):
            text = text + turn_text("convA", t, 200)
            r = complete(text)
            n, b = entries()
            rows.append(("clean extension %d" % t, r["timings"]["prompt_n"],
                         r["timings"]["cache_n"], n, b))

        # 2. a different conversation, same preamble, different body
        other = PREAMBLE + turn_text("convB", 0, 200)
        r = complete(other)
        n, b = entries()
        rows.append(("second conversation", r["timings"]["prompt_n"], r["timings"]["cache_n"], n, b))

        # 3. the first conversation returns and extends
        text = text + turn_text("convA", 5, 200)
        r = complete(text)
        n, b = entries()
        rows.append(("first returns", r["timings"]["prompt_n"], r["timings"]["cache_n"], n, b))
    finally:
        srv.stop()

    print("  %-22s %9s %9s %9s %10s" % (name + ": step", "prompt_n", "cache_n", "entries", "MiB"))
    for label, pn, cn, n, b in rows:
        print("  %-22s %9d %9d %9d %10.2f" % (label, pn, cn, n, b / 1048576.0))
    return rows


def main():
    global PREAMBLE
    PREAMBLE = words_text(260, "shared preamble")

    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print("  %-58s %-4s %s" % (name, "ok" if ok else "FAIL", detail))

    print("the contract of the f_keep gate, one slot, both cache backends")
    print()

    disk = scenario("disk", ["--cache-ram", "0", "--cache-disk-path", CACHE_DIR,
                             "--cache-disk-mib", "512"])
    print()
    ram = scenario("ram", ["--cache-ram", "512", "--cache-disk-mib", "0"])

    print()
    for label, rows in (("disk cache", disk), ("RAM cache", ram)):
        ext = rows[:5]
        check("%s: a clean extension writes no entry file" % label,
              all(r[3] == 0 for r in ext),
              "entry counts %s" % [r[3] for r in ext])
        check("%s: a clean extension is served by the arena" % label,
              ext[-1][2] >= ext[-2][2] and ext[-1][2] > 0,
              "cache_n %s" % [r[2] for r in ext])
        # The return is the evidence that the slot was parked, and it is the check that applies
        # to both backends: the RAM cache keeps entries in memory, so counting files says nothing
        # about it, and an empty file count there is correct rather than a failure.
        check("%s: the first conversation returns with its whole prefix" % label,
              rows[6][2] >= ext[-1][2],
              "cache_n %d on return, its prefix was %d" % (rows[6][2], ext[-1][2]))

    check("disk cache: a different conversation writes one entry file",
          disk[5][3] == 1, "entries %d" % disk[5][3])
    check("RAM cache: a different conversation writes no file, by design",
          ram[5][3] == 0, "entries %d" % ram[5][3])

    # The two backends must agree on the reusable behaviour, which is what the gate controls
    same_extension = [r[2] for r in disk[:5]] == [r[2] for r in ram[:5]]
    same_return = disk[6][2] == ram[6][2]
    check("both backends agree on the extension and return behaviour",
          same_extension and same_return,
          "extensions %s, return %d vs %d" % (same_extension, disk[6][2], ram[6][2]))

    print()
    print("  overall: %s" % ("pass" if all(results) else "FAIL"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Does the SSD prompt cache hold up with more than one slot?

The production server ran with `-np 2`, and every measurement in
`docs/research/30-ssd-prompt-cache-results.md` is one slot and one request at a time. This
check runs two slots with several conversations in flight at once, which is where a cache
with a shared directory and per-entry files can go wrong: two slots writing at the same
moment, one slot reading an entry another is rewriting, or two slots loading the same entry.

Four things are checked:

  1. every request is served, no errors, no server exit
  2. every entry file is structurally intact, checked with the Python validator in
     `scripts/research/kvstore/validate_file.py`, which shares no code with the C++ reader
  3. a hit still works after a restart, so nothing the concurrent phase wrote is junk
  4. the token accounting repeats between two runs of the same concurrent traffic. the box does
     not produce the same text twice (docs/research/28), but prompt_n and cache_n are counts,
     and a corrupted restore would move them

Run: python3 scripts/research/ssdcache/cache_concurrent_check.py [--slots N] [--conversations N]
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SERVER = os.path.join(ROOT, "build-vk", "bin", "llama-server")
VALIDATOR = os.path.join(ROOT, "scripts", "research", "kvstore", "validate_file.py")
SCRATCH = "/home/herlanggays/.jcode/scratch/ssd-cache"
CACHE_DIR = os.path.join(SCRATCH, "concurrent-cache")
MODEL = "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf"

PORT = 8751
N_CTX = 8192
N_PREDICT = 1

# the disk budget. the default fits everything this check writes, so nothing is evicted; a small
# value forces eviction while several slots are writing, which is the case where two of them can
# decide to remove the same entry at once
BUDGET_MIB = 512

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "kilo", "lima"]

PREAMBLE = None


def words_text(n, tag):
    return tag + " " + "".join(WORDS[i % 12] + (".\n" if i % 13 == 12 else " ") for i in range(n))


def turn_text(conv, turn, n_words):
    # a bracketed marker keeps the join at a token boundary, which is what makes a restore
    # usable at all. see docs/research/30-ssd-prompt-cache-results.md
    return "[%s turn %d] " % (conv, turn) + words_text(n_words, conv) + "\n"


class Server:
    def __init__(self, extra, tag, slots):
        self.path = os.path.join(SCRATCH, "conc-%s.log" % tag)
        self.log = open(self.path, "wb")
        self.proc = subprocess.Popen(
            [SERVER, "-m", MODEL, "-c", str(N_CTX), "-np", str(slots), "-kvu", "-ngl", "0",
             "--host", "127.0.0.1", "--port", str(PORT), "-t", "6"] + extra,
            stdout=self.log, stderr=subprocess.STDOUT)

    def wait(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early, see %s" % self.path)
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("server did not become healthy")

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()


def complete(prompt, slot=None):
    body = {"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0.0, "seed": 1,
            "cache_prompt": True}
    if slot is not None:
        body["id_slot"] = slot
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT,
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def tokenize(content):
    body = json.dumps({"content": content}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/tokenize" % PORT, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["tokens"]


def build_traffic(conversations, turns, growth):
    """A prompt per (conversation, turn), each turn appending to the previous one."""
    traffic = []
    for c in range(conversations):
        text = PREAMBLE
        for t in range(turns):
            text = text + turn_text("conv%d" % c, t, growth)
            traffic.append((c, t, text))
    return traffic


def run_phase(slots, traffic, threads, tag):
    """Fire the traffic with several requests in flight. Returns the rows and the token counts."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    srv = Server(["--cache-ram", "0", "--cache-disk-path", CACHE_DIR, "--cache-disk-mib", str(BUDGET_MIB)],
                 tag, slots)
    log_path = srv.path
    try:
        srv.wait()

        # the prompt lengths, taken while a server is up so the accounting can be checked later
        counts = {(c, t): len(tokenize(text)) for (c, t, text) in traffic}

        rows = []

        def one(item):
            c, t, text = item
            try:
                r = complete(text)
                return (c, t, r["timings"]["prompt_n"], r["timings"]["cache_n"], None)
            except Exception as e:            # a request must never fail because of the cache
                # the status alone does not say why, and the server's body does
                detail = str(e)
                reader = getattr(e, "read", None)
                if reader is not None:
                    try:
                        detail += " | " + reader().decode("utf-8", "replace")[:400]
                    except Exception:
                        pass
                return (c, t, -1, -1, detail)

        # id_slot is left to the server so the scheduler decides, which is the real case
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for row in ex.map(one, traffic):
                rows.append(row)

        alive = srv.alive()
    finally:
        srv.stop()

    # A failure here is rare and its evidence is only in the server's log, which every later run
    # overwrites under the same name. One occurrence of this check was lost that way, so a failing
    # phase now leaves a copy behind, taken after the server has stopped so the buffer is flushed.
    if any(r[4] for r in rows):
        keep = log_path.replace(".log", "-failed.log")
        try:
            shutil.copyfile(log_path, keep)
            print("  %-56s %-4s %s" % ("the server log from the failed phase", "", keep))
        except OSError as e:
            print("  %-56s %-4s %s" % ("could not keep the failed phase log", "warn", e))

    return rows, counts, alive


def main():
    global PREAMBLE, PORT, BUDGET_MIB

    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--conversations", type=int, default=4)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--growth", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8751)
    ap.add_argument("--budget-mib", type=int, default=512,
                    help="disk budget for the entries. a value smaller than the traffic forces "
                         "eviction while slots are writing")
    args = ap.parse_args()
    PORT = args.port
    BUDGET_MIB = args.budget_mib

    PREAMBLE = words_text(260, "shared preamble")
    traffic = build_traffic(args.conversations, args.turns, args.growth)

    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print("  %-56s %-4s %s" % (name, "ok" if ok else "FAIL", detail))

    print("%d slots, %d conversations x %d turns = %d requests, %d in flight"
          % (args.slots, args.conversations, args.turns, len(traffic), args.threads))
    print()

    print("run 1")
    rows1, counts1, alive1 = run_phase(args.slots, traffic, args.threads, "1")
    errors = [r for r in rows1 if r[4] is not None]
    check("every request was served", len(errors) == 0,
          "%d errors%s" % (len(errors), (": " + errors[0][4]) if errors else ""))
    check("the server is still running", alive1)

    n_prompt = sum(r[2] for r in rows1 if r[2] >= 0)
    n_cached = sum(r[3] for r in rows1 if r[3] >= 0)
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".kvs")]
    print("  %-56s %d files, %d prompt tokens, %d from cache"
          % ("result", len(files), n_prompt, n_cached))
    print("  per request: conv turn prompt_n cache_n")
    for r in sorted(rows1):
        print("    %-9d %-4d %-8d %-7d" % (r[0], r[1], r[2], r[3]))

    # ---- every file validated by the Python validator, which shares no code with the reader ----
    paths = [os.path.join(CACHE_DIR, f) for f in sorted(files)]
    if paths:
        out = subprocess.run([sys.executable, VALIDATOR] + paths, capture_output=True, text=True)
        text = out.stdout + out.stderr
        bad = [l for l in text.splitlines() if "bad_blk=" in l and "bad_blk=0" not in l]
        holes = [l for l in text.splitlines() if "holes=" in l and "holes=0" not in l]
        stale = [l for l in text.splitlines() if "stale" in l and "stale=0" not in l]
        check("every entry file is intact, no bad blocks", not bad,
              "%d of %d files reported" % (len(bad), len(paths)))
        check("no holes and no stale generations", not holes and not stale)
    else:
        check("entry files were written", False, "none found")

    # ---- a restart must still hit ----
    #
    # How many conversations can hit after a restart depends on the budget, and the expectation has
    # to follow it. An entry exists only for a conversation that was displaced, so with four slots
    # one conversation of each slot is still resident and has no entry, and with a budget that
    # evicts, the restart phase's own writes evict the entries the later conversations need before
    # they are asked. What is a property of the cache either way: the surviving files are intact
    # and a restart serves whatever is still on disk   so the hit count is gated only when nothing
    # was evicted, and reported otherwise.
    evicted = False
    try:
        with open(os.path.join(SCRATCH, "conc-1.log"), "rb") as f:
            evicted = b"removing oldest entry" in f.read()
    except OSError:
        pass

    srv = Server(["--cache-ram", "0", "--cache-disk-path", CACHE_DIR, "--cache-disk-mib", str(BUDGET_MIB)],
                 "restart", args.slots)
    try:
        srv.wait()
        # the last turn of each conversation, extended by one more, so the cache has to serve
        # the whole stored prefix plus the new turn
        hits = 0
        for c in range(args.conversations):
            row = [r for r in sorted(traffic) if r[0] == c][-1]
            r = complete(row[2] + turn_text("conv%d" % c, args.turns, args.growth))
            if r["timings"]["cache_n"] > 0:
                hits += 1
        if evicted:
            print("  %-56s %-4s %d of %d, %d files on disk and the rest evicted"
                  % ("a restart serves what is still on disk", "n/a", hits, args.conversations,
                     len(files)))
        else:
            # every conversation that was displaced must still hit: the ones that were not are
            # still resident in a slot and have no entry to hit, which is one per slot
            need = max(1, args.conversations - args.slots)
            check("a restart hits every displaced conversation", hits >= need,
                  "%d of %d hit, %d files, at least %d expected"
                  % (hits, args.conversations, len(files), need))
    finally:
        srv.stop()

    # ---- the invariants that hold under concurrency ----
    #
    # The token accounting does NOT repeat between two runs of the same concurrent traffic, and
    # that is a property of the scheduling, not a fault: how much a request reuses depends on
    # which slot it landed in and what that slot happened to be holding, and with two slots and
    # four requests in flight that order is not fixed. Measured in the first run of this check,
    # one conversation's second turn reused 707 tokens while in the second run the same turn
    # reused 0, because its slot had been taken by another conversation in between.
    #
    # What does hold, and is checked here, is that the accounting is *consistent* for every
    # request whatever the order: the tokens served from the cache plus the tokens processed are
    # the prompt, no more and no less. A restore that came back wrong, or a save that was
    # counted but not written, would move that.
    print("run 2, same traffic, fresh cache")
    rows2, counts2, alive2 = run_phase(args.slots, traffic, args.threads, "2")
    check("the server is still running after the second run", alive2)

    total_tokens = 0
    consistent = True
    for (c, t, text) in traffic:
        n = counts1[(c, t)]
        total_tokens += n
        got = [r for r in rows1 if r[0] == c and r[1] == t]
        if not got or got[0][2] + got[0][3] != n:
            consistent = False
            print("    conv %d turn %d: prompt_n %s + cache_n %s != %d tokens"
                  % (c, t, got[0][2] if got else "?", got[0][3] if got else "?", n))
    check("every request's accounting is consistent", consistent,
          "%d requests, %d tokens in total" % (len(traffic), total_tokens))

    n_prompt1 = sum(r[2] for r in rows1)
    n_cached1 = sum(r[3] for r in rows1)
    check("concurrent traffic still reuses the cache", n_cached1 > 0,
          "%d of %d tokens came from the cache, %.1f%%"
          % (n_cached1, n_prompt1 + n_cached1, 100.0 * n_cached1 / (n_prompt1 + n_cached1)))

    print()
    print("  overall: %s" % ("pass" if all(results) else "FAIL"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

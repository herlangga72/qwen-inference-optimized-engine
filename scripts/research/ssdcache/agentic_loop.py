"""The agentic loop the SSD prompt cache is for, measured end to end.

Several conversations share a server with fewer slots than conversations, which is what a
shared agentic setup looks like. Each turn of each conversation therefore displaces the
previous conversation from the slot, so every request has to come from the prompt cache
rather than from the arena. That is the difference between this and the four request check in
`cache_hit_check.py`: here the cache is on the critical path every single turn.

Two runs, same traffic:

  with the disk cache     --cache-ram 0 --cache-disk-path DIR --cache-disk-mib N
  without any cache       --cache-ram 0

The second is the honest baseline for the question "how much prompt processing does the
agentic loop cost me": with no cache and one slot, every request processes its whole prompt.

Run: python3 scripts/research/ssdcache/agentic_loop.py [--conversations N] [--turns N]
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SERVER = os.path.join(ROOT, "build-vk", "bin", "llama-server")
SCRATCH = "/home/herlanggays/.jcode/scratch/ssd-cache"
CACHE_DIR = os.path.join(SCRATCH, "loop-cache")
MODEL = "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf"

PORT = 8749
N_CTX = 8192
N_PREDICT = 1

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "kilo", "lima"]

preamble = None


def words_text(n, tag):
    return tag + " " + "".join(WORDS[i % 12] + (".\n" if i % 13 == 12 else " ") for i in range(n))


def turn_text(conv, turn, n_words):
    # every appended turn starts with a bracketed marker on its own line, which keeps the join
    # at a token boundary. a join that merges one token makes the cached prefix one token short
    # of the input prefix and the restore cannot be used at all
    return "[%s turn %d] " % (conv, turn) + words_text(n_words, conv) + "\n"


class Server:
    def __init__(self, extra, tag):
        self.path = os.path.join(SCRATCH, "loop-%s.log" % tag)
        self.log = open(self.path, "wb")
        self.proc = subprocess.Popen(
            [SERVER, "-m", MODEL, "-c", str(N_CTX), "-np", "1", "-kvu",
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

    def entry_bytes(self):
        if not os.path.isdir(CACHE_DIR):
            return 0
        return sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR))

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
    body = json.dumps({
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0.0,
        "seed": 1,
        "cache_prompt": True,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    out["wall_s"] = time.time() - t0
    return out


def run_loop(args, extra, tag):
    """One pass over the traffic. Returns the totals."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    srv = Server(extra, tag)
    try:
        srv.wait()

        # each conversation is a growing transcript, reset at the start of its own sequence
        prompts = {c: preamble for c in range(args.conversations)}

        n_prompt = 0
        n_cached = 0
        wall = 0.0
        peak_entry_bytes = 0
        per_request = []

        try:
            for turn in range(args.turns):
                for conv in range(args.conversations):
                    prompts[conv] = prompts[conv] + turn_text("conv%d" % conv, turn, args.growth)
                    r = complete(prompts[conv])
                    t = r["timings"]
                    n_prompt += t["prompt_n"]
                    n_cached += t["cache_n"]
                    wall += r["wall_s"]
                    peak_entry_bytes = max(peak_entry_bytes, srv.entry_bytes())
                    per_request.append((conv, turn, t["prompt_n"], t["cache_n"], r["wall_s"]))
        finally:
            srv.stop()

        return {
            "tag": tag,
            "requests": len(per_request),
            "prompt_n": n_prompt,
            "cache_n": n_cached,
            "wall_s": wall,
            "entry_bytes": peak_entry_bytes,
            "per_request": per_request,
        }
    except Exception:
        srv.stop()
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=4)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--growth", type=int, default=200)
    ap.add_argument("--preamble", type=int, default=260)
    args = ap.parse_args()

    global preamble
    preamble = words_text(args.preamble, "shared preamble")

    total = args.conversations * args.turns
    print("%d conversations x %d turns = %d requests over one slot"
          % (args.conversations, args.turns, total))
    print("each turn appends %d words, preamble %d words" % (args.growth, args.preamble))
    print()

    cached = run_loop(args, ["--cache-ram", "0", "--cache-disk-path", CACHE_DIR,
                             "--cache-disk-mib", "1024"], "with-cache")
    plain = run_loop(args, ["--cache-ram", "0"], "no-cache")

    print("  %-34s %12s %12s" % ("", "with cache", "no cache"))
    print("  %-34s %12d %12d" % ("requests", cached["requests"], plain["requests"]))
    print("  %-34s %12d %12d" % ("prompt tokens processed", cached["prompt_n"], plain["prompt_n"]))
    print("  %-34s %12d %12d" % ("tokens served from the cache", cached["cache_n"], plain["cache_n"]))
    print("  %-34s %11.2fs %11.2fs" % ("wall clock", cached["wall_s"], plain["wall_s"]))
    print("  %-34s %11.1fMiB %11s" % ("entries on disk, peak", cached["entry_bytes"] / 1048576.0, "-"))
    print()
    print("  prompt processing removed:  %.1f%%"
          % (100.0 * (1.0 - cached["prompt_n"] / plain["prompt_n"])))
    print("  prompt processing speedup:  %.2fx" % (plain["prompt_n"] / cached["prompt_n"]))
    print("  wall clock speedup:         %.2fx" % (plain["wall_s"] / cached["wall_s"]))
    print()

    # the last turn of each conversation is the steady state: the prefix is long and every
    # conversation already has an entry, so this is the ratio a long loop converges to. the
    # average above is diluted by the first turns of each conversation, which have nothing
    # cached to reuse yet
    last = args.turns - 1
    cl = [r for r in cached["per_request"] if r[1] == last]
    pl = [r for r in plain["per_request"] if r[1] == last]
    c_pn = sum(r[2] for r in cl)
    c_cn = sum(r[3] for r in cl)
    p_pn = sum(r[2] for r in pl)
    c_wall = sum(r[4] for r in cl)
    p_wall = sum(r[4] for r in pl)
    print("  last turn of each conversation, the steady state:")
    print("  %-34s %12s %12s" % ("", "with cache", "no cache"))
    print("  %-34s %12d %12d" % ("prompt tokens processed", c_pn, p_pn))
    print("  %-34s %12d %12d" % ("tokens served from the cache", c_cn, sum(r[3] for r in pl)))
    print("  %-34s %11.2fs %11.2fs" % ("wall clock", c_wall, p_wall))
    print("  %-34s %11.1f%% %11s" % ("prompt tokens removed",
                                     100.0 * (1.0 - c_pn / p_pn), "-"))
    print("  %-34s %11.2fx %11s" % ("prompt processing speedup", p_pn / c_pn, "-"))
    print("  %-34s %11.2fx %11s" % ("wall clock speedup", p_wall / c_wall, "-"))
    print()

    print("  per request, with cache: conv turn prompt_n cache_n wall")
    for i, (conv, turn, pn, cn, w) in enumerate(cached["per_request"]):
        if i < 8 or i >= len(cached["per_request"]) - 4:
            print("    %-18d %-4d %-8d %-7d %.3f" % (conv, turn, pn, cn, w))
        elif i == 8:
            print("    ...")

    return 0


if __name__ == "__main__":
    sys.exit(main())

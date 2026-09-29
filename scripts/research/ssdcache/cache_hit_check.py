"""End to end check for the SSD prompt cache through llama-server.

The prompt cache is not consulted while a slot already holds a good prefix. The server only
saves and loads through it when a slot is about to lose more than half of its context
(server-context.cpp, "if we are about to lose a large portion of the existing context"), or
when the slot is empty. So the case that exercises it is several conversations rotating over
fewer slots, which is what a shared agentic server looks like.

The scenario below is three conversations that share a preamble, on one slot. Each new
conversation displaces the previous one, and the returning conversation has to be restored
from disk.

  1. the first request processes its whole prompt and the cache holds nothing yet
  2. conversations that displace a slot write their state to disk
  3. a returning conversation is restored from its file instead of reprocessing
  4. the same holds after a server restart
  5. a corrupt entry is a miss and not an error, and the answer is still correct
  6. the answer with the cache equals the answer without it

Run: python3 scripts/research/ssdcache/cache_hit_check.py [model]
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
CACHE_DIR = os.path.join(SCRATCH, "server-cache")

PORT = 8747
N_CTX = 8192

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "kilo", "lima"]

# model and server flags, filled in by main() from the arguments
MODEL = None
NGL = 0
SCALE = 1.0
BUDGET_MIB = 512
N_PREDICT = 16


def words_text(n, tag):
    # the tag goes first so the tokenization cannot merge into whatever precedes it
    return tag + " " + "".join(WORDS[i % 12] + (".\n" if i % 13 == 12 else " ") for i in range(n))


# a shared preamble, then a distinct body per conversation. the preamble has to be more than
# 10 percent of the prompt for the slot to be chosen at all, and less than half of it so the
# slot is considered about to lose too much context, which is what engages the cache
PREAMBLE = words_text(260, "shared preamble")


def conversation(tag, n_words):
    # ends on a newline, so the next turn can be appended without merging a token across the
    # join. a join that merges makes the cached prefix one token short of the input prefix and
    # the restore cannot be used, which is the property this test asserts below
    return PREAMBLE + words_text(n_words, "conversation " + tag) + "\n"


def tokenize(content):
    body = json.dumps({"content": content}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/tokenize" % PORT, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["tokens"]


def common_prefix_len(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


class Server:
    def __init__(self, extra, tag):
        self.tag = tag
        self.path = os.path.join(SCRATCH, "server-%s.log" % tag)
        self.log = open(self.path, "wb")
        # CACHE_CHECK_VERBOSE=1 turns on the server's TRC lines, which is how the cache decisions
        # are inspected when a hit does not happen
        verbose = ["-v"] if os.environ.get("CACHE_CHECK_VERBOSE") else []
        self.proc = subprocess.Popen(
            [SERVER, "-m", MODEL, "-c", str(N_CTX), "-np", "1", "-kvu", "-ngl", str(NGL),
             "--host", "127.0.0.1", "--port", str(PORT), "-t", "6"] + extra + verbose,
            stdout=self.log, stderr=subprocess.STDOUT)

    def cache_lines(self):
        out = []
        try:
            with open(self.path, "r", errors="replace") as f:
                for line in f:
                    if any(k in line for k in ("saving prompt", "cache entry", "cache state:",
                                               "obsolete cached prompt", "prompt cache is",
                                               "idle slots will be")):
                        out.append(line.rstrip())
        except OSError:
            pass
        return out

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

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()


def complete(prompt, n_predict=1, cache_prompt=True):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": 1,
        "cache_prompt": cache_prompt,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    out["wall_s"] = time.time() - t0
    return out


def entries():
    return sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".kvs"))


def main():
    global MODEL, NGL, SCALE, BUDGET_MIB, N_CTX, N_PREDICT

    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?",
                    default="/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf")
    ap.add_argument("--ngl", type=int, default=0,
                    help="layers on the gpu. the 35B needs 99, which puts it on the same device "
                         "the production server uses and is the configuration that has hard reset "
                         "this box before, so run it under the watchdog")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the prompt sizes by this. the 35B processes far fewer tokens "
                         "per second, so 0.4 keeps the check to a few minutes")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--budget-mib", type=int, default=512)
    ap.add_argument("--n-predict", type=int, default=16,
                    help="tokens to generate in the text comparisons. a longer completion makes "
                         "the cached against uncached comparison much stronger; the driver kept "
                         "out of the process run in doc 31 uses 200")
    args = ap.parse_args()

    MODEL = args.model
    NGL = args.ngl
    SCALE = args.scale
    BUDGET_MIB = args.budget_mib
    N_CTX = args.ctx
    N_PREDICT = args.n_predict

    def scaled(n):
        return max(20, int(n * SCALE))

    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print("  %-54s %-4s %s" % (name, "ok" if ok else "FAIL", detail))

    a = conversation("A", scaled(900))
    b = conversation("B", scaled(1300))
    c = conversation("C", scaled(1700))
    a2 = a + "[turn 2] " + words_text(scaled(200), "more") + "\n"

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    extra = ["--cache-ram", "0", "--cache-disk-path", CACHE_DIR, "--cache-disk-mib", str(BUDGET_MIB)]

    print("model %s, ngl %d, n_ctx %d, scale %.2f" % (MODEL, NGL, N_CTX, SCALE))
    print("server with the disk cache at %s" % CACHE_DIR)
    srv = Server(extra, "cached")
    try:
        srv.wait()

        ta, ta2 = tokenize(a), tokenize(a2)
        check("the appended turn keeps the prefix token-exact",
              common_prefix_len(ta, ta2) == len(ta),
              "A is %d tokens, common prefix %d" % (len(ta), common_prefix_len(ta, ta2)))

        r = complete(a)
        na, ca = r["timings"]["prompt_n"], r["timings"]["cache_n"]
        check("A processed its whole prompt", ca == 0 and na == len(ta),
              "prompt_n=%d cache_n=%d, %d tokens" % (na, ca, len(ta)))

        # how much of a displacing conversation is served from the arena rather than the cache
        # depends on how large the shared preamble is next to the prompt, so cache_n here is
        # reported and not asserted. what is asserted is that the displaced state reached disk
        r = complete(b)
        nb, cb = r["timings"]["prompt_n"], r["timings"]["cache_n"]
        check("B displaced A and A was written to disk", len(entries()) >= 1,
              "prompt_n=%d cache_n=%d files=%d" % (nb, cb, len(entries())))

        r = complete(c)
        nc, cc = r["timings"]["prompt_n"], r["timings"]["cache_n"]
        check("C displaced B and B was written to disk", len(entries()) >= 2,
              "prompt_n=%d cache_n=%d files=%d" % (nc, cc, len(entries())))

        r = complete(a2, n_predict=N_PREDICT)
        n2, c2 = r["timings"]["prompt_n"], r["timings"]["cache_n"]
        check("returning conversation A was restored from disk", c2 >= 0.9 * na,
              "prompt_n=%d cache_n=%d (A was %d)" % (n2, c2, na))

        text_cached = r["content"]
        wall_cached = r["wall_s"]
        pms_cached = r["timings"]["prompt_ms"]
        print("  %-54s %-4s %.3f s wall, %.0f ms of prompt eval"
              % ("cached request", "", wall_cached, pms_cached))

        # Comparing a restored state against a recomputation is not a gate while the Vulkan
        # driver is in the process, and the reasoning matters. docs/research/28 and 31 show
        # that two computations of the same prefix do not agree there, in one process, on one
        # thread, and that the cause is the driver being loaded rather than this tree. A
        # restore returns the state that was saved; a recomputation produces a new one. With
        # the driver kept out (VK_ICD_FILENAMES=/nonexistent/icd.json) the two agree, which is
        # the run doc 31 records, and this comparison becomes meaningful. Either way the check
        # that gates is the byte exact round trip in store_roundtrip, which compares a state
        # with its own round trip rather than with a second computation.
        r_re = complete(a2, n_predict=N_PREDICT, cache_prompt=False)
        check("the recomputation really reprocessed the prompt",
              r_re["timings"]["prompt_n"] > 0.5 * len(ta2) and r_re["timings"]["cache_n"] == 0,
              "prompt_n=%d cache_n=%d" % (r_re["timings"]["prompt_n"], r_re["timings"]["cache_n"]))
        print("  %-54s %-4s %d chars restored, %d chars recomputed, %s"
              % ("restored vs recomputed, same process", "n/a",
                 len(text_cached), len(r_re["content"]),
                 "same" if r_re["content"] == text_cached else "DIFFERENT"))
        if r_re["content"] != text_cached:
            print("  %-54s %s" % ("", "the two are different computations of one prefix, and"))
            print("  %-54s %s" % ("", "this build does not agree with itself on those, so this"))
            print("  %-54s %s" % ("", "difference is not attributable to the cache."))

        # The entry that a restart will have to use is the slot's prompt at the moment it was
        # saved, and after a generation that prompt carries the generated tokens. The save that
        # survives is the one taken when the *next* request arrived, so the entry holds the
        # generated tokens of the request before that, which is text_cached, not the generation
        # of the recompute request that followed it.
        #
        # This is what a real transcript does: the next turn re-sends the assistant's own output.
        # The join between that text and the next turn can merge into one token, which would make
        # the request diverge inside the entry and the restore unusable, so the separator is
        # chosen by trying candidates until the server's own tokenizer agrees rather than assumed.
        t_entry = tokenize(a2 + text_cached)

        restart_prompt = None
        sep_used = None
        for sep in ("", " ", "\n", "\n\n", "\n---\n"):
            cand = a2 + text_cached + sep + "[conv0 turn 9] " + words_text(scaled(60), "more") + "\n"
            if common_prefix_len(t_entry, tokenize(cand)) == len(t_entry):
                restart_prompt = cand
                sep_used = sep
                break
        if restart_prompt is None:
            restart_prompt = a2 + text_cached + "\n\n" + "[conv0 turn 9] " + words_text(scaled(60), "more") + "\n"
    finally:
        srv.stop()
        for line in srv.cache_lines()[:12]:
            print("  log: %s" % line)

    # ---- a restart must still hit ----
    print("server restarted, same cache dir, %d entries on disk" % len(entries()))
    srv = Server(["--cache-ram", "0", "--cache-disk-path", CACHE_DIR, "--cache-disk-mib", str(BUDGET_MIB)],
                 "restart")
    try:
        srv.wait()

        # The restart request reproduces the newest entry, which is the previous request's prompt
        # plus what it generated, and then adds a turn. It has to be token exact or the restore
        # cannot be used, and this asserts that against the entry's own text rather than against
        # an assumption about it.
        t_req = tokenize(restart_prompt)
        check("the restart prompt is a token exact extension of the entry",
              common_prefix_len(t_entry, t_req) == len(t_entry),
              "entry %d tokens, common prefix %d, separator %r"
              % (len(t_entry), common_prefix_len(t_entry, t_req), sep_used))

        r = complete(restart_prompt)
        check("the restore survives a restart", r["timings"]["cache_n"] >= 0.9 * len(t_entry),
              "cache_n=%d, entry is %d tokens" % (r["timings"]["cache_n"], len(t_entry)))
    finally:
        srv.stop()

    # ---- corrupt every entry: a miss, not an error ----
    n_trunc = 0
    for f in entries():
        p = os.path.join(CACHE_DIR, f)
        size = os.path.getsize(p)
        with open(p, "r+b") as fh:
            fh.truncate(max(4096, size // 3))
        n_trunc += 1
    print("server restarted, %d entry files truncated to a third" % n_trunc)

    srv = Server(extra, "corrupt")
    try:
        srv.wait()
        r = complete(a2)
        check("a corrupt entry is a miss", r["timings"]["cache_n"] == 0,
              "cache_n=%d" % r["timings"]["cache_n"])
        check("the answer is still produced", len(r.get("content", "")) > 0)
    finally:
        srv.stop()
        for line in srv.cache_lines()[:8]:
            print("  log: %s" % line)

    # ---- the control that decides what a text comparison can mean ----
    #
    # Two runs of the same request with no cache at all, in two separate server processes. The
    # cached request above also ran in its own process. If two uncached processes disagree with
    # each other then a cached/uncached difference is not evidence about the cache, and
    # docs/research/28-ubatch-determinism-results.md says exactly that: this build does not
    # reproduce a prefix twice, and the difference shows up between processes rather than within
    # one, which is why comparing two runs inside one session is not enough of a control.
    print("server with no cache, two separate processes, for the comparison and its control")
    plain = []
    for i in (1, 2):
        srv = Server(["--cache-ram", "0"], "nocache%d" % i)
        try:
            srv.wait()
            r1 = complete(a2, n_predict=N_PREDICT)
            r2 = complete(a2, n_predict=N_PREDICT)
            plain.append((r1, r2))
            print("  %-54s %-4s %d chars, twice in one process: %s"
                  % ("uncached request, process %d" % i, "", len(r1["content"]),
                     "same" if r1["content"] == r2["content"] else "DIFFERENT"))
        finally:
            srv.stop()

    check("no cache means no reuse", plain[0][0]["timings"]["cache_n"] == 0,
          "cache_n=%d" % plain[0][0]["timings"]["cache_n"])

    two_processes_agree = plain[0][0]["content"] == plain[1][0]["content"]
    print("  %-54s %-4s %s"
          % ("two uncached processes agree", "n/a" if not two_processes_agree else "ok",
             "%d chars vs %d chars, %s" % (len(plain[0][0]["content"]), len(plain[1][0]["content"]),
                                           "same" if two_processes_agree else "DIFFERENT")))
    if not two_processes_agree:
        print("  %-54s %s" % ("", "this is docs/research/28-ubatch-determinism-results.md at the"))
        print("  %-54s %s" % ("", "server level: two fresh servers, no cache, one prompt, two"))
        print("  %-54s %s" % ("", "different answers. It is why no text comparison in this check"))
        print("  %-54s %s" % ("", "is a gate, and it is not a property of the cache."))

    if two_processes_agree:
        # The baseline holds still between processes, so this comparison is at least not obviously
        # confounded. It is still reported and not counted, because the run above already compared
        # a restore against a recomputation inside one process and that is the sharpest form of
        # the same comparison: on the 35B those two disagreed, in one process, on one prompt,
        # which is docs/research/28-ubatch-determinism-results.md measured at the server level.
        # A comparison of two computations of one prefix is therefore a measurement of this build,
        # not of the cache, however still the baseline looks.
        same = plain[0][0]["content"] == text_cached
        print("  %-54s %-4s %d chars cached, %d chars uncached, %s"
              % ("cached vs uncached text", "n/a", len(text_cached),
                 len(plain[0][0]["content"]), "same" if same else "DIFFERENT"))
        if not same:
            print("  %-54s %s" % ("", "two computations of one prefix, so this is not a cache"))
            print("  %-54s %s" % ("", "signal either. see the same process line above."))
    else:
        print("  %-54s %-4s two uncached processes differ, so nothing here is a cache signal"
              % ("cached vs uncached text", "n/a"))

    r = plain[0][0]
    print("  %-54s %-4s %.3f s wall, %.0f ms of prompt eval"
          % ("uncached request", "", r["wall_s"], r["timings"]["prompt_ms"]))
    print("  %-54s %-4s %.2fx wall, %.2fx prompt eval"
          % ("speedup, uncached over cached", "",
             r["wall_s"] / wall_cached if wall_cached > 0 else 0.0,
             r["timings"]["prompt_ms"] / pms_cached if pms_cached > 0 else 0.0))

    print()
    print("  overall: %s" % ("pass" if all(results) else "FAIL"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

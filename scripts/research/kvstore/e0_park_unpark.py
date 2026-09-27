"""E0: measure the existing park and unpark path.

The server already parks idle slots into a host RAM prompt cache
(`--cache-ram`, `--cache-idle-slots`). This measures what that costs and what it
saves, so the decision about building a new store is made on numbers.

Sequence per case:

  A  slot 0, long prompt P1     full prompt eval, slot 0 now busy
  B  slot 1, long prompt P2     starting a new task parks idle slot 0
  C  slot 0, prompt P1 again    unpark, expected to be a prefix cache hit

Run twice, once with the prompt cache enabled and once disabled. The disabled run
is the control: if C is fast there too, the speedup is not coming from the cache.

Run: python3 scripts/research/kvstore/e0_park_unpark.py
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SERVER = os.path.join(ROOT, "build-cpu", "bin", "llama-server")
MODEL = "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf"
SCRATCH = "/home/herlanggays/.jcode/scratch/kvstore"
PORT = 8731

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "system", "context", "cache", "memory", "layer",
]


def make_prompt(seed, n_words):
    out = []
    x = seed
    for _ in range(n_words):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(WORDS[x % len(WORDS)])
    return " ".join(out) + "\n"


def post(path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get(path, timeout=30):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_health(proc, deadline=180):
    start = time.time()
    while time.time() - start < deadline:
        if proc.poll() is not None:
            return False
        try:
            get("/health")
            return True
        except Exception:
            time.sleep(0.3)
    return False


def rss_mib(pid):
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def completion(prompt, slot, n_predict=2, pin=True):
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": True,
    }
    if pin:
        body["id_slot"] = slot
    t0 = time.time()
    res = post("/completion", body)
    return {
        "wall_ms": (time.time() - t0) * 1000.0,
        "prompt_n": res.get("timings", {}).get("prompt_n"),
        "prompt_ms": res.get("timings", {}).get("prompt_ms"),
        "predicted_ms": res.get("timings", {}).get("predicted_ms"),
        "cached": res.get("tokens_cached"),
        "cached_detail": (res.get("prompt_tokens_details") or {}).get("cached_tokens"),
    }


def log_stats(logpath):
    stats = {"idle_saves": 0, "load_attempts": 0, "load_failures": 0,
             "restore_failures": 0, "perfect_match": 0}
    try:
        with open(logpath) as f:
            text = f.read()
    except OSError:
        return stats
    stats["idle_saves"] = text.count("__TEST_TAG_CACHE_IDLE_SLOT__")
    stats["load_attempts"] = text.count("looking for better prompt")
    stats["perfect_match"] = text.count("f_keep = 1.000, f_sim = 1.000")
    stats["load_failures"] = text.count("failed to load prompt from cache")
    stats["restore_failures"] = text.count("failed to restore state with size")
    return stats


def run_case(label, extra_args, prompt_words, ctx=8192, unified=False, pin=True):
    os.makedirs(SCRATCH, exist_ok=True)
    logpath = os.path.join(SCRATCH, "e0-%s.log" % label)
    log = open(logpath, "w")

    cmd = [SERVER, "-m", MODEL, "-c", str(ctx), "-np", "2", "-ngl", "0",
           "--port", str(PORT), "--no-webui", "-v"]
    cmd += ["-kvu"] if unified else []
    cmd += extra_args
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    try:
        if not wait_health(proc):
            print("%s: server did not become healthy, see %s" % (label, logpath))
            return None

        p1 = make_prompt(1, prompt_words)
        p2 = make_prompt(999, prompt_words)

        out = {"label": label, "pin": pin, "args": ("-c %d -np 2 %s %s %s"
                                                   % (ctx, "-kvu" if unified else "",
                                                      "pinned" if pin else "auto",
                                                      " ".join(extra_args))).strip()}
        out["rss_start"] = rss_mib(proc.pid)
        out["t_base"] = time.time()

        out["A"] = completion(p1, 0, pin=pin)
        out["A"]["t"] = time.time() - out["t_base"]
        out["rss_after_A"] = rss_mib(proc.pid)

        out["B"] = completion(p2, 1, pin=pin)
        out["B"]["t"] = time.time() - out["t_base"]
        out["rss_after_B"] = rss_mib(proc.pid)

        out["C"] = completion(p1, 0, pin=pin)
        out["C"]["t"] = time.time() - out["t_base"]
        out["rss_after_C"] = rss_mib(proc.pid)

        out["log_stats"] = log_stats(logpath)

        try:
            out["slots"] = get("/slots")
        except Exception as e:
            out["slots_error"] = str(e)

        _report(out)
        return out
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        with open(logpath) as f:
            for line in f:
                if "KV self size" in line or "n_ctx_per_seq" in line or "kv_unified" in line:
                    print("    log: %s" % line.strip())


def _report(out):
    print("%s  (%s)" % (out["label"], out["args"]))
    print("    %-4s %10s %10s %10s %10s" % ("req", "wall_ms", "prompt_n", "cached", "prompt_ms"))
    for k in ("A", "B", "C"):
        r = out[k]
        print("    %-4s %10.1f %10s %10s %10s"
              % (k, r["wall_ms"], r["prompt_n"], r["cached"],
                 ("%.1f" % r["prompt_ms"]) if r["prompt_ms"] else "-"))
    print("    rss MiB: start %.0f  after A %.0f  after B %.0f  after C %.0f"
          % (out["rss_start"], out["rss_after_A"], out["rss_after_B"], out["rss_after_C"]))
    s = out["log_stats"]
    print("    path: idle_saves=%d load_attempts=%d perfect_match=%d load_failures=%d restore_failures=%d"
          % (s["idle_saves"], s["load_attempts"], s["perfect_match"],
             s["load_failures"], s["restore_failures"]))
    a, c = out["A"], out["C"]
    if a["prompt_ms"] and c["prompt_ms"]:
        print("    C vs A prompt_ms: %.1fx" % (a["prompt_ms"] / c["prompt_ms"]))


def main():
    if not os.path.exists(SERVER):
        print("missing %s" % SERVER)
        return 1
    if not os.path.exists(MODEL):
        print("missing %s" % MODEL)
        return 1

    words = int(os.environ.get("E0_WORDS", "900"))

    # Separate KV per slot: each slot owns its own stream, so nothing is evicted and
    # the cache has nothing to do. Kept as the baseline pair.
    split_on = run_case("split_cache_on", ["--cache-ram", "512", "--cache-idle-slots"], words)
    split_off = run_case("split_cache_off", ["--cache-ram", "0"], words)

    # Unified KV with a shared buffer too small for both prompts, which is the case
    # where an idle slot actually has to be parked and later restored.
    on_args = ["--cache-ram", "512", "--cache-idle-slots"]
    uni_on = run_case("unified_cache_on", on_args, words, ctx=2048, unified=True)
    uni_off = run_case("unified_cache_off", ["--cache-ram", "0"], words, ctx=2048, unified=True)

    # Same, without pinning the slot. Only here does the server pick the slot itself
    # and therefore run the restore path. Repeated to check the result is stable.
    uni_on_auto = run_case("unified_cache_on_auto", on_args, words, ctx=2048,
                           unified=True, pin=False)
    uni_on_auto2 = run_case("unified_cache_on_auto2", on_args, words, ctx=2048,
                            unified=True, pin=False)
    uni_off_auto = run_case("unified_cache_off_auto", ["--cache-ram", "0"], words,
                            ctx=2048, unified=True, pin=False)

    # Same as the auto case but with room. This is the one that tells apart two
    # explanations for a failed restore: a broken restore path, or a restore that
    # simply cannot fit next to the state it is replacing.
    uni_on_roomy = run_case("unified_cache_on_roomy", on_args, words, ctx=8192,
                            unified=True, pin=False)
    uni_off_roomy = run_case("unified_cache_off_roomy", ["--cache-ram", "0"], words,
                             ctx=8192, unified=True, pin=False)

    if not all((split_on, split_off, uni_on, uni_off, uni_on_auto, uni_on_auto2,
                uni_off_auto, uni_on_roomy, uni_off_roomy)):
        return 1

    print()
    print("summary, request C is the unpark")
    for a, b, title in ((split_on, split_off, "separate KV per slot"),
                        (uni_on, uni_off, "unified KV, 2048 ctx, slot pinned"),
                        (uni_on_auto, uni_off_auto, "unified KV, 2048 ctx, slot auto"),
                        (uni_on_roomy, uni_off_roomy, "unified KV, 8192 ctx, slot auto")):
        print("    %s" % title)
        print("        C prompt eval:   cache on %7.1f ms   cache off %7.1f ms"
              % (a["C"]["prompt_ms"] or -1, b["C"]["prompt_ms"] or -1))
        print("        B->C gap ms:     cache on %7.1f      cache off %7.1f"
              % ((a["C"]["t"] - a["B"]["t"]) * 1000.0,
                 (b["C"]["t"] - b["B"]["t"]) * 1000.0))
        print("        restore path:    idle_saves=%d load_attempts=%d perfect_match=%d restore_failures=%d"
              % (a["log_stats"]["idle_saves"], a["log_stats"]["load_attempts"],
                 a["log_stats"]["perfect_match"], a["log_stats"]["restore_failures"]))

    print()
    print("    auto repeat: restore_failures run 1 = %d, run 2 = %d"
          % (uni_on_auto["log_stats"]["restore_failures"],
             uni_on_auto2["log_stats"]["restore_failures"]))
    print("    state size: see prompt_save lines in the logs, one entry per saved slot")
    return 0


if __name__ == "__main__":
    sys.exit(main())

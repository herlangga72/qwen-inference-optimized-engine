# MTP speculative decoding Implementation Plan

**Outcome: measured, and the sub-project pays.** 1.27x on CPU and 1.36x on Vulkan, single stream,
against a gate of above about 10%. Measuring it also showed that no separate draft file is needed,
because the MTP head ships inside the model as its last blocks. Results in
`docs/research/09-mtp-speculation-results.md`.
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the MTP head from the target GGUF into a standalone draft model, get `draft-mtp` speculation running on it, and measure whether it pays; productionize only if it does.

**Architecture:** One new tool (`gguf-py`) copies the source metadata and a subset of tensors byte for byte into `draft-mtp.gguf`. The fork's existing `draft-mtp` speculative implementation consumes it through `-md` plus `--spec-type draft-mtp`. No engine changes.

**Tech Stack:** Python 3 with `gguf-py`, and the fork's existing binaries (`llama-completion`, `llama-gguf`), on `build-cpu` and `build-vk`.

**Spec:** `docs/superpowers/specs/2026-09-26-mtp-speculative-design.md`

## Global Constraints

- Commits need explicit human approval for each action (`AGENTS.md`). Every commit step is a checkpoint: ask, and if approval is not given, leave the tree uncommitted and report. Never push.
- Do not add files under `tests/`. The new tool goes in `scripts/research/`.
- ASCII only, no em dashes.
- Models: target `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`. The draft is generated, not downloaded.
- Scratch dir: `/home/herlanggays/.jcode/scratch/mtp/`. The draft file is about 750 MB (326 MB of head plus `token_embd` at 417 MB), which fits the free space on that disk.
- The head is 19 tensors: everything named `blk.40.*`. The loader additionally requires `token_embd.weight` and `output_norm.weight` unconditionally (`src/models/qwen35moe.cpp:load_arch_tensors`), while `output.weight` is `TENSOR_NOT_REQUIRED` and falls back to `token_embd`.
- Generation text leaves `llama-completion` through the log callback (stderr); capture with `2>&1` and strip timestamped lines before diffing.
- The gate for keeping this work is a single-stream speedup above about 10%. Below that, record the negative result and stop.

---

### Task 1: The extraction tool

**Files:**
- Create: `scripts/research/extract_mtp_head.py`

**Interfaces:**
- Consumes: `gguf.GGUFReader` (`fields`, `tensors`, `ReaderTensor.tensor_type`, `.shape`, `.data`), `gguf.GGUFWriter` (`add_architecture`, `add_key_value`, `add_tensor`, `write_header_to_file`, `write_kv_data_to_file`, `write_tensors_to_file`, `close`).
- Produces: a GGUF at the given output path that loads as a qwen35moe model whose only layer is the nextn layer.

- [ ] **Step 1: Write the tool**

Create `scripts/research/extract_mtp_head.py`:

```python
#!/usr/bin/env python3
"""Extract the MTP head of a qwen35moe GGUF into a standalone draft model.

usage: extract_mtp_head.py SOURCE.gguf OUTPUT.gguf

The draft keeps the source metadata and copies the nextn layer's tensors plus the tensors the loader
requires unconditionally, byte for byte, so the quantized types are preserved.
"""
import sys

import numpy as np

from gguf import GGUFReader, GGUFWriter, GGUFValueType

# the nextn layer of the models this fork supports
MTP_LAYER = 40

# required regardless of the mtp-only path, see load_arch_tensors
REQUIRED_GLOBALS = ("token_embd.weight", "output_norm.weight")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    src_path, dst_path = sys.argv[1], sys.argv[2]

    reader = GGUFReader(src_path)
    arch = str(reader.fields["general.architecture"].contents())

    writer = GGUFWriter(dst_path, arch, use_temp_file=False)
    writer.add_architecture()

    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub_type)

    prefix = f"blk.{MTP_LAYER}."
    n_copied = 0
    for tensor in reader.tensors:
        if not (tensor.name.startswith(prefix) or tensor.name in REQUIRED_GLOBALS):
            continue
        writer.add_tensor(tensor.name, np.asarray(tensor.data), raw_shape=list(tensor.shape),
                          raw_dtype=tensor.tensor_type)
        n_copied += 1

    print(f"copied {n_copied} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/mtp
mkdir -p "$D"
PYTHONPATH=gguf-py python3 scripts/research/extract_mtp_head.py "$M" "$D/draft-mtp.gguf"
ls -l "$D/draft-mtp.gguf"
```

Expected: `copied 21 tensors` (19 head tensors plus the two required globals) and a file of about 750 MB.

- [ ] **Step 3: Verify the copy**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/mtp
PYTHONPATH=gguf-py python3 - "$M" "$D/draft-mtp.gguf" <<'PY'
import sys
from gguf import GGUFReader

a = GGUFReader(sys.argv[1])
b = GGUFReader(sys.argv[2])

skipped = {"general.architecture"}
missing = [k for k in a.fields if k not in skipped and k not in b.fields]
differ = [k for k in a.fields if k not in skipped and k in b.fields
          and str(a.fields[k].contents()) != str(b.fields[k].contents())]
print("metadata keys missing in draft:", missing)
print("metadata keys differing in draft:", differ)

bt = {t.name: t for t in b.tensors}
bad = []
for t in a.tensors:
    if not (t.name.startswith("blk.40.") or t.name in ("token_embd.weight", "output_norm.weight")):
        continue
    o = bt.get(t.name)
    if o is None:
        bad.append((t.name, "missing"))
    elif o.tensor_type != t.tensor_type or list(o.shape) != list(t.shape) or bytes(o.data) != bytes(t.data):
        bad.append((t.name, "payload or shape differs"))
print("tensor problems:", bad)
PY
```

Expected: no missing metadata keys, no differing keys, no tensor problems. If metadata keys differ, fix the copy loop before continuing; a value that does not survive the round trip will not survive loading either.

- [ ] **Step 4: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add scripts/research/extract_mtp_head.py
git commit -m "scripts: extract the MTP head into a standalone draft model"
```

---

### Task 2: Load gate

This is where the sub-project can die. If the draft does not load, record the error and stop.

**Files:**
- Modify if needed: `scripts/research/extract_mtp_head.py` (the copy set)

**Interfaces:**
- Consumes: the draft from Task 1.
- Produces: a command line that loads the draft and generates, with the speculative implementation active.

- [ ] **Step 1: Try to load it**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/mtp
./build-cpu/bin/llama-completion -m "$M" -md "$D/draft-mtp.gguf" --spec-type draft-mtp \
    -p "The capital of France is" -n 16 -s 1 --temp 0 -c 256 -ngl 99 2>&1 | tail -25
```

Note: `-ngl 99` is harmless on `build-cpu` (no GPU device) and correct for `build-vk`.

Expected: the log names the speculative implementation, the model loads, and text is generated. If instead the loader reports missing tensors, add each named tensor to the copy set in `extract_mtp_head.py`, re-run Task 1 step 2 and step 3, and try again. Two tensors are already known to be required and are copied. A report that `blk.40` is unused means the draft was not picked up, check the `-md` path.

- [ ] **Step 2: Confirm the draft model's shape matches what the implementation asserts**

```bash
cd /home/herlanggays/RISET/llama.cpp
D=/home/herlanggays/.jcode/scratch/mtp
./build-cpu/bin/llama-gguf "$D/draft-mtp.gguf" r 2>&1 | grep -iE "nextn|block_count|mtp" | head -8
./build-cpu/bin/llama-completion -m "$D/draft-mtp.gguf" -p "hello" -n 1 -c 64 2>&1 | grep -iE "nextn|nextn_predict|n_layer|error|assert" | head -12
```

Expected: `qwen35moe.nextn_predict_layers = 1` and a load that does not assert. `common/speculative.cpp` asserts that the draft's and target's output embedding widths match; if that fires, the head cannot serve as this target's draft and the sub-project stops here with that finding.

- [ ] **Step 3: Commit (checkpoint)**

If the tool changed, ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add scripts/research/extract_mtp_head.py
git commit -m "scripts: complete the MTP head copy set"
```

---

### Task 3: Measure

**Files:**
- Create: `docs/research/09-mtp-speculation-results.md`
- Create (scratch): `/home/herlanggays/.jcode/scratch/mtp/*.txt`

**Interfaces:**
- Consumes: the working command line from Task 2.
- Produces: the acceptance and throughput table that the gate in Task 4 reads.

- [ ] **Step 1: Baselines without speculation, both backends**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/mtp
P="The capital of France is"

./build-cpu/bin/llama-completion -m "$M" -p "$P" -n 128 -s 1 --temp 0 -c 2048 > "$D/base-cpu.txt" 2>&1
./build-vk/bin/llama-completion  -m "$M" -p "$P" -n 128 -s 1 --temp 0 -c 2048 -ngl 99 > "$D/base-vk.txt" 2>&1
grep -E "eval time" "$D/base-cpu.txt" "$D/base-vk.txt"
```

Expected: `eval time` lines giving ms per token; that is the number to beat, about 44 ms on Vulkan.

- [ ] **Step 2: With speculation, per backend and per draft length**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/mtp
P="The capital of France is"

for n in 1 2 3; do
  ./build-cpu/bin/llama-completion -m "$M" -md "$D/draft-mtp.gguf" --spec-type draft-mtp --draft $n \
      -p "$P" -n 128 -s 1 --temp 0 -c 2048 > "$D/spec-cpu-n$n.txt" 2>&1
  ./build-vk/bin/llama-completion -m "$M" -md "$D/draft-mtp.gguf" --spec-type draft-mtp --draft $n \
      -p "$P" -n 128 -s 1 --temp 0 -c 2048 -ngl 99 > "$D/spec-vk-n$n.txt" 2>&1
done
grep -E "eval time|accept|draft" "$D"/spec-*.txt | head -40
```

Expected: each run reports an `eval time` per token and acceptance statistics (accepted drafts, accepted tokens, per position). If the statistics do not appear, add `-v` to one run to confirm they are printed at all before concluding anything about acceptance.

- [ ] **Step 3: Correctness gate**

```bash
cd /home/herlanggays/RISET/llama.cpp
D=/home/herlanggays/.jcode/scratch/mtp
for f in "$D"/spec-*.txt; do
  diff <(grep -v "^0\." "$D/base-vk.txt" | grep -v "eval time" | grep -v "total time" | grep -v "sampling time") \
       <(grep -v "^0\." "$f" | grep -v "eval time" | grep -v "total time" | grep -v "sampling time") > /dev/null \
    && echo "IDENTICAL $(basename "$f")" || echo "DIFFERS  $(basename "$f")"
done
```

Expected: all identical at temperature 0, since verification is exact. A difference means a defect in the speculative path or in the extracted head, and it stops the measurement until explained.

- [ ] **Step 4: A second prompt and a longer context**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/mtp

./build-vk/bin/llama-completion -m "$M" -md "$D/draft-mtp.gguf" --spec-type draft-mtp --draft 1 \
    -p "Write one sentence about the sea." -n 128 -s 2 --temp 0 -c 2048 -ngl 99 > "$D/spec-vk-p2.txt" 2>&1
./build-vk/bin/llama-completion -m "$M" -md "$D/draft-mtp.gguf" --spec-type draft-mtp --draft 1 \
    -p "$(head -c 4000 /dev/zero | tr '\0' 'a')" -n 64 -s 1 --temp 0 -c 16384 -ngl 99 > "$D/spec-vk-long.txt" 2>&1
grep -E "eval time|accept" "$D/spec-vk-p2.txt" "$D/spec-vk-long.txt"
```

Expected: acceptance in the same range as the first prompt. A very different rate is a finding, not a failure.

- [ ] **Step 5: Write the results**

Create `docs/research/09-mtp-speculation-results.md` with: the baseline ms per token, the speculation table (backend, `--draft` value, ms per token, speedup, accepted tokens per step, per-position acceptance), the identity check results, and the second-prompt and long-context numbers. State plainly whether the speedup clears 10%.

- [ ] **Step 6: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add docs/research/09-mtp-speculation-results.md
git commit -m "docs: record the MTP speculation measurements"
```

---

### Task 4: Gate and documentation

**Files:**
- Modify: `docs/research/09-mtp-speculation-results.md` (add the decision)
- Modify: `QWEN_ONLY.md` (verification log) only if speculation is kept

**Interfaces:**
- Consumes: Task 3's table.
- Produces: either a documented configuration or a recorded negative result.

- [ ] **Step 1: Apply the gate**

If the best single-stream speedup is above about 10%: add a section to `09-mtp-speculation-results.md` titled "How to run it" with the exact command line, the recommended backend and `--draft` value, and the measured speedup, and add an entry to the verification log in `QWEN_ONLY.md` in the same table format as the existing entries.

If it is at or below 10%: add a section titled "Decision: not kept" with the acceptance numbers and the reason, and do not touch `QWEN_ONLY.md`.

- [ ] **Step 2: Confirm nothing else moved**

```bash
cd /home/herlanggays/RISET/llama.cpp
./build-cpu/bin/test-llama-archs -a qwen35moe -s 1
git status --short | grep -v "^??"
```

Expected: the arch test passes and no tracked file outside the documents and the new script has changed. This sub-project adds no engine code, so a diff anywhere else means something unintended happened.

- [ ] **Step 3: Commit (checkpoint)**

Ask for approval, then commit whichever documentation changed:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add docs/research/09-mtp-speculation-results.md QWEN_ONLY.md
git commit -m "docs: record the MTP speculation decision"
```

---

## Self-Review

**Spec coverage:** the spec's tool maps to Task 1; its load risk and the missing-tensor discovery loop map to Task 2; its measurement, acceptance and identity requirements map to Task 3; its productionize gate maps to Task 4; its out-of-scope list produces no tasks, as intended.

**Placeholder scan:** no TBD. The one open-ended step (Task 2 step 1) is a discovery loop with an exact command, the expected error shape, and the bounded fix, because the loader's required tensor set cannot be fully enumerated without running it; the two known-required globals are already in the copy set.

**Type consistency:** the tool writes `draft-mtp.gguf`, and every later task uses that exact name; `MTP_LAYER = 40` matches `nextn_predict_layers = 1` with `block_count = 41`; the flag names `--spec-type draft-mtp`, `--spec-draft-model`/`-md` and `--draft` match `common/arg.cpp`.

**Known gap, stated rather than hidden:** Task 2 can end the sub-project, and Task 3 can end it at the gate. Both outcomes are documented rather than left as failures, which is the point of measuring before productionizing.

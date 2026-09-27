# Recurrent state view (phase 1): pre-change baselines

Date: 2026-09-27
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB, Vulkan0 + CPU.
Tree: `qwen-only-backends` at 4f5831ba2, both build trees rebuilt from it.
Plan: `docs/superpowers/plans/2026-09-26-recurrent-state-view-phase1.md`.
Spec: `docs/superpowers/specs/2026-09-26-recurrent-state-copies-design.md`.

Models: dense `/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf`,
MoE `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`
(13.09 GiB, qwen35moe, IQ3_S 3.4375 bpw).

## Method

The plan's Task 1 commands, run one at a time through
`/home/herlanggays/.jcode/scratch/rs-view/capture.sh`, which logs `MemAvailable`, GTT, VRAM and
Cached around every step and kills the child if `MemAvailable` falls under 1.5 GiB. The machine
froze seven times today under this workload (see `docs/research/freeze-notes-2026-09-27.md`), so
every step that touches the 35B model is serialized and watched.

Two additions to the plan's command set, both recorded below:

- `llama-passkey -np 2` is not the multi-sequence check the plan assumes. The tool runs one group
  (`n_grp = grp_attn_n`, default 1) and overrides `-c` with `n_ctx_train*n_grp + n_keep`, so the
  plan's `-c 4096` has no effect and only one sequence is ever decoded.
- `llama-batched -np N` is used instead for the multi-sequence case. It needs `-kvu` on this tree:
  without it the run stops at `split_equal: sequential split is not supported when there are
  coupled sequences in the input batch`.

Raw logs: `/home/herlanggays/.jcode/scratch/rs-view/*.txt`. Every number below comes from those
files, and each file's digest is the reference Task 2 diffs against.

## Single sequence generation

`llama-completion -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64`, CPU and Vulkan.

| file | backend | stripped digest (16 hex) | eval |
| --- | --- | --- | --- |
| gen-0.8b-cpu.txt | CPU | `a599102cce02ed30` | 57.10 t/s |
| gen-0.8b-vk.txt | Vulkan0 | `a599102cce02ed30` | 76.84 t/s |
| gen-35b-vk.txt | Vulkan0 | `dfbe0e02dec9d892` | 21.36 t/s |

The digest is `grep -v "^0\." <file> | sha256sum`, which drops the timestamped log lines and keeps
the generated text. The 0.8B runs are byte identical on both backends. The generated text of the
dense model is `The capital of France is **Paris**.` followed by an explanation; the MoE model
answers the same, in a shorter reply.

## Multi-sequence generation

`llama-batched -p "The capital of France is" -np N -n 16 -c 512 -s 1 --temp 0 -kvu` with `-ngl 99`
for Vulkan0, two repeats per configuration.

| configuration | digest | repeat 2 | CPU vs Vulkan0 |
| --- | --- | --- | --- |
| 0.8B, np=2, CPU | `031fec546e66` | identical | identical |
| 0.8B, np=2, Vulkan0 | `031fec546e66` | identical | identical |
| 0.8B, np=4, CPU | `166ad7dab0f5` | identical | identical |
| 0.8B, np=4, Vulkan0 | `166ad7dab0f5` | identical | identical |

Digest of `python3 norm-msgen.py <file>`. The raw logs cannot be diffed directly: the sequence
blocks and the log lines interleave in a different order between runs, and the speed line varies.
Both sequences in every configuration generate the same text:

```
sequence 0:
The capital of France is the capital of the country.
The capital of France
```

Per-sequence text is identical at np=2 and np=4, which is what `n_seqs = 1` and `n_seqs > 1` are
expected to do when each sequence carries its own state row.

## Passkey

`llama-passkey -m <0.8B> -np 2 -n 16 -c 4096 -s 1`, CPU and Vulkan0.

| backend | inserted | retrieved | n_ctx actually used | eval |
| --- | --- | --- | --- | --- |
| CPU | 30887 | `The pass key is 30887.` | 262624 | 52.69 t/s |
| Vulkan0 | 30887 | `The pass key is 30887.` | 262624 | 24.77 t/s |

Digest `ae5f3046d0bfef0e` for both. Both backends insert and retrieve the same key. As noted above
this is a single-sequence run.

## Throughput, 35B, Vulkan0

`llama-bench -m <35B> -r 3 -p 0 -n 128 -t 8 -ngl 99`:

| test | t/s |
| --- | --- |
| tg128 | 22.33 +/- 0.21 |

`llama-batched-bench -m <35B> -ngl 99 -t 8 -c 4096 -npp 128 -ntg 32 -npl 1,2,4,8`:

| PP | TG | B | N_KV | S_PP t/s | S_TG t/s | S t/s |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 32 | 1 | 160 | 143.23 | 21.46 | 67.08 |
| 128 | 32 | 2 | 320 | 190.05 | 32.54 | 96.57 |
| 128 | 32 | 4 | 640 | 235.95 | 40.51 | 120.07 |
| 128 | 32 | 8 | 1280 | 239.60 | 44.30 | 127.33 |

## Correction to the plan's expected numbers

Task 1 step 3 expects `S_TG` at B=8 to be "about 31.6 t/s", and Task 2 step 10 expects it to rise
to "about 44.5". The baseline already reads 44.30. Three readings of the same table make the point:

- this capture, `-c 4096 -npp 128`: S_TG at B=8 = 44.30
- the delta-net output projection capture, `-c 16384 -npp 512`: S_TG at B=8 = 34.66
- the spec's ablation table: pristine 31.61, ablation B 44.52

Two candidate explanations, neither of which the recorded evidence can separate: the ablation
command is not written down anywhere (it is not in the spec and not in
`docs/research/04-stage5-diagnosis.md`), so its shapes are unknown; and the tree has since gained
the delta-net output projection change, which the op-level measurement shows to be worth 2.2x on
the Vulkan np=8 decode shape, the same phase and case the ablation attacked.

Consequence for Tasks 2 and 3: the comparison basis is the table above, taken with the plan's own
command, not the spec's 31.61/44.52 pair. If `S_TG` at B=8 does not move after the change, do not
conclude that the predicate is false; first check whether the gain the spec predicted was already
banked by the delta-net output projection change. Task 2 step 10's "if nothing moves" branch needs
that check added before it means anything.

Also worth noting for Task 2 step 9: `llama-passkey` cannot show that the multi-sequence state
mapping did not change, because it decodes one sequence. Use `llama-batched` with the digests above.

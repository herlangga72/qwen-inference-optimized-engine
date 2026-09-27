# Recurrent state copies in the delta-net path: Implementation Design

Date: 2026-09-26
Status: implemented and measured
Outcome: `build_rs` returns a view of the cache instead of gathering rows when the active rows are
already contiguous, detected by `state_rows_are_contiguous` in `src/llama-memory-recurrent.cpp`. The
per-token gather and its write-back are gone, and outputs are byte identical to before, checked
across CPU and Vulkan and across batch shapes. Results in
`docs/research/08-recurrent-state-phase1-results.md`.
Scope: `src/llama-graph.cpp` (`build_rs`), `src/llama-memory-recurrent.cpp`, `src/models/delta-net-base.cpp`, and, for phase 2, the GATED_DELTA_NET op in `ggml` plus its CPU and Vulkan implementations
Related evidence: `docs/research/04-stage5-diagnosis.md`

## Context

The gated delta net layers keep a per-sequence recurrent state of 2 MB (`524288` f32 elements per
layer, `blk.N.ssm_*`). Every layer every token touches that state three times:

1. `build_rs` gathers the active rows out of the cache into a graph buffer (`GET_ROWS`).
2. the delta-net op reads that buffer and produces a new state.
3. `ggml_cpy` writes the new state back into the cache rows
   (`src/models/delta-net-base.cpp:555-558`).

Both copies are per layer per token, so a 40 layer model with 30 delta-net layers pays 60 extra
graph ops per decode step.

## Evidence

Measured by ablation on the Vulkan backend, `-ngl 99`, single build tree, same commands in one
session. Ablation A comments out the write-back `ggml_cpy`; ablation B replaces the gather with a
view of the head rows. Both are timing probes; the tree was restored and the baseline reproduced
afterwards (tg128 22.35 before, 22.67 after; batch 8 aggregate 31.61 before, 32.28 after).

| build | tg128 | vs baseline | batch 8 aggregate | vs baseline |
| --- | --- | --- | --- | --- |
| pristine | 22.35 / 22.67 | - | 31.61 / 32.28 | - |
| A: no state write-back | 23.67 | +5.9% | 43.11 | +36% |
| B: gather replaced by head-row view | 24.01 | +7.4% | 44.52 | +41% |

Readings:

- Neither copy is hidden; both cost real decode time.
- Ablation B is equivalent to the phase 1 fast path rather than merely a probe: with `n_rs == 1` and
  `get_head() == 0` the head rows are the gathered row by construction, so a view is the same tensor
  contents. The plan still has to verify output identity explicitly, the ablation run did not settle
  it (the generated text leaves through the log callback, so the redirection used there dropped it).
- At batch 8 both ablations win about 40%, far above the 18% the copy bytes predict, and the whole
  gain lands in the decode phase (prefill time is unchanged). In decode these ops are latency and
  serialization bound.
- The two are not additive; they remove the same serialization.

## Phase 1: gather to view

**Requirement.** When the active state rows for the ubatch are exactly `[head, head+n_seqs)` in
order, `build_rs` must return a view of the cache instead of gathering.

**Design.**

- `llama_memory_recurrent_context` exposes a predicate, for example
  `bool state_rows_are_contiguous() const`, computed where the row mapping is decided
  (`llama_memory_recurrent::init_batch`, which fills `cells[i].src0`), true when
  `cells[i].src0 == rs_head + i` for every `i < n_seqs`. The existing `get_head()` and `get_n_rs()`
  already return `0` and `size` in the `is_full` case, so the fully occupied cache satisfies the
  predicate by construction.
- `build_rs` (src/llama-graph.cpp:2574) returns
  `ggml_view_2d(ctx0, s, state_size, n_seqs, s->nb[1], (size_t) rs_head * s->nb[1])` when the
  predicate holds, and the existing gather otherwise. The extra-state copy at the end of `build_rs`
  stays; it is zero sized when `n_rs == n_seqs`.
- The predicate is added to `llm_graph_input_rs::can_reuse` alongside the existing `rs_z` check, so a
  graph built for a contiguous mapping is never reused after the mapping changes.

**Correctness.** The view must cover exactly the rows that `s_copy` would have gathered. That is the
definition of the predicate. The gather path remains for every other case.

**Expected result.** About +7% single stream and the batch 8 gain that ablation B measured, for the
cases where the predicate holds. In the single-sequence case the predicate holds always, and
ablation B already demonstrated both the speed and a correct result there.

**Risk.** A wrong or stale predicate makes the delta-net read another sequence's state. This fails
silently with plausible output, so the verification must include a case where the predicate is false
(multi-sequence, slow path, output must be unchanged) and a case where it is true.

## Phase 2: dropped, the arithmetic does not support it

Phase 2 was scoped as "remove the write-back copy". Reading the op's contract before writing its plan
shows the goal is not reachable at a profit, so it is dropped and recorded here.

`ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K)` takes the state as `src[5]` and produces a
single dst holding the attention output followed by `K` state snapshots
(`ggml/src/ggml.c:6349`). One node has one dst, so the new state has exactly two possible
itineraries:

1. **Today**: the op writes the state into its dst (2 MB), then `ggml_cpy` reads it (2 MB) and writes
the cache row (2 MB). The op also reads the old state (2 MB). State traffic: 8 MB per layer per token.
2. **In place**: split into an attention op (`K = 0`, reads the old state, writes only the attention
output) plus an in-place state update op (reads the old state 2 MB, writes the cache row 2 MB).
State traffic: 6 MB.

Ablation A measured 5.9% single stream by deleting the copy, but it also stopped updating the state,
so it removed 4 MB of movement that a correct implementation must still perform somewhere. The gain
available to a correct phase 2 is therefore one graph edge per layer, not 4 MB per layer per token,
and the in-place split additionally re-reads `k`, `v`, `g`, `b` for the state half.

The only shape that could both write the state into the cache and keep the attention output is a
single op whose dst covers two regions of the state buffer, the cache row and a scratch area for the
attention output. That is expressible but intrusive: the attention scratch would live inside the
recurrent state buffer, and both backends would have to accept a split dst. It is not proposed here.

**Phase 1 stands on its own**: ablation B replaced the gather with a head-row view and produced
correct output, +7.4% single stream and +41% at batch 8. That is the part of this sub-project that is
worth implementing, and it is unaffected by the finding above.

**What would change the conclusion:** a measured, correct in-place prototype showing a gain above a
few percent. That requires the op work to exist first, so the order would be to build it, measure it,
and keep it only if it pays; on the arithmetic above, that is not a good use of effort.

**History, kept because it bounds the idea.** Two mechanisms were considered before the arithmetic
above was worked out, and both are recorded so the ground is not re-covered:

- Unfusing the whole delta net to get in-place primitives costs too much. Measured: with the fused
  autoregressive GDN op disabled, tg128 falls from 22.67 to 16.20 t/s and the batch 8 aggregate from
  32.28 to 23.32, 28.5% and 27.8%. That cannot be paid for by removing a copy.
- A new in-place state op is the shape the split would take, and it is what the arithmetic in the
  section above rejects on traffic grounds rather than on feasibility: an in-place op aliasing the
  state tensor is expressible with `ggml_view_tensor`, the way `ggml_scale_inplace` works.

**Risks that remain relevant if phase 2 is ever revived:** in-place state updates create
write-after-read hazards between the attention computation and the state update, which the graph
scheduler only respects through explicit dependencies; both backends must agree on the aliasing
rules; and the rollback path (`n_rs_seq > 0`) and the chunked prefill path must be re-verified rather
than assumed.

## Verification (both phases), per phase

1. `test-llama-archs -a qwen35 -a qwen35moe -a qwen4exp -s 1`.
2. Byte identical generation against the pre-change build on the 35B MoE and the 0.8B dense, both
   backends, fixed seed, using `llama-completion` with `--temp 0`. Outputs go to the terminal through
   the log callback, so capture with `2>&1` and strip timestamped lines before diffing.
3. A multi-sequence case with the predicate false, to prove the slow path is unchanged:
   `llama-parallel -m MODEL -np 4 -n 16 --temp 0 -s 1 -ngl 99` prints one text per sequence and is
   deterministic at temperature 0, so diff its output against the pre-change build. `llama-parallel`
   exists in both build trees.
4. `llama-bench` tg128 and pp256 with `-ngl 99`, and `llama-batched-bench` at 1, 2, 4, 8, 16
   sequences, compared against the numbers in the ablation table.
5. `ctest -L main` on both builds, and `test-backend-ops` for the changed ops.
6. Record everything in `QWEN_ONLY.md` and in `docs/research/`.

## Out of scope

- The f16 or otherwise reduced precision recurrent state (a quality trade, and it needs the op type
  path).
- The chunked prefill path.
- The `SCALE` zeroing op, which only runs when a sequence slot is cleared, not in steady state.
- MoE expert kernels (measured byte optimal) and the delta-net output projection (its own spec).

## Follow-ups

The delta-net output projection spec (`docs/superpowers/specs/2026-09-26-delta-net-output-projection-design.md`)
is independent: it removes a per-sequence weight re-read in the same layers, worth about 10% at batch
8. Both changes touch the same builder, so whichever lands second must re-measure.

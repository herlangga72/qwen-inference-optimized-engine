#!/usr/bin/env python3
"""Reproduce the planar3_0 Lloyd-Max codebook and print it as GLSL literals.

The Vulkan flash attention shader cannot solve Lloyd-Max at runtime, so the 8 centroids have to be
literals there. This mirrors planar3_0_solve_lloyd_max() from ggml/src/ggml-quants.c exactly: the same
Abramowitz and Stegun erf, the same starting points, the same 300 iteration cap and the same 1e-12
stopping rule. The final float() cast is a round trip through float32, which is what the C code does
when it writes `out[i] = (float) c[i]`.

Usage:
    planar3_codebook.py
"""

import math
import struct


def erf_as(x):
    """Abramowitz and Stegun 7.1.26, the same one the host uses."""
    sign = -1.0 if x < 0.0 else 1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * math.exp(-x * x)
    return sign * y


def norm_pdf(x, sigma):
    return math.exp(-(x * x) / (2.0 * sigma * sigma)) / (sigma * math.sqrt(2.0 * math.pi))


def norm_cdf(x, sigma):
    return 0.5 * (1.0 + erf_as(x / (sigma * math.sqrt(2.0))))


def solve_lloyd_max(sigma, levels):
    lo = -3.5 * sigma
    hi = 3.5 * sigma
    outer = 12.0 * sigma
    c = [lo + (hi - lo) * (i + 0.5) / levels for i in range(levels)]
    iters = 0
    for it in range(300):
        iters = it + 1
        nxt = []
        shift = 0.0
        for i in range(levels):
            a = -outer if i == 0 else 0.5 * (c[i - 1] + c[i])
            b = outer if i == levels - 1 else 0.5 * (c[i] + c[i + 1])
            mass = norm_cdf(b, sigma) - norm_cdf(a, sigma)
            if mass > 1e-15:
                nxt.append(sigma * sigma * (norm_pdf(a, sigma) - norm_pdf(b, sigma)) / mass)
            else:
                nxt.append(c[i])
            shift = max(shift, abs(nxt[-1] - c[i]))
        c = nxt
        if shift < 1e-12:
            break
    return [f32(v) for v in c], iters


def f32(v):
    """Round a double to the nearest float32, as the C cast does."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def splitmix64(state):
    """Mirror of planar3_0_splitmix64() in ggml-quants.c, returning (new_state, value)."""
    mask = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & mask
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return state, z ^ (z >> 31)


def givens_table():
    """Mirror of planar3_0_givens(): cos, sin per pair, angles from splitmix64 seed 42."""
    state = 42
    table = []
    for _ in range(256 // 2):
        state, r = splitmix64(state)
        u = (r >> 11) / float(1 << 53)
        t = u * 2.0 * math.pi
        table.append(f32(math.cos(t)))
        table.append(f32(math.sin(t)))
    return table


def main():
    sigma = 1.0 / math.sqrt(256.0)
    cb, iters = solve_lloyd_max(sigma, 8)
    print(f"sigma = {sigma!r} (1/sqrt(256)), converged in {iters} iterations")
    print()
    print("// Lloyd-Max centroids for sigma = 1/16, from ggml-quants.c planar3_0_codebook().")
    print("// Reproduced by scripts/research/kvplanar3/planar3_codebook.py, which mirrors the solver.")
    print("const float PLANAR3_0_CB[8] = {")
    for i, v in enumerate(cb):
        print(f"    {v:+.9e}, // cb[{i}]")
    print("};")
    print()
    print("values:", cb)
    # a Gaussian 3 bit Lloyd-Max codebook must be symmetric about zero and cover about +-3.5 sigma
    print(f"max |cb| = {max(abs(v) for v in cb):.9e}, 3.5*sigma = {3.5*sigma:.9e}")
    print(f"symmetric: {all(abs(cb[i] + cb[7 - i]) < 1e-9 for i in range(8))}")

    # The write side needs the same rotation the host quantizer applies, and a shader cannot
    # reproduce it: the angles come from a 64 bit PRNG and a double precision cosine.
    rot = givens_table()
    print()
    print("// The Givens table from planar3_0_givens(): cos, sin per coordinate pair, angles drawn from")
    print("// splitmix64 with seed 42. 128 pairs, one 2x2 block diagonal rotation per pair.")
    print("const float PLANAR3_0_GIVENS[256] = {")
    for p in range(0, 256, 8):
        row = ", ".join(f"{v:+.9e}" for v in rot[p:p + 8])
        print(f"    {row},")
    print("};")
    # a rotation table must satisfy cos^2 + sin^2 == 1 per pair
    worst = max(abs(rot[2*i]*rot[2*i] + rot[2*i + 1]*rot[2*i + 1] - 1.0) for i in range(128))
    print(f"worst |cos^2 + sin^2 - 1| over 128 pairs = {worst:.3e}")


if __name__ == "__main__":
    main()

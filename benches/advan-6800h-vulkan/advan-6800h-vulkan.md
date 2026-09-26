## System info

```bash
Linux herlangga-laptop 7.1.13-2-MANJARO #1 SMP PREEMPT_DYNAMIC Fri, 04 Sep 2026 15:41:39 +0000 x86_64 GNU/Linux
16 threads, AMD Ryzen 7 6800H with Radeon Graphics, 27.1 GiB RAM
```

## Memory ceiling

```
membw: threads=8 run=0 46.81 GB/s (chk=1736164148112261120)
membw: threads=8 run=1 46.38 GB/s (chk=1736164148112261120)
membw: threads=8 run=2 45.03 GB/s (chk=1736164148112261120)
membw: threads=8 BEST 46.81 GB/s
membw: threads=16 run=0 44.88 GB/s (chk=1736164148112261120)
membw: threads=16 run=1 42.89 GB/s (chk=1736164148112261120)
membw: threads=16 run=2 42.29 GB/s (chk=1736164148112261120)
membw: threads=16 BEST 44.88 GB/s
```

## Model

```
-rw-r--r-- 1 herlanggays herlanggays 14069266720 Sep 26 18:33 /home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
pagecache: /home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
pagecache: resident 2663153/3434879 pages = 77.5% (10.16 GiB)
```

## llama-bench

#### prompt processing, 8 threads

ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon 680M (RADV REMBRANDT) (radv) | uma: 1 | fp16: dot2 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | Vulkan     |  99 |           pp256 |        202.41 ± 2.99 |

build: 175ade4cf (11207)

#### token generation against thread count

ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon 680M (RADV REMBRANDT) (radv) | uma: 1 | fp16: dot2 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | Vulkan     |  99 |       4 |           tg128 |         22.67 ± 0.12 |

build: 175ade4cf (11207)
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon 680M (RADV REMBRANDT) (radv) | uma: 1 | fp16: dot2 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | Vulkan     |  99 |           tg128 |         22.46 ± 0.10 |

build: 175ade4cf (11207)
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon 680M (RADV REMBRANDT) (radv) | uma: 1 | fp16: dot2 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | Vulkan     |  99 |      16 |           tg128 |         22.48 ± 0.06 |

build: 175ade4cf (11207)

#### token generation against load mode, 8 threads

ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon 680M (RADV REMBRANDT) (radv) | uma: 1 | fp16: dot2 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl |         lm |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | Vulkan     |  99 |       mmap |           tg128 |         21.62 ± 0.98 |

build: 175ade4cf (11207)
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon 680M (RADV REMBRANDT) (radv) | uma: 1 | fp16: dot2 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl |         lm |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | Vulkan     |  99 |       none |           tg128 |         22.40 ± 0.13 |

build: 175ade4cf (11207)
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon 680M (RADV REMBRANDT) (radv) | uma: 1 | fp16: dot2 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl |         lm |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | Vulkan     |  99 |      mlock |           tg128 |         22.37 ± 0.05 |

build: 175ade4cf (11207)


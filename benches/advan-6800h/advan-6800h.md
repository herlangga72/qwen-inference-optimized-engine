## System info

```bash
Linux herlangga-laptop 7.1.13-2-MANJARO #1 SMP PREEMPT_DYNAMIC Fri, 04 Sep 2026 15:41:39 +0000 x86_64 GNU/Linux
16 threads, AMD Ryzen 7 6800H with Radeon Graphics, 27.1 GiB RAM
```

## Memory ceiling

```
membw: threads=8 run=0 43.85 GB/s (chk=1736164148112261120)
membw: threads=8 run=1 46.54 GB/s (chk=1736164148112261120)
membw: threads=8 run=2 44.23 GB/s (chk=1736164148112261120)
membw: threads=8 BEST 46.54 GB/s
membw: threads=16 run=0 42.18 GB/s (chk=1736164148112261120)
membw: threads=16 run=1 41.42 GB/s (chk=1736164148112261120)
membw: threads=16 run=2 44.04 GB/s (chk=1736164148112261120)
membw: threads=16 BEST 44.04 GB/s
```

## Model

```
-rw-r--r-- 1 herlanggays herlanggays 14069266720 Sep 26 18:33 /home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
pagecache: /home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
pagecache: resident 1960429/3434879 pages = 57.1% (7.48 GiB)
```

## llama-bench

#### prompt processing, 8 threads

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | CPU        |       8 |           pp256 |         61.27 ± 0.55 |

build: 04e10983e (11209)

#### token generation against thread count

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | CPU        |       4 |           tg128 |         12.97 ± 0.08 |

build: 04e10983e (11209)
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | CPU        |       8 |           tg128 |         15.26 ± 0.44 |

build: 04e10983e (11209)
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | CPU        |      16 |           tg128 |         11.29 ± 0.44 |

build: 04e10983e (11209)

#### token generation against load mode, 8 threads

| model                          |       size |     params | backend    | threads |         lm |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | CPU        |       8 |       mmap |           tg128 |         15.88 ± 0.12 |

build: 04e10983e (11209)
| model                          |       size |     params | backend    | threads |         lm |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | CPU        |       8 |       none |           tg128 |         15.71 ± 0.16 |

build: 04e10983e (11209)
| model                          |       size |     params | backend    | threads |         lm |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ3_S - 3.4375 bpw |  13.09 GiB |    35.51 B | CPU        |       8 |      mlock |           tg128 |         15.71 ± 0.28 |

build: 04e10983e (11209)


# OEA Partial Paper Tables

This view includes only committed value evidence for paper Tables 2, 3 and 12-15. It excludes all extension experiments. Multiple predeclared protocols remain separate; no protocol is selected after observing proximity to the paper value.

## Table 2

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Nemo3B (+Cl) | Clotho | [CODE] all-caption | 21.57 / 47.16 / 60.36 | 21.7225 / 47.1196 / 60.4402 | 0.152488 |
| OEA-Nemo3B (+Cl) | Clotho | [CODE] seed-0 one-caption | 21.57 / 47.16 / 60.36 | 21.6268 / 46.7943 / 59.8086 | 0.551388 |
| OEA-Qwen3B | Clotho | [CODE] all-caption | 19.18 / 42.05 / 55.85 | 19.0048 / 41.9522 / 55.9617 | 0.175215 |
| OEA-Qwen3B | Clotho | [CODE] seed-0 one-caption | 19.18 / 42.05 / 55.85 | 19.3301 / 41.7225 / 55.6938 | 0.327512 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] all-caption | 22.87 / 49.78 / 63.25 | 22.7368 / 49.4737 / 63.3110 | 0.306316 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] seed-0 one-caption | 22.87 / 49.78 / 63.25 | 21.9139 / 49.7608 / 64.2105 | 0.960526 |

## Table 3

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Nemo3B (+Cl) | Clotho | [INFERRED] all-caption sensitivity | 63.77 / 75.29 / 80.11 | 63.6938 / 75.2344 / 80.0191 | 0.090861 |
| OEA-Nemo3B (+Cl) | Clotho | [CODE] seed-0 one-caption/self-exclusion | 63.77 / 75.29 / 80.11 | 62.9665 / 75.9809 / 80.8612 | 0.803493 |
| OEA-Qwen3B | Clotho | [INFERRED] all-caption sensitivity | 62.81 / 73.76 / 78.22 | 62.7943 / 73.8756 / 78.3158 | 0.115598 |
| OEA-Qwen3B | Clotho | [CODE] seed-0 one-caption/self-exclusion | 62.81 / 73.76 / 78.22 | 61.3397 / 74.1627 / 79.3301 | 1.470287 |
| OEA-Qwen3B (+Cl) | Clotho | [INFERRED] all-caption sensitivity | 64.52 / 75.25 / 79.71 | 64.3636 / 75.1388 / 79.7321 | 0.156364 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] seed-0 one-caption/self-exclusion | 64.52 / 75.25 / 79.71 | 64.4019 / 76.1722 / 80.7656 | 1.055550 |

## Table 12

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Qwen3B | Clotho | [CODE] released positive UIQ | 21.34 / 44.11 / 59.71 | 21.5311 / 44.4976 / 59.5215 | 0.387608 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] released positive UIQ | 25.74 / 54.26 / 66.79 | 25.8373 / 54.0670 / 66.6029 | 0.193014 |

## Table 13

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Qwen3B | Clotho | [CODE] released positive UIQ | 22.30 / 46.12 / 59.33 | 22.4880 / 46.2201 / 59.1388 | 0.191244 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] released positive UIQ | 25.45 / 55.69 / 67.56 | 25.8373 / 55.6938 / 66.9856 | 0.574354 |

## Table 14

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Qwen3B | Clotho | [CODE] released positive UIQ | 21.24 / 46.79 / 60.19 | 20.9569 / 46.7943 / 60.0957 | 0.283062 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] released positive UIQ | 27.56 / 55.50 / 70.81 | 26.9856 / 55.1196 / 70.2392 | 0.574354 |

## Table 15

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Qwen3B | Clotho | [CODE] released positive UIQ | 24.50 / 49.76 / 61.72 | 24.3062 / 49.6651 / 61.7225 | 0.193780 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] released positive UIQ | 27.66 / 57.99 / 71.87 | 27.5598 / 58.0861 / 71.6746 | 0.195359 |

## Boundaries

- [PAPER] Clotho caption selection is not specified.
- [PAPER] T2T self-exclusion and tie handling are not fully specified.
- [CODE] The public audio path omits the `passage:` prefix stated by the paper.
- [OBSERVED] Nemo3B (+Cl) T2T is listed from a completed CPU suite; positive UIQ remains pending.

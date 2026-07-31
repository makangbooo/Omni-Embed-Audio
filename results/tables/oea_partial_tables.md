# Partial Paper Tables

This view includes only committed value evidence for paper Tables 2, 3 and 12-15. It excludes all extension experiments. Multiple predeclared protocols remain separate; no protocol is selected after observing proximity to the paper value.

## Table 2

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| LAION-CLAP | Clotho | [CODE] all-caption | 13.34 / 33.49 / 46.22 | 14.0478 / 37.2057 / 49.8182 | 3.715742 |
| LAION-CLAP | Clotho | [CODE] seed-0 one-caption | 13.34 / 33.49 / 46.22 | 13.6842 / 38.5646 / 49.9522 | 5.074593 |
| M2D-CLAP | Clotho | [CODE] all-caption | 17.55 / 42.91 / 55.54 | 16.4211 / 40.7081 / 53.7033 | 2.201866 |
| M2D-CLAP | Clotho | [CODE] seed-0 one-caption | 17.55 / 42.91 / 55.54 | 17.1292 / 40.4785 / 53.7799 | 2.431531 |
| MGA-CLAP | Clotho | [CODE] all-caption | 18.78 / 43.79 / 56.63 | 21.1100 / 46.9474 / 60.0766 | 3.446555 |
| MGA-CLAP | Clotho | [CODE] seed-0 one-caption | 18.78 / 43.79 / 56.63 | 21.4354 / 46.5072 / 59.7129 | 3.082919 |
| Nemotron-3B | Clotho | [CODE] all-caption | 7.20 / 21.57 / 30.12 | 7.2536 / 21.3014 / 30.1818 | 0.268565 |
| Nemotron-3B | Clotho | [CODE] seed-0 one-caption | 7.20 / 21.57 / 30.12 | 7.4641 / 21.2440 / 31.4833 | 1.363254 |
| OEA-Nemo3B | Clotho | [CODE] all-caption | 19.04 / 40.57 / 54.24 | 18.2775 / 41.3206 / 54.2010 | 0.762488 |
| OEA-Nemo3B | Clotho | [CODE] seed-0 one-caption | 19.04 / 40.57 / 54.24 | 17.9904 / 41.6268 / 55.7895 | 1.549474 |
| OEA-Nemo3B (+Cl) | Clotho | [CODE] all-caption | 21.57 / 47.16 / 60.36 | 21.7225 / 47.1196 / 60.4402 | 0.152488 |
| OEA-Nemo3B (+Cl) | Clotho | [CODE] seed-0 one-caption | 21.57 / 47.16 / 60.36 | 21.6268 / 46.7943 / 59.8086 | 0.551388 |
| OEA-Qwen3B | Clotho | [CODE] all-caption | 19.18 / 42.05 / 55.85 | 19.0048 / 41.9522 / 55.9617 | 0.175215 |
| OEA-Qwen3B | Clotho | [CODE] seed-0 one-caption | 19.18 / 42.05 / 55.85 | 19.3301 / 41.7225 / 55.6938 | 0.327512 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] all-caption | 22.87 / 49.78 / 63.25 | 22.7368 / 49.4737 / 63.3110 | 0.306316 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] seed-0 one-caption | 22.87 / 49.78 / 63.25 | 21.9139 / 49.7608 / 64.2105 | 0.960526 |
| OEA-Qwen7B | Clotho | [CODE] all-caption | 19.77 / 44.78 / 57.65 | 19.8852 / 44.6890 / 57.0909 | 0.559091 |
| OEA-Qwen7B | Clotho | [CODE] seed-0 one-caption | 19.77 / 44.78 / 57.65 | 20.5742 / 44.9761 / 55.8852 | 1.764833 |
| OEA-Qwen7B (+Cl) | Clotho | [CODE] all-caption | 22.53 / 48.80 / 62.70 | 22.0287 / 48.5359 / 61.6459 | 1.054067 |
| OEA-Qwen7B (+Cl) | Clotho | [CODE] seed-0 one-caption | 22.53 / 48.80 / 62.70 | 22.0096 / 48.2297 / 60.4785 | 2.221531 |
| Qwen2.5-Omni-3B | Clotho | [CODE] all-caption | 0.17 / 0.63 / 1.13 | 0.1722 / 0.6316 / 1.1483 | 0.018325 |
| Qwen2.5-Omni-3B | Clotho | [CODE] seed-0 one-caption | 0.17 / 0.63 / 1.13 | 0.0957 / 0.5742 / 1.1483 | 0.074306 |
| Qwen2.5-Omni-7B | Clotho | [CODE] all-caption | 0.10 / 0.65 / 1.32 | 0.0957 / 0.6507 / 1.2440 | 0.075981 |
| Qwen2.5-Omni-7B | Clotho | [CODE] seed-0 one-caption | 0.10 / 0.65 / 1.32 | 0.0000 / 0.6699 / 1.3397 | 0.100000 |

## Table 3

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| LAION-CLAP | Clotho | [INFERRED] all-caption sensitivity | 50.05 / 65.88 / 72.67 | 50.0478 / 65.8756 / 72.6699 | 0.004402 |
| LAION-CLAP | Clotho | [CODE] seed-0 one-caption/self-exclusion | 50.05 / 65.88 / 72.67 | 49.3780 / 64.8804 / 72.2488 | 0.999617 |
| M2D-CLAP | Clotho | [INFERRED] all-caption sensitivity | 55.85 / 69.05 / 74.76 | 53.1675 / 67.9809 / 73.7990 | 2.682536 |
| M2D-CLAP | Clotho | [CODE] seed-0 one-caption/self-exclusion | 55.85 / 69.05 / 74.76 | 52.9187 / 67.2727 / 73.0144 | 2.931340 |
| MGA-CLAP | Clotho | [INFERRED] all-caption sensitivity | 63.27 / 74.70 / 78.79 | 63.2727 / 74.6986 / 78.7943 | 0.004258 |
| MGA-CLAP | Clotho | [CODE] seed-0 one-caption/self-exclusion | 63.27 / 74.70 / 78.79 | 62.0096 / 73.1100 / 77.2249 | 1.589952 |
| Nemotron-3B | Clotho | [INFERRED] all-caption sensitivity | 57.84 / 69.36 / 74.26 | 57.8947 / 69.2823 / 74.2010 | 0.077703 |
| Nemotron-3B | Clotho | [CODE] seed-0 one-caption/self-exclusion | 57.84 / 69.36 / 74.26 | 57.1292 / 69.3780 / 74.3541 | 0.710813 |
| OEA-Nemo3B | Clotho | [INFERRED] all-caption sensitivity | 62.79 / 74.12 / 78.91 | 64.6316 / 74.9856 / 79.2536 | 1.841579 |
| OEA-Nemo3B | Clotho | [CODE] seed-0 one-caption/self-exclusion | 62.79 / 74.12 / 78.91 | 63.3493 / 74.8325 / 80.0957 | 1.185694 |
| OEA-Nemo3B (+Cl) | Clotho | [INFERRED] all-caption sensitivity | 63.77 / 75.29 / 80.11 | 63.6938 / 75.2344 / 80.0191 | 0.090861 |
| OEA-Nemo3B (+Cl) | Clotho | [CODE] seed-0 one-caption/self-exclusion | 63.77 / 75.29 / 80.11 | 62.9665 / 75.9809 / 80.8612 | 0.803493 |
| OEA-Qwen3B | Clotho | [INFERRED] all-caption sensitivity | 62.81 / 73.76 / 78.22 | 62.7943 / 73.8756 / 78.3158 | 0.115598 |
| OEA-Qwen3B | Clotho | [CODE] seed-0 one-caption/self-exclusion | 62.81 / 73.76 / 78.22 | 61.3397 / 74.1627 / 79.3301 | 1.470287 |
| OEA-Qwen3B (+Cl) | Clotho | [INFERRED] all-caption sensitivity | 64.52 / 75.25 / 79.71 | 64.3636 / 75.1388 / 79.7321 | 0.156364 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] seed-0 one-caption/self-exclusion | 64.52 / 75.25 / 79.71 | 64.4019 / 76.1722 / 80.7656 | 1.055550 |
| OEA-Qwen7B | Clotho | [INFERRED] all-caption sensitivity | 63.66 / 74.32 / 79.37 | 62.6603 / 74.1053 / 79.2153 | 0.999713 |
| OEA-Qwen7B | Clotho | [CODE] seed-0 one-caption/self-exclusion | 63.66 / 74.32 / 79.37 | 60.8612 / 72.6316 / 78.7560 | 2.798756 |
| OEA-Qwen7B (+Cl) | Clotho | [INFERRED] all-caption sensitivity | 63.58 / 75.04 / 79.90 | 63.2344 / 74.9856 / 79.6555 | 0.345550 |
| OEA-Qwen7B (+Cl) | Clotho | [CODE] seed-0 one-caption/self-exclusion | 63.58 / 75.04 / 79.90 | 60.8612 / 72.8230 / 78.4689 | 2.718756 |
| Qwen2.5-Omni-3B | Clotho | [INFERRED] all-caption sensitivity | 38.35 / 51.14 / 57.07 | 38.4115 / 51.1388 / 57.1100 | 0.061483 |
| Qwen2.5-Omni-3B | Clotho | [CODE] seed-0 one-caption/self-exclusion | 38.35 / 51.14 / 57.07 | 39.5215 / 52.2488 / 59.0431 | 1.973062 |
| Qwen2.5-Omni-7B | Clotho | [INFERRED] all-caption sensitivity | 40.23 / 54.26 / 60.27 | 39.4067 / 52.9761 / 59.1005 | 1.283923 |
| Qwen2.5-Omni-7B | Clotho | [CODE] seed-0 one-caption/self-exclusion | 40.23 / 54.26 / 60.27 | 40.7655 / 54.1627 / 59.8086 | 0.535550 |

## Table 12

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Nemo3B | Clotho | [CODE] released positive UIQ | 19.81 / 43.54 / 57.61 | 18.9474 / 43.6364 / 56.3636 | 1.246364 |
| OEA-Nemo3B (+Cl) | Clotho | [CODE] released positive UIQ | 23.44 / 50.62 / 63.73 | 23.6364 / 50.8134 / 63.9234 | 0.196364 |
| OEA-Qwen3B | Clotho | [CODE] released positive UIQ | 21.34 / 44.11 / 59.71 | 21.5311 / 44.4976 / 59.5215 | 0.387608 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] released positive UIQ | 25.74 / 54.26 / 66.79 | 25.8373 / 54.0670 / 66.6029 | 0.193014 |

## Table 13

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Nemo3B | Clotho | [CODE] released positive UIQ | 18.85 / 43.25 / 57.03 | 19.0431 / 43.3493 / 56.8421 | 0.193062 |
| OEA-Nemo3B (+Cl) | Clotho | [CODE] released positive UIQ | 24.02 / 51.29 / 64.69 | 24.0191 / 51.1005 / 64.9761 | 0.286077 |
| OEA-Qwen3B | Clotho | [CODE] released positive UIQ | 22.30 / 46.12 / 59.33 | 22.4880 / 46.2201 / 59.1388 | 0.191244 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] released positive UIQ | 25.45 / 55.69 / 67.56 | 25.8373 / 55.6938 / 66.9856 | 0.574354 |

## Table 14

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Nemo3B | Clotho | [CODE] released positive UIQ | 20.96 / 44.50 / 57.22 | 20.7656 / 44.3062 / 56.6507 | 0.569282 |
| OEA-Nemo3B (+Cl) | Clotho | [CODE] released positive UIQ | 23.83 / 50.14 / 64.50 | 23.5407 / 49.9522 / 64.4019 | 0.289330 |
| OEA-Qwen3B | Clotho | [CODE] released positive UIQ | 21.24 / 46.79 / 60.19 | 20.9569 / 46.7943 / 60.0957 | 0.283062 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] released positive UIQ | 27.56 / 55.50 / 70.81 | 26.9856 / 55.1196 / 70.2392 | 0.574354 |

## Table 15

| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |
|---|---|---|---:|---:|---:|
| OEA-Nemo3B | Clotho | [CODE] released positive UIQ | 21.53 / 45.45 / 59.14 | 21.5311 / 45.8373 / 60.1914 | 1.051388 |
| OEA-Nemo3B (+Cl) | Clotho | [CODE] released positive UIQ | 25.93 / 52.54 / 66.03 | 25.8373 / 52.6316 / 66.1244 | 0.094402 |
| OEA-Qwen3B | Clotho | [CODE] released positive UIQ | 24.50 / 49.76 / 61.72 | 24.3062 / 49.6651 / 61.7225 | 0.193780 |
| OEA-Qwen3B (+Cl) | Clotho | [CODE] released positive UIQ | 27.66 / 57.99 / 71.87 | 27.5598 / 58.0861 / 71.6746 | 0.195359 |

## Boundaries

- [PAPER] Clotho caption selection is not specified.
- [PAPER] T2T self-exclusion and tie handling are not fully specified.
- [CODE] The public audio path omits the `passage:` prefix stated by the paper.
- [OBSERVED] Nemo3B (+Cl) T2T and all four released positive-UIQ protocols are listed from completed suites.

# Reproduction run — repro-20260428-0210

## Environment
- Instance: g5.16xlarge (A10G 24 GB, 64 vCPU, 256 GB RAM)
- CUDA: 11.8 (torch 2.4.0+cu118)
- Python: 3.10, conda env neo
- Date: 2026-04-28T02:10:49Z

## What was run
| Figure | Script | Model | Note |
|--------|--------|-------|------|
| Fig 6b | evaluation/reproduce-fig10a.py | Llama-3.1-8B | base + ours, 2000 req |
| Fig 6c | evaluation/reproduce-fig6c.py  | Llama-2-7b   | ours only (vllm skipped: CUDA 12 mismatch) |

## Artifacts
- `01_baseline/fig10a.pdf` — Fig 6b reproduction
- `01_baseline/fig6c.pdf`  — Fig 6c reproduction (NEO curve only)
- `01_baseline/raw_fig10a/` — raw throughput JSON
- `01_baseline/raw_fig6c/`  — raw latency JSON
- `01_baseline/server_logs/` — per-run server stderr/stdout
- `01_baseline/fig10a.log`  — full stdout of reproduce-fig10a.py
- `01_baseline/fig6c.log`   — full stdout of reproduce-fig6c.py
- `02_smoke/unit_tests.log` — pytest 15/15 passing
- `02_smoke/int8_smoke.log` — int8-cpu-kv smoke (16 iters, baseline + int8)

## Known limitations
- vllm comparison in fig6c omitted: vllm 0.5.5 wheel is cu121, won't load on cu118
- fig6c uses 100 requests (paper used 2000); latency numbers will be lower due to less queuing

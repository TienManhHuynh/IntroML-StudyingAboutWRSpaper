# Runners cần triển khai

| Script dự kiến | Trách nhiệm |
|---|---|
| `run_reproduction.py` | Tuning, baseline và MAIN WRS |
| `run_ablations.py` | Top-K, moving/EMA, benchmark |
| `summarize_results.py` | Curves, tables, win count, Wilcoxon, deviations |
| `compare_reference.py` | Code tác giả ở môi trường tách biệt |
| `run_noise_analysis.py` | Noise bổ sung |

Các script trên chưa tồn tại; không có runner giả trả thành công. Xem `docs/full-scope.md` cho EMBER/NONPA. `scripts/plan_experiments.py` chỉ lập kế hoạch, không chạy thuật toán.

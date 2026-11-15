# Experiment runners trên tv3

| Script | Trách nhiệm / trạng thái |
|---|---|
| `run_reproduction.py` | Paired baseline/WRS trial; WRS nhận factory thật từ TV2 |
| `run_tuning.py` | Grid B.2, 5 trials/config và lựa chọn tham số |
| `run_ablations.py` | Chưa triển khai; cần wrappers tương ứng từ nhóm |
| `summarize_results.py` | Grouped measured summaries và native-source plots |
| `compare_reference.py` | CSV comparison helper; code tác giả phải được chạy riêng |
| `run_noise_analysis.py` | Train-only label noise, paired masks |

Baseline development và noise/tuning smoke đã chạy thật, xem [hướng dẫn](../docs/tv3-running.md). Chưa chạy WRS/full paper/reference implementation. `scripts/plan_experiments.py` vẫn chỉ lập kế hoạch, không chạy thuật toán.

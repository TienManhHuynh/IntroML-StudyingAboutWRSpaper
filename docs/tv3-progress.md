# Tiến độ TV3 - nhánh tv3

Ngày14/11/2026; phạm vi làm đầy đủ paper theo yêu cầu nhóm trưởng. Checklist gốc: [TV3 tasks](tv3-tasks.md). Đây là tiến độ có bằng chứng, không phải xác nhận hoàn thành deadline toàn phần.

| Hạng mục deadline | Đã thực hiện | Còn thiếu |
|---|---|---|
| Tích hợp learner TV1 | Merge nguyên trạng PAC/FSOL, tests pass | WAT TV2 và tests tích hợp thật |
| Loader/preprocessing | HF pin revision/hash; SVMLight/NPZ/parquet; sparse/finite/labels guards; split/CountVectorizer/noise | Nguồn đủ 16 datasets và equivalence; Kaggle backend nếu dùng |
| Accuracy/ROP/sparsity | Implement và tests tính tay; ROP âm; oracle/checkpoints; deviations | Review protocol với reference trên dữ liệu đầy đủ |
| Experiment runner | Baseline/paired WRS-factory interface; lưu config/env/checkpoints/masks/status; từ chối overwrite | Chạy WRS thật sau TV2; ablation wrappers |
| Fixed seeds/checkpoints | Split, stream, subset/noise seeds; đầu/cuối và logging grid | Chốt grid paper/reference chính xác |
| Tuning | Grid PAC/FSOL và lựa chọn B.2, tests/35-trial synthetic smoke | Tuning datasets thật, đối chiếu round/tie convention |
| W&B | Logging code disabled/offline/online, offline metrics/artifact smoke pass | Người dùng tạo/login public project, chạy online và xác minh run URLs |
| PAC vs PAC-WRS / FSOL vs FSOL-WRS | 10 baseline local development trials trên SST-2 | Thiếu WAT: chưa có paired WRS results, chưa đủ Full scope |
| Label noise | Runner 5 rates, train-only paired masks, synthetic smoke | Chạy noise trên real datasets + WRS, phân tích kết quả |
| Bảng/plot/tái lập | Grouped summary, run index, SVG/PNG, evidence, environment lock | Full runs, public links, reference comparison thực tế |
| Deviation/statistics | Delta/near-zero handling; normalized reference CSV helper; paired Wilcoxon/win count | Reference run outputs; đủ16 dataset means; chọn convention thống kê theo paper |
| Slides22-30 | 9-slide PPTX nháp có notes/tables/charts và provenance | Thay phần pending bằng kết quả thật/public runs; ghép theme chung/QA deck cuối |

**Không tự thay code TV2 bằng thuật toán giả.** Factory identity chỉ tồn tại trong test để xác minh giao diện, không xuất số WRS. Code TV1 được giữ nguyên; finite guard của loader TV3 bảo vệ đường vào runner.

Hướng dẫn chạy và W&B cho người mới: [TV3 running](tv3-running.md). Bằng chứng local: [SST-2 development](../results/development/sst2-baseline/README.md). Bản nháp: [PPTX](../slides/tv3_experiments_draft.pptx).

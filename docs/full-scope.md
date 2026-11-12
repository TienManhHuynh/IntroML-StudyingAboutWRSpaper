# Full scope: ma trận bao phủ

Nguồn: paper Section 4, Table 1, Figures 1-4, Tables 2-4 và Appendices B-G. Full scope do nhóm trưởng chốt 12/11/2026.

| ID | Nội dung | Bao phủ / artifact |
|---|---|---|
| MAIN | PAC/FSOL và WRS | 16 dataset, mọi biến thể Section 4; accuracy/sparsity curves, ROP, final accuracy/sparsity; Figures 1-4, Table 2, Appendix G |
| TOPK | Deterministic top-K | K=64, simple/weighted average, 2 learners trên 16 dataset; Tables 3-4 |
| AVERAGING | Moving/EMA | K=64, gamma=.9, 2 learners; kế hoạch phủ 16 dataset; Appendix D, Figures 6-7 |
| RUNTIME | Training time per iteration | WRS/moving/EMA, tách evaluation/I/O, xác minh variants; Table 5 |
| STATS | Wilcoxon và win count | Trung bình 5 trials, ghép theo 16 dataset, K=64; Tables 2-4,6-7 |
| REFERENCE | Code tác giả | Cùng data/config/seed khi có thể, lưu commit/môi trường và divergence |
| EMBER | Appendix C, Figure 5 | PAC/FSOL +/-WRS, xác minh edition/split/variant trước lập jobs |
| NONPA | Appendix E, Figures 8-10 | Modified WAT cho SGD+Momentum, AdaGrad, TGD; cần xác minh datasets/defaults |
| NOISE | Phân tích nhóm | Noise train [0,.05,.10,.20,.30], giữ test sạch; bổ sung, không thay MAIN |

Không dùng subset dataset hay label noise thay thí nghiệm chính rồi ghi Full scope hoàn tất. Nếu dataset gốc không truy cập được, ghi blocker, tìm equivalent có lý do và công khai deviation. EMBER/NONPA vẫn ở coverage Planned, chưa có manifest vì còn thiếu xác minh cấu hình; không âm thầm loại khỏi kế hoạch. Appendix G là kết quả mở rộng MAIN, không phải trial độc lập.

## Lưới kế hoạch

- 16 dataset; learners PAC/FSOL; weighting standard/exponential; averaging simple/weighted; voting_zeroing false/true; K=[1,4,16,64]; 5 trials.
- Seeds nhóm chọn: 42,43,44,45,46; paper không ấn định các số seed này.
- WRS: 16 x 2 x 2 x 2 x 2 x 4 x 5 = **5.120 jobs**.
- Baseline: **160 jobs**. TOPK dự kiến **320**. Moving/EMA dự kiến **320**.
- Manifest hiện tại **5.920 jobs dự kiến**, chưa tuning/runtime riêng/reference/EMBER/NONPA/noise. Chi tiết TOPK/averaging phải đối chiếu paper/reference trước thực thi.
- Tuning: PAC 7 C_err; FSOL 13 eta x 8 lambda; 16 dataset x 5 trials = **8.880 lượt đánh giá cấu hình**. Có thể tái sử dụng run đã tuning nếu đúng provenance, không báo đếm trùng là trial độc lập.

`configs/full_scope.json` giữ lưới; `configs/datasets.json` giữ 16 dataset và số Table 1. Hosted source chưa xác minh để null. `python scripts/plan_experiments.py` chỉ in/sinh kế hoạch. `results/coverage.csv` chỉ chuyển Completed khi có config, raw metrics, public W&B, bảng/plot và deviation.

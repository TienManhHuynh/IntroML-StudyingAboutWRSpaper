# Tiến độ TV3 - nhánh intergration-v1

Cập nhật ngày 18/11/2026. Hoàn tất **Đợt 1 gồm 8/16 datasets Table 1** (`dexter`, `dorothea`, `pcmac`, `w8a`, `newsgroups-binary-cs`, `sst2`, `real-sim`, `news20`) với **1.275 runs `completed_online`** trên W&B public (100% có `wandb_url` mở được không cần đăng nhập).

| Hạng mục deadline | Đã thực hiện | Còn thiếu / Ghi nhận giới hạn |
|---|---|---|
| Tích hợp learner TV1 & WAT TV2 | Tích hợp `PAC`, `FSOL`, `src.wat.wat:WAT`; toàn bộ `pytest` pass (`41 passed, 4 subtests passed`) | Giữ nguyên ranh giới module TV1/TV2 |
| Loader/preprocessing | Pin HF + SHA256 cho `stanfordnlp/sst2` và 7 bộ `sparse_npz` trên `TienManh2307/wrs-paper-datasets` khớp 100% `(N, D, sparsity)` Table 1 | 8 bộ lớn/khổng lồ còn lại (`webspam`, `rcv1`, `mnist8-4-9`, `url`, `kdd2010-algebra`, `avazu-*`, `criteo`) hoãn do giới hạn RAM/thời gian |
| Accuracy/ROP/sparsity | Đầy đủ metrics, checkpoint grid 200 điểm + đầu/cuối, deviation helper, plots min/max error bars | — |
| Experiment runner & Ablations | Hoàn thành `run_reproduction.py`, `run_ablations.py` (Top-K, Moving Avg, EMA), `run_noise_analysis.py`, `run_batch1.py` | Appendix C (`EMBER`) và Appendix E (`Non-PA`) chưa chạy |
| Tuning (Appendix B.2) | Chạy xong tuning PAC (280 runs) & FSOL (4.160 runs) trên 8 bộ Đợt 1 (`results/tuning/`) | Phát hiện sai khác `ceil(2.5%)` vs `top_acc - 0.025` trong code tác giả; đã chạy 35 runs kiểm chứng `deviation_band` |
| W&B public | **1.275 runs online** (`720` main + `320` ablations + `200` noise + `35` deviation_band), `0` link lỗi | — |
| PAC vs PAC-WRS / FSOL vs FSOL-WRS | FSOL-WRS K=64 thắng ROP **6/8** bộ; PAC-WRS thắng **3/8** (`ceil(2.5%)`) và **4/8** ROP, **5/8** Acc (`top_acc - 0.025`) | Kết quả Wilcoxon đang là partial (`n=8/16`) |
| Label noise bổ sung | Hoàn thành **200 runs** (`dexter`, `pcmac`, `w8a`, `sst2` × 5 rates); `w8a` 30% noise: WRS giữ `96.45%–96.55%` vs Base `60.23%–61.07%` | — |
| Reference comparison | Chuẩn hóa `author_batch1_means.csv`, xuất `reference_comparison.csv`, phát hiện 4 điểm divergence trong `analyzer.ipynb` & `PAC_main.py` | — |
| Slides 22–30 | Xuất `slides/tv3_experiments.pptx` từ số liệu thật & W&B URLs | Còn bước phối hợp TV1/TV2 ghép deck cuối 20–40 slides & QA Slide Show 100% |

Chi tiết kết quả tổng hợp nằm tại [results/batch1/summary.json](../results/batch1/summary.json), [results/batch1/run-index.csv](../results/batch1/run-index.csv), [results/batch1/wilcoxon_stats.json](../results/batch1/wilcoxon_stats.json), [results/batch1/runtime_table5.json](../results/batch1/runtime_table5.json) và thư mục [figures/](../figures/README.md).

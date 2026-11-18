# Ma trận tham chiếu 16 bộ Table 1 & phạm vi thực thi Đợt 1

Nguồn: paper Section 4, Table 1, Figures 1-4, Tables 2-4 và Appendices B-G. Khung cấu hình (`configs/full_scope.json`) liệt kê đầy đủ **16 bộ dữ liệu Table 1** để quản lý độ phủ (`results/coverage.csv`). Do giới hạn phần cứng thực tế (`8.55 GB` RAM trống so với mức `>10.3 GB` chỉ riêng cho reservoir dense `K=64` của `kdd2010-algebra` hoặc `51.88M` mẫu của `criteo`) và hạn nộp 21/11, nhóm chốt thực thi thực tế theo chuẩn **Core scope mở rộng (Đợt 1 / Batch 1)**: hoàn tất **8/16 bộ Table 1** (`dexter`, `dorothea`, `pcmac`, `w8a`, `newsgroups-binary-cs`, `sst2`, `real-sim`, `news20`), phân tích bổ sung **Label Noise** (4 bộ, 200 runs) và kiểm chứng **Deviation Band** (`top_acc - 0.025`, 35 runs) — tổng cộng **1.275 runs W&B online** (cùng 4.440 runs tuning).

| ID | Nội dung | Phạm vi tham chiếu & trạng thái thực tế |
|---|---|---|
| MAIN | PAC/FSOL và WRS | Khung tham chiếu 16 bộ; **Đợt 1 hoàn tất 8/16 bộ** (720 runs W&B online); 8 bộ lớn/khổng lồ giữ `planned` |
| TOPK | Deterministic top-K | Khung 16 bộ; **Đợt 1 hoàn tất 8/16 bộ** (`K=64`, simple/weighted, 2 learners × 5 seeds = 160 runs) |
| AVERAGING | Moving/EMA | Khung 16 bộ; **Đợt 1 hoàn tất 8/16 bộ** (`K=64`, `gamma=.9`, 2 learners × 5 seeds = 160 runs) |
| RUNTIME | Training time per iteration | Đo trực tiếp trên 8/16 bộ Đợt 1 (xác nhận Table 5: `WRS` nhanh hơn `2.36x–9.00x` trên `news20`) |
| STATS | Wilcoxon và win count | Trung bình 5 trials trên tập con ghép cặp `n=8/16` Đợt 1 (`wilcoxon_stats.json`, ghi rõ partial `n=8/16`) |
| REFERENCE | Code tác giả | Đối chiếu `analyzer.ipynb` / `PAC_main.py` / `metric_extractors.py`, chạy thêm 35 runs `deviation_band` |
| EMBER | Appendix C, Figure 5 | Chưa thực hiện trước hạn 21/11 — giữ nguyên `planned` |
| NONPA | Appendix E, Figures 8-10 | Chưa thực hiện trước hạn 21/11 — giữ nguyên `planned` |
| NOISE | Phân tích nhóm (bổ sung) | **Hoàn tất 4 bộ** (`dexter`, `pcmac`, `w8a`, `sst2`) × 5 rates = 200 runs W&B online |

Không dùng subset dataset hay label noise thay thí nghiệm chính rồi báo đã xong cả 16 bộ. Những dataset chưa chạy được giữ nguyên trạng thái `planned` trong `results/coverage.csv` kèm lý do giới hạn RAM/thời gian, và Wilcoxon được ghi rõ là kết quả từng phần (`partial n=8/16`).

## Lưới kế hoạch tham chiếu (16 bộ Table 1) và thực thi Đợt 1

- Lưới tham chiếu 16 dataset (`configs/full_scope.json`): learners PAC/FSOL; weighting standard/exponential; averaging simple/weighted; voting_zeroing false/true; K=[1,4,16,64]; 5 trials.
- Seeds nhóm chọn: 42,43,44,45,46; paper không ấn định các số seed này.
- Tổng lưới kế hoạch tham chiếu 16 bộ: **5.920 jobs** (`5.120` WRS + `160` baseline + `320` TOPK + `160` Moving + `160` EMA) và `8.880` lượt tuning.
- Thực thi thực tế **Đợt 1 (8/16 bộ)**: **4.440 lượt tuning** (`280` PAC + `4.160` FSOL) + **1.275 runs W&B online** (`720` MAIN + `320` ablations TOPK/Moving/EMA + `200` NOISE + `35` `deviation_band`).

`configs/full_scope.json` giữ lưới tham chiếu; `configs/datasets.json` giữ 16 dataset Table 1 (8 bộ Đợt 1 đã pin HF + SHA256, 8 bộ lớn để `unverified`). `python scripts/plan_experiments.py` chỉ sinh kế hoạch. `results/coverage.csv` chỉ chuyển `completed` cho các mục đã hoàn tất với đủ W&B URL công khai và artifact tổng hợp; 8 bộ còn lại và Appendix C/E giữ `planned`.

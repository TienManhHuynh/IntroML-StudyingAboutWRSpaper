# Slide bàn giao

Bản slide hoàn chỉnh của TV3 là [slides/tv3_experiments.pptx](tv3_experiments.pptx) (9 slides, Slides 22–30), dựng từ số liệu thật **1.275 runs W&B online** cho **Đợt 1 (8/16 bộ Table 1 + Ablations/Runtime + Label Noise + Deviation Band)** qua `scripts/build_tv3_slides.mjs`. Bản hoàn chỉnh này thay thế bản nháp cũ (`tv3_experiments_draft.pptx`); hướng dẫn chạy trong [TV3 running](../docs/tv3-running.md).

Deck cuối gộp cả nhóm cần đạt 20-40 slides kể cả appendix, dưới 20 MB. Mỗi slide có notes <=500 từ; citation IEEE chân slide; không References riêng. Hình tự vẽ, cùng theme 16:9; QA overlap/clipping trong Slide Show.

| Slides | Chủ sở hữu | File bàn giao & trạng thái |
|---|---|---|
| 01-11 | TV1: context, predecessors, notation, PAC/FSOL, theory | `TV1.pptx` (đề nghị đổi `tv1_background_theory.pptx`, chốt ranh giới Slide 12) |
| 12-21 | TV2: survival, WRS/WAT, training/inference, variants, complexity | `tv2.pptx` (cần đánh số 12–21 và bổ sung citation IEEE chân slide) |
| 22-30 | TV3: 8/16 bộ Đợt 1 (khung 16 bộ), protocol, ablations, deviations, noise | `tv3_experiments.pptx` (đã hoàn tất, QA pass) |

Dùng contract cuối phân công; TV3 ghép final sau khi thống nhất sửa slide TV1/TV2.

## Cấu trúc 9 slides TV3 đã hoàn thiện (`tv3_experiments.pptx`)

22. Độ phủ 8/16 bộ Đợt 1 trên Hugging Face và giải trình hoãn 8 bộ lớn (RAM trống 8.55 GB).
23. Protocol 1-pass, splits, lưới tuning Appendix B.2 và ma trận 1.275 runs online W&B.
24. Định nghĩa Accuracy, ROP (chấp nhận ROP âm), Model Sparsity và Wilcoxon (`n=8/16`).
25. Kết quả PAC-WRS: đối chiếu `ceil(2.5%)` (3/8 thắng ROP) với `top_acc - 0.025` (4/8 ROP, 5/8 Acc).
26. Kết quả FSOL-WRS: thắng ROP 6/8 bộ, bảo toàn độ thưa với Voting-based Zeroing.
27. Ablations & Runtime Table 5: WRS nhanh hơn Moving/EMA từ `2.36x–9.00x` trên `news20`.
28. Tổng hợp Wilcoxon (`n=8/16`) và đánh giá mức độ xác nhận luận điểm paper.
29. Đối chiếu code gốc tác giả và 4 nguyên nhân divergence (`analyzer.ipynb`, `PAC_main.py`, `metric_extractors.py`).
30. Kết quả Label Noise bổ sung (`w8a` 30%: 96.5% vs 60.2%–61.1%), kết luận và giới hạn phạm vi cho Milestone 3 & Poster.

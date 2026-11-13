# Slide bàn giao

Trên nhánh tv3 đã có [bản nháp 9 slide TV3](tv3_experiments_draft.pptx), dùng baseline SST-2 development thực tế, có notes và tables/charts native. Chưa có kết quả WRS/public W&B/full reproduction, nên **chưa phải PPTX nghiệm thu cuối**. Source builder: `scripts/build_tv3_slides.mjs`; hướng dẫn và giới hạn trong [TV3 running](../docs/tv3-running.md).

Deck cuối 20-40 slides kể cả appendix, dưới 20 MB. Mỗi slide có notes <=500 từ; citation IEEE chân slide; không References riêng. Hình tự vẽ, cùng theme 16:9; QA overlap/clipping trong Slide Show.

| Slides | Chủ sở hữu | File bàn giao |
|---|---|---|
| 01-11 | TV1: context, predecessors, notation, PAC/FSOL, theory | `tv1_background_theory.pptx` |
| 12-21 | TV2: survival, WRS/WAT, training/inference, variants, complexity | `tv2_method.pptx` |
| 22-30 | TV3: Full scope, 16 dataset, protocol, results/deviations | `tv3_experiments.pptx` |

Dùng contract cuối phân công; cần nhóm xác nhận khác biệt với cách chia đầu file. TV3 ghép final, không tự thay học thuật người khác khi chưa thống nhất.

## Outline TV3 đề xuất

22. Dataset coverage 16 bộ, nguồn/preprocessing.
23. Full-scope matrix, trials/split/K/protocol.
24. Accuracy/ROP/sparsity và cách đọc.
25. PAC-WRS stability và ROP.
26. FSOL-WRS stability và sparsity.
27. Ablations, averaging, runtime.
28. Paper vs reproduction, win count/statistics.
29. Deviations, reference comparison, appendix extensions khi có kết quả.
30. Noise bổ sung, kết luận và giới hạn.

Chi tiết dataset/variants có thể dùng appendix trong giới hạn 40 slide và dẫn bảng repo/W&B. Không chỉ trình bày dataset thuận lợi để giấu coverage thiếu. Outline là kế hoạch, chưa có PPTX/số liệu.

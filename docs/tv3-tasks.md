# TV3 / nhóm trưởng: danh sách công việc

Đã triển khai một phần trên nhánh tv3: xem [tiến độ có bằng chứng](tv3-progress.md) và [hướng dẫn chạy](tv3-running.md). Các checkbox dưới đây chỉ nghiệm thu khi đủ toàn bộ phạm vi, không đánh dấu hoàn tất chỉ từ development smoke.

Cập nhật 14/11/2026. Nguồn: PDF phân công `requirements/course-project.pdf`, yêu cầu môn `requirements/project_description.txt`, paper Section 4 và phụ lục. PDF còn Core Scope; quyết định trực tiếp của nhóm trưởng thay bằng **đầy đủ paper**. Không sửa bản nguồn.

## 1. Tích hợp và điều phối

- [ ] Review/nhận PAC và FSOL từ TV1, chốt API loss/update/get_weights/predict, quy ước labels và tie score.
- [ ] Yêu cầu TV2 bàn giao reservoir/WAT và tests; phần này hiện là blocker chạy PAC-WRS/FSOL-WRS.
- [ ] Chốt snapshot trước update, passive theo hinge loss, empty reservoir, zero total weights, terminal candidate và chống overflow.
- [ ] Merge qua review, ghi commit; không tự thay học thuật TV1/TV2 khi chưa thống nhất.
- [ ] Phân bổ top-K/moving/EMA và non-PA learners cho người thực hiện phù hợp; TV3 điều phối chạy, không mặc định tự viết toàn bộ core của TV1/TV2.

## 2. Dữ liệu - `src/data/loader.py`

- [ ] Xác minh nguồn HF/Kaggle của đủ 16 dataset, hosted revision và checksum; bổ sung EMBER và datasets cho appendix E sau xác minh.
- [ ] Nếu chưa có bản phù hợp, chuẩn bị/upload hợp lệ rồi cho code tải tự động.
- [ ] Reconstruct labels/features đúng paper; phân biệt News20 với Newsgroups Binary CS, xử lý SST-2 đúng quy trình vector hóa.
- [ ] Kiểm tra shape, nhãn -1/+1, NaN/Inf, duplicate sparse indices và feature alignment; không leak label vào features.
- [ ] Sparse giữ sparse, tránh densify toàn dataset; lưu cache và provenance.
- [ ] Random split 70/30, stream order/seeds tái lập; các model ghép dùng cùng split/stream; fit preprocessing có ghi rõ tập fit.

## 3. Metrics - `src/evaluation/metrics.py`

- [ ] Accuracy tại checkpoints và cuối pass; model sparsity (không nhầm dataset sparsity).
- [ ] ROP dùng cumulative maximum của baseline tương ứng trên cùng checkpoint grid; chấp nhận ROP âm.
- [ ] Mean qua 5 trials, min/max error bars theo paper; số liệu fractions/% nhất quán.
- [ ] Tests tính tay cho ROP/oracle/sparsity; missing metrics không điền 0.

## 4. Runners và tracking

- [ ] `experiments/run_reproduction.py`: tuning PAC/FSOL Appendix B.2, chọn cấu hình rồi chạy baseline/WRS các variants đầy đủ.
- [ ] `experiments/run_ablations.py`: top-K, moving/EMA, timing; phân tách train/evaluation/I/O.
- [ ] Fixed seeds, một pass, weights ban đầu 0, K/weighting/averaging/zeroing theo config; khoảng 200 checkpoint cộng đầu/cuối theo B.3.
- [ ] Lưu config/môi trường/commit/seeds/data revision/checkpoints/final metrics/failure reason; job IDs không trùng, retry không đếm trùng trials.
- [ ] Tạo một W&B project public, tích hợp logging thật, sync offline và kiểm tra run links.
- [ ] Chạy smoke end-to-end trước để kiểm chứng pipeline, sau đó đủ dataset/config/trials; smoke không báo như reproduction.
- [ ] Kiểm kê toàn bộ paper và extensions C/D/E/F/G; hoàn thiện cấu hình còn thiếu, chạy và ghi bằng chứng, không bỏ mục âm thầm.

## 5. So sánh và phân tích

- [ ] `experiments/compare_reference.py`: chạy code tác giả ở môi trường riêng, lưu commit/config và khoanh vùng divergence; không copy core.
- [ ] `experiments/summarize_results.py`: curves, win counts, Wilcoxon và paper comparison có nguồn table/figure/variant.
- [ ] Báo absolute/relative deviation khi có nghĩa; gần/bằng 0 để N/A kèm lý do; số đọc từ chart ghi là xấp xỉ.
- [ ] Giải thích lệch do preprocessing, split/order, hyperparameters, implementation, logging grid, versions và hardware.
- [ ] `experiments/run_noise_analysis.py`: noise train theo mask/seed, giữ test sạch, dùng cùng mask giữa models; phân tích thêm theo phân công, không thay thí nghiệm paper.

## 6. Sản phẩm bàn giao

- [ ] Bảng run-level và README experiment -> script/notebook -> public W&B run -> kết luận; coverage cập nhật theo bằng chứng.
- [ ] Plot/bảng tự tạo có mã nguồn, trục/legend/đơn vị và provenance.
- [ ] `slides/tv3_experiments.pptx`, slides 22-30: dữ liệu/protocol/metrics/baselines/parameters, kết quả, ablations/noise, paper comparison/deviations và giới hạn.
- [ ] Notes <=500 từ/slide, citations IEEE và run links ở slide liên quan.
- [ ] Nhận slides TV1/TV2, thống nhất dải slide theo contract cuối; ghép deck 20-40 slides, <20 MB, đủ notes, không overlap/clipping.
- [ ] Nộp PPTX và link repo riêng Moodle; đối chiếu deadline Moodle.

## Trạng thái ban đầu

Tại lúc lập checklist, main mới có scaffold/config/protocol. Trên tv3 hiện có loader/metrics/runners/tuning/tracking code, baseline development evidence và PPTX nháp; chưa có WAT/full reproduction/public W&B. Không thay checklist nghiệm thu toàn phần bằng trạng thái code đã viết. Review đầu vào: [TV1/TV2](reviews/tv1-tv2-2026-11-14.md).

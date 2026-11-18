# TV3 / nhóm trưởng: danh sách công việc

Cập nhật ngày 18/11/2026: đã hoàn tất toàn bộ **Đợt 1 (8/16 bộ Table 1 + Label Noise + Deviation Band = 1.275 runs W&B online công khai)**, chi tiết tại [tiến độ có bằng chứng](tv3-progress.md) và [hướng dẫn chạy](tv3-running.md). Khung tham chiếu ban đầu gồm 16 bộ Table 1; do giới hạn RAM trống `8.55 GB` và hạn Moodle 21/11, 8 bộ rất lớn cùng Appendix C/E được giữ nguyên ở trạng thái chưa chạy (`planned`) và không đánh dấu hoàn tất khống.

Nguồn: PDF phân công `requirements/course-project.pdf`, yêu cầu môn `requirements/project_description.txt`, paper Section 4 và phụ lục. Không sửa bản nguồn.

## 1. Tích hợp và điều phối

- [x] Review/nhận PAC và FSOL từ TV1 (PR #2), chốt API `loss`/`update`/`get_weights`/`predict`, quy ước labels và tie score.
- [x] Nhận bàn giao `reservoir`/`WAT` và tests từ TV2 (PR #4), chạy tích hợp `PAC-WRS` và `FSOL-WRS`.
- [x] Chốt snapshot trước update, passive theo hinge loss, empty reservoir, zero total weights, terminal candidate và chống overflow (NaN/Inf).
- [x] Merge qua review trên `intergration-v1`; giữ nguyên ranh giới module TV1/TV2.
- [x] Triển khai và kiểm thử `Top-K`, `Moving Avg`, `EMA` (`experiments/run_ablations.py`) cho Đợt 1; giữ Appendix E (non-PA) ở trạng thái `planned`.

## 2. Dữ liệu - `src/data/loader.py`

- [x] Xác minh nguồn HF (`stanfordnlp/sst2` và `TienManh2307/wrs-paper-datasets`), revision SHA và checksum cho **8/16 bộ Đợt 1**; giữ 8 bộ rất lớn + EMBER/Appendix E ở trạng thái `planned` do giới hạn RAM/thời gian.
- [x] Chuẩn hóa `sparse_npz` qua `scripts/prepare_datasets.py`, upload công khai lên Hugging Face và tải tự động qua `src/data/loader.py`.
- [x] Reconstruct labels/features khớp `(N, D, sparsity)` Table 1 cho 7 bộ `sparse_npz`; phân biệt `news20` với `newsgroups-binary-cs`, xử lý `sst2` qua `CountVectorizer` fit trên train.
- [x] Kiểm tra shape, nhãn `-1/+1`, NaN/Inf, duplicate sparse indices và feature alignment; không leak label vào features.
- [x] Giữ sparse CSR trong `load_dataset` và quá trình đánh giá; lưu cache và provenance (`metadata`, SHA256).
- [x] Random split 70/30, stream order/seeds `42–46` tái lập; các model ghép dùng cùng split/stream.

## 3. Metrics - `src/evaluation/metrics.py`

- [x] Accuracy tại checkpoints và cuối pass; model sparsity (tách biệt với dataset sparsity).
- [x] ROP dùng cumulative maximum của baseline tương ứng trên cùng checkpoint grid; hỗ trợ ROP âm.
- [x] Mean qua 5 trials (`seeds 42–46`), min/max error bars (`summarize_results.py`, `aggregate_batch1.py`); số liệu fractions/% nhất quán.
- [x] Unit tests tính tay cho ROP/oracle/sparsity (`tests/test_tv3.py`, `41 passed, 4 subtests passed`).

## 4. Runners và tracking

- [x] `experiments/run_reproduction.py` & `run_tuning.py`: hoàn tất tuning Appendix B.2 (4.440 lượt) và baseline/WRS trên 8/16 bộ Đợt 1 (`720` runs `main`).
- [x] `experiments/run_ablations.py`: hoàn tất Top-K, Moving Avg, EMA và timing Table 5 trên 8/16 bộ Đợt 1 (`320` runs `ablations`).
- [x] Fixed seeds `42–46`, một pass, `w_0 = 0`, ~200 checkpoints theo B.3.
- [x] Lưu `config`/`environment`/`checkpoints`/`metrics`/`split-indices`/`run.json`; không trùng ID hay seed.
- [x] Tạo W&B project public (`wrs-reproduction-group8`), đồng bộ online thực tế **1.275 runs** Đợt 1 (0 URL lỗi).
- [x] Chạy smoke end-to-end trước, sau đó hoàn tất toàn bộ 8/16 bộ Đợt 1 (8 bộ lớn còn lại giữ `planned`, không báo vượt phạm vi thực chạy).
- [ ] Hoàn thiện chạy 8 bộ rất lớn (`webspam`..`criteo`) và Appendix C/E (đang hoãn ở trạng thái `planned` do RAM trống `8.55 GB` và hạn 21/11).

## 5. So sánh và phân tích

- [x] `experiments/compare_reference.py` & `scripts/build_reference_csv.py`: chạy đối chiếu repo tác giả (`tmp/author-ref`), khoanh vùng 4 điểm divergence và chạy kiểm chứng 35 runs `deviation_band`.
- [x] `experiments/summarize_results.py` & `scripts/aggregate_batch1.py`: xuất curves, 5 bộ biểu đồ PNG/SVG, Wilcoxon partial `n=8/16` (`wilcoxon_stats.json`) và `reference_comparison.csv`.
- [x] Báo absolute/relative deviation và giải thích rõ trường hợp ROP near-zero.
- [x] Giải trình nguyên nhân sai lệch (`top_acc - 0.025` vs `ceil(2.5%)`, thiếu `+1` trong `||x||^2 + 1`, `loss > 1e-4`, `sign(0)=0`, seeds/grid/SST-2).
- [x] `experiments/run_noise_analysis.py`: hoàn tất 200 runs noise train trên 4 bộ (`dexter`, `pcmac`, `w8a`, `sst2` × 5 rates), giữ test sạch.

## 6. Sản phẩm bàn giao

- [x] Bảng run-level (`results/batch1/run-index.csv`), [README.md](../README.md) và [results/coverage.csv](../results/coverage.csv) cập nhật bằng chứng thật.
- [x] Plot/bảng tự tạo (`figures/batch1_*.{png,svg}`, `scripts/aggregate_batch1.py`) đầy đủ trục/legend/đơn vị/provenance.
- [x] `slides/tv3_experiments.pptx`, Slides 22–30: đầy đủ dữ liệu Đợt 1, protocol, metrics, kết quả, ablations/runtime, đối chiếu tác giả/deviations, noise và giới hạn phạm vi 8/16 bộ.
- [x] Notes `107–161` từ/slide (`<=500`), citations IEEE và W&B run links công khai trên từng slide.
- [ ] Nhận lại/chỉnh sửa đồng bộ từ TV1 & TV2 (sửa citation TV2, chốt Slide 12), ghép deck cuối 20–40 slides (`< 20 MB`) và QA Slide Show.
- [ ] Nộp PPTX gộp cuối và link repo lên Moodle trước hạn 21/11/2026.

## Trạng thái hiện tại

Cập nhật 18/11/2026: nhánh `intergration-v1` đã hoàn tất tích hợp TV1 + TV2 + TV3, pass `41 passed, 4 subtests passed` (`pytest`) và `check_repository.py` PASS, hoàn tất **Đợt 1 (8/16 bộ Table 1, 1.275 runs W&B online)** cùng bộ slide `slides/tv3_experiments.pptx`. 8 bộ dữ liệu rất lớn và Appendix C/E được ghi nhận giới hạn RAM trống `8.55 GB` và giữ trạng thái `planned`. Review đầu vào ban đầu: [TV1/TV2](reviews/tv1-tv2-2026-11-14.md).

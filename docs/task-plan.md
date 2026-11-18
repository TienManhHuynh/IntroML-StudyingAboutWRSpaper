# Phân công và tiến độ

Khung kế hoạch ban đầu kiểm kê đủ 16 bộ dữ liệu Table 1. Sau khi đo phần cứng thực tế (`8.55 GB` RAM trống so với mức `>10.3 GB` chỉ riêng cho reservoir `K=64` của `kdd2010-algebra` hay `51.88M` mẫu của `criteo`), nhóm thực thi theo chuẩn **Core scope mở rộng (Đợt 1 / Batch 1)**: hoàn tất **8/16 bộ Table 1** + **Label Noise** + **Deviation Band** (`1.275` runs W&B online), và giữ 8 bộ rất lớn cùng Appendix C/E ở trạng thái `planned`. Không sửa bản nguồn yêu cầu.

| Người | Trách nhiệm | Bàn giao & Trạng thái | Slide |
|---|---|---|---|
| TV1 / Nguyên | Background, theory, PAC/PA-II, FSOL | `src/models/pac.py`, `fsol.py`, tests (đã merge); `slides/TV1.pptx` | 01-11 |
| TV2 / Tùng | Survival, WRS/WAT, averaging/zeroing, complexity | `src/wat/reservoir.py`, `wat.py`, tests (đã merge); `slides/tv2.pptx` (cần bổ sung citation & đồng bộ số slide) | 12-21 |
| TV3 / TienManhHuynh | Dataset HF, metrics, runners, tuning, ablations, W&B, reference comparison & deviations | Đã hoàn tất 8/16 bộ Đợt 1 (`1.275` runs W&B online), `slides/tv3_experiments.pptx`; chờ ghép deck cuối | 22-30, ghép cuối |

Các module `PAC`/`FSOL` (TV1), `WRS`/`WAT` (TV2), `loader`/`metrics`/`runners`/`run_ablations` (TV3) đều đã tích hợp và pass `41 passed, 4 subtests passed` (`pytest`). Appendix C (`EMBER`) và Appendix E (`Non-PA`) giữ ở trạng thái `planned` do giới hạn thời gian và tài nguyên trước 21/11.

| Mốc 2026 | Mục tiêu | Trạng thái cập nhật (18/11/2026) |
|---|---|---|
| 11/11 | Đọc paper bước đầu | Hoàn tất |
| 12/11 | Khung tham chiếu 16 bộ, API và coverage | Hoàn tất |
| 13/11 | Theory/method và tích hợp API | Đã merge PR #2 (TV1) & PR #4 (TV2) |
| 14–18/11 | TV3 benchmark phần cứng, tuning (4.440 lượt) & chạy Đợt 1 (8/16 bộ, 1.275 runs W&B online), đối chiếu tác giả | Hoàn tất Đợt 1; 8 bộ lớn giữ `planned` |
| 15–18/11 | Bàn giao PPTX, notes, nguồn | Đã có `slides/TV1.pptx`, `slides/tv2.pptx`, `slides/tv3_experiments.pptx` |
| 16–20/11 | Ghép/QA bộ slide chung 20–40 slides | Đang thực hiện (chốt TV2 citation & ranh giới Slide 12) |
| 21/11 | Nộp Moodle (PPTX + link repo) | Sắp tới |

Thí nghiệm lưới 16 bộ có tổng `5.920` jobs dự kiến vượt quá tài nguyên máy cá nhân; phạm vi thực thi được chuẩn hóa thành Đợt 1 (8/16 bộ) trong `results/coverage.csv`, `results/batch1/run-index.csv` và Slides 22–30.

## Điểm cần chốt khi tích hợp

- File gốc có hai cách chia slide TV1/TV2. Khung dùng contract cuối, ghi nhận đồng thuận trong PR bàn giao.
- Chốt loss/update, tie prediction, bias, normalization, terminal reservoir và exponential weighting từ paper/reference.
- Xác minh dataset hosted revision/reconstruction và W&B entity/project public thực tế.
- Thay API cập nhật contract; không tự sửa học thuật người khác khi chưa thống nhất.

TV3 quản lý tiến độ bằng `results/coverage.csv`, run index và PR.

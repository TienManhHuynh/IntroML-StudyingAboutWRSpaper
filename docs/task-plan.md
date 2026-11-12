# Phân công và tiến độ

Nhóm trưởng chốt **Full scope** ngày 12/11/2026, thay mục Core Scope trong nguồn cũ; không sửa bản nguồn.

| Người | Trách nhiệm | Bàn giao dự kiến | Slide |
|---|---|---|---|
| TV1 / Nguyên | Background, theory, PAC/PA-II, FSOL | `src/models/pac.py`, `fsol.py`, tests | 01-11 |
| TV2 / Tùng | Survival, WRS/WAT, averaging/zeroing, complexity | `src/wat/reservoir.py`, `wat.py`, tests | 12-21 |
| TV3 / nhóm trưởng | Dataset, metrics, runners, tuning, W&B, statistics/deviations | `src/data/loader.py`, `src/evaluation/metrics.py`, experiment scripts/plots | 22-30, ghép cuối |

Các file thuật toán/runner trên là deliverables chưa triển khai. Top-K/moving/EMA cần TV2 phối hợp TV3. Appendix E cần TV1/TV2 mở rộng learners; Appendix C cần TV3 bổ sung EMBER.

| Mốc 2026 | Mục tiêu | Trạng thái khởi tạo |
|---|---|---|
| 11/11 | Đọc paper bước đầu | Chưa xác minh từ repo |
| 12/11 | Khung main, Full scope, API và coverage | Dựng trong commit khung |
| 13/11 | Theory/method và tích hợp API | Planned |
| 14/11 | TV3 code/thí nghiệm theo phân công | Planned; cần benchmark và điều phối tài nguyên |
| 15/11 | Bàn giao PPTX, notes, nguồn | Planned |
| 16-20/11 | Ghép/QA | Planned |
| 21/11 | Mốc trong phân công | Cần đối chiếu Moodle |

Full scope nhiều jobs; các mốc là mục tiêu nội bộ, không bảo đảm đủ tài nguyên. TV3 benchmark một job mỗi nhóm kích thước trước khi phân phối máy, dùng job IDs khác nhau, cùng manifest và W&B project.

## Điểm cần chốt khi tích hợp

- File gốc có hai cách chia slide TV1/TV2. Khung dùng contract cuối, ghi nhận đồng thuận trong PR bàn giao.
- Chốt loss/update, tie prediction, bias, normalization, terminal reservoir và exponential weighting từ paper/reference.
- Xác minh dataset hosted revision/reconstruction và W&B entity/project public thực tế.
- Thay API cập nhật contract; không tự sửa học thuật người khác khi chưa thống nhất.

TV3 quản lý tiến độ bằng `results/coverage.csv`, run index và PR.

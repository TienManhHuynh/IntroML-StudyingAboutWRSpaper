# Quy trình làm việc nhóm

`main` là nhánh tích hợp do TV3/nhóm trưởng điều phối. Khung ban đầu được đưa lên `main`; các phần tiếp theo nên qua PR để kiểm tra và ghi nhận đóng góp.

1. `git switch main` rồi `git pull --ff-only`.
2. Tạo nhánh, ví dụ `codex/tv1-base-learners`, `codex/tv2-wat`, `codex/tv3-experiments`.
3. Triển khai phần phụ trách, thêm docstrings purpose/parameters/returns và tests có ý nghĩa.
4. Chạy `python scripts/check_repository.py` và tests liên quan nếu có implementation.
5. Commit từng bước thực tế, push nhánh, mở PR vào `main` kèm mô tả và validation.
6. Nhóm trưởng kiểm tra tích hợp rồi merge. Không force-push `main` hay commit khóa/dataset lớn.

TV1 sở hữu models và slide 01-11; TV2 sở hữu WAT và slide 12-21; TV3 sở hữu dữ liệu/evaluation/experiments/tracking, slide 22-30 và ghép cuối. Thay API phải cập nhật `docs/implementation-contract.md` và thống nhất. Người khác nhận xét lỗi học thuật, người phụ trách sửa; nhóm trưởng không tự thay học thuật khi chưa thống nhất.

PR ghi rõ scaffold, implementation, smoke test hay reproduction thật. Core algorithm tự viết từ paper; code tác giả chỉ so sánh. Nếu có số liệu, kèm config/seeds/commit/dataset revision và W&B run public. Nếu có hình, kèm source hoặc mã tạo plot. Không báo Completed chỉ vì script không lỗi.

Một thí nghiệm hoàn tất khi runner end-to-end, jobs đủ coverage, kết quả có W&B public, plot/bảng tự tạo, đối chiếu paper và giải thích deviations. CI khung không xác nhận các điều kiện đó.

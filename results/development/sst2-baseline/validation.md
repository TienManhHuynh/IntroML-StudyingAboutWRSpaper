# Validation - 14/11/2026

- `python -m pytest -q`: 17 tests pass và 4 unittest subtests pass (TV1 + TV3).
- `python scripts/check_repository.py`: 16 dataset IDs, 5.920 planned jobs, source checksums, local links và coverage pass. Không phải test reproduction.
- Hosted development: PAC và FSOL, seeds42-46, mỗi trial 600 SST-2 samples (train420/test180), một pass, 41 checkpoints; tổng 10 runs completed_local. Không có WRS hoặc W&B public.
- `run_noise_analysis.py`: synthetic n=100, PAC, rates0/.05/.1/.2/.3, 5 local smoke runs; không phải noise results paper.
- `run_tuning.py`: synthetic n=60, PAC, đủ 7 C_err x 5 seeds = 35 local trials, selection.json tạo được. Không dùng selection synthetic như tuned parameters SST-2.
- `run_reproduction.py --wandb-mode offline`: synthetic n=80, FSOL; metrics và artifact ghi thành công. Lần đầu staging ngoài workspace thất bại; runner giữ run.json=failed và local evidence. Đã sửa cache/data/config paths vào wandb/ trong repo và chạy lại vào output mới thành công. Không có public URL.
- Slide nháp: 9 slides (phần22-30), 9 notes nonempty, word counts100/99/101/79/79/79/86/89/95, dưới20MB, charts/tables editable, package/layout/native chart validation pass. Đã render và quan sát từng slide; chưa mở trong PowerPoint. Không phải deck nộp cuối.

Môi trường dependencies đúng lúc thử có trong `requirements-tv3-dev.lock.txt`; code giữ Python>=3.9 syntax và môi trường thử Python3.11.0. Kết quả chạy giữa lúc code đang dirty, chỉ dùng development validation. Public W&B, paper-equivalent data, main WRS/ablations, reference outputs và final result slides chưa có bằng chứng hoàn tất.

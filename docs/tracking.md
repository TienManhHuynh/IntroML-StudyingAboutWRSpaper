# W&B và bằng chứng

TV3 chọn một project W&B **public** dùng chung. `.env.example` chỉ là biến đề xuất, chưa được runner đọc và chưa tạo project. Dùng `wandb login` tại máy chạy, không commit API key.

## Run metadata

- job_id, experiment group, dataset/revision/checksum, config hash.
- Git commit, dirty-state, command, Python/dependency versions, OS/CPU/RAM.
- Trial seed, split/stream/reservoir/noise seeds, split checksum.
- Learner, tuned parameters, K, weighting/averaging/zeroing/noise.
- Checkpoint index, base/method/oracle accuracy, ROP components, sparsity.
- Final metrics, train/evaluation timings, status/failure reason.
- Artifacts config, metrics, plots và manifest; không tự upload raw dataset lớn.

## Output đề xuất

```text
results/raw/<job_id>/config.json
results/raw/<job_id>/environment.json
results/raw/<job_id>/checkpoints.csv
results/raw/<job_id>/metrics.json
results/raw/<job_id>/run.json
results/run-index.csv
results/paper-comparison.csv
figures/<plot>.svg hoặc .png + mã tạo plot
```

run-index chỉ thêm run thật: job_id/group/dataset/learner/seed/commit/status/wandb_url/artifact. Paper comparison có source/variant/metric/paper_value/reproduced_value/delta/relative_deviation/reason/run_urls. Missing không được điền 0.

Offline phải sync và kiểm tra public access trước trích dẫn. Disabled/offline chưa sync không đủ nộp. Aggregate report dẫn mọi run thành phần, n_trials và code lọc/tổng hợp. Cập nhật bảng README bằng URL thật; project-level link không thay run-specific link cho một con số.

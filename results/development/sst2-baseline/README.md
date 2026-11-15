# SST-2 baseline development evidence

**Đây là kiểm tra pipeline phát triển, chưa phải tái hiện paper để nộp.** Không có WRS hoặc public W&B runs. Chạy trên subset 600 mẫu từ HF Stanford SST-2, train 420/test 180, subset seed 0, split seeds 42-46. CountVectorizer fit train riêng từng trial. C=1, eta=1, lambda=.01 là cấu hình thử chưa được chọn bằng tuning paper.

- `summary.json`: mean/min/max của 5 trials cho mỗi learner; có provenance và cờ public URLs còn thiếu.
- `run-index.csv`: nguồn từng trial; `path` ghi đường dẫn raw lúc chạy. Sau checkout, dùng bản copy `evidence/<learner>/<seed>/` tương ứng.
- `evidence/`: 10 local runs, mỗi run có config/environment/checkpoint metrics/final metrics/status/split-noise masks; **không chứa raw features hay văn bản dataset**.
- `trajectory-*.png` / `.svg`: plots tạo từ checkpoints thật bằng `experiments/summarize_results.py`; xem thứ tự trong run index.
- `validation.md`: lệnh và giới hạn kiểm chứng.

Source HF revision pin và SHA256 được ghi trong mỗi config. Dòng asset gốc 67.349 khác paper 67.337. Không sửa paper metadata cho khớp dữ liệu hiện tại. Metadata `dirty=true`: code đang được phát triển khi chạy, nên commit cha trong environment không tự xác nhận đúng toàn bộ source. Số liệu này không được dùng như proof final implementation.

Để dựng lại summary sau checkout, truyền các thư mục evidence (chọn output mới):

```powershell
python experiments/summarize_results.py --runs results/development/sst2-baseline/evidence/pac/42 results/development/sst2-baseline/evidence/pac/43 results/development/sst2-baseline/evidence/pac/44 results/development/sst2-baseline/evidence/pac/45 results/development/sst2-baseline/evidence/pac/46 --output results/rebuilt-pac
```

Source plots/metrics là code TV3 trong repo; không lấy số từ paper làm số nhóm. WRS/full data/tuning/public W&B còn phải hoàn thiện.

# Giao diện tích hợp đề xuất

Đặc tả cần thống nhất trong PR, không phải implementation.

## TV1: learner

| API | Input | Output / quy ước |
|---|---|---|
| constructor | n_features, hyperparameters có tên | Khởi tạo weights=0, không dùng test |
| `loss(x,y)` | Một sample sparse/dense, y=-1/+1 | Scalar loss đúng thuật toán, không mutate |
| `update(x,y)` | Một sample/label | Một bước paper; return cần thống nhất |
| `get_weights()` | Không | Bản copy weights, không alias |
| `decision_function(X)` | Batch features | Scores 1D |
| `predict(X)` | Batch features | Labels -1/+1; tie=0 cần xác minh |

PAC/PA-II và FSOL có docstrings purpose/parameters/returns, tests tính tay, no-update và sparsity. Bias/normalization phải công khai.

## TV2: WAT

Nhận base learner, K, weighting, averaging, voting_zeroing và seed. Cung cấp update/get_weights/decision_function/predict cho runner. Tách w_t với w_WRS; inference không dùng test labels để chọn reservoir.

Passive/aggressive theo tiêu chí loss/update paper; không tự đồng nhất dự đoán đúng với passive. Pseudo-passive chỉ dùng NONPA. Snapshot trước base update, đúng survival/sampling priority, tests tie/empty/terminal candidate. Bộ nhớ <=K candidate, tránh dense không cần thiết. Top-K/moving/EMA chung API, phối hợp TV3.

## TV3: loader/runner

Loader trả X_train,y_train,X_test,y_test và provenance; không leak label. Cùng split/stream cho model ghép, RNG reproducible. Oracle đúng baseline/trial và checkpoint. Lưu config/environment/checkpoints/final metrics/timings/run URL/failure reason. Run hoàn tất chỉ khi artifacts/tracking cần thiết thành công, lỗi trả nonzero.

Quy ước chưa rõ cần xác minh paper/reference trước đóng băng, không biến giả định thành claim từ paper. Khi đổi API cập nhật callers/tests và docs cùng PR.

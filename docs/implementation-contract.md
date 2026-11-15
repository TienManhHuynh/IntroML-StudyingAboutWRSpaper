# Giao diện tích hợp đề xuất

TV1/TV3 vẫn là đặc tả cần thống nhất. TV2 đã có implementation theo contract bên dưới; chưa kiểm chứng tích hợp learner thật TV1.

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

Implementation: `src.wat.WAT(base_learner, K=64, weighting="standard", averaging="simple", voting_zeroing=False, seed=None)`. Weighting nhận `standard`/`exponential`; averaging nhận `simple`/`weighted`. `update(x,y)` trả self, không phụ thuộc giá trị trả về của base update. Base `get_weights()` trả dense vector 1D hữu hạn; sparse X được chuyển nguyên trạng vào learner và hỗ trợ inference. Base update chỉ gọi khi loss dương; learner giữ mọi state phụ như FSOL theta. Ensemble trả về là bản copy.

Quy ước: dùng số candidate thực tế m; reservoir rỗng dùng active weights; tổng averaging weight bằng 0 dùng simple average; VZ là đa số nghiêm ngặt các giá trị zero chính xác; score 0 trả +1; không flush terminal candidate. Quy ước xử lý biên cần đối chiếu reference trước đóng băng. Xem [README TV2](../src/wat/README.md).

Passive/aggressive theo tiêu chí loss/update paper; không tự đồng nhất dự đoán đúng với passive. Pseudo-passive chỉ dùng NONPA. Snapshot trước base update, đúng survival/sampling priority, tests tie/empty/terminal candidate. Bộ nhớ <=K candidate, tránh dense không cần thiết. Top-K/moving/EMA chung API, phối hợp TV3.

## TV3: loader/runner

Loader trả X_train,y_train,X_test,y_test và provenance; không leak label. Cùng split/stream cho model ghép, RNG reproducible. Oracle đúng baseline/trial và checkpoint. Lưu config/environment/checkpoints/final metrics/timings/run URL/failure reason. Run hoàn tất chỉ khi artifacts/tracking cần thiết thành công, lỗi trả nonzero.

Quy ước chưa rõ cần xác minh paper/reference trước đóng băng, không biến giả định thành claim từ paper. Khi đổi API cập nhật callers/tests và docs cùng PR.

# Protocol thực nghiệm

Nguồn: paper Section 4, Appendix B. Đặc tả cần triển khai; scaffold chưa chạy protocol.

## Dataset và split

- Xác minh 16 dataset với Table 1. News20 khác Newsgroups (Binary, CS).
- Runner tải tự động từ HF/Kaggle. Hosted source null nghĩa chưa xác minh, không tự bịa URL. Nếu thiếu bản phù hợp, nhóm chuẩn bị/upload hợp lệ.
- Lưu revision/checksum, features/schema, label mapping, preprocessing và khác biệt kích thước; không commit raw data.
- Random 70/30, một pass train; cùng split/stream cho base/WRS/ablations của trial. Không tự thêm stratification rồi coi đúng paper.
- Khởi tạo 0; dữ liệu sparse giữ sparse, tránh densify toàn dataset lớn. Ghi rõ mọi preprocessing fit trên tập nào.
- Newsgroups Binary CS gộp Computers/Science thành +1, còn lại -1. SST-2 dùng CountVectorizer. Xác minh vocabulary/tokenization/normalization/bias chưa rõ.

## Hyperparameter tuning

- PAC C_err=10^-3,...,10^3.
- FSOL eta=2^-3,...,2^9; lambda=0,10^-3,...,10^3.
- Mỗi cấu hình 5 trials. Appendix B.2 chọn trong top 2.5% final test accuracy cấu hình có ROP cao nhất, metrics trung bình 5 trials.
- Paper dùng test accuracy trong chọn cấu hình: nhánh tái hiện ghi rõ điều này và phân tích hạn chế. Validation-only là protocol cải tiến riêng, không trộn vào bảng tái hiện.
- Làm tròn top 2.5%, ties và chi tiết reference phải xác minh/lưu lại.
- WRS dùng hyperparameters của base tương ứng; mọi lựa chọn variant theo test phải công khai.

## Seeds, logging, metrics

- Seeds nhóm chọn 42-46; tách RNG split/stream/reservoir/noise, lưu split indices hoặc checksum.
- K=[1,4,16,64], standard/exponential, simple/weighted, zeroing false/true.
- Khoảng 200 checkpoint cách đều cộng đầu/cuối theo B.3. Lưu index chính xác, tránh trùng, xác minh cách chia nguyên với reference.
- Final accuracy: tỷ lệ test đúng sau một pass. Final sparsity: tọa độ zero / kích thước vector suy luận; ghi bias/zero tolerance, không dùng dataset sparsity.
- Oracle_j = max accuracy của baseline tương ứng từ đầu đến checkpoint j. ROP xấp xỉ = mean(oracle_j - method_accuracy_j) trên grid logging B.3. Cùng oracle/grid cho các biến thể; ROP thấp hơn tốt hơn, có thể âm.
- Lưu fractions và quy tắc hiển thị %, không trộn với percentage points. Đối chiếu mean 5 trials; Figures 2-4 dùng error bars min/max, không tự thay bằng standard deviation.

## Ablations, runtime, statistics

- Top-K: K=64 survival lớn nhất, simple/weighted; xác minh ties và terminal handling.
- Moving average: 64 vector gần nhất. EMA gamma=.9; xác minh thời điểm lấy vector/averaging.
- Benchmark training riêng, ghi có/không evaluation, I/O, W&B, sparse operations và peak RAM.
- Wilcoxon ghép 16 dataset sau mean 5 trials; xác minh alternative/ties/zero policy với reference, lưu SciPy version. Không chỉ báo datasets thuận lợi.

## Reference và deviations

Chạy code tác giả trong môi trường tách biệt, lưu commit và lệnh; không nhập core code vào repo nhóm. So small-case outputs, loss/update, survival/reservoir, checkpoints/metrics để khoanh vùng divergence.

Bảng paper comparison có nguồn table/figure/variant, paper value, reproduced value, absolute delta, relative deviation khi có nghĩa, giải thích và run URLs. Paper value bằng/gần 0 (đặc biệt ROP) thì relative deviation N/A kèm lý do; vẫn báo absolute delta. Số đọc từ chart ghi là xấp xỉ, có phương pháp và sai số.

## Noise bổ sung

Nhóm chọn rates [0,.05,.10,.20,.30], chỉ lật train labels sau split, giữ test sạch; lưu noise mask/seed, cùng mask cho các model. Dùng hyperparameters chốt clean protocol hoặc giải thích rõ lựa chọn khác. Không thay MAIN bằng noise analysis.

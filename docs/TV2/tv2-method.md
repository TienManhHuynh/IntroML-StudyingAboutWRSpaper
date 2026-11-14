Nguồn học thuật: Wu et al., _Stabilizing Linear Passive-Aggressive Online Learning with Weighted Reservoir Sampling_, NeurIPS 2024, Section 2–3, Algorithm 1 (trang 3–5 của [paper trong repo](../../papers/wrs-neurips2024.pdf)).

## 1. Bài toán WAT giải quyết

PAC và FSOL học online: nhận từng cặp `(x_t, y_t)`, rồi quyết định giữ nguyên hoặc cập nhật model. Một mẫu nhiễu có thể làm model cập nhật quá mạnh, khiến chất lượng trên dữ liệu chưa thấy giảm đột ngột. Nếu chuyện này xảy ra cuối luồng, dùng riêng vector cuối cùng có thể cho kết quả kém.

WAT (WRS-Augmented Training) giữ một reservoir gồm tối đa K vector model từng xuất hiện trong quá trình học. Các model có nhiều bước Passive liên tiếp được ưu tiên lấy mẫu. Model suy luận là trung bình các vector trong reservoir, tùy chọn thêm voting-based zeroing. Base learner vẫn cập nhật theo quy tắc gốc; vector trung bình không được dùng thay thế trạng thái đang học.

## 2. Passive, Aggressive và Survival Time

Với nhãn `y_t ∈ {-1,+1}`, score của model hiện tại là:

$$f_t(x_t)=w_t^\top x_t.$$

Paper dùng hinge loss:

$$\ell_t=\max(0,1-y_t w_t^\top x_t).$$

| Điều kiện            | Trạng thái             | Xử lý                                                              |
| -------------------- | ---------------------- | ------------------------------------------------------------------ |
| `y_t w_t^T x_t >= 1` | Passive, loss bằng 0   | Giữ nguyên model, tăng survival counter                            |
| `y_t w_t^T x_t < 1`  | Aggressive, loss dương | Kết thúc candidate cũ, xét đưa vào reservoir, rồi cập nhật learner |

Ví dụ: `y=+1` và score `0.3` cho dự đoán đúng dấu, nhưng hinge loss là `0.7`, nên vẫn Aggressive. Không dùng điều kiện “predict đúng” để thay cho `loss == 0`.

**Survival Time s** trong Algorithm 1 là số bước Passive liên tiếp sau khi một candidate được hình thành, trước bước Aggressive tiếp theo. Khởi tạo `s=0`; mỗi Passive tăng một; sau Aggressive đặt về 0. Bước Aggressive chấm dứt candidate không được tính vào s.

Ví dụ một candidate gặp chuỗi `Passive → Passive → Passive → Aggressive`: survival của candidate là 3. Ngay trước lần cập nhật cuối, xét lưu **candidate cũ với s=3**; model mới bắt đầu với s=0.

### Vì sao dùng survival làm trọng số?

Model tồn tại lâu mà không cần sửa đã vượt qua nhiều mẫu tiếp theo với đủ margin. Vì thế survival là tín hiệu về chất lượng tương đối, có sẵn từ quá trình train, không cần giữ riêng validation set để đánh giá từng candidate.

Trực giác minh họa: nếu một model cố định gặp các mẫu i.i.d., mỗi mẫu có xác suất gây Aggressive là q, thì số lần Passive trước lần Aggressive đầu tiên có phân phối hình học, với `E[s]=(1-q)/q` khi `0<q<=1`. q càng nhỏ thì survival kỳ vọng càng lớn. Đây là giải thích trực giác, không phải khẳng định survival bằng test accuracy; q ở đây là xác suất vi phạm margin.

Survival có thể bị ảnh hưởng bởi may mắn và thứ tự stream. Vì vậy paper dùng lấy mẫu và tổng hợp nhiều model thay vì chỉ chọn một model sống lâu nhất. Lập luận tổng quát hóa trong Section 3.2 có các giả định, gồm i.i.d.; không suy ra bảo đảm tương tự cho mọi luồng có concept drift.

## 3. Weighted Reservoir Sampling (WRS)

Mục tiêu: từ một luồng candidate có thể chưa biết độ dài, lấy tối đa K candidate với xác suất ưu tiên trọng số lớn, chỉ qua một lượt và không lưu toàn bộ lịch sử.

Mỗi candidate i có trọng số `b_i`, số ngẫu nhiên độc lập `u_i ∈ (0,1)` và priority:

$$k_i=u_i^{1/(b_i+\epsilon)},\qquad \epsilon=10^{-8}.$$

Giữ K priority lớn nhất: nếu reservoir chưa đầy thì thêm candidate; nếu đầy thì chỉ thay phần tử có priority nhỏ nhất khi priority mới lớn hơn. Candidate bị loại không cần lưu tiếp.

Với cùng u, b lớn làm priority gần 1 hơn, nên tăng cơ hội được giữ. Đây vẫn là lấy mẫu ngẫu nhiên: candidate có survival lớn không chắc chắn được chọn. Khi K>1, không mô tả xác suất inclusion đơn giản là `K*b_i/sum(b)`.

Hai cách tạo trọng số trong paper:

| Weighting scheme | Trọng số b | Ý nghĩa                                 |
| ---------------- | ---------- | --------------------------------------- |
| Standard         | `s`        | Ưu tiên survival theo mức tuyến tính    |
| Exponential      | `exp(s)`   | Ưu tiên mạnh hơn cho candidate sống lâu |

Lưu ý `exp(0)=1` trong Exponential, còn Standard cho `b=0` khi không có bước Passive. Epsilon tránh chia cho 0 trong priority; không tự động giải quyết mẫu số bằng 0 khi weighted averaging.

### Ví dụ tính tay với Standard, K=2

Bỏ epsilon trong phép tính minh họa:

| Candidate | s=b |      u | Priority xấp xỉ | Reservoir sau khi xét                |
| --------- | --: | -----: | --------------: | ------------------------------------ |
| A         |   2 |   0.25 |           0.500 | A                                    |
| B         |   4 | 0.0625 |           0.500 | A, B                                 |
| C         |   1 |    0.8 |           0.800 | C và một trong A/B, theo quy tắc tie |

C có survival ngắn hơn nhưng vẫn có thể được chọn. Đây là khác biệt so với deterministic Top-K theo survival. Tie giữa các resident cần quy tắc xác định để tái lập.

## 4. Phân tích Algorithm 1 từ đầu đến cuối

| Dòng     | Nội dung                                         | Giải thích khi triển khai                                                       |
| -------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| Input, 1 | Nhận w, stream, K, WS, AS, VZ, H                 | H là hyperparameters của learner; thiết kế nhóm bổ sung seed cho RNG            |
| 2–5      | Khởi tạo `s=0`, reservoir R và hai mảng b, k     | Mỗi entry là bộ `(snapshot, weight, priority)`; ba thành phần luôn đi cùng nhau |
| 7–8      | Đọc từng mẫu, tính loss của w hiện tại           | Tính loss trước khi mutate learner                                              |
| 9–10     | Khi Aggressive, kết thúc candidate và sinh u     | Candidate đang xét là model trước cập nhật                                      |
| 11–15    | Đổi survival thành trọng số và priority          | Standard dùng s; Exponential dùng exp(s)                                        |
| 16–18    | So priority với phần tử nhỏ nhất, thêm hoặc thay | Nếu R rỗng/chưa đầy, append; chỉ tìm minimum để thay khi đã đầy                 |
| 19–20    | Cập nhật base learner                            | Dùng đúng quy tắc PAC hoặc FSOL, không dùng vector ensemble làm đầu vào update  |
| 21       | Reset `s=0`                                      | Bắt đầu đếm survival của candidate mới                                          |
| 22–23    | Nếu Passive, tăng s                              | Không tạo một entry mới cho mỗi bước Passive                                    |
| 24–28    | Tạo vector ensemble từ R                         | Chọn simple hoặc weighted average                                               |
| 29–31    | Tùy chọn voting-based zeroing                    | Zero một tọa độ nếu đa số candidate có giá trị 0 ở tọa độ đó                    |
| 32       | Trả vector ổn định `w_WRS`                       | Phân biệt với vector cuối w của learner                                         |

PAC trong paper là PA-II, có cập nhật:

$$w_{t+1}=w_t+\frac{\ell_t}{\|x_t\|^2+1/(2C_{err})}y_tx_t.$$

FSOL giữ thêm trạng thái theta và khi Aggressive cập nhật:

$$\theta_{t+1}=\theta_t+\eta y_t x_t,$$
$$w_{t+1}=\operatorname{sign}(\theta_{t+1})\odot[|\theta_{t+1}|-\eta\lambda]_+.$$

Snapshot dùng cho suy luận chỉ cần w; để resume train FSOL phải lưu cả theta.

### Tạo model suy luận

Gọi m là số candidate đang có thực tế. Khi reservoir đầy, `m=K`:

$$w_{WRS}=\frac{1}{m}\sum_{j=1}^{m}R_j \quad\text{(simple average)},$$

$$w_{WRS}=\frac{\sum_{j=1}^{m}b_jR_j}{\sum_{j=1}^{m}b_j}\quad\text{(weighted average)}.$$

Đừng nhầm sampling weight với averaging: dù đã lấy mẫu có trọng số, vẫn có thể chọn simple average. Weighted average tiếp tục ưu tiên b lớn khi tổng hợp.

Nếu bật VZ, với mỗi tọa độ d, đặt `w_WRS[d]=0` khi số candidate có `R_j[d]=0` **lớn hơn m/2**. Đây là voting về độ thưa, không phải bỏ phiếu nhãn dự đoán. Ví dụ 3 trong 4 candidate có tọa độ đó bằng 0 thì zero; 2 trong 4 không phải đa số.

## 5. Training Pipeline và Testing/Inference Pipeline

xem ảnh cùng thư mục

<!-- ## 7. Các trường hợp biên cần chốt trước code

| Trường hợp | Algorithm 1 thể hiện | Đề xuất / hành động cần làm |
|---|---|---|
| Reservoir chưa đầy | Công thức dùng K; chỉ số minimum khi rỗng chưa được mô tả đầy đủ | Append khi chưa đầy; ensemble chia theo số entry thực tế m, công khai quy ước |
| R rỗng khi yêu cầu predict | Chưa có fallback | Đề xuất dùng bản copy w hiện tại, không đưa nó vào reservoir chỉ vì có predict |
| Standard có tổng b bằng 0 | Epsilon chỉ nằm trong công thức priority | Đề xuất fallback simple average; cần đối chiếu reference |
| Candidate còn sống ở cuối stream | Không có bước flush cuối trong pseudocode | Theo Algorithm 1 không tự thêm candidate cuối; xác minh reference trước khi đổi chính sách |
| Priority bằng nhau | Điều kiện mới phải lớn hơn ngưỡng | Không thay khi bằng; tie giữa resident chọn theo chỉ số/ID cố định |
| `exp(s)` quá lớn hoặc priority làm tròn thành 1 | Pseudocode dùng số thực trực tiếp | Dùng so sánh priority trong log-space; không cắt survival tùy ý |

Để so sánh ổn định, có thể giữ `log(k)=log(u)/(b+epsilon)` thay cho k. Với Exponential, tránh tạo exp(s) bằng cách tính `log(-log(k))=log(-log(u))-log(exp(s)+epsilon)` qua log-sum-exp; đại lượng này càng nhỏ tương ứng k càng lớn. Khi averaging exponential weights, chuẩn hóa theo `exp(s_j-max(s))`. Đây là thiết kế số học tương đương, không đổi ý nghĩa sampling.

## 8. Chi phí và điểm kiểm tra khi hiện thực

Với vector dense D chiều, reservoir dùng O(KD) bộ nhớ; thêm O(K) metadata. Mỗi Aggressive cần tính priority và tìm minimum O(K) nếu dùng mảng, hoặc O(log K) nếu dùng min-heap, cộng chi phí copy snapshot O(D) khi nhận vào reservoir. Tính ensemble trực tiếp O(KD); không cần làm lại ở mỗi bước Passive nếu reservoir chưa đổi. Inference một mẫu dense O(D). Với dữ liệu/model sparse cần giữ biểu diễn thích hợp, tránh densify toàn bộ dataset.

Các kiểm tra quan trọng cho phần code sau: đúng dấu nhưng thiếu margin vẫn Aggressive; s đếm đúng; snapshot trước update và không alias; reservoir không quá K; inference không mutate; xử lý R rỗng/tổng trọng số 0; VZ dùng đa số nghiêm ngặt; cùng seed và stream cho cùng reservoir; quy tắc terminal candidate được ghi rõ.

## 9. Gợi ý đưa lên slides 12–21

| Slide | Nội dung |
|---|---|
| 12 | Động cơ: final model nhạy với outlier cuối luồng |
| 13 | Passive/Aggressive, hinge loss và margin |
| 14 | Survival Time: ví dụ chuỗi và trực giác chất lượng |
| 15 | WRS: reservoir, trọng số và priority |
| 16 | Algorithm 1: khởi tạo và nhánh Passive/Aggressive |
| 17 | Algorithm 1: sampling rồi mới cập nhật base learner |
| 18 | Simple/weighted average, exponential weights và VZ |
| 19 | Training Pipeline, ranh giới trách nhiệm TV1/TV2/TV3 |
| 20 | Testing/Inference Pipeline và phân biệt hai vector w |
| 21 | Complexity, giới hạn và các quy ước triển khai |

Lời giải thích ngắn khi thuyết trình: “Thay vì chỉ dùng model cuối cùng, WAT giữ một tập nhỏ các model trung gian. Model nào trải qua nhiều bước Passive liên tiếp sẽ được ưu tiên lấy mẫu vì đó là dấu hiệu model hoạt động tốt trên các mẫu tiếp theo. Khi dự đoán, chúng ta tổng hợp các model đã lưu thành một vector duy nhất để giảm phụ thuộc vào một lần cập nhật cuối luồng.” -->

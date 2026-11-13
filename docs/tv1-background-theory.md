# TV1 — Background, Theory & Base Learners
Người phụ trách: Hà Huy Nguyên

## 1. Bài toán, động lực và ứng dụng

### 1.1. Vấn đề nghiên cứu (Research Problem)

Bài báo nghiên cứu sự thiếu ổn định của mô hình phân loại tuyến tính trong học trực tuyến (online learning). Ở cách học này, dữ liệu đến lần lượt; mô hình tiếp nhận từng mẫu và cập nhật trọng số trước khi xử lý mẫu tiếp theo.

Các thuật toán như bộ phân loại thụ động–tích cực (Passive-Aggressive Classifier, PAC) có khả năng phản ứng nhanh với lỗi. Tuy nhiên, khi gặp mẫu nhiễu hoặc mẫu ngoại lệ, chúng có thể điều chỉnh trọng số quá mạnh để phù hợp với mẫu vừa quan sát. Hiện tượng này được gọi là điều chỉnh quá mức (over-correction): mô hình sửa được lỗi trên một mẫu nhưng lại dự đoán kém hơn trên nhiều mẫu khác.

Vì vậy, độ chính xác trên dữ liệu chưa thấy có thể giảm đột ngột giữa hai thời điểm học liên tiếp. Các mẫu tiếp theo có thể giúp mô hình phục hồi, nhưng nếu quá trình học dừng đúng lúc độ chính xác vừa giảm, mô hình cuối cùng sẽ có chất lượng thấp hơn kỳ vọng. Vấn đề cũng ảnh hưởng đến hệ thống phải đưa ra dự đoán liên tục trong quá trình cập nhật. [1, Sec. 1]

Bài toán mà tác giả đặt ra là: làm thế nào tận dụng các phiên bản mô hình xuất hiện trong quá trình học để tạo ra mô hình dự đoán ổn định hơn, mà không cần:

- Duyệt lại dữ liệu huấn luyện nhiều lần.
- Dùng một tập xác thực riêng (validation set) để chấm điểm và lựa chọn các phiên bản mô hình.
- Lưu tất cả phiên bản, khiến bộ nhớ tăng theo độ dài luồng dữ liệu.

Trọng tâm là giảm những lần suy giảm độ chính xác trong suốt quá trình học, chứ không chỉ tăng độ chính xác tại bước cuối cùng.

### 1.2. Động lực nghiên cứu (Motivation)

Học trực tuyến phù hợp với dữ liệu lớn, dữ liệu nhiều chiều và những trường hợp không thể đưa toàn bộ dữ liệu vào bộ nhớ. Tuy nhiên, lợi thế cập nhật nhanh cũng khiến mô hình dễ bị ảnh hưởng bởi một quan sát bất thường.

Một giới hạn tốt về độ chênh mất mát tích lũy (regret) không đồng nghĩa với việc mọi phiên bản trung gian đều dự đoán tốt trên dữ liệu mới. Vì vậy, chỉ xem kết quả cuối quá trình học có thể bỏ qua những lần mô hình hoạt động kém ở giữa quá trình. [1, Sec. 1–2]

**Ví dụ tự xây dựng: bộ lọc thư rác.**

Giả sử bộ lọc đang nhận diện tốt các email chứa dấu hiệu “trúng thưởng” và “nhận tiền”. Một email như vậy bị gán nhầm nhãn thành thư thường. Khi học theo nhãn sai, mô hình có thể giảm quá mạnh mức độ nghi ngờ đối với các dấu hiệu này, khiến nhiều thư rác thật sau đó bị bỏ sót.

Ví dụ cho thấy việc học thêm một mẫu không bảo đảm mô hình lập tức tốt hơn. Nếu giữ lại và kết hợp được những phiên bản đáng tin cậy trước đó, hệ thống có thể giảm ảnh hưởng của một lần cập nhật bất lợi.

Tuy nhiên, việc liên tục đánh giá mọi phiên bản trên một tập xác thực sẽ tốn tài nguyên. Đây là động lực để tác giả tìm một tín hiệu chất lượng có sẵn trong quá trình học: số bước liên tiếp mà mô hình không cần cập nhật. Tín hiệu này là nền tảng cho phương pháp WAT được trình bày ở phần sau. [1, Sec. 3]

### 1.3. Ứng dụng thực tế

**Phát hiện mã độc (malware detection).**

Đây là một động lực thực tế được tác giả trực tiếp đề cập. Dữ liệu lớn khiến học trực tuyến trở nên hữu ích, trong khi quá trình gán nhãn có nhiễu có thể gây khó khăn cho các thuật toán thụ động–tích cực. Một lần cập nhật bất lợi có thể làm tăng nguy cơ phân loại sai các tệp tiếp theo. Tác giả còn đánh giá phương pháp trên bộ dữ liệu mã độc EMBER trong Appendix C. [1, Sec. 2; App. C]

**Phân loại thư rác và nội dung web.**

Bài báo nêu phân loại web spam như một ứng dụng phù hợp của học trực tuyến. Với bài toán này, mô hình cần xử lý lượng dữ liệu lớn và tiếp tục đưa ra dự đoán trong khi học. Việc hạn chế dao động độ chính xác giúp giảm nguy cơ hệ thống hoạt động kém ngay sau một lần cập nhật. [1, Sec. 1]

Các ứng dụng trên có chung nhu cầu: mô hình không chỉ cần học nhanh mà còn phải duy trì chất lượng dự đoán tương đối ổn định trong quá trình vận hành.

## 2. Các phương pháp trước và khoảng trống nghiên cứu

Trong quá trình học trực tuyến, mỗi lần cập nhật trọng số tạo ra một phiên bản mới của mô hình. Phiên bản cuối cùng chưa chắc là phiên bản dự đoán tốt nhất. Trước WAT, đã có nhiều cách chọn một mô hình từ chuỗi phiên bản này. [1, Sec. 2]

### 2.1. Giữ phiên bản tồn tại lâu nhất (Pocket)

Một cách tiếp cận trong các nghiên cứu về Perceptron là giữ lại phiên bản có chuỗi dự đoán đúng liên tiếp dài nhất. Chẳng hạn, nếu phiên bản A dự đoán đúng 20 mẫu liên tiếp còn phiên bản B chỉ đúng 5 mẫu trước khi mắc lỗi, ta có lý do để ưu tiên A. Ưu điểm là có thể chọn mô hình ngay trong lúc học mà không cần tập dữ liệu riêng để chấm điểm.

Hạn chế nằm ở việc quyết định dựa trên một chuỗi quan sát và chỉ giữ một phiên bản. A có thể đạt kỷ lục vì tình cờ gặp 20 mẫu dễ. Nếu chọn A làm mô hình cuối cùng, các phiên bản khác có chất lượng tương đương cũng không đóng góp vào dự đoán. [1, Sec. 2–3]

### 2.2. Chọn mô hình bằng tập xác thực (validation set)

Một cách khác là dành riêng một tập dữ liệu có nhãn, đánh giá các phiên bản mô hình trên cùng tập đó rồi chọn phiên bản đạt kết quả tốt nhất. Cách này cho phép so sánh trực tiếp các phiên bản thay vì suy đoán chất lượng từ thời gian tồn tại.

Tuy nhiên, hệ thống phải giữ tập xác thực và tốn chi phí đánh giá lại nhiều phiên bản. Khi dữ liệu đến nhanh, có nhiều đặc trưng hoặc hệ thống phải trả kết quả ngay, việc chấm điểm sau mỗi lần cập nhật có thể quá tốn kém. Nếu chỉ chấm ở một vài thời điểm thì lại có thể bỏ qua những phiên bản tốt xuất hiện giữa hai lần đánh giá. [1, Sec. 1–2]

### 2.3. Khoảng trống nghiên cứu (research gap)

| Cách tiếp cận | Điểm có ích | Hạn chế và hệ quả |
|---|---|---|
| Giữ phiên bản tồn tại lâu nhất | Chọn được ngay trong lúc học | Một chuỗi dữ liệu thuận lợi có thể khiến hệ thống quá tin vào một phiên bản |
| Chọn bằng tập xác thực | So sánh các phiên bản trên cùng dữ liệu | Cần giữ thêm dữ liệu và lặp lại việc đánh giá |
| Lấy trung bình đều các phiên bản | Giảm phụ thuộc vào một lần cập nhật | Phiên bản yếu cũng được tính ngang với phiên bản tốt |

Từ đó, bài báo đặt ra câu hỏi: có thể chọn và kết hợp **nhiều phiên bản có triển vọng** ngay trong một lượt học, với bộ nhớ cố định và không cần dùng tập xác thực để chọn phiên bản hay không? Ý tưởng của tác giả là dùng thời gian tồn tại làm tín hiệu ưu tiên, nhưng vẫn lấy mẫu ngẫu nhiên thay vì luôn chọn các phiên bản có kỷ lục cao nhất. [1, Sec. 2–3]

## 3. Những đóng góp của bài báo

### 3.1. Đóng góp lý thuyết (theoretical contributions)

Tác giả phân tích vì sao thời gian tồn tại (survival time) có thể cung cấp thông tin về chất lượng của một phiên bản mô hình. Nếu một mô hình có xác suất mắc lỗi thấp trên dữ liệu mới, nó có nhiều khả năng duy trì được một chuỗi dự đoán đúng dài. Tuy nhiên, một chuỗi dài vẫn có thể xuất hiện do may mắn; do đó không thể kết luận phiên bản sống lâu nhất luôn là phiên bản tốt nhất.

Ngoài việc phân tích cách chọn ứng viên, tác giả còn xây dựng giới hạn trên cho rủi ro kỳ vọng (expected risk) của mô hình tổ hợp. Kết quả liên hệ chất lượng của mô hình dùng để dự đoán với mất mát tích lũy và bảo đảm về regret của thuật toán học cơ sở. Các giả định và giới hạn của từng kết quả được trình bày ở mục 4.4. [1, Sec. 3.2]

### 3.2. Đóng góp kỹ thuật (technical contributions)

Tác giả đề xuất phương pháp huấn luyện tăng cường bằng lấy mẫu có trọng số (WRS-Augmented Training, WAT). Kỹ thuật lấy mẫu WRS đã có từ trước; điểm mới của bài báo là dùng nó để chọn **các véc-tơ trọng số của mô hình trung gian**, với mức ưu tiên dựa trên số bước liên tiếp mà phiên bản đó không cần cập nhật. [1, Sec. 2–3]

Phương pháp giữ tối đa K phiên bản trong một bộ chứa (reservoir). Mô hình cơ sở tiếp tục học theo quy tắc ban đầu, còn mô hình dùng để dự đoán được tạo bằng cách lấy trung bình các véc-tơ trong bộ chứa. Nhờ tách hai vai trò này, thuật toán cơ sở vẫn có thể phản ứng nhanh với dữ liệu mới, trong khi kết quả dự đoán ít phụ thuộc hơn vào một lần cập nhật riêng lẻ.

### 3.3. Đóng góp thực nghiệm (empirical contributions)

Tác giả áp dụng WAT cho hai thuật toán cơ sở PAC và FSOL, rồi so sánh từng thuật toán trước và sau khi thêm WAT trên 16 bộ dữ liệu phân loại nhị phân. Thí nghiệm theo dõi cả độ ổn định trong quá trình học, độ chính xác cuối cùng và độ thưa của mô hình. [1, Sec. 4]

Với K = 64 và lấy trung bình thường, FSOL-WRS dùng trọng số chuẩn cải thiện chỉ số ổn định ROP trên 16/16 bộ dữ liệu; PAC-WRS dùng trọng số hàm mũ cải thiện ROP trên 14/16 bộ dữ liệu. Đây là **kết quả tác giả báo cáo**, chưa phải kết quả tái hiện của nhóm. Các con số này cũng không có nghĩa WAT luôn làm tăng độ chính xác cuối cùng: đối với PAC, tác giả chưa quan sát được mức cải thiện cuối cùng có ý nghĩa thống kê. [1, Table 2; Sec. 4.1]

## 4. Thuật toán nền và cơ sở lý thuyết

### 4.1. Bài toán hình thức và các ký hiệu

Bài báo xét bài toán phân loại nhị phân tuyến tính. Tại bước t, mô hình có trọng số w_t và nhận mẫu có đặc trưng x_t, nhãn thật y_t.

| Ký hiệu | Ý nghĩa |
|---|---|
| t | Thứ tự của mẫu trong luồng dữ liệu |
| D | Số đặc trưng của một mẫu |
| x_t | Véc-tơ đặc trưng của mẫu thứ t |
| y_t | Nhãn thật, nhận giá trị -1 hoặc +1 |
| w_t | Véc-tơ trọng số trước khi học mẫu thứ t |
| score | Điểm dự đoán: score = dot(w_t, x_t) |
| margin | Biên trên mẫu đang xét: margin = y_t * score |
| loss | Mất mát bản lề: loss = max(0, 1 - margin) |

Dấu của score quyết định lớp dự đoán. Tuy nhiên, **đoán đúng chưa chắc đã đủ biên**. Ví dụ, với y_t = +1 và score = 0.2, mô hình đoán đúng lớp nhưng margin = 0.2 và loss = 0.8, nên vẫn cần cập nhật. Khi margin >= 1 thì loss = 0. [1, Sec. 2]

### 4.2. Bộ phân loại thụ động–tích cực (PAC) và biến thể PA-II

#### 4.2.1. Nguyên lý hoạt động

Khi loss = 0, thuật toán ở trạng thái thụ động (passive): giữ nguyên trọng số. Khi loss > 0, thuật toán chuyển sang trạng thái tích cực (aggressive): điều chỉnh trọng số theo hướng cải thiện biên của mẫu hiện tại.

PA gốc tìm véc-tơ mới gần véc-tơ cũ nhất nhưng yêu cầu mất mát trên mẫu hiện tại phải bằng 0. Điều kiện này có thể quá mạnh khi mẫu bị nhiễu. Hai biến thể PA-I và PA-II cho phép còn một phần sai lệch sau cập nhật. Gọi phần sai lệch được phép là epsilon, PA-I phạt theo epsilon, còn PA-II phạt theo epsilon². **Trong các thí nghiệm của bài báo, tên PAC chỉ biến thể PA-II.** [1, Sec. 2]

#### 4.2.2. Công thức cập nhật PA-II

Tính lần lượt:

- score = dot(w_t, x_t)
- loss = max(0, 1 - y_t * score)
- tau = loss / (dot(x_t, x_t) + 1 / (2 * C_err))
- w_next = w_t + tau * y_t * x_t

Ở đây, tau là độ lớn bước cập nhật; C_err > 0 điều khiển mức phạt đối với sai lệch. Với cùng một mẫu và cùng loss, C_err càng nhỏ thì bước cập nhật càng nhỏ. Khi loss = 0, ta có tau = 0 và trọng số không đổi.

#### 4.2.3. Ví dụ tính tay

Giả sử w_t = (0, 0), x_t = (1, 0), y_t = +1 và C_err = 0.5.

- score = 0, nên loss = 1.
- dot(x_t, x_t) = 1 và 1 / (2 * C_err) = 1.
- tau = 1 / (1 + 1) = 0.5.
- w_next = (0, 0) + 0.5 * (+1) * (1, 0) = (0.5, 0).

Sau cập nhật, điểm của mẫu tăng từ 0 lên 0.5. Mô hình đã điều chỉnh theo đúng hướng nhưng biên vẫn dưới 1. Đây là ví dụ cho thấy PA-II cho phép còn sai lệch sau một lần cập nhật.

### 4.3. Học trực tuyến thưa bậc nhất (FSOL)

#### 4.3.1. Nguyên lý hoạt động

FSOL hướng đến một mô hình thưa (sparse model), tức là nhiều trọng số bằng 0. Thuật toán giữ hai véc-tơ: theta tích lũy các lần điều chỉnh, còn w là trọng số thực sự dùng để dự đoán. Khi cần cập nhật, FSOL điều chỉnh theta rồi dùng phép co ngưỡng mềm (soft thresholding) để tạo w. Những tọa độ có độ lớn chưa vượt ngưỡng sẽ trở về 0. [1, Sec. 2]

#### 4.3.2. Công thức cập nhật

Nếu mất mát bản lề bằng 0, giữ nguyên trạng thái. Nếu mất mát lớn hơn 0:

- theta_next = theta_t + eta * y_t * x_t
- threshold = eta * lambda
- Với từng đặc trưng j: w_next[j] = sign(theta_next[j]) * max(abs(theta_next[j]) - threshold, 0)

Trong đó eta là tốc độ học (learning rate), lambda là tham số điều khiển độ thưa và j là chỉ số đặc trưng. Điểm cần phân biệt: dự đoán bằng w, nhưng ở lần cập nhật tiếp theo phải cộng dồn trên theta. Nếu dùng w thay cho theta thì đã thay đổi thuật toán.

#### 4.3.3. Ví dụ tính tay

Giả sử theta_t = w_t = (0, 0), x_t = (2, -0.25), y_t = +1, eta = 1 và lambda = 0.5. Do điểm hiện tại bằng 0 nên loss = 1 và thuật toán cập nhật:

- theta_next = (2, -0.25).
- Ngưỡng bằng 1 * 0.5 = 0.5.
- Tọa độ thứ nhất có độ lớn 2, sau khi co còn 1.5.
- Tọa độ thứ hai có độ lớn 0.25, nhỏ hơn ngưỡng nên trở thành 0.
- Kết quả: w_next = (1.5, 0).

Như vậy, mô hình tạm thời không dùng đặc trưng thứ hai khi dự đoán, nhưng theta vẫn giữ giá trị -0.25 của đặc trưng đó.

### 4.4. Định lý 1–4: giả định, ý nghĩa và giới hạn

Phần lý thuyết trả lời hai câu hỏi: (1) thời gian tồn tại có giúp chọn mô hình tốt hay không, và (2) có thể bảo đảm điều gì về rủi ro của mô hình tổ hợp trên dữ liệu mới? [1, Sec. 3.2; App. A]

| Định lý | Giả định chính | Ý nghĩa và giới hạn |
|---|---|---|
| 1 | Dữ liệu độc lập, cùng phân phối (i.i.d.); xét mất mát đúng/sai (0–1 loss) và một cận dưới của rủi ro | Chặn trên xác suất mô hình sống lâu nhất có rủi ro cao hơn mô hình tốt nhất trong bộ chứa quá một mức cho trước. Sống lâu là tín hiệu có ích, không bảo đảm chọn đúng mô hình tốt nhất. |
| 2 | Tiếp tục xét rủi ro phân loại; giả sử các ứng viên được chia thành nhóm tốt và nhóm kém theo các khoảng rủi ro mà định lý quy định | Khi đang xét B phiên bản sống lâu nhất, thêm phiên bản xếp thứ B+1 có một xác suất làm rủi ro trung bình không tăng. Định lý cho cận dưới của xác suất này, không khẳng định cứ thêm mô hình là tốt hơn. |
| 3 | Luồng dữ liệu i.i.d.; hàm mất mát lồi, bị chặn trên; xét tại thời điểm dừng cố định | Chặn trên rủi ro của mô hình tổ hợp dựa trên mất mát tích lũy có trọng số, số mô hình được giữ và thời gian tồn tại tối thiểu của chúng. Đây là bảo đảm theo xác suất. |
| 4 | Kế thừa bối cảnh của cận tổng quát hóa và sử dụng cận regret của thuật toán PA cơ sở | Liên hệ rủi ro của mô hình WRS với rủi ro của mô hình tối ưu, cùng các số hạng phụ thuộc vào thuật toán cơ sở và lượng dữ liệu. Kết quả không bảo đảm WRS luôn đạt accuracy cao hơn ở bước cuối. |

Cần phân biệt hai cách hiểu về thời gian tồn tại: ở phần phân tích với mất mát 0–1, tác giả đếm chuỗi dự đoán đúng trước lỗi đầu tiên; trong Algorithm 1 áp dụng cho PAC, bộ đếm tăng khi mô hình ở trạng thái passive, tức là **đúng và đủ biên**. Hai điều kiện này không hoàn toàn giống nhau.

### 4.5. Độ phức tạp của các thuật toán nền

Dưới đây là phân tích từ các phép toán ở mục 4.2 và 4.3. Giả sử mỗi véc-tơ có D đặc trưng và được lưu dạng đặc (dense).

| Thuật toán | Thời gian xử lý một mẫu | Bộ nhớ trạng thái mô hình | Lý do |
|---|---|---|---|
| PAC/PA-II | O(D) | O(D) | Tính tích vô hướng để dự đoán; khi cần thì tính độ dài mẫu và cập nhật w |
| FSOL | O(D) | O(D) | Tính điểm bằng w; khi cập nhật thì điều chỉnh theta và co ngưỡng để tạo w |

Hai phương pháp cùng bậc phức tạp nhưng chi phí thực tế có thể khác: PAC chủ yếu giữ một véc-tơ trọng số, còn FSOL giữ cả theta và w. Một mô hình có nhiều trọng số bằng 0 cũng không tự động tiết kiệm RAM nếu chương trình vẫn lưu nó bằng mảng đặc.

Với đầu vào thưa (sparse input), nếu một mẫu chỉ có k đặc trưng khác 0, phép tính điểm có thể chỉ cần O(k) thay vì O(D). Chi phí của cả bước cập nhật còn phụ thuộc vào cách cài đặt: chẳng hạn, FSOL vẫn mất O(D) nếu quét toàn bộ véc-tơ để co ngưỡng. Chi phí lưu nhiều phiên bản trong bộ chứa của WAT là phần bổ sung, được phân tích riêng ở phần phương pháp của TV2.

---

**Tài liệu tham khảo**

[1] S. Wu, F. Lu, E. Raff, and J. Holt, “Stabilizing Linear Passive-Aggressive Online Learning with Weighted Reservoir Sampling,” in Advances in Neural Information Processing Systems, vol. 37, 2024.
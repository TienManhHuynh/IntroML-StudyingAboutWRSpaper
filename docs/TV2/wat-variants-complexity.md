# Các biến thể WAT và chi phí triển khai

Nguồn: Wu et al., NeurIPS 2024, Section 3.1 và Algorithm 1, trang 4–5 của [paper](../../papers/wrs-neurips2024.pdf). Đây là giải thích phương pháp và phân tích implementation, chưa phải kết quả benchmark.

## 1. Standard và Exponential Weighting

Survival s là số bước Passive liên tiếp trước khi candidate bị thay bởi một bước Aggressive. Passive đòi hỏi hinge loss bằng 0, không chỉ dự đoán đúng dấu.

| Scheme | Sampling weight b | Tác động |
|---|---|---|
| Standard | b=s | Ưu tiên tuyến tính theo survival |
| Exponential | b=exp(s) | Ưu tiên mạnh hơn cho survival lớn, vẫn ngẫu nhiên |

Ví dụ s=2 và s=4: tỷ lệ trọng số Standard là 2; Exponential là exp(2), khoảng 7.39. Khi s=0, Standard có b=0, Exponential có b=1.

Sinh u độc lập từ (0,1), priority `k=u**(1/(b+1e-8))`. Giữ tối đa K priority lớn nhất. Candidate survival ngắn vẫn có thể thắng; đây không phải deterministic Top-K. Với K=1 và b dương, xác suất chọn mỗi item tỷ lệ với b+epsilon; với K>1 không tự suy inclusion probability thành K*b/sum(b).

Implementation giữ `log(-log(u)) - log(b+epsilon)` và chọn giá trị nhỏ nhất, tương đương chọn k lớn nhất. Với Exponential dùng log-sum-exp để tính log(exp(s)+epsilon), không tạo exp(s) trực tiếp. Min-heap trên giá trị đối dấu có candidate tệ nhất ở root.

## 2. Simple và Weighted Averaging

Với m candidate thực tế trong reservoir, vector model R_j và trọng số b_j:

- Simple: `w_WRS = sum(R_j)/m`. Mỗi candidate được giữ có đóng góp như nhau.
- Weighted: `w_WRS = sum(b_j*R_j)/sum(b_j)`. Candidate trọng số lớn tiếp tục đóng góp nhiều hơn khi tổng hợp.

Ví dụ R1=(2,0), R2=(0,8), b1=1, b2=3: Simple cho (1,4); Weighted cho (0.5,6).

Sampling và averaging là hai lựa chọn độc lập: lấy mẫu có trọng số vẫn dùng được Simple Average. Weighted Average dùng chính b của scheme đã chọn, không dùng priority k hoặc cộng epsilon. Khi Exponential, hệ số được chuẩn hóa qua `exp(s_j-max(s))` để tránh overflow. Nếu Standard có mọi b bằng 0, implementation fallback Simple Average; đây là quy ước nhóm cho tình huống paper chưa đặc tả.

## 3. Voting-based Zeroing (VZ)

Averaging có thể làm vector FSOL mất độ thưa: các candidate zero ở các tọa độ khác nhau nhưng tổng của chúng không còn zero. Sau averaging, VZ đặt một tọa độ thành 0 nếu **hơn một nửa** candidate có giá trị 0 tại tọa độ đó.

Ví dụ bốn candidate có một tọa độ là [0,0,0,8]: sau simple average là 2, sau VZ thành 0. Nếu [0,0,4,8] thì không zero vì chỉ có một nửa. Voting dùng số candidate, không dùng trọng số b. Đây là vote về sparsity, không phải majority vote nhãn.

VZ hỗ trợ giữ độ thưa nhưng có thể loại một feature hữu ích; không bảo đảm accuracy tăng. Implementation dùng zero chính xác, không tự đặt tolerance.

## 4. Thời gian và bộ nhớ phát sinh

Gọi T là số mẫu train, A là số lần Aggressive, D là số features, P là số lần tổng hợp ensemble và m<=K là số candidate đã lưu. Đây là overhead ngoài state/update vốn có của base learner.

| Công việc | Thời gian | Bộ nhớ bổ sung |
|---|---|---|
| Passive | O(1) ngoài tính loss | Counter O(1) |
| Mỗi Aggressive | O(D + log K) | Snapshot tạm O(D) |
| Simple/Weighted Average | O(mD), Weighted thêm O(m) hệ số | O(D+m) tạm |
| VZ | Thêm O(mD) đếm zero | O(D) counters |
| Inference bằng vector đã tổng hợp | Dense O(D)/mẫu; sparse dot theo nnz | Vector O(D) |
| Đọc thuộc tính candidates để debug | O(mD) do trả bản copy | O(mD) ở caller |

Reservoir lưu tối đa K vector dense: O(KD+K), ngoài active learner và state riêng như FSOL theta. Khi thay resident, có thể có thêm một snapshot O(D) tạm thời; reservoir vẫn không vượt K entry. Aggregate không stack thành ma trận K×D.

Tổng overhead xấp xỉ `O(T + A(D+log K) + P*K*D)`, chưa tính loss. Wrapper phải tính loss trước update; nếu learner cũng tính lại loss trong update thì Aggressive phát sinh thêm một lần tính loss. Với sparse sample, loss tuyến tính thường theo nnz(x).

Dù dùng heap O(log K), `offer` hiện kiểm tra vector O(D) kể cả candidate bị loại. Vì vậy không mô tả toàn bộ offer chỉ là O(log K).

`get_weights()` tổng hợp lại mỗi lần gọi. `predict`/`decision_function` hiện tốn O(KD) để tổng hợp cộng chi phí nhân X với vector. Gọi mỗi timestep có thể tốn O(TKD); runner nên đo ở checkpoint. Khi deploy cố định, lấy vector một lần rồi tái dùng cho nhiều batch để đạt inference một vector tuyến tính.

Standard/Exponential cùng O(1) phép toán scalar để tạo priority; Exponential có hằng số số học khác. Simple/Weighted cùng O(KD); VZ thêm một lượt quét và O(D) bộ đếm. K lớn tăng bộ nhớ và chi phí tổng hợp; không suy ra K lớn luôn tốt hơn.

## 5. Trạng thái bàn giao

Code: [reservoir](../../src/wat/reservoir.py), [WAT](../../src/wat/wat.py), [tests](../../tests/test_wat.py). API và ví dụ: [README](../../src/wat/README.md).

Base TV1 chưa có implementation trong checkout khi thực hiện. Unit tests dùng learner PA-II nhỏ để kiểm tra tích hợp contract, không thay kiểm chứng PAC/FSOL thật.

Các quy ước bổ sung: ensemble chưa đầy chia m; rỗng dùng active weights; score 0 trả +1; không flush candidate cuối theo Algorithm 1; tổng Standard weight bằng 0 dùng Simple Average. Cần đối chiếu code tác giả trước chốt reproduction.

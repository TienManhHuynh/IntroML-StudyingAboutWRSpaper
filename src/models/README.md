# TV1 — Base learners
PAC(n_features, C_err=...) trong src.models.pac; FSOL(n_features, eta=..., lambda_=...) trong src.models.fsol. Tự hiện thực theo Section 2 của Wu et al. (2024).

Quy ước tích hợp đề xuất: weights khởi tạo bằng 0, không thêm bias hay chuẩn hóa. Nhãn -1/+1; score bằng 0 chọn +1. loss(x,y) không đổi trạng thái. update(x,y) trả True nếu hinge loss trước bước học > 0, kể cả khi vector không đổi về số học. WAT phải lấy snapshot trước update bằng get_weights() (bản sao). decision_function(X) và predict(X) trả mảng 1 chiều. Hỗ trợ dense và sparse CSR. FSOL giữ theta riêng với vector dùng để dự đoán.

Chạy từ gốc repo: python -m unittest discover -s tests -p "test_*.py" -v

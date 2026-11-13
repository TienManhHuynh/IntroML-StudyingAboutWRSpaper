# Tests cần bổ sung cùng implementation

TV1: update PAC/FSOL tính tay, no-update, sparse/dense. TV2: reservoir <=K, snapshot không alias, survival boundaries, RNG, averaging/zeroing, terminal candidate. TV3: không leak label, split reproducible, ROP đúng oracle, sparsity đúng model, end-to-end/logging failure.

Trên tv3 đã tích hợp tests TV1 và có tests data/metrics/runner/tracking/comparison/tuning của TV3. Chạy `python -m pytest -q`; tests không cần tài khoản hoặc mạng. Factory giả trong test chỉ kiểm giao diện, không phải WAT. Checker cấu trúc không thay kiểm chứng học thuật.

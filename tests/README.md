# Tests cần bổ sung cùng implementation

TV1: update PAC/FSOL tính tay, no-update, sparse/dense. TV2: reservoir <=K, snapshot không alias, survival boundaries, RNG, averaging/zeroing, terminal candidate. TV3: không leak label, split reproducible, ROP đúng oracle, sparsity đúng model, end-to-end/logging failure.

Đã tích hợp tests TV1, TV2 (`test_wat.py`: reservoir, priority reference, sampling distribution, RNG, snapshot, margin/survival, averaging/VZ, sparse inference và xử lý biên) và tests data/metrics/runner/tracking/comparison/tuning của TV3. Chạy `python -m pytest -q` (hoặc `python -m pytest -q -p no:cacheprovider`); tests không cần tài khoản hoặc mạng. Factory giả trong test TV3 chỉ kiểm giao diện; tests TV2 chưa thay kiểm chứng tích hợp end-to-end với learner thật TV1. Checker cấu trúc không thay kiểm chứng học thuật.

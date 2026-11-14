# Tests cần bổ sung cùng implementation

TV1: update PAC/FSOL tính tay, no-update, sparse/dense. TV2: reservoir <=K, snapshot không alias, survival boundaries, RNG, averaging/zeroing, terminal candidate. TV3: không leak label, split reproducible, ROP đúng oracle, sparsity đúng model, end-to-end/logging failure.

TV2 đã có `test_wat.py`: reservoir, priority reference, sampling distribution, RNG, snapshot, margin/survival, averaging/VZ, sparse inference và xử lý biên. Chạy `python -m pytest -q -p no:cacheprovider`. Tests chưa thay kiểm tra learner thật TV1. Checker chỉ kiểm tra cấu trúc/kế hoạch, không thay kiểm chứng học thuật.

# TV2 - WRS / WAT

Đã hiện thực [reservoir.py](reservoir.py), [wat.py](wat.py) và [unit tests](../../tests/test_wat.py). Chưa tích hợp learner thật TV1 hoặc chạy reproduction. Top-K/moving/EMA và NONPA chưa triển khai.

```python
from src.wat import WAT

# learner của TV1 có loss(x,y), update(x,y), get_weights().
model = WAT(learner, K=64, weighting="standard", averaging="simple",
            voting_zeroing=False, seed=42)
for x, y in train_stream:
    model.update(x, y)
weights = model.get_weights()
labels = model.predict(X_test)
```

Weighting: `standard`/`exponential`; averaging: `simple`/`weighted`.
`WeightedReservoir(capacity, weighting, seed).offer(weights, survival)` trả bool
cho biết candidate có được nhận hay không. `candidates` trả các bản copy;
`aggregate(averaging, voting_zeroing)` trả vector ensemble mới.

Learner sở hữu state riêng; `get_weights()` phải trả vector dense 1D hữu hạn.
Sample sparse/dense được chuyển nguyên trạng vào learner, không densify X.
Snapshot copy trước update, kể cả learner mutate weights tại chỗ.

Quy ước: chưa đầy chia theo m thực tế; rỗng dùng active weights; tổng standard
weight bằng 0 dùng simple average; không flush candidate cuối; score 0 trả +1;
VZ dùng zero chính xác và đa số nghiêm ngặt. Những quy ước xử lý biên này cần
đối chiếu reference. Priority log-space giữ epsilon 1e-8; averaging không cộng
epsilon. Inference không lấy mẫu hoặc thay đổi reservoir.

Xem [biến thể và chi phí](../../docs/TV2/wat-variants-complexity.md) và
[API tích hợp](../../docs/implementation-contract.md).

Chạy `python -m pytest -q -p no:cacheprovider`.

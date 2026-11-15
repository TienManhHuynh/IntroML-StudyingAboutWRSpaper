# Chạy phần TV3 trên nhánh tv3

## Trạng thái

Đã tích hợp nguyên trạng code PAC/FSOL từ nhánh TV1 (`d95f70d`) vào nhánh tv3; không sửa code của thành viên. TV2 chưa có WAT implementation ở thời điểm bàn giao. Loader chặn NaN/Inf trước learner. Full reproduction và W&B public chưa hoàn tất.

## Cài đặt và kiểm tra

```powershell
git switch tv3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/check_repository.py
```

HF Hub tải file với revision SHA cố định; pyarrow đọc parquet; python-dotenv đọc cấu hình local, môi trường đã có ưu tiên hơn .env. Không commit khóa hoặc raw dataset. Thư viện ngoài scientific stack phục vụ tải dữ liệu, tracking và đồ thị theo yêu cầu môn, không thay thuật toán core.

## Chạy baseline thật

```powershell
python experiments/run_reproduction.py --dataset sst2 --max-samples 600 --learner pac --seed 42 --output results/raw/my-pac-42
python experiments/run_reproduction.py --dataset sst2 --max-samples 600 --learner fsol --seed 42 --output results/raw/my-fsol-42
```

`--max-samples` là subset phát triển, không phải tái hiện đầy đủ. Subset seed=0 giữ cùng tập con giữa trials; --seed điều khiển random split và stream order. Không truyền max-samples sẽ tải toàn asset đã khai báo, nhưng vẫn phải xác minh equivalence và tuning trước báo reproduction. Không ghi đè output đã có; lỗi training/tracking lưu run.json=failed và trả exit khác 0.

Source hiện xác minh: `stanfordnlp/sst2`, revision `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`, file train parquet. Asset 67.349 dòng khác paper 67.337; preprocessing train-only CountVectorizer cũng cần đối chiếu reference. Registry còn 15 dataset chưa có source xác minh. Không sửa count paper cho khớp asset đang có.

Loader hỗ trợ pinned HF assets ở dạng SVMLight, numeric NPZ (X,y) hoặc text parquet qua `configs/data_sources.json`. Để thêm nguồn, xác minh repo/revision/filename/format/label mapping/checksum và paper equivalence; không chỉ điền tên dataset. Kaggle chưa được triển khai như backend; cần chuyển nguồn hợp lệ sang HF hoặc bổ sung backend có kiểm thử khi chọn Kaggle.

Output từng run: config.json, environment.json, checkpoints.csv, metrics.json, split-indices.npz và run.json. Không có raw features trong artifacts. `reportable=false` là mặc định cần review dữ liệu/protocol và public run links, không tự đổi true vì process thành công.

## Nhận WAT từ TV2

Factory đề xuất trong API contract:

```python
def build_wat(*, base_learner, K, weighting, averaging, voting_zeroing, seed):
    # TV2 returns its implemented wrapper; not a placeholder.
    ...
```

Wrapper có update(x,y), predict(X), get_weights(). Base riêng cho WRS, không chia sẻ weights với baseline. Ví dụ sau chỉ chạy khi TV2 đã cung cấp module thật:

```powershell
python experiments/run_reproduction.py --dataset sst2 --learner pac --include-wrs --wat-factory src.wat.wat:build_wat --K 16 --weighting standard --averaging simple --output results/raw/paired-pac-42
```

Nếu thiếu factory, runner dừng với lỗi rõ ràng trước tạo output; không tạo số WRS giả. Test identity factory chỉ kiểm hợp đồng API trong tests, không được dùng như WAT experiment.

TV1 dùng tie score=0 -> +1, không thêm bias/normalization. Các quy ước này phải đối chiếu reference trước báo paper equivalence; đặc biệt tie có thể ảnh hưởng accuracy tại checkpoint ban đầu.

## Noise analysis

```powershell
python experiments/run_noise_analysis.py --dataset sst2 --max-samples 600 --learner pac --rates 0 .05 .1 .2 .3 --output results/raw/noise-pac
```

Chỉ flip train labels; cùng split/subset/noise seed cho các models/rates; test sạch. Các probabilities dùng cùng RNG nên masks lồng nhau. Ghi rate yêu cầu và số flip thực tế. Thêm --include-wrs và --wat-factory khi TV2 sẵn sàng. Đây là phân tích thêm, không thay Full scope.

## Tuning và tổng hợp

```powershell
python experiments/run_tuning.py --dataset sst2 --learner pac --output results/raw/tuning-pac
python experiments/summarize_results.py --runs results/raw/my-pac-42 results/raw/my-fsol-42 --output results/my-summary
```

Tuning chạy toàn grid của learner trong full_scope.json, 5 seeds/cấu hình. Chọn highest mean ROP trong top 2.5% mean final accuracy theo B.2; làm tròn ceil, tie theo parameter JSON là quy ước triển khai công khai còn cần đối chiếu reference. Đọc selection.json và truyền tham số được chọn vào runner (--C-err hoặc --eta/--lambda). Không bỏ tham số thử mặc định rồi coi đã tuned.

Summarizer không gộp configurations khác source/algorithm parameters/noise/provenance; từ chối trùng seed/config. Báo n_trials và cờ đủ 5 trials, không gọi một run là mean 5 trials. Tạo summary JSON, run index CSV, PNG/SVG từ measured checkpoints. Wilcoxon dùng `src/evaluation/statistics.py`: đầu vào là mean của từng dataset ghép cặp, không phải các seeds độc lập; ghi rõ alternative/zero handling. Phải đối chiếu convention paper trước báo p-values.

`compare_reference.py` so CSV đã normalize có keys dataset/learner/model/seed/config_id, metrics final_accuracy/rop/final_sparsity và source. Phải thực sự chạy code tác giả ở môi trường riêng và cung cấp output/provenance trước. Helper không tự chạy code tác giả và không xác nhận equivalence từ config_id tùy ý. Unmatched rows giữ rõ, ROP near-zero báo relative delta trống thay vì chia 0.

## W&B cho người mới

W&B là nơi lưu và chia sẻ một lần chạy: tham số, metrics theo bước và files kết quả. **Entity** là tên tài khoản/nhóm; **project** là thư mục dự án. Giáo viên cần mở được run để kiểm chứng số liệu.

1. Đăng ký/đăng nhập trên https://wandb.ai, tạo một project cho nhóm và đặt visibility Public trong project settings.
2. Trên máy chạy đã activate .venv: `wandb login`. Nhập credential trực tiếp tại terminal, không gửi vào chat/commit.
3. Điền tên entity/project vào .env (copy .env.example nếu chưa có).
4. Chạy với `--wandb-mode online`. Runner đọc .env nhưng mode mặc định vẫn disabled; chọn online rõ ràng cho runs cần nộp.
5. Kiểm tra URL trong run.json mở được ở trạng thái không đăng nhập trước khi đưa lên slide.

```powershell
python experiments/run_reproduction.py --dataset sst2 --max-samples 600 --learner pac --wandb-mode online --wandb-project TEN_PROJECT --wandb-entity TEN_TAI_KHOAN --output results/raw/online-check
```

Offline đã được kiểm chứng bằng synthetic run, lưu trong wandb/runs/wandb/offline-run-*. Có thể sync bằng `wandb sync PATH_TO_RUN`, nhưng chỉ sync development run khi muốn chia sẻ kiểm tra pipeline, không đặt nó vào bảng reproduction như kết quả cuối. Cần cập nhật public URL sau sync. Không tự tạo account hoặc khai API key thay người dùng.

## Slide và bằng chứng phát triển

`slides/tv3_experiments_draft.pptx` là phần TV3 9 slides (22-30), có notes, tables/charts native và các trạng thái thiếu dữ liệu rõ ràng. Chart minh họa seed 42 hiển thị 11/41 checkpoints, làm tròn 6 decimals; summary tính từ toàn bộ checkpoints/5 trials. Source builder: scripts/build_tv3_slides.mjs, cần artifact-tool runtime; dựng từ evidence raw hoặc bản copy development/evidence sau checkout mới. Đây chưa phải deck nộp cuối.

Builder từ chối ghi đè final đã có; muốn tạo revision mới, truyền đường dẫn output mới làm argument thứ nhất, ví dụ `node scripts/build_tv3_slides.mjs slides/tv3_experiments_draft_v2.pptx`. Runtime paths có thể đặt qua RUNTIME_NODE_MODULES / PRESENTATIONS_SKILL_DIR / RUNTIME_PYTHON; dùng Node của runtime artifact-tool.

`results/development/sst2-baseline/` lưu 10 baseline development runs (PAC/FSOL x seeds42-46), summary/plots và bản sao evidence. Không có WRS/public W&B; metadata dirty=true vì chạy trong lúc viết code, nên không dùng như proof final code commit. Xem `validation.md` và không diễn giải kết quả này là reproduction paper.

## Phần còn chờ

- WAT/reservoir/tests từ TV2; ablation wrappers và non-PA learners theo phân công thống nhất.
- Hosted sources + equivalence đầy đủ 16 dataset, EMBER/appendix E.
- Project W&B public do nhóm đăng nhập, rồi chạy đầy đủ tuned trials/variants.
- Chạy code tác giả, đối chiếu paper và hoàn thiện result slides có run URLs.

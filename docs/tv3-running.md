# Chạy phần TV3 trên nhánh intergration-v1

## Trạng thái

Đã tích hợp hoàn chỉnh code PAC/FSOL từ nhánh TV1 (`src/models/`), WRS/WAT từ nhánh TV2 (`src/wat/`), cùng pipeline TV3 (`src/data/`, `src/evaluation/`, `experiments/`). Toàn bộ bộ kiểm thử `pytest` pass (`41 passed, 4 subtests passed`). Đợt 1 (**8/16 bộ dữ liệu Table 1** + Label Noise + Deviation Band) đã chạy xong **1.275 runs W&B online công khai** tại project `wrs-reproduction-group8`; 8 bộ rất lớn còn lại giữ trạng thái `planned` do giới hạn RAM trống `8.55 GB`.

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

Các nguồn đã xác minh và ghim SHA256 cho **Đợt 1 (8/16 bộ Table 1)**: `stanfordnlp/sst2` (revision `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`, `text_parquet`, 67.349 dòng HF vs 67.337 Table 1 do train-only `CountVectorizer`) và 7 bộ `sparse_npz` trên `TienManh2307/wrs-paper-datasets` (revision `27597e3350168b3786470fe15b6d93db15f8cd20`: `dexter`, `dorothea`, `pcmac`, `newsgroups-binary-cs`, `news20`, `w8a`, `real-sim`) khớp chính xác `(N, D, sparsity)` Table 1. 8 bộ rất lớn còn lại giữ `hosted_source: null` (`unverified`) do giới hạn RAM trống `8.55 GB`. Không sửa count paper cho khớp asset đang có.

Loader hỗ trợ pinned HF assets ở dạng SVMLight, numeric NPZ (X,y), `sparse_npz` hoặc `text_parquet` qua `configs/data_sources.json`. Để thêm nguồn, xác minh repo/revision/filename/format/label mapping/checksum và paper equivalence; không chỉ điền tên dataset. Kaggle chưa được triển khai như backend; cần chuyển nguồn hợp lệ sang HF hoặc bổ sung backend có kiểm thử khi chọn Kaggle.

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

Chỉ flip train labels; cùng split/subset/noise seed cho các models/rates; test sạch. Các probabilities dùng cùng RNG nên masks lồng nhau. Ghi rate yêu cầu và số flip thực tế. Đây là phân tích bổ sung (200 runs trên 4 bộ `dexter`, `pcmac`, `w8a`, `sst2`), trình bày song song với kết quả chính chứ không thay thế các bộ `planned`.

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

## Slide và bằng chứng thực nghiệm Đợt 1

- Bản slide hoàn chỉnh của TV3 là `slides/tv3_experiments.pptx` gồm 9 slides (Slides 22–30), dựng từ số liệu thật 1.275 runs Đợt 1 (`results/batch1/`) qua `node scripts/build_tv3_slides.mjs`, có 3 native charts, 6 native tables, speaker notes 107–161 từ/slide (`<=500`), trích dẫn IEEE và URL W&B công khai.
- Bản hoàn chỉnh `slides/tv3_experiments.pptx` đã thay thế bản nháp phát triển cũ (`tv3_experiments_draft.pptx`); lệnh `scripts/run_batch1.py` và `scripts/aggregate_batch1.py` cho phép tái sinh toàn bộ bảng số liệu, kiểm định Wilcoxon, `runtime_table5.json` và 5 bộ biểu đồ PNG/SVG tại `figures/`.

`results/development/sst2-baseline/` lưu 10 baseline development runs ban đầu (PAC/FSOL x seeds42-46) trước khi hoàn thiện WRS và Đợt 1. Không dùng bản nháp phát triển này thay cho 1.275 runs chính thức trong `results/batch1/`.

## Phần hoãn lại & việc nhóm còn lại

- 8 bộ rất lớn trong Table 1 (`webspam`, `rcv1`, `mnist8-4-9`, `url`, `kdd2010-algebra`, `avazu-app`, `avazu-site`, `criteo`) và Appendix C/E (`EMBER`, `NONPA`): giữ ở trạng thái `planned` do giới hạn RAM trống `8.55 GB` và thời gian trước hạn 21/11.
- Việc còn lại trước khi nộp Moodle: ghép `slides/TV1.pptx` + `slides/tv2.pptx` + `slides/tv3_experiments.pptx`, bổ sung citation/số thứ tự slide đồng bộ cho TV1/TV2 và duyệt Slide Show toàn deck.

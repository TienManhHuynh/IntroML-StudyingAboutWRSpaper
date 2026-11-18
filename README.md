# Weighted Reservoir Sampling / WAT

Repo đồ án Nhập môn Học máy của **24TNT1 - Nhóm 8**, nghiên cứu bài **Stabilizing Linear Passive-Aggressive Online Learning with Weighted Reservoir Sampling (NeurIPS 2024)**.



## Cấu trúc repo

```text
IntroML-StudyingAboutWRSpaper/
├── README.md                         
├── .gitignore                        
├── .gitattributes                    
├── .env.example                      Cấu hình W&B đề xuất
├── pyproject.toml                    
│
├── papers/
│   ├── wrs-neurips2024.pdf            
│   ├── CITATION.bib                 
│   └── NOTICE.md                 
│
├── docs/
│   ├── project-checklist.md          
│   ├── task-plan.md                  
│   ├── tv3-tasks.md                  
│   ├── full-scope.md                 Ma trận tham chiếu 16 bộ Table 1
│   ├── experiment-protocol.md        Dữ liệu, tuning, seeds, metrics
│   ├── implementation-contract.md    Giao diện tích hợp TV1 / TV2 / TV3
│   ├── tracking.md                   W&B và quy tắc lưu bằng chứng
│   ├── reviews/                      
│   └── requirements/                
│
├── configs/                          
├── src/
│   ├── models/                       TV1: PAC / FSOL
│   ├── wat/                          TV2: reservoir / WAT
│   ├── data/                         TV3: downloader và preprocessing
│   └── evaluation/                   TV3: metrics và thống kê
├── experiments/                      TV3: runners cần triển khai
├── scripts/                          
├── tests/                            
├── results/                          
├── figures/                         
└── slides/                           
```


## Mở nhanh

- [Paper gốc](papers/wrs-neurips2024.pdf) · [BibTeX](papers/CITATION.bib) · [Trích dẫn và nguồn](papers/NOTICE.md)
- [Yêu cầu môn](docs/requirements/project_description.txt) · [PDF phân công và slide contract](docs/requirements/course-project.pdf)
- [Checklist nộp bài](docs/project-checklist.md) · [Outline slide](slides/README.md)
- [Phân công nhóm](docs/task-plan.md) · [Giao diện tích hợp](docs/implementation-contract.md) · [TV3 cần làm gì](docs/tv3-tasks.md)
- [Phương pháp và WAT (TV2)](src/wat/README.md) · [Tài liệu TV2](docs/TV2/tv2-method.md) · [Slide TV2](slides/tv2.pptx)
- [Chạy code TV3 và thiết lập W&B](docs/tv3-running.md) · [Tiến độ TV3](docs/tv3-progress.md) · [Slide TV3 hoàn chỉnh](slides/tv3_experiments.pptx)
- [Kế hoạch 16 bộ & phạm vi](docs/full-scope.md) · [Protocol thực nghiệm](docs/experiment-protocol.md) · [W&B và bằng chứng](docs/tracking.md)
- [Hướng dẫn đóng góp](CONTRIBUTING.md) · [Review TV1 / TV2](docs/reviews/tv1-tv2-2026-11-14.md)

Nguồn bài: [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/38d88c3296eafb09d3971d091a19cf6b-Abstract-Conference.html) · [Repo tác giả](https://github.com/FutureComputing4AI/Weighted-Reservoir-Sampling-Augmented-Training).

## Thành viên và trạng thái

| Thành viên | Vai trò | Trạng thái |
|---|---|---|
| Nguyên / TV1 | Background, theory, PAC / FSOL; slides 01-11 | Code PAC/FSOL đã tích hợp, 4 tests pass; có `slides/TV1.pptx` (cần đổi tên `tv1_background_theory.pptx` & chốt ranh giới Slide 12) |
| Tùng / TV2 | WRS / WAT, training / inference; slides 12-21 | Đã merge `src/wat/` (`reservoir.py`, `wat.py`), `tests/test_wat.py`, `docs/TV2/` (2 PNG) và `slides/tv2.pptx` (cần đánh lại số slide & thêm citation) |
| TienManhHuynh / TV3 | Dữ liệu, metrics, experiments, W&B; slides 22-30; tích hợp | Đã hoàn tất 8/16 bộ Đợt 1 (**1.275 runs online W&B**), `run_ablations.py`, `aggregate_batch1.py`, bảng/plots, đối chiếu code tác giả và `slides/tv3_experiments.pptx` |



## Chuẩn bị môi trường (Python >=3.9)

Windows PowerShell, tại gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python scripts/check_repository.py
python scripts/plan_experiments.py
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
python scripts/check_repository.py
python scripts/plan_experiments.py
```

Hai script trên chỉ kiểm tra tài liệu/cấu hình và sinh **kế hoạch tham chiếu**, không tải data, train hoặc log W&B. Muốn xuất manifest: `python scripts/plan_experiments.py --output results/plans/full_scope.jsonl`; lệnh từ chối ghi đè. Việc chạy thực nghiệm thực tế Đợt 1 (4.440 lượt tuning và 1.275 runs W&B online) được thực hiện qua `experiments/run_tuning.py`, `experiments/run_reproduction.py`, `experiments/run_ablations.py`, `experiments/run_noise_analysis.py`, `scripts/run_batch1.py` và tổng hợp bằng `scripts/aggregate_batch1.py`.

Chạy toàn bộ kiểm thử bằng `python -m pytest -q` (hoặc `python -m pytest -q -p no:cacheprovider`, gồm tests TV1, TV2 và TV3; hiện pass `41 passed, 4 subtests passed`). Tests không cần dữ liệu mạng hoặc tài khoản W&B. Chạy smoke test local: `python experiments/run_reproduction.py --dataset sst2 --max-samples 600 --learner pac --output results/raw/my-pac-42`.

### Ngăn xếp thư viện, lý do sử dụng & tái sử dụng mã nguồn

- **Thuật toán cốt lõi tự hiện thực 100% từ mô tả bài báo**: `PAC` / `PA-II`, `FSOL` (`src/models/`), `WeightedReservoir` / `WAT` (`src/wat/`), các baseline đối chứng `Top-K` / `MovingAverage` / `ExponentialAverage` (`experiments/run_ablations.py`) và thước đo `accuracy` / `sparsity` / `ROP` (`src/evaluation/metrics.py`).
- **Tái sử dụng trong scientific stack chuẩn (`numpy`, `scipy`, `scikit-learn`)**:
  - `numpy` và `scipy.sparse` (`csr_matrix`, `issparse`): lưu trữ ma trận thưa và tính tích vô hướng trong `src/models/_linear.py`, `src/data/loader.py`, `scripts/prepare_datasets.py`.
  - `scipy.stats.wilcoxon`: kiểm định ghép cặp Wilcoxon signed-rank trong `src/evaluation/statistics.py` và `scripts/aggregate_batch1.py`.
  - `sklearn` (`CountVectorizer` fit chỉ trên train cho `sst2`, `train_test_split`, `load_svmlight_file`, `fetch_20newsgroups`, `make_classification`): vector hóa, đọc/tải và chia dữ liệu chuẩn hóa trong `src/data/loader.py`, `scripts/prepare_datasets.py`, `experiments/*.py`.
- **Các gói ngoài scientific stack (`pyproject.toml`) và lý do sử dụng**:
  - `matplotlib`: vẽ biểu đồ đường cong, min/max error bars và đồ thị tổng hợp (`experiments/summarize_results.py`, `scripts/aggregate_batch1.py`).
  - `wandb`: theo dõi từng lượt chạy và đồng bộ 1.275 runs công khai lên Weights & Biases theo yêu cầu tracking của môn (`src/evaluation/tracking.py`).
  - `huggingface_hub` và `pyarrow`: tải tự động có ghim commit SHA/SHA256 các bộ dữ liệu từ Hugging Face và đọc file Parquet cho `stanfordnlp/sst2` (`src/data/loader.py`).
  - `python-dotenv`: nạp biến cấu hình `WANDB_PROJECT`/`WANDB_ENTITY` từ `.env` cục bộ, tránh ghi cứng thông tin tài khoản vào mã nguồn (`experiments/*.py`, `scripts/run_batch1.py`).

Mỗi run đều ghi `environment.json` (phiên bản thư viện thật và commit SHA); danh sách gói phát triển tham chiếu nằm trong [requirements-tv3-dev.lock.txt](requirements-tv3-dev.lock.txt).

## Dataset và input/output

`configs/datasets.json` lưu **16 dataset Table 1**. `configs/data_sources.json` đã pin nguồn Hugging Face + SHA256 cho **8/16 bộ Đợt 1**: `sst2` (`stanfordnlp/sst2`, `text_parquet`) và 7 bộ chuẩn hóa chính xác `(N, D, sparsity)` Table 1 (`dexter`, `dorothea`, `pcmac`, `newsgroups-binary-cs`, `news20`, `w8a`, `real-sim`) tại [`TienManh2307/wrs-paper-datasets`](https://huggingface.co/datasets/TienManh2307/wrs-paper-datasets) (qua `scripts/prepare_datasets.py`, định dạng `sparse_npz`). 8 bộ lớn còn lại (`webspam`, `rcv1`, `mnist8-4-9`, `url`, `kdd2010-algebra`, `avazu-app`, `avazu-site`, `criteo`) và Appendix C/E được giữ nguyên trạng thái `planned` do giới hạn RAM trống `8.55 GB` và thời gian trước hạn nộp. Không commit raw data; Kaggle backend chưa triển khai.

Protocol: random 70/30, một pass, 5 trials (`seeds 42–46`), K=[1,4,16,64], các biến thể weighting/averaging/zeroing và khoảng 200 checkpoint cộng đầu/cuối. Chi tiết tuning/ROP trong [protocol](docs/experiment-protocol.md).

Input: features dense/sparse, nhãn -1/+1 và config/seed. Loader kiểm tra shape/labels/NaN/Inf và fit text vocabulary trên train. Output từng run: config, environment, checkpoint/final metrics, timings, split/noise masks và run status. Toàn bộ **1.275 runs** Đợt 1 đã đồng bộ public tại [W&B Project](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8) (xem được ẩn danh không cần đăng nhập) và lập chỉ mục trong [run-index.csv](results/batch1/run-index.csv).

## Bảng ánh xạ thí nghiệm & W&B (Đợt 1 — 8 datasets, 1.275 runs online)

| Experiment | Script / Notebook | Public W&B Run (mẫu đại diện) | Kết quả ngắn gọn (1 dòng) |
|---|---|---|---|
| Tuning PAC & FSOL (App. B.2) | `experiments/run_tuning.py` | Local grid → đầu vào các run W&B bên dưới (`results/tuning/`) | Chọn được tham số tối ưu 8/8 bộ Đợt 1 (`sst2` PAC `C_err=0.1`, FSOL `eta=0.25, lambda=0.1`) |
| Main PAC / FSOL vs WRS (`8 datasets × 9 variants × 5 seeds`) | `experiments/run_reproduction.py`, `scripts/run_batch1.py --suite main` | [FSOL SST-2 K=64](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/a52skr4u) · [PAC SST-2 K=64](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/fyluqj2o) · [FSOL News20](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/todr3o2s) | FSOL-WRS (K=64) thắng ROP ở 6/8 bộ; PAC-WRS thắng 3/8 do `ceil(2.5%)` chọn baseline ít dao động |
| Ablations (Top-K, Moving Avg, EMA & Runtime) | `experiments/run_ablations.py`, `scripts/run_batch1.py --suite ablations` | [Top-K SST-2](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/o5ayrgkq) · [Moving News20](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/19h8uhu9) · [EMA News20](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/z7y6ri8j) | Trên `news20` (`D=1.35M`), Moving Avg chậm hơn `2.36x–4.26x` và EMA chậm hơn `4.96x–9.00x` so với WRS |
| Kiểm chứng chính sách tuning `top_acc - 0.025` | `scripts/run_batch1.py --suite deviation_band` | [PAC Real-Sim band](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/ps5joe9j) · [PAC SST-2 band](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/39ouhtyd) | Khi đổi sang biên 2.5% điểm như `analyzer.ipynb`, PAC-WRS tăng lên thắng 4/8 ROP và 5/8 Accuracy |
| Label noise bổ sung (`0%, 5%, 10%, 20%, 30%`) | `experiments/run_noise_analysis.py`, `scripts/run_batch1.py --suite noise` | [W8A PAC 30% noise](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/7mbaow9o) · [SST-2 FSOL 30% noise](https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8/runs/by7grnde) | Tại 30% nhiễu trên `w8a`, WRS giữ `96.45%–96.55%` accuracy trong khi baseline tụt còn `60.23%–61.07%` |
| Tổng hợp, Wilcoxon & đối chiếu code tác giả | `scripts/aggregate_batch1.py`, `experiments/compare_reference.py` | Liên kết qua [run-index.csv](results/batch1/run-index.csv) & [reference_comparison.csv](results/batch1/reference_comparison.csv) | Định vị 4 điểm rẽ nhánh với repo tác giả (`top_acc - 0.025`, bias thiếu `+1` ở `||x||^2`, `loss > 1e-4`, `sign(0)=0`) |
| EMBER / Non-PA (Appendix C/E) & 8 bộ lớn | Chưa triển khai runner cho App. C/E | Trạng thái `planned` trong [coverage.csv](results/coverage.csv) | Hoãn do giới hạn RAM |




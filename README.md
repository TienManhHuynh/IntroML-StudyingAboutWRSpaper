# Weighted Reservoir Sampling / WAT

Repo đồ án Nhập môn Học máy của **24TNT1 - Nhóm 8**, nghiên cứu bài **Stabilizing Linear Passive-Aggressive Online Learning with Weighted Reservoir Sampling (NeurIPS 2024)**.

Nhóm làm **đầy đủ paper / Full scope**: phương pháp, lý thuyết, training và inference pipelines, thí nghiệm chính cùng các mở rộng được kiểm kê trong kế hoạch.

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
│   ├── full-scope.md                 Ma trận thí nghiệm đầy đủ
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

- [Paper gốc](papers/wrs-neurips2024.pdf) · [Trích dẫn và nguồn](papers/NOTICE.md)
- [Yêu cầu môn](docs/requirements/project_description.txt) · [PDF phân công và slide contract](docs/requirements/course-project.pdf)
- [Checklist nộp bài](docs/project-checklist.md) · [Outline slide](slides/README.md)
- [Phân công nhóm](docs/task-plan.md) · [TV3 cần làm gì](docs/tv3-tasks.md)
- [Chạy code TV3 và thiết lập W&B](docs/tv3-running.md) · [Slide TV3 bản nháp](slides/tv3_experiments_draft.pptx)
- [Kế hoạch đầy đủ](docs/full-scope.md) · [Protocol thực nghiệm](docs/experiment-protocol.md)
- [Hướng dẫn đóng góp](CONTRIBUTING.md) · [Review TV1 / TV2](docs/reviews/tv1-tv2-2026-11-14.md)

Nguồn bài: [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/38d88c3296eafb09d3971d091a19cf6b-Abstract-Conference.html) · [Repo tác giả để đối chiếu](https://github.com/FutureComputing4AI/Weighted-Reservoir-Sampling-Augmented-Training).

## Thành viên và trạng thái

| Thành viên | Vai trò | Trạng thái đã kiểm tra ngày 14/11/2026 |
|---|---|---|
| Nguyên / TV1 | Background, theory, PAC / FSOL; slides 01-11 | Code PAC/FSOL đã tích hợp vào tv3, 4 tests pass; loader TV3 chặn non-finite input; PPTX vẫn chờ bàn giao |
| Tùng / TV2 | WRS / WAT, training / inference; slides 12-21 | Có tài liệu và 2 PNG; thiếu reservoir/WAT, tests, source hình chỉnh sửa và PPTX |
| TienManhHuynh / TV3 | Dữ liệu, metrics, experiments, W&B; slides 22-30; tích hợp | Đã có pipeline và baseline development; W&B offline đã kiểm chứng, slide nháp có notes; còn Full scope và public runs |

[Nhánh TV1](https://github.com/TienManhHuynh/IntroML-StudyingAboutWRSpaper/tree/TV1-Background-Theory-and-Base-Learners) · [Nhánh TV2](https://github.com/TienManhHuynh/IntroML-StudyingAboutWRSpaper/tree/TV2-Proposed-Method-and-WAT-Implementation). Tests pass trên snapshot TV1 không có nghĩa WAT hay reproduction đã hoàn tất. Dải slide theo contract cuối; bản phân công còn cách chia cũ cần nhóm thống nhất.

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

Hai script hiện có chỉ kiểm tra tài liệu/cấu hình và sinh **kế hoạch**, không tải data, train hoặc log W&B. Muốn xuất manifest: `python scripts/plan_experiments.py --output results/plans/full_scope.jsonl`; lệnh từ chối ghi đè. Tuning và các mục chưa có trong manifest được theo dõi riêng trong coverage.

Trên `tv3`, chạy toàn bộ kiểm thử bằng `python -m pytest -q` (gồm tests TV1 và TV3). Tests không cần dữ liệu mạng hoặc tài khoản W&B. Chạy baseline hosted: `python experiments/run_reproduction.py --dataset sst2 --max-samples 600 --learner pac --output results/raw/my-pac-42`. Đây là development subset, không phải reproduction đầy đủ.

NumPy/SciPy/scikit-learn phục vụ scientific stack; Matplotlib tạo đồ thị; W&B phục vụ yêu cầu tracking của môn. Chưa có environment lock cho reproduction: phải ghi phiên bản thật và commit SHA của từng run.

## Dataset và input/output

`configs/datasets.json` lưu **16 dataset Table 1**. `configs/data_sources.json` đã pin SST-2 HF source/revision/checksum; 15 dataset khác còn thiếu nguồn xác minh. Loader tải/reconstruct từ HF; source SST-2 và train-only preprocessing còn khác paper cần giải thích/xác minh. Không commit raw data. Kaggle backend chưa triển khai.

Protocol: random 70/30, một pass, 5 trials, K=[1,4,16,64], các biến thể weighting/averaging/zeroing và khoảng 200 checkpoint cộng đầu/cuối. Chi tiết tuning/ROP trong [protocol](docs/experiment-protocol.md).

Input: features dense/sparse, nhãn -1/+1 và config/seed. Loader kiểm tra shape/labels/NaN/Inf và fit text vocabulary trên train. Output từng run: config, environment, checkpoint/final metrics, timings, split/noise masks và run status. URL chỉ có khi chạy W&B online thành công. Xem [development evidence](results/development/sst2-baseline/README.md).

## W&B kiểm chứng

`src/evaluation/tracking.py` đã gọi W&B với disabled/offline/online modes. CLI đọc `.env`, nhưng mode mặc định disabled; chọn `--wandb-mode online` rõ ràng khi cần upload. Nhóm cần tạo project **public**, đăng nhập bằng `wandb login`, cung cấp entity/project. Offline logging + artifacts đã chạy thử thành công; không có public run URL để trích dẫn. Xem [hướng dẫn W&B cho người mới](docs/tv3-running.md).

Mỗi run cần dataset revision/checksum, config, seeds, commit, môi trường, metrics/plots. Mỗi kết quả tổng hợp dẫn đủ runs thành phần. Chi tiết: [tracking](docs/tracking.md).

| Experiment | Script / API dự kiến | Input → output | W&B / trạng thái |
|---|---|---|---|
| PAC / FSOL và WRS | `experiments/run_reproduction.py` | Data/config → curves, ROP, final accuracy/sparsity | Baseline SST-2 development đã chạy; WRS chờ TV2 |
| Top-K, moving/EMA, runtime | `experiments/run_ablations.py` | Cùng data/trials → metrics/timings | Chưa có script/run |
| Statistics / paper comparison | `experiments/summarize_results.py`, `src/evaluation/statistics.py` | Runs thật → bảng/plots, paired statistics | Có summary/plots development; chưa có dữ liệu đầy đủ cho paper comparison |
| Code tác giả | `experiments/compare_reference.py` | CSV đã normalize → matched/unmatched deltas | Có comparison helper; chưa chạy reference implementation |
| EMBER / non-PA extensions | Cần xác minh và thiết kế | Cấu hình appendix C/E → kết quả tương ứng | Chưa có runner/run |
| Label noise bổ sung | `experiments/run_noise_analysis.py` | Noisy train, clean test → metrics theo rates | Có script, synthetic smoke 5 rates; chưa có WRS noise results |

## Bàn giao

Code tự viết từ paper; code tác giả chỉ dùng kiểm chứng. Hình tự vẽ và có source. Milestone Reading + lab Reproduction nộp chung một PPTX 20-40 slides, dưới 20 MB, notes mỗi slide <=500 từ, citation IEEE chân slide, không References slide riêng; link repo nộp riêng Moodle.

Mốc nội bộ: TV3 code/thí nghiệm 14/11/2026, slide 15/11, ghép/QA 16-20/11. Ngày 21/11 ghi trong phân công cần đối chiếu deadline Moodle. Bản nguồn còn chữ Core Scope; quyết định trực tiếp của nhóm trưởng là **làm đầy đủ paper**.


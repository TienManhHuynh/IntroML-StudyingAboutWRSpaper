# Weighted Reservoir Sampling / WAT

Repo đồ án Nhập môn Học máy của **24TNT1 - Nhóm 8**, nghiên cứu bài **Stabilizing Linear Passive-Aggressive Online Learning with Weighted Reservoir Sampling (NeurIPS 2024)**.

Nhóm làm **đầy đủ paper / Full scope**: phương pháp, lý thuyết, training và inference pipelines, thí nghiệm chính cùng các mở rộng được kiểm kê trong kế hoạch. Label-noise analysis là phần làm thêm, không thay thí nghiệm paper.

## Cấu trúc repo

```text
IntroML-StudyingAboutWRSpaper/
├── README.md                         Sơ đồ repo, trạng thái và hướng dẫn bắt đầu
├── .gitignore                        Bỏ qua môi trường, khóa, cache và dữ liệu lớn
├── .gitattributes                    Quy ước file, giữ nguyên bản nguồn
├── .env.example                      Cấu hình W&B đề xuất, chưa có runner đọc
├── pyproject.toml                    Dependencies cho giai đoạn triển khai
│
├── papers/
│   ├── wrs-neurips2024.pdf            Paper gốc
│   ├── CITATION.bib                  Thông tin trích dẫn
│   └── NOTICE.md                     Nguồn và ghi công
│
├── docs/
│   ├── project-checklist.md          Yêu cầu môn và checklist nộp
│   ├── task-plan.md                  Phân công và mốc nội bộ
│   ├── tv3-tasks.md                  Danh sách công việc TV3 / nhóm trưởng
│   ├── full-scope.md                 Ma trận thí nghiệm đầy đủ
│   ├── experiment-protocol.md        Dữ liệu, tuning, seeds, metrics
│   ├── implementation-contract.md    Giao diện tích hợp TV1 / TV2 / TV3
│   ├── tracking.md                   W&B và quy tắc lưu bằng chứng
│   ├── reviews/                      Kết quả kiểm tra nhánh thành viên
│   └── requirements/                 Bản nguồn yêu cầu và phân công
│
├── configs/                          Registry 16 dataset và cấu hình Full scope
├── src/
│   ├── models/                       TV1: PAC / FSOL
│   ├── wat/                          TV2: reservoir / WAT
│   ├── data/                         TV3: downloader và preprocessing
│   └── evaluation/                   TV3: metrics và thống kê
├── experiments/                      TV3: runners cần triển khai
├── scripts/                          Kiểm tra khung, lập kế hoạch jobs
├── tests/                            Tests thuật toán cần tích hợp
├── results/                          Coverage và kết quả có nguồn
├── figures/                          Hình nhóm tự vẽ và mã tạo plot
└── slides/                           Outline và PPTX bàn giao
```

**Trên `main` hiện có khung dự án.** PAC/FSOL và tests nằm trên nhánh TV1, chưa merge; TV2 có tài liệu và hai ảnh pipeline trên nhánh riêng, chưa có code WAT. Chưa có kết quả reproduction hoặc W&B run public trong repo. Xem [review ngày 14/11/2026](docs/reviews/tv1-tv2-2026-11-14.md).

## Mở nhanh

- [Paper gốc](papers/wrs-neurips2024.pdf) · [Trích dẫn và nguồn](papers/NOTICE.md)
- [Yêu cầu môn](docs/requirements/project_description.txt) · [PDF phân công và slide contract](docs/requirements/course-project.pdf)
- [Checklist nộp bài](docs/project-checklist.md) · [Outline slide](slides/README.md)
- [Phân công nhóm](docs/task-plan.md) · [TV3 cần làm gì](docs/tv3-tasks.md)
- [Kế hoạch đầy đủ](docs/full-scope.md) · [Protocol thực nghiệm](docs/experiment-protocol.md)
- [Hướng dẫn đóng góp](CONTRIBUTING.md) · [Review TV1 / TV2](docs/reviews/tv1-tv2-2026-11-14.md)

Nguồn bài: [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/38d88c3296eafb09d3971d091a19cf6b-Abstract-Conference.html) · [Repo tác giả để đối chiếu](https://github.com/FutureComputing4AI/Weighted-Reservoir-Sampling-Augmented-Training).

## Thành viên và trạng thái

| Thành viên | Vai trò | Trạng thái đã kiểm tra ngày 14/11/2026 |
|---|---|---|
| Nguyên / TV1 | Background, theory, PAC / FSOL; slides 01-11 | Có code, tài liệu và 4 tests pass trên nhánh TV1; cần bổ sung kiểm tra dữ liệu không hữu hạn, hoàn thiện PPTX và tích hợp |
| Tùng / TV2 | WRS / WAT, training / inference; slides 12-21 | Có tài liệu và 2 PNG; thiếu reservoir/WAT, tests, source hình chỉnh sửa và PPTX |
| TienManhHuynh / TV3 | Dữ liệu, metrics, experiments, W&B; slides 22-30; tích hợp | Có kế hoạch/protocol; loader, metrics, runners và kết quả chưa triển khai |

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

Trên nhánh TV1 đã được review, chạy tests từ gốc nhánh bằng `python -m unittest discover -s tests -p "test_*.py" -v`. Lệnh này trên `main` hiện chưa kiểm thử PAC/FSOL vì code/tests chưa merge. Tham khảo snapshot và môi trường trong review.

NumPy/SciPy/scikit-learn phục vụ scientific stack; Matplotlib tạo đồ thị; W&B phục vụ yêu cầu tracking của môn. Chưa có environment lock cho reproduction: phải ghi phiên bản thật và commit SHA của từng run.

## Dataset và input/output

`configs/datasets.json` lưu thông tin **16 dataset từ Table 1 của paper**, không phải dữ liệu nhóm đã tải. Hosted source/revision còn để null khi chưa xác minh. Runner cần tự tải/reconstruct từ Hugging Face hoặc Kaggle theo yêu cầu môn; không commit raw data.

Protocol: random 70/30, một pass, 5 trials, K=[1,4,16,64], các biến thể weighting/averaging/zeroing và khoảng 200 checkpoint cộng đầu/cuối. Chi tiết tuning/ROP trong [protocol](docs/experiment-protocol.md).

Input dự kiến: features dense/sparse, nhãn -1/+1 và config/seed. Loader phải kiểm tra shape, labels, NaN/Inf, feature schema và tránh leakage. Output dự kiến: config, environment, checkpoint metrics, final metrics, timings, public run URL; **loader/runner/output thực nghiệm hiện chưa có**.

## W&B dùng ở đâu?

TV3 phải tạo một project **public** dùng chung và tích hợp logging vào runners. `.env.example` chỉ là mẫu; chưa có code tự đọc `.env` hoặc gọi W&B. Đăng nhập tại máy chạy, không commit API key. Run offline cần sync và kiểm tra public access trước khi dẫn trong slide.

Mỗi run cần dataset revision/checksum, config, seeds, commit, môi trường, metrics/plots. Mỗi kết quả tổng hợp dẫn đủ runs thành phần. Chi tiết: [tracking](docs/tracking.md).

| Experiment | Script / API dự kiến | Input → output | W&B / trạng thái |
|---|---|---|---|
| PAC / FSOL và WRS | `experiments/run_reproduction.py` | 16 datasets + config → curves, ROP, final accuracy/sparsity | Chưa có script/run |
| Top-K, moving/EMA, runtime | `experiments/run_ablations.py` | Cùng data/trials → metrics/timings | Chưa có script/run |
| Statistics / paper comparison | `experiments/summarize_results.py` | Runs thật → bảng, Wilcoxon, deviations | Chưa có script/run |
| Code tác giả | `experiments/compare_reference.py` | Cùng config/data → divergence report | Chưa có script/run |
| EMBER / non-PA extensions | Cần xác minh và thiết kế | Cấu hình appendix C/E → kết quả tương ứng | Chưa có runner/run |
| Label noise bổ sung | `experiments/run_noise_analysis.py` | Noisy train, clean test → noise curves | Chưa có script/run |

## Bàn giao

Code tự viết từ paper; code tác giả chỉ dùng kiểm chứng. Hình tự vẽ và có source. Milestone Reading + lab Reproduction nộp chung một PPTX 20-40 slides, dưới 20 MB, notes mỗi slide <=500 từ, citation IEEE chân slide, không References slide riêng; link repo nộp riêng Moodle.

Mốc nội bộ: TV3 code/thí nghiệm 14/11/2026, slide 15/11, ghép/QA 16-20/11. Ngày 21/11 ghi trong phân công cần đối chiếu deadline Moodle. Bản nguồn còn chữ Core Scope; quyết định trực tiếp của nhóm trưởng là **làm đầy đủ paper**.

## Nguồn và tái sử dụng

Tài liệu nguồn giữ nguyên và có checksum trong `docs/requirements/source-manifest.json`. Hiện chưa nhập core code tác giả hoặc nhóm khác. README tham khảo cách bố trí của [online-iforest-reproduction](https://github.com/BrownieOro/online-iforest-reproduction); nội dung được viết cho WRS. Bổ sung nguồn mọi dataset/thư viện/phần tái sử dụng khi triển khai.

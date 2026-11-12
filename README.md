# WRS / WAT Reproduction - Intro to Machine Learning

**24TNT1 - Group 8 | Phạm vi đã chốt: FULL SCOPE | Nhánh tích hợp: `main`**

Nhóm nghiên cứu và tự hiện thực bài **Stabilizing Linear Passive-Aggressive Online Learning with Weighted Reservoir Sampling**, Skyler Wu, Fred Lu, Edward Raff, James Holt, NeurIPS 2024.

**Trạng thái:** đã dựng khung tài liệu, phân công, cấu hình và công cụ lập kế hoạch. Thuật toán, downloader, experiment runners, W&B runs và slide kết quả **chưa triển khai/chạy**. CI kiểm tra khung không xác nhận tái hiện thành công.

## Bắt đầu

- [Yêu cầu môn và checklist](docs/project-checklist.md)
- [Phân công và tiến độ](docs/task-plan.md)
- [Full scope và ma trận thí nghiệm](docs/full-scope.md)
- [Protocol dữ liệu, tuning và metrics](docs/experiment-protocol.md)
- [Giao diện tích hợp](docs/implementation-contract.md)
- [W&B và bằng chứng](docs/tracking.md)
- [Cách đóng góp](CONTRIBUTING.md) · [Outline slide](slides/README.md)
- [Paper gốc](papers/wrs-neurips2024.pdf) · [BibTeX](papers/CITATION.bib) · [Nguồn và ghi công](papers/NOTICE.md)

## Nhóm

| Vai trò | Thành viên | Trách nhiệm | Slide theo contract cuối |
|---|---|---|---|
| TV1 | Nguyên | Background, theory, PAC/FSOL | 01-11 |
| TV2 | Tùng | WRS/WAT, training/inference | 12-21 |
| TV3 / nhóm trưởng | TienManhHuynh | Dữ liệu, evaluation, experiments, W&B, tích hợp | 22-30 |

Quyết định nhóm trưởng ngày 12/11/2026: **Full scope thay Core scope** trong phân công cũ. Bản phân công gốc giữ nguyên để đối chiếu. File gốc có hai cách chia slide TV1/TV2; khung dùng contract cuối, cần nhóm xác nhận khi bàn giao.

## Cấu trúc

```text
papers/                 Paper, citation và ghi công
docs/requirements/      Bản nguồn yêu cầu và phân công
docs/                   Checklist, Full scope, protocol, API, tracking
configs/                Registry 16 dataset và lưới thí nghiệm
src/models/             TV1: PAC, FSOL
src/wat/                TV2: reservoir, WAT, ensemble đối chứng
src/data/               TV3: tải, preprocessing, split
src/evaluation/         TV3: metrics và thống kê
experiments/            TV3: runners cần triển khai
scripts/                Kiểm tra khung và sinh kế hoạch jobs
tests/                  Algorithm/integration tests cần bổ sung
results/                Coverage, kết quả có nguồn, deviations
figures/                Hình nhóm tự tạo và mã tạo plot
slides/                 Outline và PPTX bàn giao
```

## Kiểm tra ngay

Python >=3.9; các lệnh sau **không tải dữ liệu, không train và không gửi W&B**:

```powershell
python scripts/check_repository.py
python scripts/plan_experiments.py
python scripts/plan_experiments.py --output results/plans/full_scope.jsonl
```

Manifest là kế hoạch, không phải measured results; công cụ từ chối ghi đè. Tuning, reference comparison, runtime riêng, appendix C/E và noise được quản lý riêng trong [coverage](results/coverage.csv).

## Chuẩn bị môi trường triển khai

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

NumPy/SciPy/scikit-learn phục vụ scientific stack; Matplotlib tạo hình; W&B tracking theo yêu cầu môn. Dependencies trong `pyproject.toml` chưa phải lockfile thực nghiệm: TV3 chốt phiên bản và lưu môi trường/commit khi runner hoạt động. `.env` chỉ là cấu hình đề xuất, chưa có code tự đọc file.

## Experiment mapping

Full scope bao phủ toàn bộ thí nghiệm chính Section 4 trên 16 dataset cùng/equivalent với paper. Appendix liên quan được theo dõi riêng để không bỏ sót. Label noise bổ sung, không thay main experiments.

| Experiment | Script dự kiến | W&B | Trạng thái / kết luận |
|---|---|---|---|
| PAC/FSOL và WRS | `experiments/run_reproduction.py` | Chưa có | Planned; chưa có số liệu |
| Top-K, moving/EMA, runtime | `experiments/run_ablations.py` | Chưa có | Planned; chưa có số liệu |
| Statistics, paper comparison | `experiments/summarize_results.py` | Chưa có | Planned; chưa có số liệu |
| Code tác giả | `experiments/compare_reference.py` | Chưa có | Planned; chưa có số liệu |
| EMBER / non-PA extensions | Cần thiết kế runners | Chưa có | Planned; cần xác minh cấu hình |
| Label noise | `experiments/run_noise_analysis.py` | Chưa có | Planned; chưa có số liệu |

Script mang nhãn **dự kiến** chưa tồn tại. Khi chạy thật, bổ sung URL run public và run index cho từng kết quả. Smoke test hoặc offline run chưa sync không đủ bằng chứng nộp bài.

## Mốc bàn giao

13/11/2026: tích hợp API/phương pháp. 14/11: mốc TV3 code và experiments. 15/11: bàn giao slide. 16-20/11: ghép và QA. 21/11 là mốc ghi trong phân công; deadline chính thức phải đối chiếu Moodle.

Nộp riêng trên Moodle một PPTX 20-40 slides, dưới 20 MB, notes mỗi slide <=500 từ và link repo. Upload GitHub không thay nộp Moodle.

## Nguồn và tái sử dụng

- [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/38d88c3296eafb09d3971d091a19cf6b-Abstract-Conference.html).
- [Code tác giả](https://github.com/FutureComputing4AI/Weighted-Reservoir-Sampling-Augmented-Training): dùng kiểm chứng/chạy đối chiếu; chưa nhập code vào repo nhóm.
- [Repo tham khảo cách tổ chức](https://github.com/BrownieOro/online-iforest-reproduction): tham khảo bố cục, chưa sao chép code/nội dung nhóm khác.
- [Yêu cầu giảng viên nguyên bản](docs/requirements/project_description.txt), [phân công nguyên bản](docs/requirements/task_distribution.txt).

Core algorithm tự viết từ paper, hình tự vẽ và có nguồn. Bổ sung mọi thư viện/dataset/phần tái sử dụng vào README khi triển khai. PDF thuộc tác giả, lưu để đọc tham khảo; không nhận ảnh/chữ sao chép là sản phẩm tự làm.

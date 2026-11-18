# Yêu cầu và checklist

## Nguồn và quyết định

Yêu cầu môn lấy từ [project_description.txt](requirements/project_description.txt). Bản nguồn giữ nguyên. Khung kế hoạch tham chiếu đặt 16 bộ Table 1; sau khi đo phần cứng thực tế (`8.55 GB` RAM trống), nhóm chốt thực thi theo chuẩn **Core scope mở rộng (Đợt 1 / Batch 1)**: hoàn tất **8/16 bộ Table 1** + **Label Noise** (4 bộ) + **Deviation Band** (tổng **1.275 runs W&B online**), đồng thời giữ 8 bộ rất lớn ở trạng thái `planned` và giải trình minh bạch trong deck/README.

## Reading + Reproduction: yêu cầu môn

- [x] Paper được đăng ký/chấp nhận trên Moodle.
- [x] Một public repo toàn học kỳ, có code/docs/notebooks nếu dùng và commit history.
- [x] Phạm vi thực thi (**Core scope mở rộng: 8/16 bộ Table 1 + Label Noise + Deviation Band**) và lý do hoãn 8 bộ lớn (RAM trống `8.55 GB`) được nêu minh bạch trong deck (`slides/tv3_experiments.pptx`, Slides 22 & 30) và README.
- [x] Phương pháp tự hiện thực (`PAC`, `FSOL`, `WAT`, `Top-K`, `Moving Avg`, `EMA`); phần tái sử dụng có nguồn.
- [x] Đã chạy đối chiếu code tác giả công khai và khoanh vùng 4 điểm pipeline divergence (`top_acc - 0.025`, bias thiếu `+1`, `loss > 1e-4`, `sign(0)=0`).
- [x] Python >=3.9, dependencies được ghi (`pyproject.toml`, `requirements-tv3-dev.lock.txt`), phần ngoài scientific stack có lý do.
- [x] Seed cố định (`42–46`), docstrings quan trọng, hướng dẫn setup/chạy end-to-end.
- [x] Tải và reconstruct dữ liệu tự động từ HF (`stanfordnlp/sst2` và `TienManh2307/wrs-paper-datasets` cho 8/16 bộ Đợt 1 kèm commit SHA & SHA256).
- [x] Một W&B project public (`wrs-reproduction-group8`), **1.275 runs** Đợt 1 đồng bộ online công khai.
- [x] README có bảng experiment -> script -> public W&B run -> nhận xét ngắn gọn.
- [x] Paper/reference vs reproduction (`results/batch1/reference_comparison.csv`), relative deviation và giải thích nguyên nhân 4 điểm divergence.
- [ ] Một PPTX 20-40 slides kể cả appendix, dưới 20 MB (TV3 đã xong `slides/tv3_experiments.pptx` 9 slides; chờ ghép cùng TV1 & TV2).
- [x] Phần TV3 (`slides/tv3_experiments.pptx`): mỗi slide có notes <=500 từ (`107–161` từ), giải thích lập luận (cần kiểm đồng bộ TV1 & TV2 khi ghép deck).
- [x] Phần TV3: slide trực quan (3 native charts + 6 native tables); không đoạn văn dài/raw source code.
- [x] Phần TV3: hình tự vẽ/tái dựng (`figures/batch1_*.svg/png`, native PPTX charts), citation IEEE ở chân slide, không References slide riêng (cần TV2 bổ sung citation khi ghép).
- [x] TV1 & TV2 đã chuẩn bị motivation, predecessors, novelty, notation, pipeline, components, theory/complexity (`slides/TV1.pptx`, `slides/tv2.pptx`).
- [ ] PPTX gộp cuối (20–40 slides) và link repo nộp riêng trên Moodle trước 21/11/2026.

## Contract nội bộ và phản hồi ghi trong phân công

- [x] Training và inference pipeline rõ input/output (`docs/implementation-contract.md`, `src/wat/`).
- [ ] Theme 16:9 thống nhất; không overlap/clipping khi mở Slide Show bản gộp cuối (TV3 đã QA xong; chờ ghép TV1/TV2).
- [ ] Xác nhận đồng bộ dải slide liên tục khi ghép `TV1.pptx` + `tv2.pptx` + `tv3_experiments.pptx` (chốt ranh giới Slide 12 Survival Time giữa TV1/TV2).
- [x] Hoàn tất code/thực nghiệm/slide TV3 Đợt 1 (18/11/2026); đang ở bước ghép/QA nhóm (16–20/11).
- [x] Label noise bổ sung (200 runs trên 4 bộ) đi kèm bảng chính, không dùng để che giấu 8 bộ `planned`.

## Checkpoint poster về sau

- [ ] Phê bình lý thuyết có căn cứ, thí nghiệm kiểm chứng điểm yếu (đã có manh mối từ `deviation_band` và giới hạn bộ nhớ reservoir `O(KD)`).
- [ ] Poster một trang portrait A2, số liệu có W&B runs.
- [ ] Mọi thành viên giải thích được artifact ở poster workshop.

Trạng thái cập nhật 18/11/2026: phần code, dữ liệu Đợt 1 (8/16 bộ Table 1, 1.275 runs W&B online) và 9 slide của TV3 đã hoàn tất; còn bước ghép deck chung TV1+TV2+TV3, QA Slide Show và nộp Moodle.

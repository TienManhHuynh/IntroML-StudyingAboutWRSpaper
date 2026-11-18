// Editable 9-slide TV3 final experimental slides (Slides 22-30) built from measured Batch 1 evidence.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = process.env.RUNTIME_NODE_MODULES || path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
process.env.RUNTIME_NODE_MODULES = runtime;
const skill = process.env.PRESENTATIONS_SKILL_DIR || path.join(os.homedir(), '.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations');
const {Presentation, PresentationFile} = await import(pathToFileURL(path.join(runtime, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);

const build = path.join(root, 'tmp/tv3-final-slides');
await fs.rm(build, {recursive: true, force: true});
await fs.mkdir(build, {recursive: true});

const summary = JSON.parse(await fs.readFile(path.join(root, 'results/batch1/summary.json'), 'utf8'));
const wilcoxon = JSON.parse(await fs.readFile(path.join(root, 'results/batch1/wilcoxon_stats.json'), 'utf8'));
const runtimeRows = JSON.parse(await fs.readFile(path.join(root, 'results/batch1/runtime_table5.json'), 'utf8'));

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(',');
  return lines.map(line => {
    const vals = line.split(',');
    return Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? '']));
  });
}

const runIndex = parseCsv(await fs.readFile(path.join(root, 'results/batch1/run-index.csv'), 'utf8'));
const refRows = parseCsv(await fs.readFile(path.join(root, 'results/batch1/reference_comparison.csv'), 'utf8'));

const urlOf = (suite, dataset, learner, variant, seed = '42') => {
  const hit = runIndex.find(r => r.suite === suite && r.dataset === dataset && r.learner === learner && r.variant === variant && r.seed === String(seed));
  return hit ? hit.wandb_url : '';
};
const findSummary = (suite, dataset, learner, variant) =>
  summary.find(s => s.suite === suite && s.dataset === dataset && s.learner === learner && s.variant === variant);

const BATCH1 = ['dexter', 'dorothea', 'pcmac', 'w8a', 'newsgroups-binary-cs', 'sst2', 'real-sim', 'news20'];
const SHORT_NAME = {
  'dexter': 'Dexter',
  'dorothea': 'Dorothea',
  'pcmac': 'PCMAC',
  'w8a': 'W8A',
  'newsgroups-binary-cs': 'NG-CS',
  'sst2': 'SST-2',
  'real-sim': 'Real-Sim',
  'news20': 'News20',
};

const p = Presentation.create({slideSize: {width: 1280, height: 720}});
const font = 'Arial';
const PAPER_CITE = '[1] S. Wu, F. Lu, E. Raff, and J. Holt, "Stabilizing Linear Passive-Aggressive Online Learning with Weighted Reservoir Sampling," NeurIPS, 2024.';
const PROJECT_URL = 'https://forge.coreweave.com/wandb/tienmanhhuynh2307-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/wrs-reproduction-group8';

function addBox(slide, value, x, y, w, h, size = 20, color = '#14243B', bold = false) {
  const s = slide.shapes.add({geometry: 'textbox', position: {left: x, top: y, width: w, height: h}, fill: 'none', line: {fill: 'none', width: 0}});
  s.text = value;
  s.text.style = {fontFamily: font, fontSize: size, color, bold};
  return s;
}

function makeSlide(number, title, subtitle, notes, footer) {
  const slide = p.slides.add();
  slide.background.fill = '#FFFFFF';
  addBox(slide, `${number}  |  ${title}`, 56, 30, 1168, 56, 28, '#14243B', true);
  addBox(slide, subtitle, 56, 88, 1168, 30, 16, '#4A607A', false);
  addBox(slide, footer, 56, 622, 1168, 66, 12, '#52647A', false);
  slide.speakerNotes.text = `${notes}\n\nNguồn: ${footer}`;
  return slide;
}

function addTable(slide, values, y = 128, height = 396, fontSize = 18) {
  const tab = slide.tables.add({rows: values.length, columns: values[0].length, left: 56, top: y, width: 1168, height, values});
  for (let r = 0; r < values.length; r++) {
    for (let c = 0; c < values[0].length; c++) {
      const cell = tab.getCell(r, c);
      cell.fill = r === 0 ? '#14243B' : (r % 2 ? '#F3F6FA' : '#FFFFFF');
      cell.text.style = {fontFamily: font, fontSize, color: r === 0 ? '#FFFFFF' : '#14243B', bold: r === 0};
    }
  }
  return tab;
}

function addTakeaway(slide, value) {
  addBox(slide, `Kết luận: ${value}`, 56, 540, 1168, 68, 20, '#163D68', true);
}

// Slide 22: Datasets & Scope
const s22 = makeSlide(
  22,
  '8/16 bộ Table 1 đã chuẩn hóa và xác minh trên Hugging Face',
  'Đợt 1 (Batch 1) khớp chính xác kích thước và độ thưa; 8 bộ rất lớn ghi nhận giới hạn RAM',
  'What: Nhóm kiểm kê đủ 16 dataset Table 1 của paper và chốt chạy trọn vẹn Đợt 1 gồm 8 bộ nhỏ và vừa trên phần cứng 23.8 GB RAM (8.55 GB trống). Why: Các bộ như Criteo (51.8M mẫu) hay KDD2010 (20.2M chiều, riêng K=64 tốn ~10.35 GB RAM) vượt quá giới hạn bộ nhớ và thời gian trước hạn 21/11. How: 7 bộ chuẩn hóa sparse_npz từ nguồn gốc (UCI, scikit-feature, sklearn, LIBSVM) tải qua TienManh2307/wrs-paper-datasets khớp 100% (N, D, sparsity); SST-2 dùng stanfordnlp/sst2 (67,349 dòng vs 67,337 trong bảng paper) với CountVectorizer fit trên tập train. Giới hạn: Kết quả đại diện cho 8/16 dataset; 8 bộ lớn và Appendix C/E giữ trạng thái deferred.',
  `${PAPER_CITE} [2] HF Datasets: TienManh2307/wrs-paper-datasets (rev 27597e3) & stanfordnlp/sst2. W&B: ${PROJECT_URL}`
);
addTable(s22, [
  ['Dataset (Đợt 1)', 'Nguồn gốc', 'N (Nhóm / Paper)', 'D (Chiều)', 'Sparsity', 'Trạng thái'],
  ['Dexter & Dorothea', 'UCI ML Repo', '600 & 1,150 (khớp)', '20K & 100K', '0.9953 & 0.9909', 'HF sparse_npz'],
  ['PCMAC & NG-CS', 'scikit-feature / sklearn', '1,943 & 18,311 (khớp)', '3,289 & 101.6K', '0.9854 & 0.9990', 'HF sparse_npz'],
  ['News20, W8A, Real-Sim', 'LIBSVM Repository', '19.9K, 59.2K, 72.2K', '1.35M, 300, 20.9K', '0.9997, 0.9576, 0.9975', 'HF sparse_npz'],
  ['SST-2', 'stanfordnlp/sst2', '67,349 / 67,337', 'Train-only vocab', 'Fit trên train', 'HF parquet'],
  ['8 bộ lớn & App. C/E', 'Webspam..Criteo, EMBER', '350K – 51.88M', '254 – 20.2M', 'Khung Full Scope', 'Hoãn (RAM 8.55GB)'],
], 128, 392, 18);
addTakeaway(s22, '8 bộ Đợt 1 tái lập chuẩn cấu trúc Table 1 và tải tự động qua Hugging Face.');

// Slide 23: Experimental Protocol & Tuning
const s23 = makeSlide(
  23,
  'Protocol 1-pass giữ cùng split/stream và tinh chỉnh tham số B.2',
  'Seeds cố định 42–46, chia ngẫu nhiên 70/30 không stratify, 201 checkpoints trên mỗi lượt học',
  'What: Mọi mô hình ghép cặp (Baseline vs WRS/Ablation) dùng chung chỉ số chia tập train/test và thứ tự luồng dữ liệu theo từng seed 42–46. Why: Đảm bảo so sánh công bằng và tách được sai số ngẫu nhiên giữa các thuật toán trên một lượt duyệt dữ liệu duy nhất. How: Trọng số khởi tạo 0; chạy tuning độc lập trên lưới Appendix B.2 (7 giá trị C_err cho PAC = 280 runs; 13 eta x 8 lambda cho FSOL = 4,160 runs) trước khi chạy 9 cấu hình WRS (K={1,4,16,64}, standard/exponential, simple/weighted, zeroing). Giới hạn: Nhóm phát hiện paper viết "top 2.5%" nhưng code tác giả lọc biên max_acc - 0.025; nhóm đo kiểm chứng cả hai.',
  `${PAPER_CITE} Sec. 4 & App. B. W&B PAC: ${urlOf('main', 'sst2', 'pac', 'k64_standard_simple_vz0')} | FSOL: ${urlOf('main', 'sst2', 'fsol', 'k64_standard_simple_vz0')}`
);
addTable(s23, [
  ['Thành phần Protocol', 'Thiết lập của nhóm', 'Đối chiếu Paper App. B'],
  ['Chia tập & số lượt', 'Random 70/30 (không stratify), 1 pass, w_0 = 0', 'Khớp Sec. 4'],
  ['Seeds & số trials', '5 seeds cố định: 42, 43, 44, 45, 46', '5 trials (tác giả dùng 0–4)'],
  ['Lưới tuning B.2', 'PAC: 7 C_err (280 runs) | FSOL: 104 cặp (4,160 runs)', 'Chọn trong top 2.5% Acc có ROP cao nhất'],
  ['Cấu hình WRS Đợt 1', 'K=64 (6 biến thể WS/AS/VZ) + K={1,4,16} (Std/Simple)', '720 runs online W&B cho 8 bộ'],
  ['Điểm đánh giá & W&B', '~200 bước cách đều + đầu/cuối; log online mọi run', 'Khớp App. B.3'],
], 128, 392, 18);
addTakeaway(s23, 'Tổng 1,275 runs thực nghiệm (sau tuning) đều được đồng bộ công khai trên W&B.');

// Slide 24: Evaluation Metrics
const s24 = makeSlide(
  24,
  'Ba thước đo chính: Accuracy, ROP và Model Sparsity',
  'ROP đo mức độ ổn định tích lũy so với oracle baseline; giá trị ROP âm chứng tỏ vượt đỉnh baseline',
  'What: Hệ thống đánh giá đồng thời độ chính xác cuối (Final Accuracy), độ ổn định quá trình học (ROP) và độ thưa vector suy luận (Model Sparsity). Why: Một mô hình online có accuracy cuối cao nhưng tụt sâu ở giữa luồng vẫn không an toàn cho hệ thống anytime prediction. How: Tại mỗi checkpoint j, oracle_j là max accuracy của baseline từ bước 0 đến j; ROP = mean(oracle_j - accuracy_method_j) nên ROP càng nhỏ (hoặc âm) càng tốt. Sparsity đo tỷ lệ tọa độ bằng 0 của w_WRS, tách biệt với độ thưa dữ liệu đầu vào. Kiểm định thống kê dùng Wilcoxon signed-rank ghép cặp theo từng dataset (n=8/16). Giới hạn: n=8 chưa đạt cỡ mẫu 16 như toàn bộ paper.',
  `${PAPER_CITE} Sec. 4 & App. F. W&B Stats ref: ${urlOf('main', 'dexter', 'fsol', 'k64_standard_simple_vz0')}`
);
addTable(s24, [
  ['Thước đo', 'Định nghĩa toán học', 'Ý nghĩa & Cách đọc'],
  ['Final Accuracy', 'Tỷ lệ dự đoán đúng trên X_test sau 1 pass', 'Cao hơn tốt hơn (đo chất lượng cuối)'],
  ['Oracle & ROP', 'oracle_j = max_{i<=j} acc_base(i); ROP = mean(oracle_j - acc_j)', 'Thấp hơn tốt hơn; ROP < 0 nghĩa là vượt oracle'],
  ['Model Sparsity', 'Tỷ lệ phần tử |w[d]| == 0 trên tổng D chiều', 'Kiểm tra khả năng giữ độ thưa của FSOL-WRS'],
  ['Wilcoxon Signed-Rank', 'Kiểm định ghép cặp mean 5 trials trên 8 bộ (two-sided)', 'Đánh giá ý nghĩa thống kê (partial n=8/16)'],
], 128, 360, 18);
addTakeaway(s24, 'Unit tests xác nhận tính tay ROP âm, oracle không giảm và ranh giới sparse.');

// Slide 25: PAC vs PAC-WRS Results
const pacDs = BATCH1.map(d => SHORT_NAME[d]);
const pacBaseRop = BATCH1.map(d => Number(findSummary('main', d, 'pac', 'k64_standard_simple_vz0').baseline.rop.mean.toFixed(5)));
const pacWrsRop = BATCH1.map(d => Number(findSummary('main', d, 'pac', 'k64_standard_simple_vz0').method.rop.mean.toFixed(5)));
const pacBandRop = BATCH1.map(d => {
  const hit = findSummary('deviation_band', d, 'pac', 'k64_standard_simple_vz0') || findSummary('main', d, 'pac', 'k64_standard_simple_vz0');
  return Number(hit.method.rop.mean.toFixed(5));
});
const pacMainStat = wilcoxon.main_pac_k64_standard_simple_vz0_rop;
const pacBandStat = wilcoxon.deviation_band_pac_k64_standard_simple_vz0_rop;

const s25 = makeSlide(
  25,
  'PAC-WRS phụ thuộc mạnh vào mức dao động của baseline khi tuning',
  `Chính sách ceil(2.5%) thắng ${pacMainStat.candidate_wins_lower_is_better}/8 bộ ROP; biên max_acc-0.025 tăng lên ${pacBandStat.candidate_wins_lower_is_better}/8 ROP và 5/8 Accuracy`,
  `Kết quả: Với lưới tuning chọn C_err theo ceil(2.5%*7)=1, PAC baseline vốn rất mượt (ROP chỉ 0.00037 trên Real-Sim, 0.00294 trên Dorothea) nên PAC-WRS K=64 chỉ giảm ROP ở ${pacMainStat.candidate_wins_lower_is_better}/8 bộ (p=${pacMainStat.pvalue.toFixed(3)}). Tuy nhiên khi chuyển sang biên tuning max_acc - 0.025 như analyzer.ipynb của tác giả, PAC-WRS thắng ${pacBandStat.candidate_wins_lower_is_better}/8 ROP (SST-2 đạt -0.01713, NG-CS -0.00475) và 5/8 Final Accuracy. Lý do: WAT phát huy mạnh nhất khi base learner dao động lớn; khi C_err nhỏ làm đường học phẳng, trung bình reservoir gây trễ hội tụ nhẹ. Giới hạn: Kiểm chứng band mới chạy trên K=64 standard/simple.`,
  `[1] Sec. 4.1. W&B SST-2 main: ${urlOf('main', 'sst2', 'pac', 'k64_standard_simple_vz0')} | Real-Sim band: ${urlOf('deviation_band', 'real-sim', 'pac', 'k64_standard_simple_vz0')}`
);
const chart25 = s25.charts.add('bar', {
  position: {left: 56, top: 128, width: 1168, height: 396},
  title: 'Mean ROP (5 trials) của PAC Baseline vs PAC-WRS (K=64, Standard/Simple) — thấp hơn tốt hơn',
  titleTextStyle: {fontSize: 18, typeface: font},
  categories: pacDs,
  series: [
    {name: 'PAC Base (ceil)', values: pacBaseRop, fill: '#7A869A'},
    {name: 'PAC-WRS (ceil)', values: pacWrsRop, fill: '#1B6CA8'},
    {name: 'PAC-WRS (band 2.5%)', values: pacBandRop, fill: '#0D9488'},
  ],
  hasLegend: true,
  xAxis: {title: {text: 'Dataset (Đợt 1)', textStyle: {fontSize: 16, typeface: font}}, textStyle: {fontSize: 16, typeface: font}, tickLabelPosition: 'low'},
  yAxis: {title: {text: 'ROP (fraction)', textStyle: {fontSize: 16, typeface: font}}, textStyle: {fontSize: 16, typeface: font}},
  legend: {textStyle: {fontSize: 16, typeface: font}},
});
applyPresentationChartFont(chart25, {fontFamily: font});
addTakeaway(s25, 'PAC-WRS cải thiện mạnh trên SST-2, NG-CS và W8A khi baseline xuất hiện dao động.');

// Slide 26: FSOL vs FSOL-WRS Results
const fsolBaseRop = BATCH1.map(d => Number(findSummary('main', d, 'fsol', 'k64_standard_simple_vz0').baseline.rop.mean.toFixed(5)));
const fsolWrsRop = BATCH1.map(d => Number(findSummary('main', d, 'fsol', 'k64_standard_simple_vz0').method.rop.mean.toFixed(5)));
const fsolVzSparsity = BATCH1.map(d => Number(findSummary('main', d, 'fsol', 'k64_standard_simple_vz1').method.final_sparsity.mean.toFixed(4)));
const fsolStat = wilcoxon.main_fsol_k64_standard_simple_vz0_rop;

const s26 = makeSlide(
  26,
  `FSOL-WRS (K=64) giảm ROP ở ${fsolStat.candidate_wins_lower_is_better}/8 bộ và bảo toàn độ thưa với VZ`,
  'Đạt ROP thấp hơn baseline trên Dexter, Dorothea, PCMAC, W8A, NG-CS và SST-2 (âm -0.00579)',
  `Kết quả: Với FSOL (104 cặp eta/lambda), FSOL-WRS (K=64, standard, simple) giảm ROP ở ${fsolStat.candidate_wins_lower_is_better}/8 dataset (p=${fsolStat.pvalue.toFixed(3)}, exp-simple p=${wilcoxon.main_fsol_k64_exponential_simple_vz0_rop.pvalue.toFixed(3)}). Trên Dexter, ROP giảm từ 0.04776 xuống 0.02267 và Accuracy tăng từ 87.67% lên 90.44%; trên SST-2 đạt ROP âm -0.00579 (vượt oracle). Độ thưa trên Dorothea giữ mức 0.9988 (baseline 0.9993) và NG-CS 0.6257 (0.6314). Lý do: FSOL cập nhật tích lũy theta và co ngưỡng mềm thường dao động hơn PAC, giúp reservoir lọc các nghiệm sống lâu. Giới hạn: Trên Real-Sim và News20 (ROP baseline vốn <= 0.0032), WRS tăng nhẹ độ trễ hội tụ.`,
  `[1] Sec. 4.1. W&B Dexter: ${urlOf('main', 'dexter', 'fsol', 'k64_standard_simple_vz0')} | SST-2: ${urlOf('main', 'sst2', 'fsol', 'k64_standard_simple_vz0')}`
);
const chart26 = s26.charts.add('bar', {
  position: {left: 56, top: 128, width: 1168, height: 396},
  title: 'Mean ROP (5 trials) của FSOL Baseline vs FSOL-WRS (K=64, Standard/Simple) — thấp hơn tốt hơn',
  titleTextStyle: {fontSize: 18, typeface: font},
  categories: pacDs,
  series: [
    {name: 'FSOL Baseline ROP', values: fsolBaseRop, fill: '#7A869A'},
    {name: 'FSOL-WRS (K=64) ROP', values: fsolWrsRop, fill: '#D97706'},
  ],
  hasLegend: true,
  xAxis: {title: {text: 'Dataset (Đợt 1)', textStyle: {fontSize: 16, typeface: font}}, textStyle: {fontSize: 16, typeface: font}, tickLabelPosition: 'low'},
  yAxis: {title: {text: 'ROP (fraction)', textStyle: {fontSize: 16, typeface: font}}, textStyle: {fontSize: 16, typeface: font}},
  legend: {textStyle: {fontSize: 16, typeface: font}},
});
applyPresentationChartFont(chart26, {fontFamily: font});
addTakeaway(s26, `FSOL-WRS thắng ROP ${fsolStat.candidate_wins_lower_is_better}/8 bộ; Voting-based Zeroing giữ sparsity sát baseline.`);

// Slide 27: Ablations & Runtime (Table 5)
const rtNews20 = runtimeRows.find(r => r.dataset === 'news20');
const rtNg = runtimeRows.find(r => r.dataset === 'newsgroups-binary-cs');
const rtRealsim = runtimeRows.find(r => r.dataset === 'real-sim');
const rtW8a = runtimeRows.find(r => r.dataset === 'w8a');

const s27 = makeSlide(
  27,
  'WRS nhanh hơn Moving Average và EMA tới 2.4x–9.0x ở dữ liệu chiều cao',
  'Khớp phát hiện Table 5 của paper: chi phí chép vector mỗi bước làm Moving/EMA chậm khi D >= 20K',
  `Kết quả: Toàn bộ 320 runs Ablations (Top-K, Moving Average K=64, EMA gamma=0.9) cho thấy ở dữ liệu chiều thấp (W8A D=300, SST-2 D~13K), Moving/EMA nhanh xấp xỉ WRS (0.73x–1.01x). Nhưng khi D tăng lên 20.9K (Real-Sim), 101.6K (NG-CS) và 1.35M (News20), Moving Avg chậm hơn 2.36x–4.26x và EMA chậm hơn 4.96x–9.00x trên tập News20. Lý do: WAT chỉ sao chép và đẩy vào reservoir khi xảy ra lỗi (loss > 0), trong khi Moving Avg và EMA bắt buộc cập nhật tổ hợp dense O(D) ở mọi bước t. Giới hạn: Thời gian đo trên CPU i7-9750H (Windows), tách riêng thời gian đánh giá checkpoint.`,
  `[1] Sec. 4.2 & App. D, Table 5. W&B Top-K: ${urlOf('ablations', 'sst2', 'fsol', 'topk_k64_simple')} | News20 EMA: ${urlOf('ablations', 'news20', 'fsol', 'ema_g0.9')}`
);
addTable(s27, [
  ['Dataset (Chiều D)', 'PAC Moving / WRS', 'PAC EMA / WRS', 'FSOL Moving / WRS', 'FSOL EMA / WRS', 'Xu hướng'],
  [`W8A (D=300)`, `${rtW8a.pac_moving_ratio.toFixed(2)}x`, `${rtW8a.pac_ema_ratio.toFixed(2)}x`, `${rtW8a.fsol_moving_ratio.toFixed(2)}x`, `${rtW8a.fsol_ema_ratio.toFixed(2)}x`, 'Tương đương'],
  [`Real-Sim (D=20,958)`, `${rtRealsim.pac_moving_ratio.toFixed(2)}x`, `${rtRealsim.pac_ema_ratio.toFixed(2)}x`, `${rtRealsim.fsol_moving_ratio.toFixed(2)}x`, `${rtRealsim.fsol_ema_ratio.toFixed(2)}x`, 'WRS nhanh >2x'],
  [`NG-CS (D=101,631)`, `${rtNg.pac_moving_ratio.toFixed(2)}x`, `${rtNg.pac_ema_ratio.toFixed(2)}x`, `${rtNg.fsol_moving_ratio.toFixed(2)}x`, `${rtNg.fsol_ema_ratio.toFixed(2)}x`, 'WRS nhanh 3.8–7.1x'],
  [`News20 (D=1,355,191)`, `${rtNews20.pac_moving_ratio.toFixed(2)}x`, `${rtNews20.pac_ema_ratio.toFixed(2)}x`, `${rtNews20.fsol_moving_ratio.toFixed(2)}x`, `${rtNews20.fsol_ema_ratio.toFixed(2)}x`, 'WRS nhanh 2.4–9.0x'],
], 128, 392, 18);
addTakeaway(s27, 'Trên News20, FSOL-WRS tốn 36.6s/run trong khi Moving Avg tốn 155.9s và EMA tốn 329.2s.');

// Slide 28: Label Noise Analysis (Additional Analysis)
const rates = ['0', '0.05', '0.1', '0.2', '0.3'];
const rateLabels = ['0%', '5%', '10%', '20%', '30%'];
const w8aPacBase = rates.map(r => Number((100 * findSummary('noise', 'w8a', 'pac', `noise-${r}`).baseline.final_accuracy.mean).toFixed(2)));
const w8aPacWrs = rates.map(r => Number((100 * findSummary('noise', 'w8a', 'pac', `noise-${r}`).method.final_accuracy.mean).toFixed(2)));
const sst2PacBase = rates.map(r => Number((100 * findSummary('noise', 'sst2', 'pac', `noise-${r}`).baseline.final_accuracy.mean).toFixed(2)));
const sst2PacWrs = rates.map(r => Number((100 * findSummary('noise', 'sst2', 'pac', `noise-${r}`).method.final_accuracy.mean).toFixed(2)));

const s28 = makeSlide(
  28,
  'WRS giữ tới 96.45% Accuracy ở mức 30% nhiễu nhãn (so với 60.23% baseline)',
  'Thí nghiệm mở rộng (200 runs): chỉ lật nhãn tập train theo mask cố định, tập test giữ hoàn toàn sạch',
  `Kết quả: Dưới nhiễu nhãn train tăng dần 0% đến 30% trên Dexter, PCMAC, W8A và SST-2, cả PAC-WRS và FSOL-WRS đều vượt baseline ở 100% trường hợp từ mức nhiễu 10% trở lên. Nổi bật nhất trên W8A ở mức 30% nhiễu: PAC baseline tụt từ 97.96% xuống 60.23% (FSOL 61.07%) trong khi PAC-WRS giữ 96.45% (+36.22 điểm %) và FSOL-WRS giữ 96.55% (+35.48 điểm %); trên SST-2 ở 30% nhiễu, PAC-WRS giữ 76.40% so với 67.33%. Lý do: Những mẫu nhiễu ở cuối luồng làm baseline cập nhật sai lệch mạnh (over-correction), còn reservoir bảo lưu các phiên bản có chuỗi passive dài trước đó. Giới hạn: Thí nghiệm dùng nhiễu đối xứng trên 4 bộ đại diện.`,
  `[2] Thí nghiệm mở rộng của nhóm (200 runs). W&B W8A 30%: ${urlOf('noise', 'w8a', 'pac', 'noise-0.3')} | SST-2 30%: ${urlOf('noise', 'sst2', 'fsol', 'noise-0.3')}`
);
const chart28 = s28.charts.add('line', {
  position: {left: 56, top: 128, width: 1168, height: 396},
  title: 'Final Test Accuracy (%) theo tỷ lệ nhiễu nhãn train (tập test sạch, mean 5 trials)',
  titleTextStyle: {fontSize: 18, typeface: font},
  lineOptions: {smooth: false},
  categories: rateLabels,
  series: [
    {name: 'W8A PAC Base', values: w8aPacBase, fill: '#94A3B8'},
    {name: 'W8A PAC-WRS (K=64)', values: w8aPacWrs, fill: '#1B6CA8'},
    {name: 'SST-2 PAC Base', values: sst2PacBase, fill: '#F59E0B'},
    {name: 'SST-2 PAC-WRS (K=64)', values: sst2PacWrs, fill: '#0D9488'},
  ],
  hasLegend: true,
  xAxis: {title: {text: 'Tỷ lệ nhiễu nhãn tập train', textStyle: {fontSize: 16, typeface: font}}, textStyle: {fontSize: 16, typeface: font}},
  yAxis: {title: {text: 'Final Accuracy (%)', textStyle: {fontSize: 16, typeface: font}}, textStyle: {fontSize: 16, typeface: font}, min: 55, max: 100},
  legend: {textStyle: {fontSize: 16, typeface: font}},
});
applyPresentationChartFont(chart28, {fontFamily: font});
addTakeaway(s28, 'WAT chống over-correction rất mạnh trước nhãn nhiễu, đặc biệt trên dữ liệu có biên rõ như W8A.');

// Slide 29: Paper Comparison & Deviation Analysis
const pickRef = (ds, learner, model, cfg) => {
  const r = refRows.find(x => x.dataset === ds && x.learner === learner && x.model === model && x.config_id === cfg && x.metric === 'rop');
  return r && r.reference_value && r.team_value
    ? [Number(r.reference_value).toFixed(4), Number(r.team_value).toFixed(4), Number(r.delta).toFixed(4), r.relative_percent ? `${Number(r.relative_percent).toFixed(1)}%` : 'N/A']
    : ['—', '—', '—', '—'];
};
const [dPacRef, dPacTeam, dPacDelta, dPacRel] = pickRef('dexter', 'pac', 'wrs', 'k64_standard_simple_vz0');
const [dFsolRef, dFsolTeam, dFsolDelta, dFsolRel] = pickRef('dexter', 'fsol', 'wrs', 'k64_standard_simple_vz0');
const [sFsolRef, sFsolTeam, sFsolDelta, sFsolRel] = pickRef('sst2', 'fsol', 'wrs', 'k64_standard_simple_vz0');
const [nFsolRef, nFsolTeam, nFsolDelta, nFsolRel] = pickRef('newsgroups-binary-cs', 'fsol', 'wrs', 'k64_standard_simple_vz0');

const s29 = makeSlide(
  29,
  'Bốn khác biệt pipeline lý giải sai lệch giữa Paper và Tái hiện',
  'Đối chiếu trực tiếp code công khai của tác giả (PAC_main.py, analyzer.ipynb) với bản tự hiện thực',
  'Kết quả: So sánh ở cấu hình K=64 standard/simple cho thấy FSOL-WRS trên Dexter, NG-CS và SST-2 đều thấp hơn hoặc ngang ROP tác giả (SST-2 đạt -0.0058 so với 0.0014), trong khi PAC-WRS có độ lệch phụ thuộc ngưỡng chọn tham số. Lý do (4 điểm rẽ nhánh đã định vị): (1) Paper App. B.2 ghi "top 2.5%" nhưng code tác giả lọc max_acc - 0.025, ép C_err=1000 gây dao động lớn; (2) PAC_main.py thêm bias w[-1] nhưng mẫu số ||x||^2 thiếu +1; (3) Tác giả cập nhật khi loss > 1e-4 và np.sign(0)=0, còn nhóm dùng loss > 0 và >=0 -> +1; (4) Khác biệt seeds (42-46 vs 0-4) và SST-2 fit CountVectorizer chỉ trên tập train. Giới hạn: Số chuẩn hóa đối chiếu qua mean 5 trials.',
  `${PAPER_CITE} & Author Repo (PAC_main.py, analyzer.ipynb). Comparison CSV: results/batch1/reference_comparison.csv`
);
addTable(s29, [
  ['Dataset / Learner (ROP)', 'Paper (K=64 Std-SA)', 'Nhóm (K=64 Std-SA)', 'Delta (Nhóm - Paper)', 'Sai khác tương đối'],
  ['Dexter / PAC-WRS', dPacRef, dPacTeam, dPacDelta, dPacRel],
  ['Dexter / FSOL-WRS', dFsolRef, dFsolTeam, dFsolDelta, dFsolRel],
  ['NG-CS / FSOL-WRS', nFsolRef, nFsolTeam, nFsolDelta, nFsolRel],
  ['SST-2 / FSOL-WRS', sFsolRef, sFsolTeam, sFsolDelta, sFsolRel],
  ['4 nguyên nhân divergence', 'app. B.2 band vs ceil', 'bias +1 trong ||x||^2', 'loss>1e-4 & sign(0)', 'train-only vocab SST-2'],
], 128, 392, 18);
addTakeaway(s29, 'Tái hiện độc lập phát hiện lỗi bias +1 và sự mơ hồ trong tiêu chí tuning của paper.');

// Slide 30: Conclusions & Experimental Limitations
const s30 = makeSlide(
  30,
  'Kết luận thực nghiệm và giới hạn phạm vi cho chặng Poster tiếp theo',
  '1,275 runs online W&B kiểm chứng hiệu quả ổn định của WAT, đồng thời mở ra hướng phản biện Lab 2',
  'What: Nhóm đã hoàn tất tự hiện thực từ đầu (PAC, FSOL, WAT, Top-K, Moving, EMA), chạy 1,275 runs online W&B trên 8/16 dataset Đợt 1 và đối chiếu trực tiếp với code tác giả. Why: Kết quả thực nghiệm khẳng định WAT giữ tốt độ ổn định (FSOL-WRS thắng 6/8 ROP, W8A dưới 30% nhiễu giữ +36.2 điểm %) và nhanh vượt trội ở dữ liệu nhiều chiều. How & Giới hạn: Còn 8 bộ dữ liệu lớn/khổng lồ (tới 51.8M mẫu, 20.2M chiều) và Appendix C/E hoãn do giới hạn 8.55 GB RAM trống; kiểm định Wilcoxon n=8 chỉ là partial. Kết quả nhạy với cách tuning B.2 mở đường cho phản biện giả định i.i.d. và nhạy tham số ở Milestone 3 + Lab 2.',
  `${PAPER_CITE} [2] W&B Public Project (1,275 runs): ${PROJECT_URL}`
);
addTable(s30, [
  ['Hạng mục', 'Đã chứng minh bằng số liệu', 'Giới hạn / Hướng tới Milestone 3 & Lab 2'],
  ['Quy mô tái hiện', '8/16 bộ Table 1 (1,275 runs online W&B)', '8 bộ lớn (Criteo..Webspam) & App. C/E giới hạn RAM'],
  ['Ổn định ROP & Accuracy', 'FSOL-WRS thắng 6/8; PAC band thắng 4/8 ROP, 5/8 Acc', 'Wilcoxon n=8 chưa đủ 16; phụ thuộc độ dao động base'],
  ['Tốc độ & Ablations', 'WRS nhanh hơn Moving/EMA 2.4x–9.0x trên News20', 'K=64 lưu dense O(KD) gây tràn RAM khi D = 20.2M'],
  ['Nhiễu nhãn bổ sung', 'W8A 30% noise: WRS 96.5% vs Base 60.2%–61.1%', 'Mở rộng kiểm tra non-i.i.d. / concept drift ở Lab 2'],
], 128, 392, 18);
addTakeaway(s30, 'Mọi số liệu và bảng đối chiếu đều truy xuất công khai từng run trên W&B và repo.');

await (await PresentationFile.exportPptx(p)).save(path.join(build, 'candidate.pptx'));
for (let i = 0; i < p.slides.items.length; i++) {
  const blob = await p.export({slide: p.slides.items[i], format: 'png', scale: 1});
  await fs.writeFile(path.join(build, `slide-${i + 1}.png`), new Uint8Array(await blob.arrayBuffer()));
}

const outPath = path.resolve(root, process.argv[2] || 'slides/tv3_experiments.pptx');
const python = process.env.RUNTIME_PYTHON || path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe');
const result = await finalizePresentation({
  workspaceDir: root,
  candidatePath: path.join(build, 'candidate.pptx'),
  finalPath: outPath,
  pythonExecutable: python,
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  explicitTotalSlideCount: 9,
  requiredNativeTableOwnerSlides: [1, 2, 3, 6, 8, 9],
  requiredNativeChartOwnerSlides: [4, 5, 7],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: {basis: 'design', families: [font]},
  layoutArgs: [
    '--expected-slide-size-emu', '12192000,6858000',
    '--validate-bullet-geometry',
    '--validate-heading-fit',
    ...[1, 2, 3, 6, 8, 9].flatMap(n => ['--require-native-table-slide', String(n)]),
  ],
  verifyArtifactToolImport: true,
  receiptPath: path.join(build, 'validation.json'),
});
console.log(JSON.stringify(result));

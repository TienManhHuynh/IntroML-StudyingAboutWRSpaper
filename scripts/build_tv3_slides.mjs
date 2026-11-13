// Editable 9-slide TV3 development draft. Requires the supplied artifact-tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath, pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.RUNTIME_NODE_MODULES || path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
process.env.RUNTIME_NODE_MODULES=runtime;
const skill=process.env.PRESENTATIONS_SKILL_DIR || path.join(os.homedir(),'.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const build=path.join(root,'tmp/tv3-slides');
await fs.mkdir(build,{recursive:true});
const data=JSON.parse(await fs.readFile(path.join(root,'results/development/sst2-baseline/summary.json'),'utf8'));
const rawRoot=path.join(root,'results/raw/tv3-hosted');
const evidenceRoot=await fs.access(rawRoot).then(()=>rawRoot).catch(()=>path.join(root,'results/development/sst2-baseline/evidence'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const paper='[1] S. Wu et al., NeurIPS 2024, Sec. 4 & App. B. doi:10.52202/079017-1019';
const development='[2] Group 8, results/development/sst2-baseline (14/11/2026). Development; no public W&B.';
const font='Arial';
function text(slide,value,x,y,w,h,size=28,color='#14243B',bold=false){
  const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  s.text=value;s.text.style={fontFamily:font,fontSize:size,color,bold};return s;
}
function start(number,title,notes,source=paper){
  const slide=p.slides.add();slide.background.fill='#FFFFFF';
  text(slide,`${number}  |  ${title}`,56,35,1168,80,38,'#14243B',true);
  text(slide,'BẢN NHÁP TV3 • DEVELOPMENT • CHƯA CÓ KẾT QUẢ WRS',56,115,1168,35,18,'#A1450B');
  text(slide,source,56,652,1168,48,15,'#52647A');
  slide.speakerNotes.text=notes+'\nNguồn: '+source+'\nPaper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/38d88c3296eafb09d3971d091a19cf6b-Abstract-Conference.html';
  return slide;
}
function table(slide,values,y=175,height=350){
 const tab=slide.tables.add({rows:values.length,columns:values[0].length,left:56,top:y,width:1168,height,values});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
   const cell=tab.getCell(r,c);cell.fill=r===0?'#14243B':(r%2?'#EFF4F8':'#FFFFFF');
   cell.text.style={fontFamily:font,fontSize:25,color:r===0?'#FFFFFF':'#14243B',bold:r===0};
 }
 return tab;
}
function takeaway(slide,value){text(slide,value,56,568,1168,66,27,'#163D68',true);}
const s1=start(22,'Nguồn dữ liệu và tập phát triển',
 'Nhóm kiểm kê đủ 16 dataset Table 1. Mới xác minh và tải được nguồn SST-2 của Stanford trên Hugging Face ở revision cố định. Asset train có 67.349 dòng, khác 67.337 dòng paper. Lần chạy này chọn ngẫu nhiên 600 mẫu với subset seed 0 để thử pipeline; không đại diện cho reproduction đầy đủ. CountVectorizer chỉ fit train để tránh leakage; việc tái dựng đúng preprocessing paper còn cần đối chiếu. Các dataset còn lại cần nguồn hosted và kiểm tra equivalence.');
table(s1,[['Hạng mục','Paper / kế hoạch','Đã chạy development'],['Dataset chính','16 bộ','SST-2: 1 bộ'],['Số mẫu SST-2','67.337','600 mẫu từ HF'],['Features','Theo Table 1','CountVectorizer fit train'],['Nguồn','Paper Table 1','HF revision cố định']]);
takeaway(s1,'Nguồn hosted đã tải được; equivalence với paper vẫn cần xác minh');
const s2=start(23,'Protocol giữ cùng dữ liệu giữa các phương pháp',
 'Mỗi trial random 70/30 không stratify. Subset seed 0 cố định tập phát triển; seeds 42 đến 46 thay split và thứ tự stream. Mỗi learner khởi tạo 0, học một lượt, đánh giá trên cùng test set tại các checkpoint đã lưu. Lần phát triển dùng target 40, tương ứng 41 checkpoint gồm đầu/cuối; paper dùng khoảng 200. PAC C_err=1, FSOL eta=1 và lambda=.01 là cấu hình thử, chưa qua tuning paper. Các biến thể WRS chưa chạy vì thiếu implementation TV2.');
table(s2,[['Bước','Thiết lập hiện tại'],['Split / stream','70/30, seeds 42–46'],['Training','Một pass, weights ban đầu 0'],['Development parameters','PAC C=1; FSOL eta=1, lambda=.01'],['Evaluation','41 checkpoints, cùng test set'],['Reproduction cần thêm','Tuning B.2 + WRS + full data']]);
takeaway(s2,'Các file split/config được lưu để đối chiếu lại từng trial');
const s3=start(24,'Accuracy, ROP và model sparsity',
 'Accuracy là tỷ lệ nhãn test dự đoán đúng. Oracle tại checkpoint j là cumulative maximum accuracy của baseline tương ứng từ đầu đến j. ROP là mean của oracle trừ accuracy method trên cùng logging grid theo approximation B.3. ROP âm có thể xảy ra nếu method vượt oracle baseline. Model sparsity là tỷ lệ tọa độ bằng 0 của vector dự đoán; không phải độ thưa dataset. Code có tests tính tay, oracle không giảm, ROP âm và từ chối dữ liệu sai.');
table(s3,[['Metric','Cách tính','Cách đọc'],['Accuracy','Số dự đoán đúng / N_test','Cao hơn tốt hơn'],['ROP','mean(oracle_j − accuracy_j)','Thấp hơn tốt hơn'],['Sparsity','Số weights bằng 0 / D','Giữ độ thưa của FSOL']]);
takeaway(s3,'Oracle phải đến từ đúng baseline và cùng trial/checkpoint');
function csvRows(contents){const lines=contents.trim().split(/\r?\n/);const keys=lines.shift().split(',');return lines.map(line=>Object.fromEntries(line.split(',').map((v,i)=>[keys[i],v])));}
for(const [number,learner] of [[25,'pac'],[26,'fsol']]){
 const item=data.find(x=>x.configuration.learner===learner);
 const rows=csvRows(await fs.readFile(path.join(evidenceRoot,learner,'42/checkpoints.csv'),'utf8')).filter(x=>x.model==='baseline');
 const plotted=rows.filter((x,i)=>i%4===0||i===rows.length-1);
 const notes=`Đồ thị lấy trực tiếp từ checkpoints.csv của ${learner.toUpperCase()} seed 42. Tập phát triển 600 mẫu, train 420/test 180; không có đường WRS vì chưa có code TV2. Mean qua 5 trials: final accuracy ${item.final_accuracy.mean.toFixed(6)}, ROP ${item.rop.mean.toFixed(6)}, sparsity ${item.final_sparsity.mean.toFixed(6)}. Hyperparameters thử chưa tuning. Dữ liệu, cấu hình và môi trường lưu trong evidence. Không có public W&B link nên chưa dùng làm kết quả reproduction nộp cuối.`;
 const slide=start(number,learner.toUpperCase()+' baseline: kiểm tra đầu cuối trên SST-2',notes,development);
 const chart=slide.charts.add('line',{position:{left:56,top:170,width:1168,height:365},title:learner.toUpperCase()+' seed 42 — 11/41 checkpoints hiển thị',titleTextStyle:{fontSize:22,typeface:font},lineOptions:{smooth:false},categories:plotted.map(x=>x.step),series:[{name:'Test accuracy',values:plotted.map(x=>Number(Number(x.accuracy).toFixed(6))),fill:'#2469A8'},...(learner==='fsol'?[{name:'Model sparsity',values:plotted.map(x=>Number(Number(x.sparsity).toFixed(6))),fill:'#B36A18'}]:[])],hasLegend:true,xAxis:{title:{text:'Training examples processed',textStyle:{fontSize:20,typeface:font}},textStyle:{fontSize:18,typeface:font}},yAxis:{title:{text:'Fraction',textStyle:{fontSize:20,typeface:font}},textStyle:{fontSize:18,typeface:font},min:0,max:1},legend:{textStyle:{fontSize:20,typeface:font}}});
 applyPresentationChartFont(chart,{fontFamily:font});
 takeaway(slide,`5 trials: accuracy mean ${(100*item.final_accuracy.mean).toFixed(1)}% • ROP ${item.rop.mean.toFixed(4)} • chưa có WRS`);
}
const s6=start(27,'Full scope còn cần các biến thể và đối chứng',
 'Phạm vi nhóm trưởng chọn là làm đầy đủ paper. Ma trận gồm 16 dataset, PAC/FSOL, standard/exponential weighting, simple/weighted averaging, zeroing bật/tắt và K 1/4/16/64, năm trials. Cần thêm top-K, moving average, EMA, runtime và thống kê. Appendix C/E cần xác minh dataset/defaults và triển khai liên quan. Pipeline TV3 hiện nhận wrapper qua factory; không dùng identity wrapper test như WAT thật.');
table(s6,[['Nhóm','Phạm vi','Trạng thái'],['PAC / FSOL','16 datasets × 5 trials','Development SST-2'],['WRS variants','K, WS, AS, VZ','Chờ WAT TV2'],['Ablations','Top-K, moving, EMA','Cần implementation'],['Statistics / extensions','Runtime, Wilcoxon, C/E','Chưa có đủ runs']]);
takeaway(s6,'Không dùng baseline development thay cho thí nghiệm WRS của paper');
const s7=start(28,'Label noise chỉ thay nhãn train',
 'Additional analysis do nhóm thiết kế: rate 0,5,10,20,30 phần trăm. Sau split, tạo noise mask từ RNG seed riêng và lật nhãn train. Test labels không đổi. Cùng seed cho các methods để có cùng mask; noise trials kết hợp clean baseline/WRS khi WAT sẵn sàng. Runner đã có và chạy thử trên synthetic. Chưa có kết quả label-noise reproduction của WRS, vì vậy slide chỉ trình bày thiết kế.');
table(s7,[['Thành phần','Thiết kế'],['Noise rates','0%, 5%, 10%, 20%, 30%'],['Train','Lật nhãn theo mask/seed cố định'],['Test','Giữ nhãn sạch'],['So sánh','Cùng split và mask giữa models'],['Bằng chứng hiện có','Synthetic smoke; chưa có WRS']]);
takeaway(s7,'Noise analysis bổ sung; không thay Full scope');
const s8=start(29,'Paper comparison cần đúng dataset và variant',
 'Bảng cuối sẽ đặt paper value cạnh reproduced value cho cùng dataset/variant, dẫn table hoặc figure và W&B run. Hiện chưa có WRS chạy thật nên không điền số paper hay số nhóm để ngụ ý đã tái hiện. Relative deviation gần paper value bằng 0 cần để N/A và báo absolute delta. Code đã có hàm deviation và bộ tổng hợp, còn các bảng đối chiếu thực nghiệm cần dữ liệu thật.');
table(s8,[['Hạng mục','Điều kiện đối chiếu','Hiện tại'],['Paper vs nhóm','Cùng data / variant / protocol','Chưa đủ điều kiện'],['Absolute delta','Reproduced − paper','Chưa có WRS'],['Relative deviation','Có nghĩa khi paper khác 0','Không tự điền 0'],['Nguồn','Paper + public W&B run','W&B chưa thiết lập']]);
takeaway(s8,'Mỗi con số nộp cuối phải truy được tới run public');
const s9=start(30,'Các bước còn lại để hoàn tất reproduction',
 'TV3 đã có loader HF pinned, finite-input guards, metrics, runner baseline/factory WAT, noise analysis, tuning và tổng hợp. Các tests và runs phát triển xác nhận pipeline hoạt động bước đầu. Còn thiếu WAT TV2, nguồn tương đương cho 15 dataset khác và xác minh SST-2 preprocessing/count, tài khoản/project W&B public, chạy grid đầy đủ, reference comparison và hoàn thiện slides kết quả. Bản này là nháp 9 slide phần TV3 để ghép 22–30, không phải deck nộp cuối.');
table(s9,[['Phụ thuộc','Người / phần việc','Bằng chứng cần có'],['WAT + tests','TV2','PAC-WRS / FSOL-WRS chạy được'],['Dataset equivalence','TV3','Hosted source + preprocessing'],['Public tracking','TV3 / nhóm trưởng','Project và run links'],['Full experiments','Nhóm','Tables / plots / deviations']]);
takeaway(s9,'Bản nháp giữ nguyên các giới hạn của bằng chứng hiện có');
await (await PresentationFile.exportPptx(p)).save(path.join(build,'candidate.pptx'));
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const python=process.env.RUNTIME_PYTHON || path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe');
const result=await finalizePresentation({workspaceDir:root,candidatePath:path.join(build,'candidate.pptx'),finalPath:path.resolve(root,process.argv[2] || 'slides/tv3_experiments_draft.pptx'),
 pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:9,requiredNativeTableOwnerSlides:[1,2,3,6,7,8,9],requiredNativeChartOwnerSlides:[4,5],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[1,2,3,6,7,8,9].flatMap(n=>['--require-native-table-slide',String(n)])],verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify(result));

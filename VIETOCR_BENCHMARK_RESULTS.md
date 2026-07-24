# VietOCR trong MinerU — kết quả benchmark & fix (2026-07-24)

File này ghi lại toàn bộ số liệu benchmark, nguyên nhân gốc, và các fix đã áp dụng
cho tích hợp VietOCR trong `mineru/model/ocr/pytorch_paddle.py`. Xem thêm
`VIETOCR_PERF_CHECKLIST.md` cho danh sách việc còn treo.

## 1. Vấn đề ban đầu: OOM khi chạy pipeline thật

Chạy `mineru_viet.py` trên file 105 trang ("Mắc ca.pdf") → lỗi:
```
[enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory:
you tried to allocate 4848615424 bytes.
```

**Nguyên nhân**: `vietocr.tool.predictor.Predictor.predict_batch()` gom các ảnh
cùng width (sau resize) vào 1 bucket rồi `torch.cat()` toàn bộ thành 1 tensor,
**không giới hạn kích thước**. MinerU gọi `predict_batch(crop_imgs)` với toàn bộ
1646-2173 crop của 1 cửa sổ xử lý (64 trang) trong 1 lần — bucket lớn nhất cần
cấp phát ~4.8GB **liên tục trên CPU RAM** (trước khi `.to(device)` chuyển qua GPU)
→ fail dù máy có 64GB RAM, vì đây là vấn đề **cấp phát 1 khối liên tục lúc RAM
đang bị chiếm dụng nhiều + có thể phân mảnh**, không phải thiếu RAM tổng.

Máy có ~64GB RAM, ~46GB free lúc đo — nhưng `torch.cat()` cần 1 khối *liên tục*
4.8GB, và pipeline đang giữ nhiều model + ảnh trang khác trong RAM cùng lúc.

## 2. Vấn đề thứ hai: "chờ nhau" trong batch

`vietocr.tool.translate.translate()` decode tự hồi quy (autoregressive), vòng
lặp `while ... not all(...==eos_token)` — **chỉ dừng khi TẤT CẢ sequence trong
batch đã ra EOS**. Nếu 1 bucket có cả ảnh chữ ngắn lẫn ảnh chữ dài, ảnh ngắn xong
sớm vẫn phải "chờ" ảnh dài chạy hết tối đa 128 bước decode.

Tăng `chunk_size` (256 → 512) **không giải quyết được** — verify thực tế: tốc độ
gần như không đổi (198.9ms/crop vs 202.9ms/crop) vì vấn đề nằm trong chính vòng
lặp decode của mỗi bucket, không phải số lượng ảnh đưa vào 1 lần gọi.

## 3. Fix: `mineru/model/ocr/vietocr_fast_batch.py`

Viết lại thay thế cho `predict_batch()`, gồm 3 phần:

1. **Sub-batch cố định** (mặc định 128 ảnh/lần) — chặn OOM.
2. **Early-exit decode thật** (`_translate_early_exit`) — mỗi bước chỉ tính cho
   những sequence chưa ra EOS (dùng `active` set, thu hẹp dần qua `memory.index_select`),
   dừng ngay khi 1 sequence xong thay vì chờ cả batch.
3. **Gom theo độ dài dự kiến** — sort theo width *chưa bị cap* trước khi chia
   sub-batch, giảm tình trạng ảnh ngắn/dài lẫn lộn trong cùng 1 sub-batch.

**Verify đúng đắn**: so với `predictor.predict()` gốc (từng ảnh 1) trên 200 mẫu
→ **0 mismatch**. Verify lại lần 2 sau khi thêm confidence: vẫn **0/200 mismatch**.

### Benchmark cô lập (chỉ load VietOCR, không model khác — 2000 crop thật)

| Cách | Tốc độ | Tổng thời gian |
|---|---|---|
| Cũ (`predict_batch` gốc, không chunk, không early-exit) | 141.4ms/crop | 282.8s |
| Mới, sub_batch=128 | **17.9ms/crop** | **35.7s** |
| Mới, sub_batch=256 | 19.6ms/crop | 39.2s |

→ **Nhanh gấp ~7.9 lần** ở quy mô cô lập. 128 nhỉnh hơn 256 nên chọn làm mặc định.

## 4. Phát hiện: tốc độ trong pipeline thật chậm hơn nhiều so với benchmark cô lập

Chạy full `mineru_viet.py` thật (105 trang) với fix trên:

| Batch (số crop) | ms/crop | Vị trí trong pipeline |
|---|---|---|
| 1183 | 8.7-9.2ms | Sớm |
| 2173 | 152-156ms | Giữa |
| 1646 | 152-155ms | Giữa |
| 4 | 50-54ms | Cuối |

**17-18x chậm hơn** so với benchmark cô lập (17.9ms/crop) ở 2 batch giữa.

### Instrument sâu hơn — tách GPU-kernel-time vs Python-glue-time

Thêm timing chi tiết vào `_translate_early_exit`/`predict_batch_grouped`
(`torch.cuda.synchronize()` bọc quanh `forward_decoder`+`softmax`):

| n | decode_total | gpu_kernel (forward_decoder+softmax) | py_glue | ms/bước (gpu_kernel) |
|---|---|---|---|---|
| 1183 | 10.31s | 6.34s (61%) | 3.97s | 6.2ms |
| 2173 | 332.73s | 292.83s (88%) | 39.90s | **103ms** |
| 1646 | 253.36s | 222.36s (88%) | 31.00s | **84ms** |
| 4 | 0.21s | 0.14s (67%) | 0.07s | 3.1ms |

**Kết luận ban đầu (SAI, đã đính chính ở mục 8): bottleneck là GPU (88% thời
gian là gpu_kernel thật, không phải Python overhead), do tranh chấp đơn vị
tính toán (SM) với các model khác chạy đồng thời trong pipeline.** Giả thuyết
này bị bác bỏ sau khi kiểm tra code: `pipeline_analyze.py` xử lý các window
**tuần tự** (vòng lặp `for`, không async/concurrent), và với 1 file PDF,
`resolve_submit_concurrency(max=3, task_count=1) = 1` — không có request nào
khác chạy cùng lúc để tranh chấp. Nguyên nhân thật: xem mục 8 (thiếu KV-cache).

## 5. Phát hiện: bước PaddleOCR recognizer không hoàn toàn vô dụng

Ban đầu tưởng bước `self.text_recognizer()` (chạy trước VietOCR, dùng
`rec_batch_num=6` cứng → CPU-bound, GPU chỉ 2-4% util) là tính toán bỏ đi hoàn
toàn (text bị VietOCR ghi đè). Đọc kỹ code phát hiện: **score được giữ lại**
(`rec_res[idx][1]`) dùng để lọc box theo `drop_score` — không hoàn toàn vô dụng.

### Fix (approach 2 — đã code, đang benchmark)
- `_translate_early_exit` giờ trả thêm **confidence tự tính** (trung bình xác
  suất top-1 softmax của các token đã chọn).
- `pytorch_paddle.py` (`ocr()` và `__call__()`): **bỏ hẳn gọi
  `self.text_recognizer()`** khi VietOCR khả dụng, dùng confidence VietOCR thay
  cho score PaddleOCR. Fallback về PaddleOCR recognizer nếu VietOCR lỗi/không
  khả dụng (giữ tương thích ngôn ngữ khác).
- Verify đúng đắn: 0/200 mismatch, confidence range hợp lý (0.91-0.94).
- **Đã chạy full pipeline, xác nhận kết quả** (xem bảng dưới).

### Kết quả full pipeline sau fix #2

Không còn dòng log `PaddleOCR text_recognizer` nào — bước này bị bỏ hoàn toàn.
So sánh cùng file, cùng 4 batch, trước/sau fix #2:

| Batch (n) | PaddleOCR recognizer (bị vứt, trước fix #2) | Sau fix #2 |
|---|---|---|
| 1183 | 11.648s | **0s (bỏ hẳn)** |
| 2173 | 92.575s | **0s** |
| 1646 | 72.856s | **0s** |
| 4 | 0.061s | **0s** |
| **Tổng tiết kiệm** | **177.14s** | |

VietOCR tự nó **không nhanh hơn đáng kể** (148.6/149.9ms/crop so với
152-156ms/crop trước đó) — đúng như dự đoán: contention GPU (mục 4) vẫn còn
nguyên, bỏ PaddleOCR recognizer không đụng tới nguyên nhân đó, chỉ loại bỏ
177.14s tính toán hoàn toàn thừa. Output vẫn đúng dấu 100%, không hồi quy chất
lượng (verify trên "Mắc ca.pdf" đầy đủ 105 trang).

### Approach 1 (tăng `rec_batch_num`) — không triển khai, không cần thiết
Lý thuyết: tăng `rec_batch_num` (6 → 128) sẽ giảm overhead round-trip CPU↔GPU
cho bước PaddleOCR recognizer, nhưng vô nghĩa khi approach 2 đã **bỏ hẳn bước
đó** — không có gì để tối ưu thêm nữa. Approach 2 luôn tốt hơn approach 1.

## 6. Tóm tắt cải thiện tích lũy

| Giai đoạn | Trạng thái |
|---|---|
| Trước fix | OOM crash trên batch lớn, dấu tiếng Việt sai ở phần bị fallback |
| Sau fix #1 (early-exit + chunking) | Không OOM, dấu đúng 100%, nhanh 7.9x ở benchmark cô lập nhưng pipeline thật vẫn chậm (nguyên nhân sai ban đầu: tưởng do contention GPU) |
| Sau fix #2 (bỏ PaddleOCR recognizer thừa) | **Đã xác nhận**: tiết kiệm 177.14s tính toán thừa, dấu vẫn đúng 100%, confidence tự tính từ VietOCR hoạt động đúng |
| Sau fix #3 (KV-cache, mục 8) | **Đã xác nhận**: pipeline thật nhanh thêm 3x (2173 crop: 148.6ms/crop → 49.8ms/crop), gpu_kernel giảm 4.1x, dấu vẫn đúng 100% |

## 7. Đính chính: nguyên nhân KHÔNG phải tranh chấp GPU với process khác

Kiểm tra lại code sau khi nghi ngờ mục 4: `pipeline_analyze.py` xử lý 2 window
(64 trang/window) **tuần tự** — log `"Pipeline processing window batch
{batch_index}/{total_batches}"` là 1 vòng `for`, không phải task async chạy
song song. Và với 1 file PDF, `resolve_submit_concurrency(max_concurrent=3,
task_count=1)` trả về `1` — không có request nào khác cùng lúc để tranh chấp
GPU. Giả thuyết "tranh chấp SM với stage khác" ở mục 4 sai.

### Xác nhận bằng dữ liệu: chi phí mỗi bước decode tăng theo prefix_len

Thêm log `MINERU_VIETOCR_STEP_DEBUG=1` (bọc mỗi bước `forward_decoder` bằng
`torch.cuda.synchronize()`, ghi `(prefix_len, active_n, elapsed_ms)`):

| prefix_len | n=2173 (ms/active) | n=1646 (ms/active) | n=1183 (ms/active) |
|---|---|---|---|
| 0-16 | 1.05 | 1.23 | 2.02 |
| 16-32 | 1.65 | 1.89 | 3.45 |
| 32-48 | 2.39 | 2.61 | 4.77 |
| 48-64 | 3.28 | 3.47 | 5.88 |
| 64-80 | 3.36 | 3.33 | *(không có bước)* |
| 80-96 | 4.00 | 4.08 | *(không có bước)* |
| 96-112 | 6.11 | 7.14 | *(không có bước)* |

Chi phí mỗi bước tăng ~5-6 lần từ prefix~8 lên prefix~104, ở cả 3 batch —
**độc lập với việc có process nào khác chạy hay không**. Batch 1183 chưa bao
giờ vượt prefix 64 (chữ ngắn, decode xong sớm) nên không bao giờ trả "thuế"
chi phí cao ở vùng prefix dài; batch 2173/1646 có chữ dài hơn → nhiều bước
decode hơn (2831/2648 vs 1021) **và** nhiều bước rơi vào vùng prefix đắt —
2 yếu tố cộng dồn, giải thích đầy đủ chênh lệch mà không cần viện đến
tranh chấp process khác.

**Nguyên nhân thật**: `forward_decoder()` gọi `self.transformer.decoder(tgt,
memory, tgt_mask=...)` — **không có KV-cache**, tính lại self-attention trên
TOÀN BỘ `tgt` (prefix đã sinh ra) mỗi bước, dù chỉ cần output của token cuối
cùng. Chi phí self-attention scale theo O(prefix_len) mỗi bước (và tích luỹ
O(prefix_len²) cho toàn bộ quá trình decode 1 sequence).

## 8. Fix #3: KV-cache thật cho decode loop

Viết `mineru/model/ocr/vietocr_fast_batch.py`: `_translate_kv_cache()` +
`_decoder_layer_step()` + `_attn_step()` + `_project_qkv()` — **tái triển
khai thủ công** forward của `TransformerDecoderLayer` (post-norm,
`norm_first=False`, đúng kiến trúc model đang dùng: 6 layer, `nhead=8`,
`d_model=256`) theo kiểu incremental:

- **Self-attention KV-cache**: mỗi bước chỉ tính Q/K/V cho token MỚI, nối vào
  cache (K,V) của các bước trước thay vì tính lại từ đầu. Cache lưu full-width
  `n` (batch gốc), index_select theo `active` mỗi bước — cùng pattern với cách
  `memory` đã được xử lý trong bản early-exit.
- **Cross-attention K/V precompute**: `memory` (encoder output) không đổi
  suốt quá trình decode → project K/V cho cross-attention **1 lần duy nhất**
  trước vòng lặp thay vì mỗi bước.
- Dùng lại đúng **trọng số** của model (`in_proj_weight`, `out_proj`,
  `linear1/2`, `norm1/2/3`) — không train lại, không đổi tham số, chỉ tổ chức
  lại *khi nào* các phép nhân ma trận được thực hiện.

### Verify đúng đắn
- 0/300 mismatch so với `predictor.predict()` gốc (test riêng).
- 0/100 spot-check mismatch trên tập 2000 crop.
- Full pipeline "Mắc ca.pdf" (105 trang): dấu tiếng Việt đúng 100%, không hồi
  quy chất lượng.

### Kết quả benchmark cô lập (2000 crop)
| Bản | ms/crop | Tổng |
|---|---|---|
| Gốc (`predict_batch` PyPI, không sửa gì) | 141.4ms | 282.8s |
| Early-exit + chunking (fix #1) | 17.9ms | 35.7s |
| **+ KV-cache (fix #3)** | **5.4ms** | **10.75s** |

→ **~26 lần nhanh hơn bản gốc**, tính lũy kế qua 2 lần tối ưu.

### Kết quả full pipeline thật (sau fix #2 + fix #3)
| Batch (n) | Trước KV-cache | Sau KV-cache | Cải thiện |
|---|---|---|---|
| 1183 | 9.1ms/crop | 6.8ms/crop | 1.3x |
| 2173 | 148.6ms/crop | 49.8ms/crop | **3.0x** |
| 1646 | 149.9ms/crop | 48.2ms/crop | **3.1x** |
| 4 | 50.4ms/crop | 48.3ms/crop | ~1x |

`gpu_kernel` riêng batch 2173: 283.57s → 68.46s (**4.1x**). Tổng thời gian
VietOCR cả file: 580.6s → 195.8s (**giảm 66%**).

Lưu ý: sau khi `gpu_kernel` giảm mạnh, `py_glue` (overhead Python: tensor
build, index_select mỗi layer × mỗi bước, giờ có 6 layer × nhiều tensor
nhỏ/bước) chiếm tỷ trọng lớn hơn hẳn (36% thay vì 12%) — **bottleneck tiếp
theo nếu muốn tối ưu thêm** sẽ là overhead Python/index_select trong vòng lặp
6 layer, không phải GPU compute nữa.

## 9. Mức độ "fork" khỏi MinerU/VietOCR gốc — cần lưu ý khi bảo trì

Sau tất cả các fix trên, code đã vượt ranh giới "patch nhỏ" sang **fork thật
sự phần decode của VietOCR**:

- `mineru/model/ocr/vietocr_fast_batch.py` — **tái triển khai thủ công** công
  thức toán của `nn.TransformerDecoderLayer`/`nn.MultiheadAttention`, không
  còn gọi `nn.TransformerDecoder.forward()` hay `vietocr.tool.translate.translate()`
  nữa. Chỉ dùng chung *trọng số* của model, không dùng chung *luồng tính toán*
  của PyTorch/vietocr.
- `mineru/model/ocr/pytorch_paddle.py` — bỏ hẳn PaddleOCR recognizer khi
  VietOCR khả dụng (2 call site: `ocr()`, `__call__()`).
- **Không đụng**: kiến trúc/trọng số model (không train lại), các model khác
  của MinerU (layout/table/formula/seal), package `vietocr` gốc (vẫn cài
  nguyên, chỉ không còn được gọi từ đường MinerU).

**Rủi ro bảo trì**: `_translate_kv_cache()` hard-code giả định `norm_first=False`
(post-norm) và cấu trúc 6-layer/`nhead=8`/`d_model=256` đọc từ model hiện tại
(`vietocr_weights/vgg_transformer.yml`). Nếu sau này **fine-tune lại VietOCR
với kiến trúc khác** (đổi `num_decoder_layers`, `nhead`, `d_model`, hoặc dùng
`norm_first=True`), code này **sẽ không tự động khớp** — cần kiểm tra/sửa lại
theo kiến trúc mới trước khi dùng.

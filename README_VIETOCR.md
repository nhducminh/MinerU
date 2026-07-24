# MinerU - Phiên bản Tích hợp VietOCR (Dành riêng cho Tiếng Việt)

## 🌟 Giới thiệu
Đây là bản *fork* (phân nhánh) từ dự án **MinerU** nguyên bản (nhánh `vietocr-integration`). Bản fork này mang trong mình toàn bộ sức mạnh trích xuất cấu trúc văn bản, bảng biểu, công thức toán học xuất sắc của MinerU, nhưng được "cấy ghép" thêm mô hình **VietOCR** để giải quyết triệt để bài toán rớt dấu, nhận sai ký tự khi xử lý tài liệu Tiếng Việt in-the-wild (bao bì, chứng từ, scan mờ).

Toàn bộ tích hợp nằm **độc lập ngay trong repo này** (`mineru/model/ocr/pytorch_paddle.py`) — không phụ thuộc vào bất kỳ fork PaddleOCR nào khác.

## 💡 Vấn đề và Giải pháp

### 1. Rớt dấu do Mô hình Nhận diện (PaddleOCR)
MinerU gốc dùng chung 1 model nhận diện ("latin") cho tất cả các ngôn ngữ Latin (Pháp, Đức, Tây Ban Nha... và cả tiếng Việt) — dù ở phiên bản PP-OCRv4 hay v5. Model gộp chung này không phủ hết bộ dấu thanh phức tạp của tiếng Việt, dẫn tới lỗi rớt dấu/nhận sai ký tự thường xuyên.
* **Giải pháp:** Can thiệp trực tiếp vào `mineru/model/ocr/pytorch_paddle.py` (class `PytorchPaddleOCR`, cả 2 nhánh `ocr()` và `__call__()`), chèn **VietOCR** (`vgg_transformer`) vào bước nhận diện (`rec`), thay cho model "latin" gộp chung. PaddleOCR vẫn đảm nhiệm bước phát hiện vùng chữ (`det`) như bình thường; bước **nhận diện (`rec`) gốc của PaddleOCR bị bỏ qua hoàn toàn** khi VietOCR khả dụng (chỉ fallback về PaddleOCR nếu VietOCR lỗi) — trước đây cả 2 vẫn chạy song song dù text của PaddleOCR bị vứt bỏ, chỉ giữ lại score để lọc box; giờ score được tự tính từ chính VietOCR (trung bình xác suất softmax), nên không cần chạy PaddleOCR recognizer nữa.
* Về hiệu năng nhận diện: dùng `mineru/model/ocr/vietocr_fast_batch.py` — **tái triển khai thủ công** vòng lặp decode của VietOCR với **KV-cache** (self-attention cache + cross-attention precompute) và **early-exit** (dừng sớm từng sequence khi ra EOS thay vì chờ cả batch), thay cho `predict_batch()`/`translate()` gốc của thư viện `vietocr` (vốn tính lại toàn bộ prefix mỗi bước, tốn kém theo O(prefix_len), và có thể OOM khi batch quá lớn do `torch.cat()` không giới hạn kích thước). Đã benchmark: **nhanh hơn ~26 lần** so với `predict_batch()` gốc (đo cô lập), **~3 lần** ngay trong pipeline thật (đo trên batch OCR lớn). Đã verify đúng đắn tuyệt đối (0 sai lệch so với `predict()` gốc từng ảnh, nhiều lần test trên hàng nghìn crop). Chi tiết đầy đủ số liệu + phương pháp đo: [`VIETOCR_BENCHMARK_RESULTS.md`](VIETOCR_BENCHMARK_RESULTS.md).

### 2. File PDF có Text Layer bị Lỗi Font
Với các file PDF được xuất từ MS Word có định dạng phông chữ lạ, lớp văn bản ẩn (text layer) thường bị vỡ dấu (Ví dụ: "QUYẾT ĐỊNH" thành "QUYT ĐNH"). MinerU mặc định sẽ bốc lớp text này ra thay vì dùng OCR, khiến kết quả sai hoàn toàn.
* **Giải pháp:** Sử dụng tệp thực thi `mineru_viet.py` thay cho `mineru.exe`. Tệp lệnh này tự động "ép phẳng" (flatten) các file PDF bằng cách render toàn bộ 100% trang thành ảnh (300 DPI) để phá hủy lớp text layer hỏng, ép hệ thống phải dùng nhãn lực VietOCR để đọc lại từ đầu một cách hoàn hảo.
* **Xử lý theo lô (batch):** khi trỏ vào một thư mục nhiều file, `mineru_viet.py` gom tất cả file vào **1 lần chạy `mineru` duy nhất** (thay vì gọi subprocess riêng cho từng file) — model chỉ load 1 lần, tránh phải trả "phí khởi động nguội" (model init + CUDA warmup, ~25-30s) lặp lại cho mỗi file.

## ⚡ Yêu cầu GPU (quan trọng)
VietOCR + PaddleOCR chạy trên **CPU sẽ rất chậm** (~10s/trang trở lên). Cần cài `torch`/`torchvision` bản có CUDA khớp với GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```
(GPU dòng Blackwell RTX 50-series cần CUDA 12.8+, tương ứng torch ≥ 2.7.0). Kiểm tra sau khi cài:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚀 Cài đặt
1. Cài đặt các gói phụ thuộc (đã bao gồm `vietocr` và `PyMuPDF` trong `pyproject.toml`):
   ```bash
   pip install -e .[all]
   ```
2. Cài `torch` bản CUDA theo hướng dẫn ở mục "Yêu cầu GPU" phía trên.
3. (Tùy chọn) Máy sẽ tự động tải trọng số VietOCR (`vgg_transformer.pth`) trong lần chạy đầu tiên.

## 💻 Sử dụng
Sử dụng tệp lệnh `mineru_viet.py` để chạy (tự động hỗ trợ quét file lẻ và quét nguyên một thư mục, kể cả thư mục con):

```bash
python mineru_viet.py -p "đường_dẫn_đến_thư_mục_hoặc_file_PDF" -o "thư_mục_đầu_ra"
```

*Tất cả cấu hình như `-l vi`, `-m ocr`, `-b pipeline` đều đã được tự động áp dụng bên trong script.*

## 📊 Hiệu năng thực đo
Đo trên RTX 5080, tài liệu kỹ thuật nông nghiệp tiếng Việt thật (11-105 trang/file):

| Cấu hình | Tốc độ |
|---|---|
| CPU (torch mặc định, chưa tối ưu) | ~10s/trang |
| GPU (cu128) + `predict_batch` gốc | ~4.5s/trang |
| GPU + KV-cache + early-exit (hiện tại) | **~26 lần nhanh hơn `predict_batch` gốc** (đo cô lập, 2000 crop); **~3 lần** trong pipeline thật trên batch OCR lớn |

Lưu ý: vùng con dấu/logo/chữ trang trí phức tạp vẫn là điểm yếu chung của mọi engine OCR (không riêng VietOCR). Số liệu chi tiết, phương pháp đo, và log debug từng bước decode: xem [`VIETOCR_BENCHMARK_RESULTS.md`](VIETOCR_BENCHMARK_RESULTS.md) và [`VIETOCR_PERF_CHECKLIST.md`](VIETOCR_PERF_CHECKLIST.md).

## 🎓 Dataset & Fine-tune
Model VietOCR đi kèm (`vietocr_weights/vietocr_finetuned.pth`, không commit vào git — vượt giới hạn 100MB của GitHub, đặt file này thủ công vào `vietocr_weights/` trước khi chạy) được fine-tune trên ~13.6k dòng dữ liệu nông nghiệp tiếng Việt thật, nhãn sinh bởi Qwen3-VL và đã qua 1 vòng audit/sửa lỗi nhãn tự động (dùng bằng chứng tần suất từ trong corpus, không chỉ tin dự đoán của 1 model). Pipeline chuẩn bị dataset + script fine-tune nằm ở `../OCR/` (repo cha) — xem `OCR/README.md`, `OCR/FINETUNE_VIETOCR.md`, `OCR/FINETUNE_PLAN_DATA_FIX.md`.

---
**Bản quyền:** Dựa trên lõi mã nguồn mở MinerU (Opendatalab). VietOCR được phát triển bởi pbcquoc.

# MinerU - Phiên bản Tích hợp VietOCR (Dành riêng cho Tiếng Việt)

## 🌟 Giới thiệu
Đây là bản *fork* (phân nhánh) từ dự án **MinerU** nguyên bản (nhánh `vietocr-integration`). Bản fork này mang trong mình toàn bộ sức mạnh trích xuất cấu trúc văn bản, bảng biểu, công thức toán học xuất sắc của MinerU, nhưng được "cấy ghép" thêm mô hình **VietOCR** để giải quyết triệt để bài toán rớt dấu, nhận sai ký tự khi xử lý tài liệu Tiếng Việt in-the-wild (bao bì, chứng từ, scan mờ).

Toàn bộ tích hợp nằm **độc lập ngay trong repo này** (`mineru/model/ocr/pytorch_paddle.py`) — không phụ thuộc vào bất kỳ fork PaddleOCR nào khác.

## 💡 Vấn đề và Giải pháp

### 1. Rớt dấu do Mô hình Nhận diện (PaddleOCR)
MinerU gốc dùng chung 1 model nhận diện ("latin") cho tất cả các ngôn ngữ Latin (Pháp, Đức, Tây Ban Nha... và cả tiếng Việt) — dù ở phiên bản PP-OCRv4 hay v5. Model gộp chung này không phủ hết bộ dấu thanh phức tạp của tiếng Việt, dẫn tới lỗi rớt dấu/nhận sai ký tự thường xuyên.
* **Giải pháp:** Can thiệp trực tiếp vào `mineru/model/ocr/pytorch_paddle.py` (class `PytorchPaddleOCR`, cả 2 nhánh `ocr()` và `__call__()`), chèn **VietOCR** (`vgg_transformer`) vào bước nhận diện (`rec`), thay cho model "latin" gộp chung. PaddleOCR vẫn đảm nhiệm bước phát hiện vùng chữ (`det`) như bình thường.
* Về hiệu năng nhận diện: dùng `predict_batch()` của VietOCR (không phải vòng lặp `predict()` từng ô một). Đã benchmark thực tế và xác nhận `predict_batch` **nhanh hơn ~1.8 lần** so với vòng lặp ở quy mô thực tế (hàng trăm crop/lần chạy), đồng thời **không gây méo khung/giảm độ chính xác** — đã verify bằng cách so khớp từng ký tự đầu ra giữa 2 cách.

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
| GPU (cu128) + `predict_batch` | ~4.5s/trang (quy mô lớn, 1 file dài hoặc nhiều file gộp 1 lần chạy) |
| GPU + `predict_batch` + gộp nhiều file nhỏ vào 1 process | thêm ~1.3-2x tùy số trang/file (file càng ngắn, gộp càng lợi) |

Lưu ý: vùng con dấu/logo/chữ trang trí phức tạp vẫn là điểm yếu chung của mọi engine OCR (không riêng VietOCR).

---
**Bản quyền:** Dựa trên lõi mã nguồn mở MinerU (Opendatalab). VietOCR được phát triển bởi pbcquoc.

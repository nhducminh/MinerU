# MinerU - Phiên bản Tích hợp VietOCR (Dành riêng cho Tiếng Việt)

## 🌟 Giới thiệu
Đây là bản *fork* (phân nhánh) từ dự án **MinerU 3.1.0** nguyên bản. Bản fork này mang trong mình toàn bộ sức mạnh trích xuất cấu trúc văn bản, bảng biểu, công thức toán học xuất sắc của MinerU, nhưng được "cấy ghép" thêm mô hình **VietOCR** để giải quyết triệt để bài toán rớt dấu, nhận sai ký tự khi xử lý tài liệu Tiếng Việt in-the-wild (bao bì, chứng từ, scan mờ).

## 💡 Vấn đề và Giải pháp

### 1. Rớt dấu do Mô hình Nhận diện (PaddleOCR)
MinerU gốc sử dụng PaddleOCR cho tiếng Việt (tùy chọn `-l vi`). Mô hình này rất yếu với văn bản Tiếng Việt có độ phức tạp cao, phông chữ lạ hoặc nhiễu sáng.
* **Giải pháp:** Can thiệp sâu vào `pytorch_paddle.py` (tại cả 2 nhánh `det=True` và `det=False`), chặn hoàn toàn mô hình nhận diện cũ và chèn thẳng **VietOCR** (với trọng số `vgg_transformer`) vào tiến trình suy luận (inference). 
* Để khắc phục lỗi méo chữ, thuật toán đã được thay đổi từ `predict_batch` (nhận diện theo cục, dễ gây bóp méo khung chữ) sang vòng lặp `predict` (nhận diện từng ô chữ một).

### 2. File PDF có Text Layer bị Lỗi Font
Với các file PDF được xuất từ MS Word có định dạng phông chữ lạ, lớp văn bản ẩn (text layer) thường bị vỡ dấu (Ví dụ: "QUYẾT ĐỊNH" thành "QUYT ĐNH"). MinerU mặc định sẽ bốc lớp text này ra thay vì dùng OCR, khiến kết quả sai hoàn toàn.
* **Giải pháp:** Sử dụng tệp thực thi `mineru_viet.py` thay cho `mineru.exe`. Tệp lệnh này tự động "ép phẳng" (flatten) các file PDF bằng cách render toàn bộ 100% trang thành ảnh (300 DPI) để phá hủy lớp text layer hỏng, ép hệ thống phải dùng nhãn lực VietOCR để đọc lại từ đầu một cách hoàn hảo.

## 🚀 Cài đặt
1. Cài đặt các gói phụ thuộc (Đã bao gồm `vietocr` và `PyMuPDF` trong `pyproject.toml`):
   ```bash
   pip install -e .[all]
   ```
2. (Tùy chọn) Máy sẽ tự động tải trọng số VietOCR (`vgg_transformer.pth`) trong lần chạy đầu tiên.

## 💻 Sử dụng
Sử dụng tệp lệnh `mineru_viet.py` để chạy (Tự động hỗ trợ quét file lẻ và quét nguyên một thư mục):

```bash
python mineru_viet.py -p "đường_dẫn_đến_thư_mục_hoặc_file_PDF_JPG" -o "thư_mục_đầu_ra"
```

*Tất cả cấu hình như `-l vi` và `-m ocr` đều đã được tự động áp dụng bên trong script.*

---
**Bản quyền:** Dựa trên lõi mã nguồn mở MinerU (Opendatalab). VietOCR được phát triển bởi pbcquoc.

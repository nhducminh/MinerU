# VietOCR / MinerU perf & correctness checklist (chưa xong, để kiểm tra tiếp)

Trạng thái tính đến 2026-07-24. Context đầy đủ nằm trong lịch sử chat; đây chỉ là danh sách việc còn treo.

## Đã xong (đã verify)
- [x] Fix OOM `predict_batch()` (bucket-concat không giới hạn size) → viết
      `mineru/model/ocr/vietocr_fast_batch.py`: `predict_batch_grouped()` +
      `_translate_early_exit()` (early-exit decode + gom theo độ dài dự kiến +
      sub-batch 128). Verify: 0/2000 mismatch so với `predict()` gốc.
- [x] Benchmark cô lập (chỉ load VietOCR, không model khác): 7.9x nhanh hơn
      (141.4ms/crop → 17.9ms/crop @ sub_batch=128).
- [x] Chạy full pipeline thật (`mineru_viet.py` trên "Mắc ca.pdf", 105 trang):
      không còn OOM, dấu tiếng Việt đúng 100% xuyên suốt.
- [x] **Bottleneck CPU hay GPU trong pipeline thật?** Đo ban đầu (gpu_kernel
      88%) khiến tưởng là tranh chấp SM với model khác — **sai, đã đính
      chính**: pipeline xử lý tuần tự, không có request nào chạy song song.
      Nguyên nhân thật: `forward_decoder()` không có KV-cache, self-attention
      tính lại toàn bộ prefix mỗi bước → chi phí tăng theo prefix_len (xác
      nhận bằng `MINERU_VIETOCR_STEP_DEBUG=1`). Chi tiết:
      `VIETOCR_BENCHMARK_RESULTS.md` mục 7.
- [x] **PaddleOCR recognizer (`self.text_recognizer`) có nên bỏ không?** →
      Quyết định: bỏ hẳn, thay confidence bằng softmax probability tự tính từ
      VietOCR. Đã code (`vietocr_fast_batch.py` + cả 2 call site trong
      `pytorch_paddle.py`), verify 0/200 mismatch, chạy full pipeline xác nhận
      tiết kiệm 177.14s tính toán thừa, dấu vẫn đúng 100%. Chi tiết:
      `VIETOCR_BENCHMARK_RESULTS.md` mục 5.
- [x] **KV-cache cho decode loop** — viết lại thủ công
      `TransformerDecoderLayer.forward` kiểu incremental (self-attn cache +
      cross-attn precompute) trong `vietocr_fast_batch.py`. Verify 0/300 +
      0/100 mismatch. Full pipeline: nhanh thêm 3x (batch 2173/1646), gpu_kernel
      giảm 4.1x, dấu đúng 100%. Chi tiết: `VIETOCR_BENCHMARK_RESULTS.md` mục 8.
      **Lưu ý bảo trì**: hard-code `norm_first=False` + kiến trúc 6-layer hiện
      tại — nếu fine-tune lại VietOCR với kiến trúc khác phải sửa lại code này
      (mục 9 trong file benchmark).

## Đang treo — cần làm tiếp

- [ ] `py_glue` (overhead Python: tensor build + index_select × 6 layer mỗi
      bước) giờ chiếm 36% thời gian decode (trước chỉ 12%, vì gpu_kernel đã
      giảm mạnh) — bottleneck tiếp theo nếu muốn tối ưu thêm, chưa xử lý.
- [ ] MinerU Docker deployment: đã hủy do mạng quá chậm (~170KB/s, base image
      vllm/vllm-openai ~15GB+). Đang dùng trực tiếp MinerU local (Windows venv,
      gọi qua WSL interop `.venv/Scripts/python.exe -X utf8 mineru_viet.py -p
      "D:\..." -o "D:\..."`, dùng path kiểu Windows). Nếu sau này mạng tốt hơn
      và muốn thử lại Docker: Dockerfile đã sửa sẵn ở
      `docker/global/Dockerfile` để build từ source (có patch VietOCR) thay vì
      pip install mineru gốc từ PyPI — cần `.dockerignore` đã tạo sẵn ở
      `MinerU/.dockerignore`. `docker/compose.yaml` service `mineru-api` đã đổi
      port host sang `2345:8000`.

import argparse
import os
import shutil
import subprocess
import sys
import fitz  # PyMuPDF
from PIL import Image

VALID_EXTS = {'.pdf', '.pptx', '.docx', '.xlsx', '.jpg', '.jpeg', '.png'}


def flatten_pdf(input_pdf_path, output_pdf_path):
    print(f"[MinerU-Viet] Đang ép phẳng PDF để xóa text layer ảo: {os.path.basename(input_pdf_path)}")
    try:
        pdf_document = fitz.open(input_pdf_path)
        images = []
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            # Render ở 300 DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        if images:
            images[0].save(
                output_pdf_path, "PDF", resolution=300.0, save_all=True, append_images=images[1:]
            )
            return True
    except Exception as e:
        print(f"[MinerU-Viet] Lỗi ép phẳng: {e}")
    return False


def collect_input_files(input_path):
    if os.path.isdir(input_path):
        found = []
        for root, _, files in os.walk(input_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in VALID_EXTS:
                    found.append(os.path.join(root, f))
        return found
    return [input_path]


def stage_for_batch(filepaths, staging_dir):
    """Ép phẳng PDF + copy các file khác vào 1 thư mục phẳng, đặt tên tránh trùng
    (nhiều file cùng tên ở các thư mục con khác nhau) để mineru xử lý cả lô
    trong 1 process duy nhất (tránh trả "thuế" cold-start model cho từng file)."""
    os.makedirs(staging_dir, exist_ok=True)
    seen_names = {}
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        stem, ext = os.path.splitext(filename)
        count = seen_names.get(filename, 0)
        seen_names[filename] = count + 1
        staged_name = filename if count == 0 else f"{stem}__{count}{ext}"
        staged_path = os.path.join(staging_dir, staged_name)

        if ext.lower() == '.pdf':
            flatten_pdf(filepath, staged_path)
        else:
            shutil.copy2(filepath, staged_path)


def main():
    parser = argparse.ArgumentParser(description="MinerU kết hợp VietOCR chuyên xử lý tiếng Việt")
    parser.add_argument("-p", "--path", required=True, help="Đường dẫn đến file hoặc thư mục cần xử lý")
    parser.add_argument("-o", "--output", required=True, help="Thư mục chứa kết quả xuất ra")
    # Các tham số khác
    args, unknown_args = parser.parse_known_args()

    input_path = os.path.abspath(args.path)
    output_dir = os.path.abspath(args.output)

    # mineru.exe của chính repo này (pipeline backend + VietOCR patch), không còn phụ thuộc Mely-PDF-Miner
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mineru_exe = os.path.join(script_dir, ".venv", "Scripts", "mineru.exe")

    os.makedirs(output_dir, exist_ok=True)

    filepaths = collect_input_files(input_path)
    if not filepaths:
        print("[MinerU-Viet] Không tìm thấy file hợp lệ để xử lý.")
        return

    staging_dir = os.path.join(output_dir, "temp_batch_staging")
    stage_for_batch(filepaths, staging_dir)

    print(f"[MinerU-Viet] Đang trích xuất {len(filepaths)} file trong 1 lần chạy (model chỉ load 1 lần)...")
    cmd = [mineru_exe, "-p", staging_dir, "-o", output_dir, "-m", "ocr", "-b", "pipeline", "-l", "vi"] + unknown_args
    try:
        subprocess.run(cmd, check=True)
        print(f"[MinerU-Viet] Thành công: {len(filepaths)} file.")
    except subprocess.CalledProcessError:
        print("[MinerU-Viet] Lỗi khi xử lý batch.")

    shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

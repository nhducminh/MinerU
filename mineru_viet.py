import argparse
import os
import subprocess
import sys
import fitz  # PyMuPDF
from PIL import Image

def flatten_pdf(input_pdf_path, output_pdf_path):
    print(f"[MinerU-Viet] Đang ép phẳng PDF để xóa text layer ảo: {os.path.basename(input_pdf_path)}")
    try:
        pdf_document = fitz.open(input_pdf_path)
        images = []
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            # Render ở 300 DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
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

def main():
    parser = argparse.ArgumentParser(description="MinerU kết hợp VietOCR chuyên xử lý tiếng Việt")
    parser.add_argument("-p", "--path", required=True, help="Đường dẫn đến file hoặc thư mục cần xử lý")
    parser.add_argument("-o", "--output", required=True, help="Thư mục chứa kết quả xuất ra")
    # Các tham số khác
    args, unknown_args = parser.parse_known_args()

    input_path = os.path.abspath(args.path)
    output_dir = os.path.abspath(args.output)
    
    # Tìm file thực thi mineru
    mineru_exe = "mineru"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Hàm xử lý 1 file
    def process_file(filepath):
        filename = os.path.basename(filepath)
        process_filepath = filepath
        temp_flattened = None
        
        if filename.lower().endswith('.pdf'):
            temp_dir = os.path.join(output_dir, "temp_flattened")
            os.makedirs(temp_dir, exist_ok=True)
            temp_flattened = os.path.join(temp_dir, filename)
            if flatten_pdf(filepath, temp_flattened):
                process_filepath = temp_flattened
                
        # Gọi mineru gốc với force ocr và ngôn ngữ vi
        cmd = [mineru_exe, "-p", process_filepath, "-o", output_dir, "-m", "ocr", "-b", "pipeline", "-l", "vi"] + unknown_args
        print(f"[MinerU-Viet] Đang trích xuất: {filename}...")
        try:
            subprocess.run(cmd, check=True)
            print(f"[MinerU-Viet] Thành công: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"[MinerU-Viet] Lỗi khi xử lý: {filename}")
        
        # Xóa file rác
        if temp_flattened and os.path.exists(temp_flattened):
            os.remove(temp_flattened)

    # Nếu là thư mục
    if os.path.isdir(input_path):
        valid_exts = {'.pdf', '.pptx', '.docx', '.jpg', '.jpeg', '.png'}
        for root, _, files in os.walk(input_path):
            for f in files:
                if any(f.lower().endswith(ext) for ext in valid_exts):
                    process_file(os.path.join(root, f))
    else:
        process_file(input_path)

if __name__ == "__main__":
    main()

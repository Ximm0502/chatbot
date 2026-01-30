# Hướng dẫn sử dụng LightOnOCR-2-1B với vLLM

## Bước 1: Tạo & Kích hoạt Virtual Environment

```bash
# Tạo virtual env
python -m venv lightocr-env

# Kích hoạt env
source lightocr-env/bin/activate

# Kiểm tra (thấy (lightocr-env) ở đầu dòng)
which python

```

## Bước 2: Cài đặt Dependencies

```bash
pip install vllm 

```
## Bước 3: Start vLLM Server

Mở terminal và chạy:

```bash
vllm serve lightonai/LightOnOCR-2-1B \
    --limit-mm-per-prompt '{"image": 1}' \
    --mm-processor-cache-gb 0 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.25
```

Server sẽ:
- Tải model từ HuggingFace (lần đầu tiên)
- Khởi động tại `http://localhost:8000/v1/chat/completions`
- Hiển thị "Application startup complete" khi sẵn sàng

## Kiểm tra Server

Kiểm tra xem server đã chạy chưa:

```bash
curl http://localhost:8000/v1/models
```

Nếu thấy response JSON với model name thì server đã sẵn sàng.

## Lưu ý

- Server cần ~4-6GB VRAM
- Lần đầu tiên sẽ tải model (~2-3GB)
- Giữ server chạy trong suốt quá trình OCR
- Dừng server: Ctrl+C

## Bước 5: Chạy OCR Script

Mở terminal MỚI (giữ server chạy ở terminal cũ) và chạy:

```bash
python batch_pdf_ocr.py
```
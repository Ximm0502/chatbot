# Hướng dẫn sử dụng LightOnOCR-2-1B với vLLM

## Bước 1: Cài đặt vLLM

```bash
pip install vllm
```

## Bước 2: Start vLLM Server

Mở terminal và chạy:

```bash
vllm serve lightonai/LightOnOCR-2-1B --trust-remote-code --dtype float16
```

Server sẽ:
- Tải model từ HuggingFace (lần đầu tiên)
- Khởi động tại `http://localhost:8000`
- Hiển thị "Application startup complete" khi sẵn sàng

## Bước 3: Chạy OCR Script

Mở terminal MỚI (giữ server chạy ở terminal cũ) và chạy:

```bash
python test-lightocr.py
```

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

import base64
import requests
from PIL import Image
import time
import os

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "lightonai/LightOnOCR-2-1B"

IMAGE_FILE = "test.png"

def main():
    print("ocr_image.py - vLLM API")
    print("=" * 70)
    
    if not os.path.exists(IMAGE_FILE):
        print(f"{IMAGE_FILE} not found!")
        return
    
    print(f" Loading {IMAGE_FILE}...")
    image = Image.open(IMAGE_FILE)
    print(f" Image size: {image.size}")
    
    with open(IMAGE_FILE, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(" OCR...", end=" ")
    t1 = time.time()

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                },
                {
                    "type": "text",
                    "text": "Extract all text from this document and convert to markdown format."
                }
            ]
        }],
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    
    response = requests.post(ENDPOINT, json=payload)
    ocr_time = time.time() - t1
    
    text = response.json()['choices'][0]['message']['content']
    
    print(f"{ocr_time:.1f}s | {len(text)} chars")
    print("-" * 60)
    print(text) 
    print("-" * 60)
    
    output_file = os.path.splitext(IMAGE_FILE)[0] + "_output.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# OCR Result\n\n{IMAGE_FILE} | {ocr_time:.1f}s\n\n{text}")
    
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()

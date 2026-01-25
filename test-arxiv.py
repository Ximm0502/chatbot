#!/usr/bin/env python3
"""Test LightOnOCR với PDF từ arXiv (như ví dụ chính thức)"""

import base64
import requests
import pypdfium2 as pdfium
import io

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "lightonai/LightOnOCR-2-1B"

# Download PDF từ arXiv
print("📥 Downloading PDF from arXiv...")
pdf_url = "https://arxiv.org/pdf/2412.13663"
pdf_data = requests.get(pdf_url).content

# Open PDF và convert trang đầu
pdf = pdfium.PdfDocument(pdf_data)
page = pdf[0]
pil_image = page.render(scale=2.77).to_pil()

print(f"📄 Image size: {pil_image.size}")

# Convert to base64
buffer = io.BytesIO()
pil_image.save(buffer, format="PNG")
image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Make request
print("🔄 Sending to vLLM...")
payload = {
    "model": MODEL,
    "messages": [{
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        }]
    }],
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
}

response = requests.post(ENDPOINT, json=payload)
text = response.json()['choices'][0]['message']['content']

print("\n" + "="*70)
print("RESULT:")
print("="*70)
print(text)
print("="*70)
print(f"\n✅ Total chars: {len(text)}")

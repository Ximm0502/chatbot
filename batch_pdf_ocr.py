#!/usr/bin/env python3
"""
batch_pdf_ocr.py - Process multiple PDFs in a folder
"""

import base64
import requests
import pypdfium2 as pdfium
import io
import time
import os
from pathlib import Path

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "lightonai/LightOnOCR-2-1B"

# ⚙️ CONFIG - Thay đổi folder ở đây
PDF_FOLDER = "./pdfs"
OUTPUT_FOLDER = "./output"  # Folder lưu kết quả

def process_pdf(pdf_path):
    """Process single PDF file"""
    print(f"\n{'='*70}")
    print(f"Processing: {pdf_path}")
    print('='*70)
    
    pdf = pdfium.PdfDocument(pdf_path)
    print(f"Pages: {len(pdf)}")
    
    results = []
    
    for i in range(len(pdf)):
        print(f"\nPage {i+1}/{len(pdf)}...", end=" ")
        page = pdf[i]
        
        pil_image = page.render(scale=2.77).to_pil()
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
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
        
        results.append({
            'page': i+1,
            'time': ocr_time,
            'text': text
        })
        
        pil_image.close()
    
    # Save results
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_filename = Path(pdf_path).stem + "_output.md"
    output_file = os.path.join(OUTPUT_FOLDER, output_filename)
    output_content = f"# OCR Results: {Path(pdf_path).name}\n\n"
    
    for r in results:
        output_content += f"## Page {r['page']}\n\n"
        output_content += r['text'] + "\n\n---\n\n"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)
    
    total_time = sum(r['time'] for r in results)
    print(f"\n✅ Saved: {output_file}")
    print(f"   Total: {total_time:.1f}s | Avg: {total_time/len(results):.1f}s/page")
    
    return len(results), total_time

def main():
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ Folder not found: {PDF_FOLDER}")
        return
    
    # Find all PDF files
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {PDF_FOLDER}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    total_pages = 0
    total_time = 0
    
    for pdf_file in pdf_files:
        pages, time_taken = process_pdf(str(pdf_file))
        total_pages += pages
        total_time += time_taken
    
    print(f"\n{'='*70}")
    print("📊 BATCH SUMMARY")
    print(f"Files: {len(pdf_files)} | Pages: {total_pages} | Time: {total_time:.1f}s")
    print(f"Avg: {total_time/total_pages:.1f}s/page")
    print('='*70)

if __name__ == "__main__":
    main()

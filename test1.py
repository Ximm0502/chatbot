import base64
import requests
import pypdfium2 as pdfium
import io
import time
import os

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "lightonai/LightOnOCR-2-1B"

PDF_FILE = "test2.pdf"

def main():
    print("test-lightocr.py - vLLM API")
    print("=" * 70)
    
    if not os.path.exists(PDF_FILE):
        print(f"{PDF_FILE} not found!")
        return
    
    print(f" Loading {PDF_FILE}...")
    pdf = pdfium.PdfDocument(PDF_FILE)
    print(f" {len(pdf)} pages")
    
    results = []
    num_pages = len(pdf)  
    
    for i in range(num_pages):
        print(f"\nPAGE {i+1}/{len(pdf)}")
        page = pdf[i]
        
        pil_image = page.render(scale=2.77).to_pil()
        print(f"   Image: {pil_image.size}")
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print("   OCR...", end=" ")
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
        
        results.append({
            'page': i+1, 'time': ocr_time, 
            'chars': len(text), 'text': text
        })
        
        pil_image.close()
    
    print("\n" + "="*70)
    total_time = sum(r['time'] for r in results)
    print("SUMMARY")
    print(f"Pages: {len(results)} | Total: {total_time:.1f}s")
    print(f"Avg: {total_time/len(results):.1f}s/page ⚡")
    
    output_file = PDF_FILE.replace(".pdf", "_output.md")
    output_content = "# OCR Results\n\n"
    output_content += f"{PDF_FILE} | {len(results)} pages | {total_time:.1f}s\n\n"
    
    for r in results:
        output_content += f"## Page {r['page']} ({r['time']:.1f}s)\n\n"
        output_content += r['text'] + "\n\n---\n"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)
    
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()

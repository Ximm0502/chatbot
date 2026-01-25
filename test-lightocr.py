#!/usr/bin/env python3
"""
LightOnOCR-2-1B + vLLM BENCHMARK
LOCAL PDF FILE + Thời gian xử lý từng page
RTX PRO 4000 Blackwell 24GB x2
"""

import time
import pypdfium2 as pdfium
import io
import base64
from PIL import Image
from pathlib import Path
from vllm import LLM, SamplingParams
import torch
import gc
import argparse
import sys

# vLLM config
MODEL_NAME = "lightonai/LightOnOCR-2-1B"
LLM_ENGINE = None

def init_vllm():
    """Khởi tạo vLLM"""
    global LLM_ENGINE
    
    print("🚀 Starting vLLM LightOnOCR-2-1B...")
    start_time = time.time()
    
    LLM_ENGINE = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,  # 20GB của 24GB
        max_model_len=4096,
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        disable_log_stats=True
    )
    
    load_time = time.time() - start_time
    print(f"✅ vLLM loaded: {load_time:.1f}s")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.memory_reserved(0)/1024**3:.1f}GB")

def pdf_to_base64_pages(pdf_path, max_pages=None):
    """Convert LOCAL PDF to base64 images"""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    print(f"📄 Loading PDF: {pdf_path.name}")
    print(f"📁 Size: {pdf_path.stat().st_size/1024**2:.1f}MB")
    
    pdf = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf)
    
    if max_pages:
        total_pages = min(total_pages, max_pages)
    
    print(f"📄 Total pages: {total_pages}")
    
    base64_pages = []
    for i in range(total_pages):
        print(f"🔄 Converting page {i+1}/{total_pages}...", end=" ")
        
        page = pdf[i]
        pil_image = page.render(scale=2.77).to_pil()  # 200 DPI
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG", optimize=True)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        base64_pages.append({
            "page": i+1,
            "image": f"data:image/png;base64,{base64_image}",
            "size": pil_image.size,
            "width": pil_image.width,
            "height": pil_image.height
        })
        
        pil_image.close()
        print(f"({pil_image.width}x{pil_image.height})")
    
    pdf.close()
    return base64_pages

def benchmark_vllm_pdf(pdf_path, max_pages=None):
    """Benchmark LOCAL PDF với vLLM"""
    print("\n" + "="*80)
    print("LIGHTONOCR-2-1B + VLLM BENCHMARK")
    print(f"PDF: {Path(pdf_path).name}")
    print("="*80)
    
    # Convert PDF pages
    pages = pdf_to_base64_pages(pdf_path, max_pages)
    
    # Prompts xoay vòng
    prompts = [
        "Convert this document page to clean markdown.",
        "Extract all text from this document page in markdown format.",
        "Convert this document page to structured markdown."
    ]
    
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=4096,
        stop=["<|im_end|>", "\n\n\n", "### END"]
    )
    
    results = []
    
    print(f"\n🔥 vLLM OCR: {len(pages)} pages")
    print("-" * 80)
    
    for page_data in pages:
        page_num = page_data["page"]
        
        print(f"\n📄 PAGE {page_num:2d} ({page_data['width']}x{page_data['height']})")
        
        # OpenAI-style message
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": page_data["image"]}
                },
                {
                    "type": "text", 
                    "text": prompts[(page_num-1) % len(prompts)]
                }
            ]
        }]
        
        # vLLM chat template
        prompt = LLM_ENGINE.llm_engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Benchmark inference
        torch.cuda.synchronize()  # Sync GPU
        start_time = time.time()
        
        outputs = LLM_ENGINE.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        result = outputs[0].outputs[0].text.strip()
        
        print(f"⏱️   Time: {inference_time:.2f}s")
        print(f"📝   Output: {len(result)} chars")
        print(f"💾   VRAM: {torch.cuda.memory_allocated(0)/1024**3:.1f}GB")
        
        # Preview first 200 chars
        preview = result[:200].replace('\n', ' ').strip()
        print(f"📄 Preview: {preview}...")
        
        results.append({
            "page": page_num,
            "time": inference_time,
            "chars": len(result),
            "vram": torch.cuda.memory_allocated(0)/1024**3,
            "text": result,
            "preview": preview
        })
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # SUMMARY
    total_time = sum(r["time"] for r in results)
    avg_time = total_time / len(results)
    total_chars = sum(r["chars"] for r in results)
    
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Page':<4} {'Time':<8} {'Chars':<8} {'Speed':<10} {'Preview'}")
    print("-"*80)
    
    for r in results:
        speed_chars_sec = r["chars"] / r["time"] if r["time"] > 0 else 0
        print(f"{r['page']:<4} {r['time']:<7.2f}s {r['chars']:<7} {speed_chars_sec:<9.0f} {r['preview'][:50]}")
    
    print("\n📈 SUMMARY")
    print(f"Pages:      {len(results)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg/page:   {avg_time:.2f}s ⚡")
    print(f"Total chars:{total_chars:,}")
    print(f"Throughput: {len(results)/total_time*60:.1f} pages/min")
    print(f"Peak VRAM:  {max(r['vram'] for r in results):.1f}GB")
    
    # Save detailed results
    output_file = Path(pdf_path).stem + "_vllm_results.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# vLLM + LightOnOCR-2-1B Results\n")
        f.write(f"**PDF:** {Path(pdf_path).name}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Summary:** {len(results)} pages, {total_time:.1f}s total, {avg_time:.2f}s/page\n\n")
        
        for r in results:
            f.write(f"\n## Page {r['page']} ({r['time']:.2f}s)\n")
            f.write(f"**Characters:** {r['chars']:,} | **Speed:** {r['chars']/r['time']:.0f} chars/s\n\n")
            f.write(r['text'] + "\n\n---\n")
    
    print(f"\n✅ Detailed results: {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="vLLM LightOnOCR Benchmark")
    parser.add_argument("pdf_path", help="Path to local PDF file")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to process")
    
    args = parser.parse_args()
    
    # GPU info
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")
    
    # Init vLLM
    init_vllm()
    
    # Benchmark
    results = benchmark_vllm_pdf(args.pdf_path, args.max_pages)
    
    print("\n🎉 PRODUCTION READY!")
    print(f"⚡ {len(results)/sum(r['time'] for r in results)*60:.1f} pages/min")

if __name__ == "__main__":
    main()

from transformers import AutoModel, AutoTokenizer
import torch
import os
import requests
from pdf2image import convert_from_path
import tempfile
import re
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import logging
import sys
from io import StringIO

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Use GPU if available and compatible, otherwise fall back to CPU
if torch.cuda.is_available():
    # Check GPU compute capability
    gpu_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{gpu_props.major}.{gpu_props.minor}"
    
    # CUDA 11.8 supports compute capability 3.5 and higher
    # CUDA 12.x requires compute capability 7.0 and higher
    cuda_version = torch.version.cuda
    min_capability = 3.5 if cuda_version and cuda_version.startswith('11') else 7.0
    
    if float(compute_capability) < min_capability:
        print(f"⚠️  GPU {torch.cuda.get_device_name(0)} has CUDA capability {compute_capability}")
        print(f"⚠️  Current PyTorch {torch.__version__} with CUDA {cuda_version} requires capability >= {min_capability}")
        print(f"⚠️  Falling back to CPU...")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {cuda_version}")
        print(f"   Compute capability: {compute_capability}")
        print(f"   GPU Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
else:
    device = torch.device('cpu')
    print(f"Using device: {device} (GPU not available)")

model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model (this may take a while)...")
# Load model with optimal settings for GPU
if device.type == 'cuda':
    # Use bfloat16 for better compatibility on GPU
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
else:
    # Use float32 for CPU - force all components to CPU
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        use_safetensors=True,
        device_map={'': 'cpu'},
    )
model = model.eval()

print("Model loaded successfully")

# Configuration
USE_GROUNDING = True  # Set to False for plain text without bounding boxes
SAVE_BBOX_IMAGES = True  # Save images with bounding boxes drawn
SAVE_PAGE_IMAGES = True  # Save original page images

if USE_GROUNDING:
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
else:
    prompt = "<image>\nFree OCR. "

# Download PDF
pdf_url = 'https://novenyzetiterkep.hu/sites/novenyzetiterkep.hu/files/ANER%20016%20C23.pdf'
print("Downloading PDF...")
response = requests.get(pdf_url)
with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
    tmp_file.write(response.content)
    pdf_path = tmp_file.name

# Convert PDF to images
print("Converting PDF to images...")
images = convert_from_path(pdf_path, dpi=150)  # Higher DPI for better quality
print(f"PDF has {len(images)} pages")

# Create output directory
output_dir = './output_enhanced'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/images', exist_ok=True)

# Initialize markdown content
markdown_content = f"# OCR Results\n\n"
markdown_content += f"**Source:** {pdf_url}\n\n"
markdown_content += f"**Total Pages:** {len(images)}\n\n"
markdown_content += "---\n\n"

def parse_grounding_output(content):
    """
    Parse the grounding OCR output to extract text content.

    Args:
        content (str): Raw OCR output with grounding tokens

    Returns:
        list: List of dictionaries with 'text' and 'bbox' keys
    """
    text_blocks = []

    # Pattern to match: <|ref|>type<|/ref|><|det|>[[coords]]<|/det|> followed by content
    # The content may have newlines before it
    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[([\d\s,]+)\]\]<\|/det\|>\s*(.*?)(?=<\|ref\||$)'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    for ref_type, coords_str, text in matches:
        text = text.strip()
        ref_type = ref_type.strip()
        
        # Parse coordinates
        bbox = None
        try:
            coords_list = [int(x.strip()) for x in coords_str.split(',')]
            if len(coords_list) == 4:
                bbox = coords_list
        except:
            pass
        
        # Keep image blocks even if they have no text (we'll extract the image region)
        # # Skip only if it's not an image and has no text
        # if not text and ref_type not in ['image', 'figure', 'chart', 'table']:
        #     continue

        text_blocks.append({
            'text': text if text else '',  # Empty string for images without captions
            'bbox': bbox,
            'type': ref_type  # e.g., 'text', 'image', 'image_caption', 'title', etc.
        })

    return text_blocks

def draw_bboxes(image, bboxes, output_path):
    """Draw bounding boxes on image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for i, bbox_info in enumerate(bboxes):
        bbox = bbox_info['bbox']
        if not bbox:
            continue
        # Draw rectangle
        draw.rectangle(bbox, outline='red', width=2)
        # Draw text index
        draw.text((bbox[0], bbox[1]-15), f"{i+1}", fill='red')
    
    img_copy.save(output_path, 'PNG')
    return output_path

def extract_image_regions(image, parsed_blocks, page_output_dir):
    """Extract and save image regions from the page"""
    images_dir = f'{page_output_dir}/images'
    os.makedirs(images_dir, exist_ok=True)
    
    extracted_images = []
    image_counter = 0
    
    for block in parsed_blocks:
        if block['type'] in ['image', 'figure', 'chart', 'table'] and block['bbox']:
            # Crop the image region
            bbox = block['bbox']
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            
            cropped = image.crop((x1, y1, x2, y2))
            
            # Save the cropped image
            img_filename = f'{image_counter}.jpg'
            img_path = f'{images_dir}/{img_filename}'
            cropped.save(img_path, 'JPEG', quality=95)
            
            extracted_images.append({
                'filename': img_filename,
                'path': img_path,
                'bbox': bbox,
                'type': block['type']
            })
            
            logger.info(f"Extracted {block['type']} region: {img_filename}")
            image_counter += 1
    
    return extracted_images


# Process all pages on GPU, or limit to 1 on CPU for testing
max_pages = len(images)
print(f"Processing {max_pages} page(s) on {device.type}")
if device.type == 'cuda':
    print(f"GPU acceleration enabled - processing will be significantly faster")

all_results = []

# Process each page
for page_idx in range(min(max_pages, len(images))):
    print(f"\nProcessing page {page_idx + 1}/{len(images)}...")
    
    image = images[page_idx]
    
    # Save original page image if requested
    if SAVE_PAGE_IMAGES:
        page_image_path = f'{output_dir}/images/page_{page_idx + 1}.png'
        image.save(page_image_path, 'PNG')
        print(f"  💾 Saved page image: {page_image_path}")
    
    # Save to temp file for OCR
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as img_tmp:
        image.save(img_tmp.name, 'JPEG', quality=95)
        image_file = img_tmp.name
    
    page_output_dir = f'{output_dir}/page_{page_idx + 1}'
    os.makedirs(page_output_dir, exist_ok=True)
    
    try:
        # Capture stdout to get the model output (since model.infer prints but returns None)
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Use larger sizes for GPU, smaller for CPU
        # NOTE: Model prints to stdout but returns None, so we capture stdout
        if device.type == 'cuda':
            logger.info(f"Running OCR inference on GPU for page {page_idx + 1}")
            res = model.infer(
                tokenizer, 
                prompt=prompt, 
                image_file=image_file, 
                output_path=page_output_dir, 
                base_size=1024,
                image_size=1024, 
                crop_mode=False, 
                save_results=False,
                test_compress=True
            )
        else:
            logger.info(f"Running OCR inference on CPU for page {page_idx + 1}")
            res = model.infer(
                tokenizer, 
                prompt=prompt, 
                image_file=image_file, 
                output_path=page_output_dir, 
                base_size=512, 
                image_size=512, 
                crop_mode=False, 
                save_results=False,
                test_compress=True
            )
        
        # Restore stdout and get the captured text
        sys.stdout = old_stdout
        result_text = captured_output.getvalue()
        
        logger.info(f"Inference completed. Return value type: {type(res)}")
        logger.info(f"Captured output length: {len(result_text)} characters")
        
        # Check if we got any output
        if not result_text or len(result_text.strip()) < 10:
            raise ValueError(f"OCR inference produced no usable output (captured {len(result_text)} chars)")
        
        logger.info(f"Successfully captured OCR output: {len(result_text)} characters")
        
        # Save raw result
        with open(f'{page_output_dir}/raw_result.txt', 'w', encoding='utf-8') as f:
            f.write(result_text)
        
        logger.info(f"Saved raw result to {page_output_dir}/raw_result.txt")
        # Parse and process results
        if USE_GROUNDING:
            parsed = parse_grounding_output(result_text)
            logger.info(f"Parsed {len(parsed)} blocks from page {page_idx + 1}")
            
            # Extract and save image regions
            extracted_images = extract_image_regions(image, parsed, page_output_dir)
            if extracted_images:
                logger.info(f"Extracted {len(extracted_images)} image regions")
            
            # Save parsed results as JSON
            import json
            with open(f'{page_output_dir}/parsed_results.json', 'w', encoding='utf-8') as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved parsed results to {page_output_dir}/parsed_results.json")
            
            # Draw bounding boxes if requested
            if SAVE_BBOX_IMAGES and parsed:
                bbox_image_path = f'{output_dir}/images/page_{page_idx + 1}_bboxes.png'
                draw_bboxes(image, parsed, bbox_image_path)
                logger.info(f"Saved bounding box visualization: {bbox_image_path}")
            
            # Extract clean text
            clean_text = '\n\n'.join([item['text'] for item in parsed])
            logger.info(f"Extracted clean text: {len(clean_text)} characters")
            
            # Post-process: Remove hyphen-space patterns (fix OCR line breaks)
            clean_text = re.sub(r'-\s+', '', clean_text)
            logger.info(f"Post-processed text: {len(clean_text)} characters")
        else:
            clean_text = result_text.replace('<|ref|>', '').replace('<|/ref|>', '').replace('<|det|>', '').replace('<|/det|>', '')
            parsed = []
        
        # Save result.mmd file (clean text without special tokens)
        with open(f'{page_output_dir}/result.mmd', 'w', encoding='utf-8') as f:
            f.write(clean_text)
        
        # Save clean markdown for this page
        with open(f'{page_output_dir}/page.md', 'w', encoding='utf-8') as f:
            f.write(clean_text)
        
        logger.info(f"Saved clean text to {page_output_dir}/page.md and result.mmd")
        
        # Add to combined markdown
        markdown_content += f"## Page {page_idx + 1}\n\n"
        
        # Check for inline images extracted by the model (images within the page content)
        page_images_dir = f'{page_output_dir}/images'
        inline_images = []
        if os.path.exists(page_images_dir):
            inline_images = sorted([f for f in os.listdir(page_images_dir) if f.endswith(('.jpg', '.png'))])
        
        # Add the extracted text content with inline images
        if clean_text.strip() or inline_images:
            # If there are inline images, insert them in the appropriate places
            if inline_images:
                # For now, add images at the beginning of the page content
                for img in inline_images:
                    img_rel_path = f'page_{page_idx + 1}/images/{img}'
                    markdown_content += f"![Inline Image]({img_rel_path})\n\n"
            
            # Add the text content
            if clean_text.strip():
                markdown_content += clean_text + "\n\n"
        else:
            markdown_content += "*No text content extracted*\n\n"
        
        # Add bounding box image if available (for debugging/reference)
        if SAVE_BBOX_IMAGES and USE_GROUNDING and parsed:
            bbox_image_path = f'images/page_{page_idx + 1}_bboxes.png'
            if os.path.exists(f'{output_dir}/{bbox_image_path}'):
                markdown_content += f"\n### Text Region Visualization\n\n"
                markdown_content += f"![Page {page_idx + 1} - Text Regions]({bbox_image_path})\n\n"
        
        markdown_content += "---\n\n"
        
        all_results.append({
            'page': page_idx + 1,
            'text': clean_text,
            'parsed': parsed
        })
        
        print(f"  ✅ Page {page_idx + 1} completed successfully")
        print(f"  📝 Extracted {len(parsed) if USE_GROUNDING else 'N/A'} text blocks")
        logger.info(f"Page {page_idx + 1} processing completed successfully")
        
        # Clear GPU cache after each page to prevent memory buildup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error processing page {page_idx + 1}: {str(e)}", exc_info=True)
        print(f"  ❌ Error processing page {page_idx + 1}: {str(e)}")
        markdown_content += f"## Page {page_idx + 1}\n\n"
        markdown_content += f"**Error:** {str(e)}\n\n"
        markdown_content += "---\n\n"
        
        # Save error info
        with open(f'{page_output_dir}/error.txt', 'w', encoding='utf-8') as f:
            f.write(f"Error: {str(e)}")
    
    # Clean up temp image
    os.unlink(image_file)

# Save combined markdown file
output_md_path = f'{output_dir}/complete_document.md'
with open(output_md_path, 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"\n✅ Processing complete!")
print(f"\n📄 Output files:")
print(f"  - Combined markdown: {output_md_path}")
print(f"  - Page images: {output_dir}/images/")
print(f"  - Per-page details: {output_dir}/page_*/")

# Clean up temp PDF
os.unlink(pdf_path)

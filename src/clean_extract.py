# this code executed in google colab 
import cv2
import numpy as np
from google.colab import files
from docling.document_converter import DocumentConverter

# 2. Upload your image
uploaded = files.upload()
img_name = list(uploaded.keys())[0]

def professional_extract(image_path):
    print("--- Step 1: Professional Image Enhancement ---")
    img = cv2.imread(image_path)

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SUPER-RESOLUTION: Upscale by 3x for tiny technical text
    # This is the secret to not missing small numbers
    enhanced = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # SHARPENING: Makes blurry letters have hard edges
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Save enhanced image for Docling to read
    enhanced_path = "enhanced_for_ai.png"
    cv2.imwrite(enhanced_path, enhanced)

    print("--- Step 2: AI Layout Analysis (Docling) ---")
    # Docling identifies columns and tables automatically
    converter = DocumentConverter()
    result = converter.convert(enhanced_path)

    # Export to Markdown (Preserves tables and lists)
    markdown_data = result.document.export_to_markdown()

    with open("final_perfect_extraction.md", "w", encoding="utf-8") as f:
        f.write(markdown_data)

    print("--- SUCCESS! ---")
    files.download("final_perfect_extraction.md")

professional_extract(img_name)
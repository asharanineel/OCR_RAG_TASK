
import os
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def clean_ocr_text(text):
    """
    Final cleaning of OCR artifacts specifically found in your cleaned file.
    """
    # Fix common word breaks found in your text
    text = text.replace("Hulu dao", "Huludao")
    text = text.replace("cutitting", "outfitting")
    text = text.replace("fist", "first")
    text = text.replace("m mles", "miles")
    text = text.replace("n mles", "n miles")
    
    # Remove random OCR characters like '二' or placeholders
    text = re.sub(r'二', '', text)
    text = re.sub(r'<!-- image -->', '', text)
    
    # Normalize extra spaces but keep Markdown table pipes (|)
    text = re.sub(r' +', ' ', text)
    
    return text

def create_vector_db():
    # 1. LOAD THE CLEANED MD FILE
    # Replace 'cleaned_final_output.md' with your exact filename if different
    input_file = "cleaned_final_output.md"
    
    if not os.path.exists(input_file):
        # Fallback for folder structure
        input_file = "src/cleaned_final_output.md"
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found. Please check the file path.")
            return

    print(f"--- Loading and Cleaning: {input_file} ---")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_content = f.read()

    cleaned_content = clean_ocr_text(raw_content)

    # 2. MARKDOWN HEADER SPLITTING
    # This is CRITICAL for RAG. It attaches the Section name to every chunk.
    headers_to_split_on = [
        ("##", "Section"),
    ]
    
    print("--- Step 1: Semantic Markdown Splitting ---")
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(cleaned_content)

    # 3. RECURSIVE CHARACTER SPLITTING
    # We use a chunk size of 600. This is large enough to hold a few table rows 
    # but small enough to be precise for retrieval.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=60,
        separators=["\n\n", "\n", "|", " ", ""]
    )
    
    final_docs = text_splitter.split_documents(md_header_splits)
    print(f"Generated {len(final_docs)} chunks for the Vector DB.")

    # 4. GENERATE EMBEDDINGS (Local & Free)
    print("--- Step 2: Generating Local Embeddings (all-MiniLM-L6-v2) ---")
    # Using local HuggingFace model to avoid OpenAI API costs/errors
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5. CREATE AND SAVE FAISS INDEX
    print("--- Step 3: Building and Saving FAISS Index ---")
    vector_db = FAISS.from_documents(final_docs, embeddings)
    
    # Save to a local folder
    output_folder = "submarine_faiss_index"
    vector_db.save_local(output_folder)
    
    print("\n" + "="*50)
    print(f"SUCCESS: Vector DB saved to '{output_folder}'")
    print("You can now run your rag_qa.py script.")
    print("="*50)

if __name__ == "__main__":
    create_vector_db()
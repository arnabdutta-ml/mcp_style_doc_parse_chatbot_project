import os
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import docx
import ollama
import json

# Load local embedding model once at startup
print("[INFO] Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("[INFO] Embedding model loaded.")

def extract_text_from_pdf(pdf_paths):
    """
    Extract text from a list of PDFs.
    """
    text = ""
    for pdf_path in pdf_paths:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    return text

def extract_text_from_docx(docx_path):
    """
    Extract text from a .docx file.
    """
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_folder(folder_path):
    """
    Find all .pdf and .docx files in a folder and extract their text.
    """
    files = os.listdir(folder_path)
    pdf_files = [os.path.join(folder_path, f) for f in files if f.lower().endswith('.pdf')]
    docx_files = [os.path.join(folder_path, f) for f in files if f.lower().endswith('.docx')]

    pdf_text = extract_text_from_pdf(pdf_files) if pdf_files else ""
    docx_text = "".join([extract_text_from_docx(path) for path in docx_files])

    return pdf_text + docx_text

def chunk_text(text, chunk_size=500):
    """
    Split text into chunks of given size.
    """
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embedding(text):
    """
    Compute embedding for a text chunk.
    """
    return np.array(embedder.encode(text), dtype='float32')

def build_faiss_index(chunks):
    """
    Create FAISS index from text chunks.
    """
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def find_top_k_chunks(question, chunks, embeddings, index, k=3):
    """
    Retrieve top-k most similar chunks to the question.
    """
    q_emb = get_embedding(question).reshape(1, -1)
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]

def ask_local_llm(question, context_chunks):
    """
    Ask the local LLM via Ollama, enforcing MCP-style JSON response.
    """
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant. You MUST ALWAYS reply in JSON with this schema:

{{
  "answer": "string",
  "citations": ["string", "string"]
}}

Use the provided context to answer as best you can.

Context:
{context}

Question:
{question}

Now respond ONLY in JSON as specified above.
"""

    response = ollama.chat(
        model='llama2',
        messages=[{"role": "user", "content": prompt}]
    )

    text_response = response['message']['content']

    # Try to parse JSON
    try:
        parsed = json.loads(text_response)
        return parsed
    except json.JSONDecodeError:
        # Fallback if model fails to produce JSON
        return {
            "answer": text_response.strip(),
            "citations": []
        }

def ask_general_local_llm(question):
    """
    Free-form general chat without document context.
    """
    response = ollama.chat(
        model='llama2',
        messages=[{"role": "user", "content": question}]
    )
    return response['message']['content']

def main():
    print("=== MCP-Style Doc Parse Chatbot (Local, Offline) ===")
    folder_path = input("Enter path to folder with .pdf and .docx files: ").strip()
    if not os.path.isdir(folder_path):
        print("[ERROR] Folder not found or is not a directory.")
        return

    # Extract and prepare documents
    text = extract_text_from_folder(folder_path)
    if not text:
        print("[ERROR] No text extracted from documents.")
        return
    print("[INFO] Extracted text from documents.")

    chunks = chunk_text(text)
    print(f"[INFO] Split text into {len(chunks)} chunks.")

    print("[INFO] Creating embeddings and building FAISS index...")
    index, embeddings = build_faiss_index(chunks)
    print("[INFO] FAISS index ready!")

    # Chat loop
    while True:
        user_input = input("\nYour input (prefix with 'doc:' or 'chat:', or 'exit'): ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower().startswith("doc:"):
            question = user_input[4:].strip()
            if not question:
                print("[ERROR] Please enter a question after 'doc:'.")
                continue

            top_chunks = find_top_k_chunks(question, chunks, embeddings, index, k=3)
            result = ask_local_llm(question, top_chunks)

            print("\nAnswer:", result.get("answer", "No answer."))
            citations = result.get("citations", [])
            if citations:
                print("\nCitations:")
                for c in citations:
                    print("-", c)

        elif user_input.lower().startswith("chat:"):
            question = user_input[5:].strip()
            if not question:
                print("[ERROR] Please enter a question after 'chat:'.")
                continue

            answer = ask_general_local_llm(question)
            print("\nAssistant:", answer)

        else:
            print("[INFO] Please prefix with 'doc:' for document questions or 'chat:' for general chat.")

if __name__ == "__main__":
    main()
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. INITIALIZATION
app = FastAPI()

# Replace with your actual OpenAI Key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Load the FAISS index (Must match the embedding model from Stage 1)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("submarine_faiss_index", embeddings, allow_dangerous_deserialization=True)

# 2. THE PROMPT (Optimized for Technical Extraction)
prompt_template = """
You are a Military Intelligence Systems Analyst. Use the following technical excerpts to answer the question.

STRICT GUIDELINES:
1. Grounding: Answer ONLY using the provided context. 
2. Precision: Include exact technical numbers, units (tonnes, MW, km), and dates.
3. Formatting: Use bullet points for lists.
4. Missing Info: If the answer is not in the context, say "Data not found in document."

CONTEXT:
{context}

QUESTION: {question}

ANALYST RESPONSE:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 3. RAG CHAIN SETUP
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 4. API ENDPOINT
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        result = rag_chain.invoke({"query": request.question})
        
        # Extract source content for citations
        sources = [doc.page_content for doc in result["source_documents"]]
        
        return {
            "answer": result["result"],
            "retrieved_sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Starts the server on http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
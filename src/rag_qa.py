
import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. API KEY SETUP
os.environ["OPENAI_API_KEY"] = "PASTE API KEY HERE"

def start_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    idx = "submarine_faiss_index" if os.path.exists("submarine_faiss_index") else "../submarine_faiss_index"
    
    if not os.path.exists(idx):
        print(f"Error: Index not found at {idx}")
        return

    vectorstore = FAISS.load_local(idx, embeddings, allow_dangerous_deserialization=True)
    
    # INCREASE K to 8: This ensures we catch both the 'Builders' table and the 'Names' table
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 2. BALANCED PROMPT: Fixes OCR while staying concise
    template = """You are a precise technical assistant. Use the provided context to answer the question.
    
    INSTRUCTIONS:
    1. Answer the question directly and briefly.
    2. Fix OCR errors in your response (e.g., 'willenter' -> 'will enter', 'countermeasuros' -> 'countermeasures', 'Narno' -> 'Name', 'Hulu dao' -> 'Huludao').
    3. If the question asks for a count, provide the number and list the names.
    4. Do not provide extra summaries or introductory text.
    5. Use proper English grammar and spacing.

    CONTEXT:
    {context}
    
    QUESTION: {question}
    
    ANSWER:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 3. THE RAG PIPELINE
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "="*50)
    print("SUBMARINE CHATBOT ONLINE (FIXED CONCISE MODE)")
    print("="*50)

    while True:
        user_query = input("\n[YOU]: ")
        if user_query.lower() in ['exit', 'quit']: break
        
        try:
            response = rag_chain.invoke(user_query)
            print(f"\n[BOT]: {response}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    start_rag()
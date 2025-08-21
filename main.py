# main.py
import streamlit as st


import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

from document_ingestor import DocumentIngestor

#  Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_conversational_chain(chat_history=None):
    prompt_template = """
    Answer the question as possible as concise but full answer., 
    if the answer is not in provided context just say, "answer is not available in the context",
    If question in Bengali then give answer in Bengali 
    don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if chat_history:
        for msg in chat_history:
            memory.save_context({"input": msg["question"]}, {"output": msg["answer"]})
    chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    return chain

@st.cache_resource(show_spinner=False)
def get_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def user_input(user_question):
    import os
    if not os.path.exists("faiss_index") or not os.path.exists(os.path.join("faiss_index", "index.faiss")):
        st.error("No FAISS index found. Please upload and process PDF files first.")
        return
    # Always reload the index after upload
    st.cache_resource.clear()
    new_db = get_faiss_index()
    # Retrieve more top chunks for better context
    docs = new_db.similarity_search(user_question, k=12)
    # Filter chunks by keyword if user asks for a specific role
    keywords = ["internal member", "internal members", "supervisor", "external member", "examiner"]
    lower_question = user_question.lower()
    filter_kw = None
    for kw in keywords:
        if kw in lower_question:
            filter_kw = kw
            break
    filtered_docs = docs
    if filter_kw:
        filtered_docs = [doc for doc in docs if filter_kw in doc.page_content.lower()]
        # If nothing matches, fall back to all docs
        if not filtered_docs:
            filtered_docs = docs
    # Conversation memory using Streamlit session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    chain = get_conversational_chain(chat_history=st.session_state["chat_history"])
    response = chain.invoke({"context": filtered_docs, "question": user_question})
    print(response)
    answer = response["output_text"] if isinstance(response, dict) and "output_text" in response else response
    st.write("Reply: ", answer)
    # Save to chat history
    st.session_state["chat_history"].append({"question": user_question, "answer": answer})
    st.write("\n**Source Chunks:**")
    for doc in filtered_docs:
        meta = doc.metadata
        page_info = f" | Page: {meta.get('page', '?')}" if 'page' in meta else ""
        st.write(f"- File: `{meta.get('filename','?')}` | Chunk: {meta.get('chunk','?')}{page_info}")
        # Show actual chunk text for debugging/clarity
        st.markdown(f"> {doc.page_content[:500]}{' ...' if len(doc.page_content) > 500 else ''}")
    # Display chat history
    st.write("\n**Conversation History:**")
    for i, msg in enumerate(st.session_state["chat_history"]):
        st.markdown(f"**Q{i+1}:** {msg['question']}")
        st.markdown(f"**A{i+1}:** {msg['answer']}")

def main():
    import streamlit as st
    from document_ingestor import DocumentIngestor
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
    import csv

    # Page configuration
    st.set_page_config(page_title="📄 Chat with PDF using Gemini", layout="wide")
    st.markdown("<h1 style='text-align: center;'>💁 Chat with Your Documents Using Gemini AI</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # User input for questions
    st.markdown("### 🧠 Ask a question about your Documents or file")
    user_question = st.text_input("Type your question and hit Submit & Process")

    if user_question:
        user_input(user_question)  


    # Sidebar for file upload, path, or URL
    with st.sidebar:
        st.markdown("## 📂 Upload, Path, or URL")
        st.markdown("Upload one or more Documents, or provide a local path/URL, then click the button below to process.")

        pdf_docs = st.file_uploader(
            label=" Select Document",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "csv", "db"],
            accept_multiple_files=True
        )

        st.markdown("---")
        st.markdown("### Or enter a local file path or URL:")
        doc_path_or_url = st.text_input("Path or URL to document (PDF)")

        if st.button(" Submit & Process"):
            processed = False
            text_chunks = []
            metadatas = []
            # Handle file upload
            if pdf_docs:
                try:
                    for uploaded_file in pdf_docs:
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        # Special handling for CSV: treat each row as a chunk
                        if uploaded_file.name.lower().endswith('.csv'):
                            with open(uploaded_file.name, newline='', encoding='utf-8') as csvfile:
                                reader = csv.reader(csvfile)
                                header = next(reader)
                                for i, row in enumerate(reader):
                                    
                                    row_dict = dict(zip(header, row))
                                    period = row_dict.get('Period', '')
                                    quarter = ''
                                    if period:
                                        try:
                                            year, month = period.split('.')
                                            q_map = {'03': 'Q1', '06': 'Q2', '09': 'Q3', '12': 'Q4'}
                                            quarter = f"{year} {q_map.get(month, month)}"
                                        except Exception:
                                            quarter = period
                                    chunk = ','.join(row) + (f"\nPeriod_as_quarter: {quarter}" if quarter else '')
                                    text_chunks.append(chunk)
                                    meta = {"filename": uploaded_file.name, "chunk": i, "period": period, "quarter": quarter}
                                    metadatas.append(meta)
                        else:
                            # Use new extract API for other types
                            chunks_with_pages = DocumentIngestor.extract(uploaded_file.name, chunk_size=200, chunk_overlap=100)
                            for i, (chunk, page) in enumerate(chunks_with_pages):
                                text_chunks.append(chunk)
                                meta = {"filename": uploaded_file.name, "chunk": i, "page": page if page is not None else i+1}
                                metadatas.append(meta)
                    processed = True
                except Exception as e:
                    st.error(f" Something went wrong with upload: {str(e)}")
            # Handle path or URL
            elif doc_path_or_url:
                import tempfile, requests
                try:
                    if doc_path_or_url.startswith("http://") or doc_path_or_url.startswith("https://"):
                        # Download file
                        response = requests.get(doc_path_or_url)
                        if response.status_code == 200:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(response.content)
                                tmp_path = tmp.name
                            chunks_with_pages = DocumentIngestor.extract(tmp_path, chunk_size=200, chunk_overlap=100)
                            for i, (chunk, page) in enumerate(chunks_with_pages):
                                text_chunks.append(chunk)
                                meta = {"filename": os.path.basename(doc_path_or_url), "chunk": i, "page": page if page is not None else i+1}
                                metadatas.append(meta)
                            processed = True
                        else:
                            st.error("Failed to download file from URL.")
                    else:
                        # Local path
                        if os.path.exists(doc_path_or_url):
                            chunks_with_pages = DocumentIngestor.extract(doc_path_or_url, chunk_size=200, chunk_overlap=100)
                            for i, (chunk, page) in enumerate(chunks_with_pages):
                                text_chunks.append(chunk)
                                meta = {"filename": os.path.basename(doc_path_or_url), "chunk": i, "page": page if page is not None else i+1}
                                metadatas.append(meta)
                            processed = True
                        else:
                            st.error("File path does not exist.")
                except Exception as e:
                    st.error(f" Something went wrong with path/URL: {str(e)}")
            else:
                st.warning(" Please upload a file or provide a path/URL.")

            if processed and text_chunks:
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
                    vectorstore.save_local("faiss_index")
                    st.cache_resource.clear()  # Clear cache for next query reloads 
                    st.success(" Successfully processed and indexed your document!")
                except Exception as e:
                    st.error(f" Something went wrong: {str(e)}")


if __name__ == "__main__":
    main()


#  uvicorn fastapi_app:app --reload --port 8000

 #  streamlit run main.py
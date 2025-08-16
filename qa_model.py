import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil

from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Setup Gemini model
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say:
    'Answer is not available in the context.' Do not provide incorrect information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Store extracted text globally
extracted_text = ""
router = APIRouter()
@router.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        global extracted_text
        temp_path = f"temp_{file.filename}"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = ""
        reader = PdfReader(temp_path)
        for page in reader.pages:
            text += page.extract_text() or ""

        extracted_text = text

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        text_chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

        os.remove(temp_path)

        return {"message": "PDF uploaded and processed successfully."}
    except:
        return {'message': "no file"}
@router.post("/ask/")
async def ask_question(request: Request, question: str = Form(...)):
    global extracted_text
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No document uploaded.")

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    if not db.docstore._dict:
        raise HTTPException(status_code=400, detail="No document found in the vector store.")

    docs = db.similarity_search(question)
    chain = get_conversational_chain()
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    return {"answer": result["output_text"]}

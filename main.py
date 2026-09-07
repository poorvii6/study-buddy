import os
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from database import SessionLocal, Message, User, Conversation
from auth import hash_password, verify_password, new_token

# Load secrets (your GROQ_API_KEY) from the .env file
load_dotenv()

app = FastAPI(title="Study Buddy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    max_output_tokens=1024,
)

# A direct Groq client for audio transcription (Whisper stays on Groq)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Study Buddy, a friendly study tutor. Explain things simply and clearly."),
    ("human", "{question}"),
])
chain = prompt | llm

# A separate prompt for when we're answering from an uploaded PDF (RAG mode)
rag_prompt = ChatPromptTemplate.from_template("""
You are Study Buddy, a friendly study tutor. The user has uploaded a document, and
relevant excerpts from it are provided below as context.

How to respond:
- If the message is a greeting or casual chat, reply warmly and naturally.
- If the question can be answered from the context, answer using it.
- If it is clearly a question about the document but the answer is not in the context,
  say you couldn't find that part in the document.
- For general questions that aren't about the document, just answer helpfully.

Context:
{context}

Question: {question}
""")
rag_chain = rag_prompt | llm

# Holds each user's uploaded-PDF search index, in memory: {user_id: vectorstore}
user_vectorstores = {}

# Embeddings run locally with FastEmbed — lightweight, no PyTorch, no API token
_embeddings = None
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import FastEmbedEmbeddings
        _embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings


# ---------- Request shapes ----------
class ChatRequest(BaseModel):
    message: str
    conversation_id: int


class AuthRequest(BaseModel):
    username: str
    password: str


# ---------- Helper: find the logged-in user from their token ----------
def get_current_user(db, authorization):
    if not authorization:
        raise HTTPException(status_code=401, detail="Not logged in")
    token = authorization.replace("Bearer ", "")
    user = db.query(User).filter(User.token == token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session, please log in again")
    return user


# ---------- Accounts ----------
@app.post("/signup")
def signup(request: AuthRequest):
    db = SessionLocal()
    if db.query(User).filter(User.username == request.username).first():
        db.close()
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(username=request.username, password_hash=hash_password(request.password))
    db.add(user)
    db.commit()
    db.close()
    return {"message": "Account created! You can now log in."}


@app.post("/login")
def login(request: AuthRequest):
    db = SessionLocal()
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not verify_password(request.password, user.password_hash):
        db.close()
        raise HTTPException(status_code=401, detail="Wrong username or password")
    token = new_token()
    user.token = token
    db.commit()
    db.close()
    return {"token": token}


# ---------- Upload a PDF (LOAD -> SPLIT -> EMBED -> STORE) ----------
@app.post("/upload")
async def upload(file: UploadFile = File(...), authorization: str = Header(None)):
    db = SessionLocal()
    user = get_current_user(db, authorization)
    user_id = user.id            # capture before closing the session
    db.close()

    # Save the uploaded PDF to a temporary file
    temp_path = f"temp_{user_id}.pdf"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # The RAG pipeline you already know
    docs = PyPDFLoader(temp_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    user_vectorstores[user_id] = FAISS.from_documents(chunks, get_embeddings())

    os.remove(temp_path)
    return {"message": f"Loaded '{file.filename}' — now ask me questions about it!"}


# ---------- Voice: transcribe recorded audio with Groq Whisper ----------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), authorization: str = Header(None)):
    db = SessionLocal()
    get_current_user(db, authorization)   # must be logged in
    db.close()

    audio_bytes = await file.read()
    result = groq_client.audio.transcriptions.create(
        file=(file.filename or "audio.webm", audio_bytes),
        model="whisper-large-v3-turbo",
        language="en",
    )
    return {"text": result.text}


# ---------- Chat (requires login; saves into a conversation; RAG if a PDF was uploaded) ----------
@app.post("/chat")
def chat(request: ChatRequest, authorization: str = Header(None)):
    db = SessionLocal()
    user = get_current_user(db, authorization)

    # Make sure this conversation exists and belongs to this user
    convo = db.query(Conversation).filter(
        Conversation.id == request.conversation_id,
        Conversation.user_id == user.id,
    ).first()
    if not convo:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    # If this is the first message, name the conversation after it
    first_message = db.query(Message).filter(Message.conversation_id == convo.id).first()
    if not first_message:
        convo.title = request.message[:40]
        db.commit()

    db.add(Message(user_id=user.id, conversation_id=convo.id, role="user", content=request.message))
    db.commit()

    if user.id in user_vectorstores:
        # RAG mode: retrieve relevant chunks, then answer from them
        retriever = user_vectorstores[user.id].as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(request.message)
        context = "\n\n".join(d.page_content for d in docs)
        reply = rag_chain.invoke({"context": context, "question": request.message}).content
    else:
        # Normal mode: no PDF uploaded yet
        reply = chain.invoke({"question": request.message}).content

    db.add(Message(user_id=user.id, conversation_id=convo.id, role="bot", content=reply))
    db.commit()
    db.close()
    return {"reply": reply}


# ---------- Conversations ----------
@app.post("/conversations")
def create_conversation(authorization: str = Header(None)):
    db = SessionLocal()
    user = get_current_user(db, authorization)
    convo = Conversation(user_id=user.id, title="New chat")
    db.add(convo)
    db.commit()
    result = {"id": convo.id, "title": convo.title}
    db.close()
    return result


@app.get("/conversations")
def list_conversations(authorization: str = Header(None)):
    db = SessionLocal()
    user = get_current_user(db, authorization)
    convos = (db.query(Conversation)
                .filter(Conversation.user_id == user.id)
                .order_by(Conversation.id.desc())
                .all())
    result = [{"id": c.id, "title": c.title} for c in convos]
    db.close()
    return result


@app.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: int, authorization: str = Header(None)):
    db = SessionLocal()
    user = get_current_user(db, authorization)
    convo = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    ).first()
    if not convo:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.id).all()
    result = [{"role": m.role, "content": m.content} for m in messages]
    db.close()
    return result


# Serve the frontend webpage (must come after the API routes above)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

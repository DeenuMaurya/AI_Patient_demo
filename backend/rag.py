import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from prompt import build_prompt

# -----------------------------
# 🔑 Load Keys
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing")

# -----------------------------
# 🧠 Pinecone Setup
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "ai-patient-data"   # ⚠️ change if your index name is different
index = pc.Index(index_name)

# -----------------------------
# 🔍 Embeddings
# -----------------------------
embeddings = OpenAIEmbeddings()

# -----------------------------
# 📦 Vector Store
# -----------------------------
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# -----------------------------
# 🤖 OpenAI Client
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# 💬 Chat Function
# -----------------------------
def ask_patient(question: str) -> str:
    try:
        docs = vectorstore.similarity_search(question, k=3)

        if not docs:
            return "I don't know based on my records."

        context = "\n".join([doc.page_content for doc in docs])

        prompt = build_prompt(context, question)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"
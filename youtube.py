
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

import os

embeddings = HuggingFaceEmbeddings()

def load_youtube_to_vector_db(video_url):
    """
    Load a YouTube video into a FAISS vector DB.
    Automatically detects Arabic transcript if available.
    Falls back to English or the first available language.
    """
    
    index_name_base = video_url.split("=")[-1].replace("?", "_").replace("&", "_")

    try:
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
        transcript = loader.load()
        if not transcript:
            raise ValueError("Transcript returned empty")
    except Exception as e:
        print("Primary loader failed:", e)
        try:
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
            transcript = loader.load()
        except Exception as e2:
            raise RuntimeError(f"Unable to load video transcript: {e2}")

    # ---- Detect Arabic automatically ----
    arabic_docs = [doc for doc in transcript if doc.metadata.get("language") == "ar"]
    english_docs = [doc for doc in transcript if doc.metadata.get("language") == "en"]

    if arabic_docs:
        selected_docs = arabic_docs
        lang_code = "ar"
    elif english_docs:
        selected_docs = english_docs
        lang_code = "en"
    else:
        selected_docs = transcript
        lang_code = transcript[0].metadata.get("language", "unknown")

    merged_text = "\n".join(doc.page_content for doc in selected_docs)

    index_name = f"{index_name_base}_{lang_code}"

    if os.path.exists(f"{index_name}/index.faiss"):
        return FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = splitter.split_documents([type(transcript[0])(page_content=merged_text, metadata={})])

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_name)

    return db

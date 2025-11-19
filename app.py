import streamlit as st
import textwrap
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
from youtube import load_youtube_to_vector_db


# APP CONFIG

def main():
    st.set_page_config(page_title="AI Video Chat", layout="wide")
    st.title("Chat With YouTube Videos")

    # Session state
    if "db" not in st.session_state:
        st.session_state.db = None

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []



    # AGENT FUNCTION

    def ask_question(query, mode="chat"):
        db = st.session_state.db
        docs = db.similarity_search(query, k=4)
        docs_text = " ".join([d.page_content for d in docs])

        chat = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )

        system_prompt = """
        You are a smart assistant that answers based ONLY on this transcript/text:

        {docs}

        If answer is not found, say: "I don't know."
        """

        # Modes for advanced tools
        extras = {
            "summary": "\nSummarize the document into 10 bullet points.",
            "key_points": "\nExtract the 8 key takeaways.",
            "entities": "\nList all named entities (people, places, orgs, tools).",
            "timeline": "\nExtract a chronological timeline of events.",
        }

        if mode in extras:
            system_prompt += extras[mode]

        system_msg = SystemMessagePromptTemplate.from_template(system_prompt)
        human_msg = HumanMessagePromptTemplate.from_template("{question}")

        prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

        chain = LLMChain(
            llm=chat,
            prompt=prompt,
            memory=st.session_state.memory,
            verbose=False
        )

        answer = chain.run(question=query, docs=docs_text)
        return answer





    # LOAD YOUTUBE

    st.subheader("🎥 Load YouTube Video")
    yt_url = st.text_input("Enter YouTube URL:")

    if st.button("Load YouTube Transcript"):
        with st.spinner("Processing YouTube transcript..."):
            st.session_state.db = load_youtube_to_vector_db(yt_url)
        st.success("YouTube transcript loaded! You can now chat.")



    # CHAT INTERFACE

    if st.session_state.db:
        user_question = st.chat_input("Ask something...")

        if user_question:
            answer = ask_question(user_question)
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("Assistant", answer))

        for role, text in st.session_state.chat_history:
            with st.chat_message(role.lower()):
                st.write(textwrap.fill(text, width=80))

        # Advanced buttons
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)

        if col1.button("📌 Summary"):
            st.write(ask_question("Summarize", mode="summary"))

        if col2.button("📝 Key Points"):
            st.write(ask_question("Key points", mode="key_points"))

        if col3.button("🧬 Named Entities"):
            st.write(ask_question("Entities", mode="entities"))

        if col4.button("⏳ Timeline"):
            st.write(ask_question("Timeline", mode="timeline"))
if __name__ == "__main__":
    main()
import streamlit as st
import ssl
import validators
import urllib.request
import re

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Function to extract YouTube transcript
def get_youtube_transcript_docs(url: str):
    def extract_video_id(youtube_url):
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", youtube_url)
        return match.group(1) if match else None

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([entry["text"] for entry in transcript])
    return [Document(page_content=full_text)]

# Streamlit App Title
st.title("YT / Website Summarizer")

# Your Groq API Key
api_key = "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"

# Setup LLM
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192",
    streaming=True
)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# URL Input
url = st.text_input("Enter YouTube or Website URL")

# Summarize Button
if st.button("Summarize"):
    if not url.strip():
        st.error("⚠️ Please provide a URL.")
    elif not validators.url(url):
        st.error("❌ Please enter a valid URL.")
    else:
        try:
            with st.spinner("⏳ Fetching and summarizing..."):
                # Load documents
                if "youtube.com" in url or "youtu.be" in url:
                    docs = get_youtube_transcript_docs(url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                    docs = loader.load()

                # Run summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success("✅ Summary Generated:")
                st.write(output_summary)

        except Exception as e:
            import traceback
            st.error(f"❌ Exception: {e}")
            st.text(traceback.format_exc())

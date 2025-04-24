import streamlit as st
from smart_research_assistant.langchain_module import ResearchAssistantLangChain
from time import time

def display_progress(stage, progress_bar, stages):
    progress = (stages.index(stage)+1)/len(stages)
    progress_bar.progress(progress)
    st.session_state.status.text(f"{stage}...")

def main():
    st.set_page_config(page_title="Research Assistant", layout="wide")
    st.title("⚡ Turbo Research Assistant")

    # Initialize session state variables
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ResearchAssistantLangChain("./research_db")
        st.session_state.url_cache = set()
        st.session_state.status = st.empty()  # Initialize status placeholder

    with st.sidebar:
        st.header("🔗 Data Sources")
        urls = st.text_area("Enter URLs (one per line)", height=150)
        
        if st.button("🚀 Process URLs"):
            if urls:
                new_urls = [url.strip() for url in urls.split('\n') 
                            if url.strip() and url not in st.session_state.url_cache]
                
                if new_urls:
                    with st.spinner("🔄 Processing..."):
                        start = time()
                        docs = st.session_state.assistant.load_urls(new_urls)
                        st.session_state.assistant.create_vector_store(docs)
                        st.session_state.url_cache.update(new_urls)
                        st.success(f"✅ Added {len(new_urls)} docs in {time()-start:.1f}s")
                else:
                    st.info("ℹ️ All URLs already processed")

    tab1, tab2 = st.tabs(["🔍 Query", "📝 Summarize"])

    with tab1:
        st.header("Ask Questions")
        query = st.text_input("Your research question:", key="query")
        
        if query:
            progress_bar = st.progress(0)
            status = st.empty()  # Use local status instead of session state
            stages = ["Initializing", "Vector Search", "Generating Answer"]
            
            try:
                status.text(f"{stages[0]}...")
                progress_bar.progress(1/len(stages))
                
                start = time()                
                result = st.session_state.assistant.query_data(query)
                
                status.text(f"{stages[1]}...")
                progress_bar.progress(2/len(stages))
                
                st.subheader("Answer")
                st.markdown(result['answer'])
                
                status.text(f"{stages[2]}...")
                progress_bar.progress(1.0)
                
                with st.expander("📚 Source Documents"):
                    for i, source in enumerate(result['sources'], 1):
                        st.caption(f"[{i}] {source}")
                
                st.caption(f"⏱️ Total time: {time()-start:.2f}s")

            except Exception as e:
                st.error(f"Query failed: {str(e)}")
            finally:
                progress_bar.empty()
                status.empty()

    with tab2:
        st.header("Document Summarization")
        text = st.text_area("Text to summarize:", height=200)
        if st.button("Generate Summary"):
            if text:
                with st.spinner("Analyzing..."):
                    start = time()
                    summary = st.session_state.assistant.summarize_document(text)
                    st.markdown(summary)
                    st.caption(f"Generated in {time()-start:.2f}s")

if __name__ == "__main__":
    main()
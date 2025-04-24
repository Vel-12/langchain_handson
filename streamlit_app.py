import streamlit as st
from smart_research_assistant.langchain_module import ResearchAssistantLangChain

def main():
    st.title("Research Assistant with Ollama and LangChain")
    
    # Initialize the research assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ResearchAssistantLangChain()
        st.session_state.processed_urls = []

    # Sidebar for URL input
    with st.sidebar:
        st.header("Configuration")
        urls = st.text_area(
            "Enter URLs to research (one per line)",
            height=150,
            help="Paste one or more URLs you want to analyze"
        )
        
        if st.button("Process URLs"):
            if urls:
                url_list = [url.strip() for url in urls.split('\n') if url.strip()]
                try:
                    with st.spinner("Loading and processing documents..."):
                        documents = st.session_state.assistant.load_urls(url_list)
                        st.session_state.assistant.create_vector_store(documents)
                        st.session_state.processed_urls = url_list
                    st.success(f"Successfully processed {len(url_list)} URLs!")
                except Exception as e:
                    st.error(f"Error processing URLs: {str(e)}")
            else:
                st.warning("Please enter at least one URL")

    # Main interface
    tab1, tab2 = st.tabs(["Research Query", "Document Summarization"])

    with tab1:
        st.header("Query Your Research")
        
        if st.session_state.processed_urls:
            st.write(f"Loaded {len(st.session_state.processed_urls)} URLs")
            query = st.text_input("Enter your research question:")
            
            if query:
                with st.spinner("Searching for answers..."):
                    try:
                        result = st.session_state.assistant.query_data(query)
                        st.subheader("Answer")
                        st.write(result["answer"])
                        
                        st.subheader("Source Documents")
                        for i, doc in enumerate(result["source_documents"], 1):
                            with st.expander(f"Source Document {i}"):
                                st.write(doc.page_content)
                                st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    except Exception as e:
                        st.error(f"Error querying data: {str(e)}")
        else:
            st.info("Please load URLs in the sidebar to begin querying")

    with tab2:
        st.header("Document Summarization")
        text_to_summarize = st.text_area(
            "Paste text to summarize",
            height=200,
            help="Enter any text you want to summarize using the LLM"
        )
        
        if st.button("Generate Summary"):
            if text_to_summarize.strip():
                with st.spinner("Generating summary..."):
                    try:
                        summary = st.session_state.assistant.summarize_documents(text_to_summarize)
                        st.subheader("Summary")
                        st.info(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
            else:
                st.warning("Please enter some text to summarize")

if __name__ == "__main__":
    main()
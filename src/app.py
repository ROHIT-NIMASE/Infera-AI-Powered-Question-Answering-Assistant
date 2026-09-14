import streamlit as st
import os
import tempfile
from rag_pipeline import handle_query
from vector_store import add_documents_to_store, clear_all_documents, get_chroma_client, COLLECTION_NAME

st.set_page_config(page_title="Document Intelligence Assistant", layout="wide")

st.title("📄 Infera — AI-Powered Question Answering Assistant")
st.caption("Upload research papers and ask questions about them.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: document upload and management ---
with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process uploaded PDFs"):
        with st.spinner("Processing documents... this may take a minute."):
            # Save uploaded files to a temp location so our existing
            # PDF-reading code (which expects a file path) can read them
            temp_paths = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for uf in uploaded_files:
                    temp_path = os.path.join(temp_dir, uf.name)
                    with open(temp_path, "wb") as f:
                        f.write(uf.getbuffer())
                    temp_paths.append(temp_path)

                num_chunks = add_documents_to_store(temp_paths)

        st.success(f"Added {num_chunks} chunks from {len(uploaded_files)} file(s).")

    st.divider()

    if st.button("🗑️ Clear all documents"):
        clear_all_documents()
        st.session_state.chat_history = []
        st.success("All documents cleared.")

    # Show currently stored documents
    st.subheader("Currently stored:")
    try:
        client = get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        all_meta = collection.get(include=["metadatas"])
        doc_names = sorted(set(m["document_name"] for m in all_meta["metadatas"]))
        if doc_names:
            for name in doc_names:
                st.write(f"- {name}")
        else:
            st.write("*No documents yet.*")
    except Exception:
        st.write("*No documents yet.*")

# --- Main chat area ---
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])

user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources, _ = handle_query(user_question, chat_history=st.session_state.chat_history)
        st.write(answer)

    st.session_state.chat_history.append({"question": user_question, "answer": answer})
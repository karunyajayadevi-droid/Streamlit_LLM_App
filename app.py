
import streamlit as st
import io
import hashlib

from pypdf import PdfReader

from pdf_processor import split_pdf_text
from embedding import generate_embeddings
from vector_database import store_chunks, search_chunks
from llm import generate_answer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PDF Question Answering AI",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# 2. APPLICATION TITLE
# ==========================================

st.title("📚 PDF Question Answering AI")

st.write(
    "Upload your PDF document and ask questions. "
    "Get AI-powered answers using Llama 3.2."
)

st.divider()


# ==========================================
# 3. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Settings")

st.sidebar.write(
    "PDF Question Answering System"
)

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()


# ==========================================
# 5. PDF UPLOAD
# ==========================================

st.header("📄 Upload Your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)


# ==========================================
# 6. PDF PROCESSING
# ==========================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    file_id = hashlib.sha256(file_bytes).hexdigest()

    if file_id != st.session_state.processed_file_id:

        with st.spinner("Processing your PDF..."):

            try:
                # Extract PDF text
                pdf_reader = PdfReader(
                    io.BytesIO(file_bytes)
                )

                extracted_text = "\n".join(
                    page.extract_text() or ""
                    for page in pdf_reader.pages
                )

                if not extracted_text.strip():
                    st.error(
                        "No readable text found in this PDF."
                    )
                    st.stop()

                # Split PDF text
                chunks = split_pdf_text(extracted_text)

                if not chunks:
                    st.error("Unable to create PDF text chunks.")
                    st.stop()

                # Generate embeddings
                embeddings = generate_embeddings(chunks)

                # Store in ChromaDB
                store_chunks(chunks, embeddings)

                # Save processing information
                st.session_state.processed_file_id = file_id
                st.session_state.pdf_chunks = chunks
                st.session_state.messages = []

                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_pages = len(pdf_reader.pages)
                st.session_state.pdf_characters = len(extracted_text)

                st.success("PDF processed successfully!")

            except Exception as e:
                st.error(f"PDF processing error: {e}")
                st.stop()


# ==========================================
# 7. DISPLAY PDF INFORMATION
# ==========================================

if st.session_state.processed_file_id is not None:

    st.success("Your PDF is ready for questions!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "PDF Pages",
            st.session_state.pdf_pages
        )

    with col2:
        st.metric(
            "Text Chunks",
            len(st.session_state.pdf_chunks)
        )

    with col3:
        st.metric(
            "Characters",
            st.session_state.pdf_characters
        )

    st.write(
        "**Uploaded File:**",
        st.session_state.pdf_name
    )

    st.divider()


    # ==========================================
    # 8. CHAT HISTORY
    # ==========================================

    st.header("💬 Ask Questions About Your PDF")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # ==========================================
    # 9. USER QUESTION
    # ==========================================

    question = st.chat_input(
        "Type your question about the PDF..."
    )

    if question:

        # Display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)


        # ==========================================
        # 10. RETRIEVE RELEVANT INFORMATION
        # ==========================================

        with st.chat_message("assistant"):

            with st.spinner("Searching your PDF..."):

                try:
                    question_embedding = generate_embeddings(
                        [question]
                    )[0]

                    results = search_chunks(
                        question_embedding,
                        top_k=3
                    )

                    documents = results["documents"][0]

                    context = "\n\n".join(documents)

                    if not context.strip():
                        st.warning(
                            "No relevant information found in the PDF."
                        )
                        st.stop()


                    # ==========================================
                    # 11. GENERATE AI ANSWER
                    # ==========================================

                    answer = generate_answer(
                        question,
                        context
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Unable to generate answer: {e}"
                    )


else:

    st.info(
        "Please upload a PDF document to start asking questions."
    )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "Powered by Streamlit | ChromaDB | "
    "Sentence Transformers | Ollama Llama 3.2"
)


from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pdf_text(text):
    """
    Split extracted PDF text into smaller chunks.
    """

    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    return chunks


def display_chunks(chunks):
    """
    Display the extracted text chunks.
    """

    print("Total text chunks:", len(chunks))

    for index, chunk in enumerate(chunks):
        print(f"\n--- Chunk {index + 1} ---")
        print(chunk)

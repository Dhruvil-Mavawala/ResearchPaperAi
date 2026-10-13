import fitz
import re


# ============================================================
# 1. CLEAN TEXT
# ============================================================

def clean_text(text):
    """
    Clean extracted PDF text.
    """

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n", text)

    # Remove spaces at beginning/end of lines
    text = "\n".join(line.strip() for line in text.splitlines())

    # Remove empty lines
    text = "\n".join(line for line in text.splitlines() if line.strip())

    return text.strip()


# ============================================================
# 2. EXTRACT TEXT FROM PDF
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF page by page.
    """

    pdf_bytes = uploaded_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        cleaned = clean_text(text)

        if cleaned:

            pages.append({
                "page": page_number,
                "text": cleaned
            })

    document.close()

    return pages


# ============================================================
# 3. CREATE CHUNKS
# ============================================================

def create_chunks(
    pages,
    paper_id,
    source,
    chunk_size=1200,
    chunk_overlap=200
):
    """
    Convert page text into smaller chunks.

    Each chunk keeps:
    - paper_id
    - source
    - page
    - chunk number
    - text
    """

    chunks = []

    chunk_counter = 1

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "chunk_id": f"{paper_id}_chunk_{chunk_counter}",
                    "paper_id": paper_id,
                    "source": source,
                    "page": page_number,
                    "chunk_number": chunk_counter,
                    "text": chunk_text,
                    "character_count": len(chunk_text)
                })

                chunk_counter += 1

            # Stop if we reached the end
            if end >= len(text):
                break

            # Move forward while keeping overlap
            start = end - chunk_overlap

    return chunks


# ============================================================
# 4. PROCESS ONE PDF
# ============================================================

def process_pdf(
    uploaded_file,
    paper_id,
    chunk_size=1200,
    chunk_overlap=200
):
    """
    Complete processing pipeline for one uploaded PDF.

    PDF
      ↓
    Extract pages
      ↓
    Clean text
      ↓
    Create chunks
    """

    pages = extract_text_from_pdf(uploaded_file)

    chunks = create_chunks(
        pages=pages,
        paper_id=paper_id,
        source=uploaded_file.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return {
        "source": uploaded_file.name,
        "pages": pages,
        "chunks": chunks
    }
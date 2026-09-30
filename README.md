# Financial Document Assistant

An offline document analysis assistant for financial documents (earnings reports, filings, prospectus PDFs) using local embeddings and language models.

## Features

- **Offline Processing:** Runs locally on your machine without external cloud API dependencies.
- **PDF Ingestion:** Extracts text from multipage PDF files and splits them into overlapping chunks.
- **Vector Search:** Stores embeddings in SQLite and performs cosine similarity retrieval.
- **Source Attribution:** Identifies the source document and page number for retrieved answers.
- **Web Interface:** Web-based chat interface for uploading PDFs and submitting queries.

## Project Structure

- `backend/main.py`: FastAPI server serving API endpoints and static frontend files.
- `backend/database.py`: SQLite setup and storage for document chunks and embeddings.
- `backend/ingestion.py`: PDF text extraction and chunk embedding pipeline.
- `backend/retrieval.py`: Vector cosine similarity search.
- `backend/llm_service.py`: Prompt assembly and model inference.
- `frontend/index.html`: Responsive chat user interface.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/alperencok/Financial-Document-Assistant.git
   cd Financial-Document-Assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. Start the server:
   ```bash
   python -m uvicorn backend.main:app --reload
   ```

5. Open `http://127.0.0.1:8000` in your browser.

## Author

- **alperencok** (https://github.com/alperencok)

## License

This project is licensed under the MIT License.

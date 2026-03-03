# rag_textbooks/ — Add Your Textbooks Here

Place any **PDF textbook** in this folder and the RAG chatbot will learn from it automatically.

## How to add a textbook

1. Copy your PDF file into this folder.
2. **Name the file so the chatbot knows which subject it belongs to** (see naming rules below).
3. Run the indexer to rebuild the knowledge base:

```bash
python manage.py build_rag_index --force
```

4. Done! The chatbot will now answer questions using content from your textbook.

---

## File Naming Convention

Include the subject code anywhere in the filename (case-insensitive):

| Subject | Code | Example filename |
|---------|------|-----------------|
| Data Structures | DS | `DS_Introduction_to_Algorithms.pdf` |
| Python | PY | `PY_Python_Crash_Course.pdf` |
| Web Development | WD | `WD_HTML_CSS_Complete_Guide.pdf` |
| Computer Networks | CN | `CN_Tanenbaum_Networks.pdf` |
| Data Science | DSC | `DSC_Hands_On_ML.pdf` |
| AI & Machine Learning | AIML | `AIML_Deep_Learning_Goodfellow.pdf` |
| General / No subject | (any name) | `Research_Paper.pdf` |

If no subject code is found in the filename, the book is indexed as general knowledge.

---

## Tips

- Scanned/image-only PDFs won't work — the PDF must have selectable text.
- Large textbooks (500+ pages) may take 1–2 minutes to index.
- You can add multiple textbooks at once — run the indexer once after adding all.
- After indexing, restart the Django server if it is already running.

# DocuMind: PDF Q&A Chatbot (RAG)

Ask questions about your own PDFs and get answers with file and page citations.
Hybrid search (BM25 + vectors), optional cross-encoder reranking, follow-up question
rewriting, and an evaluation + ablation suite.

```
PDFs -> extract -> chunk -> embed -> ChromaDB
Question -> (rewrite) -> vector + BM25 search -> (rerank) -> top-k -> LLM -> answer + sources
```

**Stack:** PyMuPDF, Sentence-Transformers, ChromaDB, rank-bm25, Claude API / Ollama / no-LLM mode, Streamlit

## Easiest start (Windows)
1. Extract the zip. Put a **text-based PDF** in the `data` folder.
2. Double-click **`run.bat`**. It creates the environment, installs packages, builds the index and starts the app.
3. Open http://localhost:8501

## Manual start
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env               # Mac/Linux: cp .env.example .env
python -m src.ingest                 # build the index (after adding PDFs to data/)
python -m streamlit run app.py
```
Always use `python -m streamlit` rather than `streamlit`.

## Choose the answering mode (`.env`)
| `LLM_PROVIDER` | What you get | Needs |
|---|---|---|
| `none` (default) | Best matching passages with page numbers | Nothing |
| `ollama` | Written answers from a free local model | Ollama from ollama.com, `ollama pull llama3.2`, `pip install ollama` |
| `anthropic` | Written answers from Claude | A real `ANTHROPIC_API_KEY` and prepaid credit |

Restart the app after changing `.env`. Run `python check_setup.py` to diagnose problems.

## Evaluation
1. Edit `eval_questions.json`: 30-50 questions with `source` and `page`, plus 5-10 with `"answerable": false`.
2. Run:
```powershell
python -m src.evaluate                  # hit rate@k and MRR
python -m src.evaluate --judge          # + faithfulness, correctness, abstention (needs an LLM)
python -m src.evaluate --ablate         # chunk size x (vector | hybrid) -> results/ablation.md
```
3. Paste your real numbers here:

| Chunk size | Overlap | Mode | Hit rate@4 | MRR |
|---|---|---|---|---|
| | | | | |

## Structure
```
run.bat               one-click setup and launch (Windows)
app.py                Streamlit UI
check_setup.py        diagnoses setup problems
src/config.py         settings
src/ingest.py         PDF loading, chunking, embedding, indexing
src/retrieve.py       vector / hybrid / reranked retrieval
src/generate.py       grounded answers, query rewriting, provider switch
src/evaluate.py       metrics and ablations
eval_questions.json   your test set
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `requirements.txt` not found | You are in the wrong folder. `cd` to the folder containing `app.py` |
| `venv\Scripts\activate` not found | Use `.\venv\Scripts\Activate.ps1` |
| Scripts disabled | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Launcher error with `streamlit` | Use `python -m streamlit run app.py` |
| No text found in PDF | The PDF is scanned. Use a text-based one or OCR it |
| 401 error | `.env` has a placeholder or wrong key |
| 400 credit too low | Add credit, or use `none` / `ollama` |

## Limitations
Scanned PDFs need OCR. Tables are extracted as plain text. Chunking is fixed-size by characters.

## Results and learnings
Fill in after your experiments.

"""Run:  python check_setup.py   -> tells you exactly what is wrong (if anything)."""
import glob
import os
import sys

print("Python:", sys.version.split()[0])

from src import config  # loads .env

pdfs = glob.glob(os.path.join(config.DATA_DIR, "*.pdf"))
print(f"PDFs in data/: {len(pdfs)}", "OK" if pdfs else "-> PROBLEM: copy a text-based PDF into the data folder")
print("Index (chroma_db):", "found" if os.path.isdir(config.DB_PATH) else "-> PROBLEM: run  python -m src.ingest")

prov = config.LLM_PROVIDER
print("LLM_PROVIDER:", prov)
if prov == "none":
    print("No LLM needed. The app will show matching passages. OK")
elif prov == "ollama":
    try:
        import ollama
        names = [m.get("model") or m.get("name") for m in ollama.list().get("models", [])]
        print("Ollama models:", names or "none -> run: ollama pull llama3.2")
    except Exception as e:
        print("-> PROBLEM: Ollama not reachable. Install it from ollama.com and run `ollama pull llama3.2`.", type(e).__name__)
else:
    k = os.getenv("ANTHROPIC_API_KEY", "")
    print("API key length:", len(k))
    if not k or k.startswith("your-key") or "xxxx" in k or len(k) < 60:
        print("-> PROBLEM: .env has a placeholder, not a real key. Create one at console.anthropic.com > API Keys.")
    else:
        try:
            import anthropic
            anthropic.Anthropic().messages.create(
                model=config.ANTHROPIC_MODEL, max_tokens=5,
                messages=[{"role": "user", "content": "hi"}])
            print("Claude API call: OK")
        except Exception as e:
            print("-> PROBLEM:", type(e).__name__, str(e)[:200])
            print("   (credit balance too low = add credit at console.anthropic.com > Plans & Billing)")

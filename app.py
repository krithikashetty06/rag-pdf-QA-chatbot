import html
import os
import time

import streamlit as st

from src import config
from src.generate import answer, rewrite_question
from src.ingest import build_index, load_pdfs
from src.retrieve import Retriever

st.set_page_config(page_title="DocuMind · Document Q&A", page_icon="🧠", layout="centered")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(139,92,246,0.28), transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(6,182,212,0.20), transparent 38%),
        radial-gradient(circle at 50% 95%, rgba(236,72,153,0.16), transparent 40%),
        linear-gradient(135deg, #0b1020 0%, #14123a 55%, #1b1650 100%);
    background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.72);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #c4b5fd; }

/* Hero */
.hero {
    position: relative; overflow: hidden;
    padding: 34px 32px; margin-bottom: 22px; border-radius: 22px;
    background: linear-gradient(120deg, rgba(139,92,246,0.35), rgba(6,182,212,0.25));
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}
.hero h1 {
    margin: 0 0 6px 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff, #c4b5fd 50%, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { margin: 0; color: #cbd5e1; font-size: 1.02rem; max-width: 80%; }
.hero .badge-row { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
    padding: 4px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.16); color: #e0e7ff;
}
.hero svg { position: absolute; right: 22px; top: 18px; opacity: 0.9; }

/* Feature + stat cards */
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 10px 0 26px 0; }
.card {
    padding: 18px; border-radius: 16px;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);
    transition: transform .2s ease, border-color .2s ease;
}
.card:hover { transform: translateY(-3px); border-color: rgba(167,139,250,0.6); }
.card .ico { font-size: 1.6rem; }
.card h4 { margin: 8px 0 4px 0; color: #f1f5f9; font-size: 1rem; }
.card p { margin: 0; color: #94a3b8; font-size: 0.85rem; line-height: 1.45; }

.stat {
    padding: 12px 14px; border-radius: 14px; margin-bottom: 10px;
    background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(6,182,212,0.18));
    border: 1px solid rgba(255,255,255,0.12);
}
.stat .num { font-size: 1.5rem; font-weight: 800; color: #fff; line-height: 1.1; }
.stat .lbl { font-size: 0.74rem; text-transform: uppercase; letter-spacing: .08em; color: #a5b4fc; }

/* Chat */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px; padding: 14px 16px; margin-bottom: 12px;
}
[data-testid="stChatInput"] { border-radius: 16px; }

/* Buttons */
.stButton > button {
    border: none; border-radius: 12px; font-weight: 600; color: white;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    box-shadow: 0 6px 18px rgba(124,58,237,0.35);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(124,58,237,0.5); color: white; }

/* Sources */
.src {
    padding: 10px 12px; margin: 8px 0; border-radius: 12px;
    background: rgba(15,23,42,0.55); border-left: 3px solid #8b5cf6;
}
.src .tag {
    display: inline-block; padding: 2px 10px; border-radius: 999px; margin-bottom: 6px;
    font-size: 0.75rem; font-weight: 700; color: #0f172a;
    background: linear-gradient(90deg, #a78bfa, #67e8f9);
}
.src .txt { color: #cbd5e1; font-size: 0.84rem; line-height: 1.5; }
.meta { color: #94a3b8; font-size: 0.78rem; margin-top: 6px; }

@media (max-width: 760px) { .cards { grid-template-columns: 1fr; } .hero p { max-width: 100%; } .hero svg { display: none; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <svg width="120" height="100" viewBox="0 0 120 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="20" y="14" width="56" height="72" rx="8" fill="rgba(255,255,255,0.12)" stroke="#c4b5fd" stroke-width="2"/>
    <rect x="30" y="28" width="36" height="4" rx="2" fill="#c4b5fd"/>
    <rect x="30" y="40" width="30" height="4" rx="2" fill="#67e8f9"/>
    <rect x="30" y="52" width="36" height="4" rx="2" fill="#c4b5fd"/>
    <circle cx="86" cy="62" r="18" fill="rgba(6,182,212,0.18)" stroke="#67e8f9" stroke-width="3"/>
    <line x1="99" y1="75" x2="112" y2="88" stroke="#67e8f9" stroke-width="4" stroke-linecap="round"/>
    <path d="M100 16l3 7 7 3-7 3-3 7-3-7-7-3 7-3z" fill="#f0abfc"/>
  </svg>
  <h1>DocuMind</h1>
  <p>Ask questions about your PDFs and get answers grounded in your documents, with page-level citations.</p>
  <div class="badge-row">
    <span class="badge">🔎 Hybrid search</span>
    <span class="badge">📌 Cited answers</span>
    <span class="badge">🧠 RAG pipeline</span>
    <span class="badge">⚡ ChromaDB</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.session_state.setdefault("history", [])
st.session_state.setdefault("index_version", 0)


@st.cache_resource(show_spinner="Loading models...")
def get_retriever(hybrid, rerank, version):
    return Retriever(hybrid=hybrid, rerank=rerank)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 📚 Documents")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("⚙️ Build / rebuild index", use_container_width=True):
        for f in uploads or []:
            with open(os.path.join(config.DATA_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        pages = load_pdfs()
        if not pages:
            st.error("No extractable text found. Scanned PDFs need OCR.")
        else:
            with st.spinner("Embedding documents..."):
                n = build_index(pages)
            st.session_state.index_version += 1
            st.success(f"Indexed {n} chunks from {len(pages)} pages.")

    st.markdown("## 🎛️ Settings")
    k = st.slider("Chunks retrieved (k)", 1, 10, config.TOP_K)
    hybrid = st.checkbox("Hybrid search (BM25 + vectors)", config.USE_HYBRID)
    rerank = st.checkbox("Cross-encoder reranking", config.USE_RERANKER)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------- retriever
try:
    retriever = get_retriever(hybrid, rerank, st.session_state.index_version)
except Exception:
    st.info("No index found yet. Upload PDFs in the sidebar (or put them in `data/`) "
            "and click **Build / rebuild index**.")
    st.stop()

n_docs = len({c["meta"]["source"] for c in retriever.store.values()})
with st.sidebar:
    st.markdown("## 📊 Index")
    st.markdown(
        f'<div class="stat"><div class="num">{n_docs}</div><div class="lbl">Documents</div></div>'
        f'<div class="stat"><div class="num">{retriever.count}</div><div class="lbl">Chunks indexed</div></div>',
        unsafe_allow_html=True)


def show_sources(sources):
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            text = html.escape(s["text"][:380]) + ("…" if len(s["text"]) > 380 else "")
            st.markdown(
                f'<div class="src"><span class="tag">{html.escape(s["meta"]["source"])} · p.{s["meta"]["page"]}</span>'
                f'<div class="txt">{text}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- chat
if not st.session_state.history:
    st.markdown("""
    <div class="cards">
      <div class="card"><div class="ico">🔎</div><h4>Smart retrieval</h4>
        <p>Combines semantic vectors with keyword search to find the exact passage.</p></div>
      <div class="card"><div class="ico">📌</div><h4>Cited answers</h4>
        <p>Every answer points to the file and page it came from.</p></div>
      <div class="card"><div class="ico">💬</div><h4>Follow-up aware</h4>
        <p>Ask “what about 2022?” and it understands the earlier context.</p></div>
    </div>
    """, unsafe_allow_html=True)

for m in st.session_state.history:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🧠"):
        st.write(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if q := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(q)
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Searching documents and thinking..."):
            t0 = time.time()
            standalone = rewrite_question(st.session_state.history, q)
            try:
                chunks = retriever.search(standalone, k)
            except Exception as e:
                st.error(f'Search failed: {e}. Click Build / rebuild index in the sidebar, or restart the app.')
                st.stop()
            reply = answer(standalone, chunks)
            elapsed = time.time() - t0
        st.write(reply)
        note = f"⏱ {elapsed:.1f}s"
        if standalone != q:
            note += f" · searched as: “{html.escape(standalone)}”"
        st.markdown(f'<div class="meta">{note}</div>', unsafe_allow_html=True)
        show_sources(chunks)
    st.session_state.history.append({"role": "user", "content": q})
    st.session_state.history.append({"role": "assistant", "content": reply, "sources": chunks})

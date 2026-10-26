"""LLM calls: grounded answering, follow-up question rewriting, and a shared helper.

LLM_PROVIDER options (set in .env):
    anthropic  Claude API (needs a real key AND credit)
    ollama     free local model (needs Ollama installed)
    none       no LLM: shows the most relevant passages with page numbers
"""
from . import config

SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "Cite sources inline like [file.pdf, p.X]. "
    f'If the context does not contain the answer, reply exactly: "{config.ABSTAIN_PHRASE}" '
    "Do not use outside knowledge."
)

_anthropic_client = None


def complete(prompt, system=None, max_tokens=600):
    """Send one prompt to the configured LLM and return the text reply."""
    global _anthropic_client
    if config.LLM_PROVIDER == "ollama":
        import ollama
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return ollama.chat(model=config.OLLAMA_MODEL, messages=messages)["message"]["content"]

    import anthropic
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    kwargs = dict(model=config.ANTHROPIC_MODEL, max_tokens=max_tokens,
                  messages=[{"role": "user", "content": prompt}])
    if system:
        kwargs["system"] = system
    msg = _anthropic_client.messages.create(**kwargs)
    return msg.content[0].text


def format_context(chunks):
    return "\n\n".join(
        f'[{c["meta"]["source"]}, p.{c["meta"]["page"]}]\n{c["text"]}' for c in chunks)


def _passages(chunks, note):
    lines = [note + "\n"]
    for c in chunks:
        lines.append(f'**[{c["meta"]["source"]}, p.{c["meta"]["page"]}]** {c["text"][:500]}...\n')
    return "\n".join(lines)


def answer(question, chunks):
    if config.LLM_PROVIDER == "none":
        return _passages(chunks, "_LLM is turned off (LLM_PROVIDER=none). Most relevant passages:_")
    prompt = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    try:
        return complete(prompt, system=SYSTEM)
    except Exception as e:  # bad key, no credit, Ollama not running, ...
        return _passages(
            chunks,
            f"⚠️ The LLM call failed ({type(e).__name__}: {str(e)[:180]}). "
            "Showing the most relevant passages instead. Run `python check_setup.py` to diagnose.")


def rewrite_question(history, question):
    """Turn a follow-up like 'what about 2022?' into a standalone question."""
    if not history or config.LLM_PROVIDER == "none":
        return question
    convo = "\n".join(f'{m["role"].capitalize()}: {m["content"]}' for m in history[-6:])
    prompt = (
        "Rewrite the final user question so it is fully self-contained, using the "
        "conversation for context. Return ONLY the rewritten question.\n\n"
        f"Conversation:\n{convo}\n\nFinal question: {question}")
    try:
        return complete(prompt, max_tokens=150).strip() or question
    except Exception:
        return question

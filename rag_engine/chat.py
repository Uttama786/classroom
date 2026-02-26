"""
RAG Chat — builds prompt with retrieved context and calls Groq LLM.
Supports both standard JSON responses and streaming SSE responses.
"""

import os
import re
import json
import threading
import urllib.request
import urllib.parse
from typing import List, Optional, Generator
from .retriever import get_context

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_CONTEXT_CHARS = 1800   # smaller context → faster first token


def _get_groq_client():
    """Initialize Groq client with API key from settings."""
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq package not installed. Run: pip install groq")

    api_key = ""
    try:
        from django.conf import settings
        api_key = getattr(settings, "GROQ_API_KEY", "")
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to settings.py or environment.")

    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are FlipLearn AI, an expert academic tutor for M.Tech CSE students.

You assist with: Data Structures (DS), Python (PY), Web Development (WD), Computer Networks (CN), Data Science (DSC), and AI & Machine Learning (AIML).

Response Guidelines:
- Use **bold** for key terms and important concepts
- Use `code` for inline code, variables, and commands
- Use bullet points or numbered lists for steps/comparisons
- Use ### for section headers when the answer is long
- Provide code blocks with ``` when showing code examples
- Be concise but thorough. Cite the source context when applicable.
- End with a one-line summary or takeaway when helpful.
"""

RELATED_SYSTEM = """You are an academic question generator. Given a student's question and an AI tutor's answer, generate exactly 3 concise follow-up questions the student might ask next.
Return ONLY a valid JSON array of 3 strings. No explanation, no extra text.
Example: ["What is the time complexity?", "How does it compare to Merge Sort?", "Can you show a code example?"]"""


def build_prompt_messages(
    user_query: str,
    chunks: list,
    chat_history: Optional[List[dict]] = None,
) -> List[dict]:
    """Build the full messages array for Groq from retrieved chunks."""
    context_parts = []
    char_count = 0

    for chunk in chunks:
        text = chunk["text"]
        if char_count + len(text) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(text)
        char_count += len(text)

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No specific context found."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history[-6:])

    user_content = (
        f"Context from FlipLearn knowledge base:\n{context_str}\n\n---\n\n"
        f"Student Question: {user_query}\n\n"
        "Answer based on the context above. Use markdown formatting."
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def _fetch_wiki_sources(query: str, limit: int = 2) -> list:
    """
    Search Wikipedia and return top `limit` results as source dicts:
    [{"title": ..., "url": ..., "snippet": ..., "source_type": "wikipedia"}, ...]
    Uses only stdlib (urllib) — no extra packages needed.
    """
    try:
        params = urllib.parse.urlencode({
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "namespace": 0,
            "format": "json",
        })
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "FlipLearnAI/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
        # data = [query, [titles], [descriptions], [urls]]
        titles = data[1]
        urls   = data[3]
        return [
            {"title": t, "url": u, "source_type": "wikipedia"}
            for t, u in zip(titles, urls)
            if t and u
        ][:limit]
    except Exception:
        return []  # Wikipedia is optional — never block the answer


def _duckduckgo_search(query: str, limit: int = 3) -> list:
    """
    Fetch results from the DuckDuckGo Instant Answer API (free, no key required).
    Returns [{"title", "url", "snippet", "source_type": "web"}, ...]
    """
    try:
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "FlipLearnAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        results = []

        # 1. Main abstract (Wikipedia-backed summary)
        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append({
                "title": data.get("AbstractSource", query[:60]),
                "url": data["AbstractURL"],
                "snippet": data["AbstractText"][:300],
                "source_type": "web",
            })

        # 2. Direct answer results
        for r in data.get("Results", []):
            if len(results) >= limit:
                break
            if r.get("FirstURL") and r.get("Text"):
                results.append({
                    "title": r["Text"][:70],
                    "url": r["FirstURL"],
                    "snippet": r["Text"][:300],
                    "source_type": "web",
                })

        # 3. Related topics (fill remaining slots)
        for topic in data.get("RelatedTopics", []):
            if len(results) >= limit:
                break
            if isinstance(topic, dict) and topic.get("FirstURL") and topic.get("Text"):
                results.append({
                    "title": topic["Text"][:70],
                    "url": topic["FirstURL"],
                    "snippet": topic["Text"][:300],
                    "source_type": "web",
                })

        return results[:limit]
    except Exception:
        return []


def _groq_web_tool_search(user_query: str, client) -> tuple:
    """
    Call Groq with a `search_web` tool definition so the model decides the best
    search query, then execute that query against DuckDuckGo.

    Returns (web_sources: list, web_context: str)
      web_sources — list of {"title", "url", "snippet", "source_type": "web"}
      web_context — plain-text block to inject into the LLM prompt
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": (
                    "Search the web for current, factual information about a topic. "
                    "Use this when the question involves technical concepts, definitions, "
                    "algorithms, frameworks, or any knowledge that benefits from a web lookup."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A focused, concise web search query (max ~10 words)",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search planning assistant. "
                        "If the user's question would benefit from a web search, "
                        "call search_web with an optimised query. "
                        "Otherwise do nothing."
                    ),
                },
                {"role": "user", "content": user_query},
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=80,
        )

        msg = resp.choices[0].message
        if not msg.tool_calls:
            return [], ""

        all_results: list = []
        for tc in msg.tool_calls:
            if tc.function.name == "search_web":
                args = json.loads(tc.function.arguments)
                sq = args.get("query", user_query)
                all_results.extend(_duckduckgo_search(sq, limit=3))

        web_context = "\n\n".join(
            f"Web result — {r['title']} ({r['url']}):\n{r.get('snippet', '')}"
            for r in all_results
        )
        return all_results[:3], web_context
    except Exception:
        return [], ""


def stream_answer(
    user_query: str,
    subject_code: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
    top_k: int = 3,          # 3 chunks is enough and faster than 5
) -> Generator[str, None, None]:
    """
    Generator that yields SSE-formatted events:
      data: {"type": "sources",  "sources": [...]}
      data: {"type": "token",    "text": "..."}
      data: {"type": "done",     "full_reply": "...", "sources": [...]}
      data: {"type": "related",  "questions": [...]}   ← arrives after done
      data: {"type": "error",    "text": "..."}
    """
    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    # ── Shared Groq client (created once, reused throughout) ────────────────
    client = _get_groq_client()

    # ── 1. Retrieve FAISS + Wikipedia + Web results in parallel ─────────────
    wiki_results: list = []
    web_results: list  = []
    web_ctx_holder: list = []   # [str] — holds web context text when found

    def _wiki():
        wiki_results.extend(_fetch_wiki_sources(user_query, limit=2))

    def _web():
        results, ctx = _groq_web_tool_search(user_query, client)
        web_results.extend(results)
        if ctx:
            web_ctx_holder.append(ctx)

    wiki_thread = threading.Thread(target=_wiki, daemon=True)
    web_thread  = threading.Thread(target=_web,  daemon=True)
    wiki_thread.start()
    web_thread.start()

    # FAISS retrieval runs in main thread while background threads work
    chunks = get_context(user_query, top_k=top_k, subject_filter=subject_code)
    wiki_thread.join(timeout=4)
    web_thread.join(timeout=6)   # web search needs a bit more time

    # ── 2. Build deduplicated source list ────────────────────────────────────
    sources = []
    seen = set()
    for c in chunks:
        s = c.get("source", "")
        if s and s not in seen:
            seen.add(s)
            sources.append({"title": s, "subject": c.get("subject", ""), "source_type": "knowledge"})

    for w in wiki_results:
        sources.append(w)

    for r in web_results:
        sources.append(r)

    yield sse({"type": "sources", "sources": sources})

    # ── 3. Build messages — inject web context if present ────────────────────
    messages = build_prompt_messages(user_query, chunks, chat_history)

    if web_ctx_holder:
        # Append web context to the last user message so the LLM can cite it
        last_msg = messages[-1]
        messages[-1] = {
            "role": "user",
            "content": (
                last_msg["content"] +
                "\n\n---\n\nAdditional web search results:\n" + web_ctx_holder[0]
            ),
        }

    # ── 4. Stream the final answer ───────────────────────────────────────────
    full_reply = ""
    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_reply += token
                yield sse({"type": "token", "text": token})

    except Exception as e:
        yield sse({"type": "error", "text": str(e)})
        full_reply = f"Error: {e}"

    # ── 5. Yield done ────────────────────────────────────────────────────────
    yield sse({"type": "done", "full_reply": full_reply, "sources": [s["title"] for s in sources]})

    # ── 6. Generate related questions in background (non-blocking) ───────────
    related_result: list = []
    related_err: list = []

    def _fetch_related():
        try:
            rel_messages = [
                {"role": "system", "content": RELATED_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {user_query}\n"
                        f"Answer: {full_reply[:400]}\n\n"
                        "Generate 3 follow-up questions as a JSON array."
                    ),
                },
            ]
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=rel_messages,
                max_tokens=150,
                temperature=0.6,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r'\[.*?\]', text, re.DOTALL)
            if m:
                related_result.extend(json.loads(m.group())[:3])
        except Exception as exc:
            related_err.append(str(exc))

    t = threading.Thread(target=_fetch_related, daemon=True)
    t.start()
    t.join(timeout=4)   # wait at most 4 s; skip if Groq is slow

    if related_result:
        yield sse({"type": "related", "questions": related_result})


def ask(
    user_query: str,
    subject_code: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
    top_k: int = 5,
) -> dict:
    """Non-streaming fallback — returns full reply at once."""
    chunks = get_context(user_query, top_k=top_k, subject_filter=subject_code)
    sources = list({c.get("source", "") for c in chunks if c.get("source")})
    messages = build_prompt_messages(user_query, chunks, chat_history)

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1200,
            temperature=0.3,
        )
        reply = response.choices[0].message.content.strip()
        return {"reply": reply, "sources": sources, "error": None}
    except Exception as e:
        return {"reply": f"⚠️ Error: {e}", "sources": [], "error": str(e)}


GROQ_MODEL = "llama-3.1-8b-instant"
MAX_CONTEXT_CHARS = 3000


def _get_groq_client():
    """Initialize Groq client with API key from settings."""
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq package not installed. Run: pip install groq")

    # Try Django settings first, then env var
    api_key = ""
    try:
        from django.conf import settings
        api_key = getattr(settings, "GROQ_API_KEY", "")
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to settings.py or set environment variable."
        )

    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are FlipLearn AI, an intelligent academic tutor for M.Tech CSE students at a flipped classroom platform.

You help students understand concepts in:
- Data Structures (DS)
- Python Programming (PY)
- Web Development (WD)
- Computer Networks (CN)
- Data Science (DSC)
- Artificial Intelligence & Machine Learning (AIML)

Guidelines:
- Give clear, concise, accurate answers based on the provided context
- Use examples and analogies to clarify difficult concepts
- For code questions, provide clean code snippets
- If the answer is not in the context, use your general knowledge but mention it
- Be encouraging and supportive to students
- Keep answers focused and structured (use bullet points when helpful)
"""


def ask(
    user_query: str,
    subject_code: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
    top_k: int = 5,
) -> dict:
    """
    Main chat function.
    
    Args:
        user_query: Student's question
        subject_code: Optional subject filter (e.g., 'DS', 'PY')
        chat_history: List of previous messages [{'role': 'user'/'assistant', 'content': '...'}]
        top_k: Number of context chunks to retrieve
    
    Returns:
        dict with keys: reply (str), sources (list of str), error (str or None)
    """
    # Retrieve relevant context
    chunks = get_context(user_query, top_k=top_k, subject_filter=subject_code)

    # Build context string
    context_parts = []
    seen_sources = set()
    char_count = 0

    for chunk in chunks:
        text = chunk["text"]
        if char_count + len(text) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(text)
        source = chunk.get("source", "Unknown")
        seen_sources.add(source)
        char_count += len(text)

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No specific context found."
    sources = sorted(seen_sources)

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add limited chat history (last 3 pairs = 6 messages)
    if chat_history:
        recent = chat_history[-6:]
        messages.extend(recent)

    # Add context + user query
    user_content = f"""Context from FlipLearn knowledge base:
{context_str}

---

Student Question: {user_query}

Please answer the student's question based on the context above."""

    messages.append({"role": "user", "content": user_content})

    # Call Groq
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )
        reply = response.choices[0].message.content.strip()
        return {"reply": reply, "sources": sources, "error": None}

    except RuntimeError as e:
        return {
            "reply": f"⚠️ Configuration error: {e}",
            "sources": [],
            "error": str(e),
        }
    except Exception as e:
        error_msg = str(e)
        return {
            "reply": f"⚠️ I'm having trouble connecting right now. Please try again shortly.\n\nError: {error_msg}",
            "sources": [],
            "error": error_msg,
        }

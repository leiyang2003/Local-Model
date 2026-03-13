"""
Character Chat Web UI — chat with a character via mlx_lm.server (OpenAI-compatible API).
Start the server separately: mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-4bit --port 8000
TTS: DashScope CosyVoice (streaming); voice longanhuan. Set DASHSCOPE_API_KEY in .env or enter in UI.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env: 先尝试当前工作目录，再加载脚本所在目录（脚本目录优先）
_env_dir = Path(__file__).resolve().parent
load_dotenv()  # cwd 下的 .env
load_dotenv(_env_dir / ".env")  # Local-Model/.env
# #region agent log
_DBG_LOG_PATH = "/Users/leiyang/Desktop/Coding/.cursor/debug.log"
_DBG_BUILD = "mem-daily-debug-v1"
def _dlog(msg, data=None, hypothesisId=None, runId="run1", location=None):
    """Write one NDJSON line to debug.log. Never log secrets."""
    try:
        import time as _t
        o = {
            "id": f"log_{int(_t.time()*1000)}",
            "timestamp": int(_t.time() * 1000),
            "location": location or "ChatBot_OpenClaw.py",
            "message": msg,
            "runId": runId,
            "hypothesisId": hypothesisId,
            "data": data or {},
        }
        with open(_DBG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    except Exception:
        pass

_env_file = _env_dir / ".env"
_dlog(
    "startup",
    {
        "build": _DBG_BUILD,
        "script_dir": str(_env_dir),
        "env_file_exists": _env_file.exists(),
        "cwd": str(Path.cwd()),
        "cwd_env_exists": (Path.cwd() / ".env").exists(),
        "CHATBOT_WORKSPACE": bool((os.environ.get("CHATBOT_WORKSPACE") or "").strip()),
        "MEMORY_FLUSH_HISTORY_TURNS": os.environ.get("MEMORY_FLUSH_HISTORY_TURNS", ""),
    },
    "H5",
    location="ChatBot_OpenClaw.py:startup",
)
# #endregion

import base64
import queue
import re
import struct
import subprocess
import tempfile
import threading
import gradio as gr
import requests

API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_SYSTEM = "You are a helpful assistant."
MAX_TOKENS = 512
TEMPERATURE = 0.7

# --- Chat: Grok AI (xAI) ---
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
GROK_CHAT_MODEL = os.environ.get("GROK_CHAT_MODEL", "grok-3")


def _get_chat_config(backend: str, xai_key: str = "") -> tuple[str, dict]:
    """Return (url, headers) for chat completion. backend: 'lan' | 'grok'."""
    if (backend or "").strip().lower() == "grok":
        key = (xai_key or "").strip() or (os.environ.get("XAI_API_KEY") or "").strip()
        return XAI_CHAT_URL, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    return API_URL, {"Content-Type": "application/json"}


def _extract_api_error_message(exc: requests.RequestException, default: str) -> str:
    """Extract error message from API response; supports OpenAI-style and plain message/detail."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None) if resp is not None else None
    if resp is None:
        _dlog("Grok request no response", {"error": str(type(exc).__name__), "message": str(exc)}, "grok_err")
        return "Grok 连接失败（未收到响应），请检查网络、代理或 api.x.ai 是否可访问。"
    try:
        _ = getattr(resp, "content", None) or getattr(resp, "text", None)
    except Exception:
        pass
    text = (getattr(resp, "text", None) or "").strip()
    if not text and getattr(resp, "content", None):
        try:
            text = (resp.content or b"").decode("utf-8", errors="replace").strip()
        except Exception:
            pass
    _dlog("Grok API error response", {"status": status, "text_preview": (text or "")[:600]}, "grok_err")
    try:
        err = resp.json()
        if isinstance(err, dict):
            msg = err.get("error") or {}
            if isinstance(msg, dict):
                msg = msg.get("message") or msg.get("detail") or ""
            else:
                msg = str(msg) if msg else ""
            if not msg:
                msg = err.get("message") or err.get("detail") or ""
            if msg and status is not None:
                return f"[{status}] {msg}"
            if msg:
                return msg
    except Exception:
        pass
    if text and len(text) <= 500:
        return f"[{status}] {text}" if status is not None else text
    status_str = str(status) if status is not None else "?"
    if text:
        return f"[{status_str}] {text[:200]}..." if len(text) > 200 else f"[{status_str}] {text}"
    return f"[{status_str}] 无错误详情，请查看 .cursor/debug.log 中 hypothesisId=grok_err 的条目。"


# --- UI: display name for chat header (set DISPLAY_MODEL_NAME in env to override) ---
DISPLAY_MODEL_NAME = os.environ.get("DISPLAY_MODEL_NAME", "MLX / Qwen")
DISPLAY_SUBTITLE = os.environ.get("DISPLAY_SUBTITLE", "LOCAL MLX SERVER")
APP_HEADER_TITLE = os.environ.get("APP_HEADER_TITLE", "Character Chat")

# --- UI: custom CSS for reference design (purple accents, chat bubbles) ---
UI_CSS = """
/* Purple primary accents */
.primary-btn, .gr-button-primary, button.primary { background: #7c3aed !important; color: white !important; border: none !important; }
.primary-btn:hover, .gr-button-primary:hover { background: #6d28d9 !important; }
/* Chat: user messages right, purple; assistant left, grey */
.gr-chatbot .message.user { margin-left: auto; max-width: 85%; background: #7c3aed !important; color: white !important; border-radius: 1rem 1rem 0.25rem 1rem !important; }
.gr-chatbot .message.bot, .gr-chatbot .message:not(.user) { margin-right: auto; max-width: 85%; background: #f3f4f6 !important; color: #1f2937 !important; border-radius: 1rem 1rem 1rem 0.25rem !important; }
/* Load persona file link-style */
.gr-file .wrap { border: none !important; }
.gr-file label, .gr-file .label { color: #7c3aed !important; text-decoration: underline; cursor: pointer; }
/* Section spacing */
.gr-form, .gr-box { border-radius: 0.5rem; }
.gr-block { margin-bottom: 0.75rem; }
/* Header/footer text */
.ui-header { font-size: 1.125rem; font-weight: 600; color: #374151; margin-bottom: 0.25rem; }
.ui-subtitle { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
.ui-footer { font-size: 0.75rem; color: #9ca3af; text-align: center; margin-top: 0.5rem; }
"""

# --- TTS (DashScope CosyVoice) ---
TTS_MAX_SEGMENT_CHARS = 2000   # CosyVoice per-call limit
TTS_SENTENCE_MAX_CHARS = 200   # max chars per segment when no sentence end found
TTS_SAMPLE_RATE = 22050        # PCM playback rate for CosyVoice default
_SCRIPT_DIR = Path(__file__).resolve().parent
TTS_OUTPUT_WAV = str(_SCRIPT_DIR / "last_tts.wav")  # 固定路径，界面可点击播放

# 局域网 TTS（MLX 兼容接口）
LAN_TTS_DEFAULT_URL = "http://192.168.31.134:9000/v1/audio/speech"
LAN_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
LAN_TTS_MAX_SEGMENT_CHARS = 800   # 单段上限，减少切碎
LAN_TTS_SENTENCE_MAX_CHARS = 400  # 无句末时截断长度

# --- Conversation log (for 调整方向 analysis) ---
CHAT_LOG_DIR = _SCRIPT_DIR / "chat_logs"
CONVERSATIONS_LOG = CHAT_LOG_DIR / "conversations.jsonl"
CONVERSATION_PROMPT_TRUNCATE = 2000  # max chars of character_prompt to store per turn
ANALYSIS_SYSTEM_PROMPT = (
    "你是一个人设优化助手。根据以下对话记录，列出 3–5 条具体、可操作的「调整方向」，"
    "用于修改角色人设/系统提示词以更符合用户期望。每条一行，简短明确。只输出调整建议，不要其他解释。"
)
ANALYSIS_MAX_TURNS = 30   # use last N turns for analysis
ANALYSIS_MAX_CHARS = 12000  # max total chars of dialogue to send to model

# --- Workspace and long-term memory (OpenClaw-style) ---
WORKSPACE_DIR = Path(os.environ.get("CHATBOT_WORKSPACE", str(_SCRIPT_DIR / "workspace")))
MEMORY_MAX_CONTEXT_CHARS = int(os.environ.get("MEMORY_MAX_CONTEXT_CHARS", "4000"))
MEMORY_FLUSH_HISTORY_TURNS = int(os.environ.get("MEMORY_FLUSH_HISTORY_TURNS", "3"))
MEMORY_FLUSH_ESTIMATED_TOKENS = int(os.environ.get("MEMORY_FLUSH_ESTIMATED_TOKENS", "8000"))
MEMORY_EXTRACT_MAX_TURNS = 5
REMEMBER_TRIGGER_PHRASES = (
    "记住", "记一下", "记着", "别忘了",
    "remember this", "save to memory", "remember that", "don't forget",
)
MEMORY_EXTRACT_SYSTEM = (
    "You are a memory extractor. Given the conversation, output what to add to long-term memory. "
    "Use exactly these section headers (one per block):\n"
    "## MEMORY.md\n<content for long-term facts>\n\n"
    "## memory/YYYY-MM-DD.md\n<content for today's log>\n\n"
    "Output only these sections with content; if nothing to store, output exactly: NOTHING"
)
MEMORY_SEARCH_ENABLED = os.environ.get("MEMORY_SEARCH_ENABLED", "").strip().lower() in ("1", "true", "yes")
MEMORY_EMBEDDING_PROVIDER = os.environ.get("MEMORY_EMBEDDING_PROVIDER", "openai").strip().lower()
MEMORY_INDEX_PATH = os.environ.get("MEMORY_INDEX_PATH", str(WORKSPACE_DIR / ".memory_index.sqlite"))
MEMORY_MAX_RESULTS = int(os.environ.get("MEMORY_MAX_RESULTS", "5"))
MEMORY_MAX_SNIPPET_CHARS = int(os.environ.get("MEMORY_MAX_SNIPPET_CHARS", "300"))
MEMORY_CHUNK_SIZE = 400
MEMORY_CHUNK_OVERLAP = 80


def _ensure_workspace_memory_dir() -> Path:
    """Ensure workspace and workspace/memory exist; return workspace dir."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    (WORKSPACE_DIR / "memory").mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR


def read_memory_file(path: Path) -> str:
    """Read a memory file; return content or empty string."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
    except Exception:
        pass
    return ""


def append_to_memory_file(path: Path, content: str, daily_prefix: bool = False) -> None:
    """Append content to a memory file. If daily_prefix, prepend date line for daily logs."""
    if not content or not content.strip():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            if daily_prefix:
                from datetime import datetime
                f.write("\n\n--- " + datetime.now().strftime("%Y-%m-%d %H:%M") + " ---\n\n")
            f.write(content.strip() + "\n")
        # #region agent log
        _dlog(
            "append_to_memory_file ok",
            {"path": str(path), "daily_prefix": bool(daily_prefix), "chars": len(content.strip())},
            "H4",
            location="ChatBot_OpenClaw.py:append_to_memory_file",
        )
        # #endregion
    except Exception:
        # #region agent log
        import traceback as _tb
        _dlog(
            "append_to_memory_file failed",
            {"path": str(path), "daily_prefix": bool(daily_prefix), "error": _tb.format_exc().splitlines()[-1]},
            "H4",
            location="ChatBot_OpenClaw.py:append_to_memory_file",
        )
        # #endregion
        pass


def get_memory_block_for_context(max_chars: int | None = None) -> str:
    """Load MEMORY.md + today's and yesterday's memory/YYYY-MM-DD.md; truncate to max_chars (recent kept)."""
    _ensure_workspace_memory_dir()
    max_chars = max_chars if max_chars is not None else MEMORY_MAX_CONTEXT_CHARS
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    parts = []
    mem_md = WORKSPACE_DIR / "MEMORY.md"
    if mem_md.exists():
        content = read_memory_file(mem_md)
        if content.strip():
            parts.append("### MEMORY.md (long-term)\n\n" + content.strip())
    for date_label in (today, yesterday):
        daily = WORKSPACE_DIR / "memory" / f"{date_label}.md"
        if daily.exists():
            content = read_memory_file(daily)
            if content.strip():
                parts.append(f"### memory/{date_label}.md\n\n" + content.strip())
    if not parts:
        return ""
    combined = "\n\n".join(parts)
    if len(combined) <= max_chars:
        return "\n\n## Long-term memory\n\n" + combined
    # Truncate from start so recent (daily) is kept
    combined = combined[-max_chars:]
    return "\n\n## Long-term memory\n\n" + combined


def _chunk_markdown(path: Path, content: str, chunk_size: int = MEMORY_CHUNK_SIZE, overlap: int = MEMORY_CHUNK_OVERLAP) -> list[dict]:
    """Split markdown into overlapping segments. Return list of {text, path, line_start}."""
    try:
        rel_path = path.relative_to(WORKSPACE_DIR)
    except ValueError:
        rel_path = path
    path_str = str(rel_path).replace("\\", "/")
    chunks = []
    start = 0
    line_start = 1
    while start < len(content):
        end = min(start + chunk_size, len(content))
        if end < len(content):
            for sep in ("\n", "。", ".", " "):
                last = content.rfind(sep, start, end + 1)
                if last >= start:
                    end = last + 1
                    break
        text = content[start:end].strip()
        if text:
            chunks.append({"text": text, "path": path_str, "line_start": line_start})
        line_start += content[start:end].count("\n") + 1
        start = end - overlap if end < len(content) else len(content)
    return chunks


def _embed_texts_openai(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    """Embed texts via OpenAI API. Returns list of vectors; empty on failure."""
    if not texts:
        return []
    key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        r = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [e.embedding for e in r.data]
    except Exception:
        return []


def _memory_index_path() -> Path:
    return Path(MEMORY_INDEX_PATH).resolve()


def _build_memory_index(api_key: str | None = None) -> None:
    """Build or rebuild SQLite index from MEMORY.md + memory/*.md. Requires embeddings."""
    _ensure_workspace_memory_dir()
    index_path = _memory_index_path()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    mem_md = WORKSPACE_DIR / "MEMORY.md"
    if mem_md.exists():
        all_chunks.extend(_chunk_markdown(mem_md, read_memory_file(mem_md)))
    memory_dir = WORKSPACE_DIR / "memory"
    if memory_dir.is_dir():
        for f in memory_dir.glob("*.md"):
            all_chunks.extend(_chunk_markdown(f, read_memory_file(f)))
    if not all_chunks:
        try:
            if index_path.exists():
                index_path.unlink()
        except Exception:
            pass
        return
    texts = [c["text"] for c in all_chunks]
    vectors = _embed_texts_openai(texts, api_key)
    if len(vectors) != len(all_chunks):
        return
    import sqlite3
    conn = sqlite3.connect(str(index_path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_chunks (id INTEGER PRIMARY KEY, path TEXT, line_start INTEGER, text TEXT, embedding BLOB)"
        )
        conn.execute("DELETE FROM memory_chunks")
        for i, (c, vec) in enumerate(zip(all_chunks, vectors)):
            blob = struct.pack("%sf" % len(vec), *vec)
            conn.execute(
                "INSERT INTO memory_chunks (id, path, line_start, text, embedding) VALUES (?,?,?,?,?)",
                (i, c["path"], c["line_start"], c["text"], blob),
            )
        conn.commit()
    finally:
        conn.close()


def memory_search(query: str, k: int | None = None, max_chars: int | None = None, api_key: str | None = None) -> str:
    """Semantic search over memory chunks. Returns formatted snippets up to max_chars."""
    k = k if k is not None else MEMORY_MAX_RESULTS
    max_chars = max_chars if max_chars is not None else MEMORY_MAX_SNIPPET_CHARS * k
    index_path = _memory_index_path()
    if not index_path.exists():
        return ""
    key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return ""
    vecs = _embed_texts_openai([query], api_key=key)
    if not vecs:
        return ""
    qvec = vecs[0]
    dim = len(qvec)
    import sqlite3
    conn = sqlite3.connect(str(index_path))
    try:
        rows = conn.execute("SELECT id, path, line_start, text, embedding FROM memory_chunks").fetchall()
    finally:
        conn.close()
    if not rows:
        return ""
    scores = []
    for row in rows:
        rid, path, line_start, text, blob = row
        vec = list(struct.unpack("%sf" % (len(blob) // 4), blob))
        if len(vec) != dim:
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        norm_q = (sum(x * x for x in qvec)) ** 0.5
        norm_v = (sum(x * x for x in vec)) ** 0.5
        if norm_q and norm_v:
            scores.append((dot / (norm_q * norm_v), path, line_start, text))
    scores.sort(key=lambda x: -x[0])
    parts = []
    total = 0
    for _, path, line_start, text in scores[:k]:
        snippet = (text[:MEMORY_MAX_SNIPPET_CHARS] + "..." if len(text) > MEMORY_MAX_SNIPPET_CHARS else text)
        block = f"[{path}#L{line_start}] {snippet}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    if not parts:
        return ""
    return "\n\n## Long-term memory (relevant)\n\n" + "\n\n".join(parts)


def get_memory_block_for_turn(user_message: str, use_vector: bool | None = None) -> str:
    """Return memory block for system prompt: vector search if enabled and available, else full load."""
    use_vector = use_vector if use_vector is not None else MEMORY_SEARCH_ENABLED
    if use_vector and MEMORY_EMBEDDING_PROVIDER == "openai" and os.environ.get("OPENAI_API_KEY", "").strip():
        idx_path = _memory_index_path()
        if not idx_path.exists():
            _build_memory_index()
        if idx_path.exists():
            block = memory_search(user_message or "")
            if block:
                return block
    return get_memory_block_for_context()


def _maybe_is_game_guide_query(message: str) -> bool:
    """Heuristic: True if user clearly asks for a game guide / 攻略."""
    if not (message or "").strip():
        return False
    text = (message or "").strip()
    lower = text.lower()
    if "攻略" not in text:
        return False
    game_like_keywords = (
        "游戏",
        "通关",
        "过关",
        "boss",
        "BOSS",
        "打法",
        "build",
        "流派",
        "加点",
        "天赋",
        "开荒",
        "前期",
    )
    return any(k in text or k in lower for k in game_like_keywords)


def _call_grok_for_game_guide(message: str) -> str:
    """
    Use Grok (xAI) chat completions API to fetch up-to-date game guide info.
    Requires GROK_API_KEY in environment.
    """
    api_key = (os.environ.get("GROK_API_KEY") or "").strip()
    if not api_key:
        return "当前未配置 GROK_API_KEY，暂时无法通过 Grok 获取游戏攻略。你可以先简单描述游戏和进度，我会用已有知识帮你分析。"
    try:
        url = "https://api.x.ai/v1/chat/completions"
        system_prompt = (
            "You are a game strategy assistant with access to the live internet via Grok. "
            "When the user asks for a game guide or strategy, you must search the web for up-to-date information "
            "and then answer in Chinese. Summarize the key strategy in about 100 Chinese characters, "
            "focusing on concrete, actionable steps (builds, skills, gear, positioning, phase mechanics, etc.). "
            "After the summary, list 3–5 useful guide links, one per line, preferring official or reputable sources."
        )
        payload = {
            "model": "grok-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            "max_tokens": 512,
            "temperature": 0.5,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        content = (content or "").strip()
        if not content:
            return "我尝试通过 Grok 查询游戏攻略，但没有得到有效内容。你可以稍后再试，或提供更多游戏和关卡信息让我用已有知识帮你分析。"
        return content
    except Exception as e:
        _dlog(
            "grok_game_guide_failed",
            {"error": str(e)},
            "H2",
            location="ChatBot_OpenClaw.py:_call_grok_for_game_guide",
        )
        return "我尝试通过 Grok 查询游戏攻略时遇到错误，暂时无法获取最新攻略。你可以先简单描述游戏和进度，我会用已有知识帮你分析。"


def _parse_skill_frontmatter(content: str) -> tuple[str, str, list[str], str | None, list[str] | None]:
    """Parse SKILL.md frontmatter. Returns (name, description, required_env_vars, script, script_args)."""
    name, desc, required_env = "", "", []
    script: str | None = None
    script_args: list[str] | None = None
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return name, desc, required_env, script, script_args
    block = match.group(1)
    for line in block.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("name:"):
            name = line_stripped[5:].strip().strip("'\"")
        elif line_stripped.startswith("description:"):
            desc = line_stripped[12:].strip().strip("'\"")
        elif "env" in line_stripped.lower() and ("requires" in line_stripped or "openclaw" in line_stripped):
            for m in re.finditer(r'["\']?env["\']?\s*:\s*\[\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']', line_stripped):
                required_env.append(m.group(1))
            for m in re.finditer(r'\[\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*\]', line_stripped):
                if "env" in line_stripped[:m.start()]:
                    required_env.append(m.group(1))
        elif re.match(r'^["\']?script["\']?\s*:', line_stripped, re.I):
            # script: "Company_Ressearch.py" or script: Company_Ressearch.py
            rest = line_stripped.split(":", 1)[1].strip().strip("'\"")
            if rest:
                script = rest
        elif re.match(r'^["\']?scriptArgs["\']?\s*:', line_stripped, re.I):
            rest = line_stripped.split(":", 1)[1].strip()
            try:
                script_args = json.loads(rest)
                if not isinstance(script_args, list):
                    script_args = None
            except (json.JSONDecodeError, TypeError):
                script_args = None
        elif line_stripped.startswith("metadata:") and ("openclaw" in line_stripped or "script" in line_stripped or "requires" in line_stripped):
            try:
                meta = json.loads(line_stripped.split(":", 1)[1].strip())
                openclaw = (meta or {}).get("openclaw") or {}
                if isinstance(openclaw, dict):
                    if not script and openclaw.get("script"):
                        script = openclaw["script"]
                    if script_args is None and openclaw.get("scriptArgs") is not None:
                        a = openclaw["scriptArgs"]
                        script_args = a if isinstance(a, list) else None
                    req = openclaw.get("requires") or {}
                    if isinstance(req, dict) and req.get("env") and not required_env:
                        e = req["env"]
                        required_env = list(e) if isinstance(e, list) else [e]
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
    return name, desc, required_env, script, script_args


def get_executable_skills() -> dict[str, tuple[Path, list[str]]]:
    """Load skills that declare a script. Returns {skill_name: (script_path, args_template)}."""
    skills_dir = WORKSPACE_DIR / "skills"
    out: dict[str, tuple[Path, list[str]]] = {}
    if not skills_dir.is_dir():
        return out
    for path in sorted(skills_dir.iterdir()):
        if not path.is_dir():
            continue
        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            continue
        try:
            raw = read_memory_file(skill_md)
            name, _desc, required_env, script, script_args = _parse_skill_frontmatter(raw)
            if not name:
                name = path.name
            if required_env and not all(os.environ.get(v) for v in required_env):
                continue
            if not script:
                continue
            script_path = path / script
            if not script_path.is_file():
                continue
            args_template = script_args if isinstance(script_args, list) else []
            out[name] = (script_path, args_template)
        except Exception:
            continue
    return out


def _skill_body_after_frontmatter(content: str) -> str:
    """Return the content of SKILL.md after the closing --- of frontmatter (trigger / Use this when...)."""
    match = re.match(r"^---\s*\n.*?\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def get_skills_block(include_exec_instructions: bool = True) -> str:
    """Load skills from workspace/skills; parse SKILL.md (frontmatter + gating by env); format for system prompt."""
    skills_dir = WORKSPACE_DIR / "skills"
    if not skills_dir.is_dir():
        return ""
    entries = []
    for path in sorted(skills_dir.iterdir()):
        if not path.is_dir():
            continue
        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            continue
        try:
            raw = read_memory_file(skill_md)
            name, description, required_env, _script, _script_args = _parse_skill_frontmatter(raw)
            if not name:
                name = path.name
            if not description:
                description = "(no description)"
            if required_env:
                if not all(os.environ.get(v) for v in required_env):
                    continue
            body = _skill_body_after_frontmatter(raw)
            entries.append((name, description, body))
        except Exception:
            continue
    if not entries:
        return ""
    lines = []
    for name, description, body in entries:
        lines.append("- **%s**: %s" % (name, description))
        if body:
            lines.append("  **Trigger (use this when):**")
            for bline in body.split("\n"):
                line_stripped = bline.strip()
                if line_stripped:
                    lines.append("    " + line_stripped)
    block = "\n\n## Available skills\n\n" + "\n".join(lines)
    # When skill execution is enabled, add instructions for executable skills
    if include_exec_instructions:
        exec_skills = get_executable_skills()
        if exec_skills:
            block += "\n\n## Skill script execution\n\n"
            block += "When you need to run a skill script, output exactly:\n"
            block += "```\n[[SKILL:<skill name>]]\nkey1=value1\nkey2=value2\n[[/SKILL]]\n```\n"
            block += "Use one line per argument (key=value). Skill names and argument keys are case-sensitive.\n"
            block += "Executable skills (argument keys to pass):\n"
            for sname, (spath, arg_tpl) in exec_skills.items():
                keys = list({m.group(1) for p in arg_tpl if isinstance(p, str) for m in re.finditer(r"\{(\w+)\}", p)})
                block += f"- **{sname}**: " + (", ".join(keys) if keys else "(no args)") + "\n"
    return block


def _parse_skill_invocation(text: str) -> tuple[str, dict[str, str]] | None:
    """If text contains [[SKILL:name]] ... [[/SKILL]], return (skill_name, args_dict) else None."""
    match = re.search(r"\[\[SKILL:\s*(.+?)\s*\]\]\s*(.*?)\s*\[\[/SKILL\]\]", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    skill_name = match.group(1).strip()
    body = (match.group(2) or "").strip()
    args = {}
    for line in body.split("\n"):
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            args[k.strip()] = v.strip()
    return (skill_name, args)


def execute_skill_script(skill_name: str, args: dict[str, str], timeout_seconds: int = 300) -> str:
    """
    Run a skill's script. Only runs scripts declared in SKILL.md under workspace/skills (allowlist).
    Returns combined stdout and stderr, or an error message string.
    """
    executable = get_executable_skills()
    if skill_name not in executable:
        return f"[Skill exec error] Unknown or non-executable skill: {skill_name!r}"
    script_path, args_template = executable[skill_name]
    skill_dir = script_path.parent
    cli_parts = []
    for part in args_template:
        if isinstance(part, str):
            for key, val in args.items():
                part = part.replace("{" + key + "}", str(val))
            cli_parts.append(part)
    try:
        result = subprocess.run(
            [os.environ.get("PYTHON", "python3"), str(script_path)] + cli_parts,
            cwd=str(skill_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ},
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode != 0:
            return f"[Skill exit {result.returncode}]\n{out}\n{err}".strip() or f"Exit code {result.returncode}"
        return f"{out}\n{err}".strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[Skill exec error] Script timed out after {timeout_seconds}s"
    except Exception as e:
        return f"[Skill exec error] {e}"


def _user_wants_remember(user_message: str) -> bool:
    """True if user message clearly asks to remember something."""
    if not (user_message or user_message.strip()):
        return False
    lower = user_message.strip().lower()
    return any(phrase in lower or phrase in user_message for phrase in REMEMBER_TRIGGER_PHRASES)


# Tracks how many turns have already been flushed to memory for the current session.
# Without this, once history exceeds the threshold, flushing would happen on every turn.
_MEMORY_FLUSH_CURSOR_TURNS = 0


def _should_flush_memory_for_history(history_tuples: list) -> bool:
    """True if *new* turns since last flush exceed turn/token thresholds."""
    global _MEMORY_FLUSH_CURSOR_TURNS
    n_turns = len(history_tuples)
    # If history was cleared/shortened, reset cursor.
    if n_turns < _MEMORY_FLUSH_CURSOR_TURNS:
        _MEMORY_FLUSH_CURSOR_TURNS = 0

    delta = history_tuples[_MEMORY_FLUSH_CURSOR_TURNS:]
    if len(delta) >= MEMORY_FLUSH_HISTORY_TURNS:
        return True

    est_chars = sum(len(u) + len(a) for u, a in delta)
    est_tokens = est_chars // 4
    return est_tokens >= MEMORY_FLUSH_ESTIMATED_TOKENS


def run_memory_extract_and_append(
    user_message: str,
    assistant_reply: str,
    character_prompt: str = "",
    extra_turns: list[tuple[str, str]] | None = None,
) -> None:
    """Non-streaming call to extract memory from exchange; append to MEMORY.md and/or memory/YYYY-MM-DD.md."""
    from datetime import datetime
    _ensure_workspace_memory_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    # Some models may not replace YYYY-MM-DD reliably; make the required header explicit.
    memory_extract_system = MEMORY_EXTRACT_SYSTEM.replace("YYYY-MM-DD", today) + f"\n\nToday's date is {today}. Use ONLY memory/{today}.md for the daily log header."
    # #region agent log
    _dlog(
        "memory_extract enter",
        {
            "today": today,
            "workspace_dir": str(WORKSPACE_DIR),
            "mem_md": str(WORKSPACE_DIR / "MEMORY.md"),
            "daily_md": str(WORKSPACE_DIR / "memory" / f"{today}.md"),
            "extra_turns": len(extra_turns or []),
            "user_len": len(user_message or ""),
            "assistant_len": len(assistant_reply or ""),
        },
        "H1",
        location="ChatBot_OpenClaw.py:run_memory_extract_and_append:enter",
    )
    # #endregion
    dialogue = f"User: {user_message}\nAssistant: {assistant_reply}"
    if extra_turns:
        for u, a in extra_turns[-MEMORY_EXTRACT_MAX_TURNS:]:
            dialogue = f"User: {u}\nAssistant: {a}\n\n" + dialogue
    payload = {
        "messages": [
            {"role": "system", "content": memory_extract_system},
            {"role": "user", "content": dialogue},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": False,
    }
    try:
        r = requests.post(API_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except Exception:
        # #region agent log
        import traceback as _tb
        _dlog(
            "memory_extract request failed",
            {"api_url": API_URL, "error": _tb.format_exc().splitlines()[-1]},
            "H2",
            location="ChatBot_OpenClaw.py:run_memory_extract_and_append:request",
        )
        # #endregion
        return
    content = content.strip()
    if not content or content.upper() == "NOTHING":
        # #region agent log
        _dlog(
            "memory_extract returned NOTHING/empty",
            {"content_len": len(content), "content_upper": content[:20].upper() if content else ""},
            "H3",
            location="ChatBot_OpenClaw.py:run_memory_extract_and_append:content",
        )
        # #endregion
        return
    # #region agent log
    header_lines = []
    for ln in content.split("\n"):
        if ln.strip().startswith("##"):
            header_lines.append(ln.strip())
        if len(header_lines) >= 10:
            break
    _dlog(
        "memory_extract got content",
        {
            "content_len": len(content),
            "headers": header_lines,
            "has_memory_header": any("MEMORY.md" in h for h in header_lines),
            "has_any_daily_header": any(h.lower().startswith("## memory/") for h in header_lines),
        },
        "H3",
        location="ChatBot_OpenClaw.py:run_memory_extract_and_append:content",
    )
    # #endregion
    mem_md = WORKSPACE_DIR / "MEMORY.md"
    daily_md = WORKSPACE_DIR / "memory" / f"{today}.md"
    in_memory = False
    in_daily = False
    buf_memory = []
    buf_daily = []
    for line in content.split("\n"):
        if re.match(r"^\s*##\s+MEMORY\.md\s*$", line, re.IGNORECASE):
            in_memory = True
            in_daily = False
            continue
        # Be tolerant: treat any "memory/<something>.md" header as today's daily log.
        # (Models sometimes emit a different date; we still want to write to today's file.)
        if re.match(r"^\s*##\s+memory/.*\.md\s*$", line, re.IGNORECASE):
            in_daily = True
            in_memory = False
            continue
        if in_memory:
            buf_memory.append(line)
        elif in_daily:
            buf_daily.append(line)
    # #region agent log
    _dlog(
        "memory_extract parsed buffers",
        {"buf_memory_lines": len(buf_memory), "buf_daily_lines": len(buf_daily)},
        "H3",
        location="ChatBot_OpenClaw.py:run_memory_extract_and_append:parse",
    )
    # #endregion
    if buf_memory:
        append_to_memory_file(mem_md, "\n".join(buf_memory).strip())
    if buf_daily:
        append_to_memory_file(daily_md, "\n".join(buf_daily).strip(), daily_prefix=True)
    else:
        # #region agent log
        _dlog(
            "memory_extract no daily buffer -> no daily write",
            {"daily_md": str(daily_md)},
            "H3",
            location="ChatBot_OpenClaw.py:run_memory_extract_and_append:write",
        )
        # #endregion


def _append_conversation_log(character_prompt: str, user_msg: str, assistant_reply: str) -> None:
    """Append one turn to conversations.jsonl. Creates chat_logs dir if needed."""
    if not (user_msg or assistant_reply):
        return
    try:
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        prompt_snippet = (character_prompt or "").strip()
        if len(prompt_snippet) > CONVERSATION_PROMPT_TRUNCATE:
            prompt_snippet = prompt_snippet[:CONVERSATION_PROMPT_TRUNCATE] + "..."
        import time as _t
        record = {
            "timestamp": _t.time(),
            "character_prompt": prompt_snippet,
            "user": user_msg.strip(),
            "assistant": (assistant_reply or "").strip(),
        }
        with open(CONVERSATIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _rule_based_prompt_ideas(entries: list) -> str:
    """Fallback: heuristics over user messages to suggest prompt adjustments."""
    lines = []
    correction_phrases = ("不对", "不是", "应该是", "别这样", "错了", "不是这样", "别这么说")
    length_short = ("太长了", "简短点", "短一点", "简短些", "不要太长", "精简")
    length_long = ("多说点", "详细点", "展开说说", "再详细")
    tone_formal = ("别这么正式", "随意点", "轻松点", "别端着")
    for e in entries:
        user = (e.get("user") or "").strip()
        if not user:
            continue
        u_lower = user.lower()
        for p in correction_phrases:
            if p in user or p in u_lower:
                lines.append("· 用户曾纠正回复内容，可在人设中更明确相关设定或禁忌。")
                break
        for p in length_short:
            if p in user:
                lines.append("· 用户希望回复更短，可在人设中增加「简短回应」「少说废话」等指示。")
                break
        for p in length_long:
            if p in user:
                lines.append("· 用户希望回复更详细，可在人设中允许或鼓励展开说明。")
                break
        for p in tone_formal:
            if p in user:
                lines.append("· 用户希望语气更随意，可在人设中强调口语化、轻松风格。")
                break
    if not lines:
        return "（根据当前对话未检测到明显的纠正或偏好，可多聊几轮后再分析，或使用 MLX 服务器做更细分析。）"
    return "\n".join(dict.fromkeys(lines))  # dedupe while preserving order


def analyze_log_for_prompt_ideas(log_path: Path) -> str:
    """Read conversation log, call MLX for 3–5 调整方向; on failure use rule-based fallback."""
    if not log_path or not log_path.exists() or log_path.stat().st_size == 0:
        return "暂无对话记录，请先进行几轮对话。"
    entries = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return "无法读取对话记录。"
    if not entries:
        return "暂无对话记录，请先进行几轮对话。"
    # Use last N turns and cap total chars
    recent = entries[-ANALYSIS_MAX_TURNS:] if len(entries) > ANALYSIS_MAX_TURNS else entries
    parts = []
    total = 0
    for e in reversed(recent):
        user = (e.get("user") or "").strip()
        asst = (e.get("assistant") or "").strip()
        block = f"用户：{user}\n助手：{asst}\n"
        if total + len(block) > ANALYSIS_MAX_CHARS:
            break
        parts.append(block)
        total += len(block)
    parts.reverse()
    dialogue = "\n".join(parts).strip()
    if not dialogue:
        return "对话内容为空，请先进行几轮对话。"
    # Try MLX
    try:
        payload = {
            "messages": [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": "对话记录：\n\n" + dialogue},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
            "stream": False,
        }
        r = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if content and content.strip():
            return content.strip()
    except Exception:
        pass
    return _rule_based_prompt_ideas(entries)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Build a minimal WAV file from raw PCM 16-bit mono little-endian bytes."""
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * n_channels * (bits_per_sample // 8)
    block_align = n_channels * (bits_per_sample // 8)
    data_size = len(pcm_bytes)
    chunk_size = 4 + 8 + 16 + 8 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        n_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _strip_parenthetical_for_tts(text: str) -> str:
    """
    TTS 前清洗：移除各类括号及其中的内容（含括号本身）。
    括号内多为动作、表情等非对白（如 扑过来抱抱），不送入 TTS，仅对白部分会被朗读。
    支持：英文 ()、中文 （）、方括号【】、［］、「」等。仅用于语音合成。
    """
    if not text:
        return ""
    s = str(text)
    for _ in range(20):
        prev = s
        s = re.sub(r"\([^()]*\)", "", s)
        s = re.sub(r"（[^（）]*）", "", s)
        s = re.sub(r"\[[^\]\[]*\]", "", s)
        s = re.sub(r"【[^】]*】", "", s)
        s = re.sub(r"［[^］]*］", "", s)
        s = re.sub(r"「[^」]*」", "", s)
        s = re.sub(r"〈[^〉]*〉", "", s)
        s = re.sub(r"《[^》]*》", "", s)
        s = re.sub(r"〔[^〕]*〕", "", s)
        s = re.sub(r"﹙[^﹚]*﹚", "", s)
        if s == prev:
            break
    s = re.sub(r"[()（）\[\]【】［］「」〈〉《》〔〕﹙﹚]", "", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _strip_complete_round_parens(text: str) -> str:
    """Remove complete round-bracket pairs （）() and mixed; leaves unpaired brackets."""
    if not text:
        return ""
    s = str(text)
    for _ in range(20):
        prev = s
        s = re.sub(r"\([^()]*\)", "", s)
        s = re.sub(r"（[^（）]*）", "", s)
        s = re.sub(r"[（(][^）)]*[）)]", "", s)
        if s == prev:
            break
    return s


def _process_segment_for_tts(seg: str, inside_parens: bool) -> tuple[str, bool]:
    """
    Process one TTS segment with cross-segment parenthesis state.
    Case 1: segment has complete () pairs -> strip them, send the rest.
    Case 2: segment has unclosed （ or ( -> send only before first such; set inside_parens.
    When inside_parens: skip until first ） or ), then send content after; clear state.
    Returns (text_to_send, new_inside_parens).
    """
    if not seg:
        return "", inside_parens
    if inside_parens:
        for i, c in enumerate(seg):
            if c in "）)":
                after = _strip_complete_round_parens(seg[i + 1 :])
                idx_open = next((j for j, ch in enumerate(after) if ch in "（("), -1)
                if idx_open >= 0:
                    to_send = after[:idx_open].strip()
                    return (to_send, True) if to_send else ("", True)
                return (after.strip(), False) if after.strip() else ("", False)
        return "", True
    s = _strip_complete_round_parens(seg)
    idx_open = next((j for j, ch in enumerate(s) if ch in "（("), -1)
    if idx_open >= 0:
        to_send = s[:idx_open].strip()
        return (to_send, True) if to_send else ("", True)
    return (s.strip(), False) if s.strip() else ("", False)


def _split_next_tts_segment(
    buffer: str,
    max_segment_chars: int | None = None,
    sentence_max_chars: int | None = None,
) -> tuple[str, str]:
    """
    从 buffer 中取出下一段用于 TTS 的文本（到句号/问号/感叹号/换行或长度上限），
    返回 (segment, remaining)。单段不超过 max_segment_chars（默认 TTS_MAX_SEGMENT_CHARS）。
    """
    max_seg = max_segment_chars if max_segment_chars is not None else TTS_MAX_SEGMENT_CHARS
    sent_max = sentence_max_chars if sentence_max_chars is not None else TTS_SENTENCE_MAX_CHARS
    if not buffer or not buffer.strip():
        return "", ""
    s = buffer.strip()
    if len(s) <= max_seg:
        for sep in ("。", "！", "？", "!", "?", ".", "\n"):
            i = s.rfind(sep)
            if i != -1:
                return s[: i + 1].strip(), s[i + 1 :].strip()
        if len(s) <= sent_max:
            return s, ""
        seg = s[:sent_max]
        return seg, s[sent_max:].strip()
    chunk = s[:max_seg]
    for sep in ("。", "！", "？", "!", "?", ".", "\n"):
        i = chunk.rfind(sep)
        if i != -1:
            return chunk[: i + 1].strip(), s[i + 1 :].strip()
    seg = chunk[:sent_max]
    return seg, s[len(seg) :].strip()


def _play_wav_file(wav_path: str) -> None:
    """Play a WAV file with system player (afplay on macOS, aplay on Linux)."""
    import sys
    try:
        if sys.platform == "darwin":
            subprocess.run(["afplay", wav_path], check=True, timeout=300, capture_output=True)
        elif sys.platform.startswith("linux"):
            subprocess.run(["aplay", "-q", wav_path], check=True, timeout=300, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _dlog("TTS playback failed", {"path": wav_path, "error": str(type(e).__name__), "message": str(e)}, "play_wav")


def _tts_lan_one_segment(text: str, url: str, voice: str = "", instruction_prompt: str = "") -> bool:
    """局域网 TTS：对一段文本请求 WAV，写入临时文件并播放。CustomVoice 支持 extra_body.instruction_prompt。"""
    if not text or not url:
        return False
    text = _strip_parenthetical_for_tts(text)
    if not text:
        return False
    body: dict = {
        "model": LAN_TTS_MODEL,
        "input": text,
        "response_format": "wav",
        "speed": 1.0,
    }
    if (voice or "").strip():
        body["voice"] = voice.strip()
    if (instruction_prompt or "").strip():
        body["extra_body"] = {"instruction_prompt": instruction_prompt.strip()}
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        content_type = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        raw = r.content
        _dlog("LAN TTS response", {"url": url, "status": r.status_code, "content_type": content_type, "len": len(raw), "first_bytes": raw[:20].hex() if len(raw) >= 20 else raw.hex()}, "lan_tts")
        r.raise_for_status()
        wav_bytes = raw
        if len(raw) >= 1 and raw[:1] == b"{":
            try:
                body = json.loads(raw.decode("utf-8"))
                b64 = body.get("audio") or body.get("data") or body.get("content")
                if b64 is not None:
                    wav_bytes = base64.b64decode(b64)
                    _dlog("LAN TTS decoded base64", {"decoded_len": len(wav_bytes)}, "lan_tts")
            except Exception as e:
                _dlog("LAN TTS JSON/base64 decode failed", {"error": str(e), "preview": raw[:200]}, "lan_tts")
                return False
        if len(wav_bytes) < 100:
            _dlog("LAN TTS audio too short", {"len": len(wav_bytes)}, "lan_tts")
            return False
        if wav_bytes[:4] != b"RIFF":
            _dlog("LAN TTS not WAV (no RIFF)", {"first4": wav_bytes[:4].hex()}, "lan_tts")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        _play_wav_file(path)
        try:
            os.unlink(path)
        except Exception:
            pass
        return True
    except requests.RequestException as e:
        resp = getattr(e, "response", None)
        _dlog("LAN TTS request failed", {"url": url, "error": str(e), "response_preview": (resp.text[:300] if resp is not None else None)}, "lan_tts")
        return False
    except Exception as e:
        _dlog("LAN TTS error", {"url": url, "error": str(type(e).__name__), "message": str(e)}, "lan_tts")
        return False


def _tts_lan_sender_worker(segment_queue: queue.Queue, lan_url: str, lan_voice: str = "", lan_instruction: str = "") -> None:
    """从 segment_queue 取段，逐段调用局域网 TTS 并播放。"""
    while True:
        try:
            seg = segment_queue.get(timeout=30)
        except queue.Empty:
            continue
        if seg is None:
            break
        cleaned = _strip_parenthetical_for_tts(seg)
        if cleaned:
            _dlog("LAN TTS segment", {"len": len(cleaned), "preview": cleaned[:50]}, "lan_tts")
            _tts_lan_one_segment(cleaned, lan_url, voice=lan_voice, instruction_prompt=lan_instruction)
        else:
            _dlog("LAN TTS segment empty after clean", {"orig_len": len(seg)}, "lan_tts")


def _tts_playback_worker(
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    fallback_wav_path: str | None = None,
) -> None:
    """
    Consume PCM from audio_queue. Prefer pyaudio for real-time playback.
    If pyaudio is unavailable or fails, collect PCM and play via WAV file (afplay/aplay).
    """
    use_pyaudio = False
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=TTS_SAMPLE_RATE,
            output=True,
        )
        use_pyaudio = True
    except (ImportError, OSError, Exception):
        pa = None
        stream = None

    collected: list[bytes] = []
    while not stop_event.is_set():
        try:
            data = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if data is None:
            break
        if use_pyaudio and stream is not None:
            try:
                stream.write(data)
            except Exception:
                use_pyaudio = False
                collected.append(data)
        else:
            collected.append(data)

    if pa and stream is not None:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass

    # Fallback: write collected PCM to WAV and play with system player（不删除，供界面点击播放）
    if collected:
        pcm = b"".join(collected)
        if pcm:
            wav_bytes = _pcm_to_wav(pcm, TTS_SAMPLE_RATE)
            path = fallback_wav_path or tempfile.mktemp(suffix=".wav")
            try:
                with open(path, "wb") as f:
                    f.write(wav_bytes)
                _play_wav_file(path)
            except Exception:
                pass


def _tts_sender_worker(
    api_key: str,
    segment_queue: queue.Queue,
    audio_queue: queue.Queue,
) -> None:
    """Run CosyVoice streaming: consume segments, streaming_call each, then streaming_complete."""
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    try:
        from dashscope.audio.tts_v2 import (
            AudioFormat,
            ResultCallback,
            SpeechSynthesizer,
        )
    except ImportError:
        return
    # Wrap our queue-putting callback so it matches ResultCallback interface
    class Callback(ResultCallback):
        def __init__(self, q: queue.Queue):
            self._q = q

        def on_data(self, data: bytes) -> None:
            if data:
                self._q.put(data)

        def on_open(self) -> None:
            pass

        def on_complete(self) -> None:
            pass

        def on_error(self, message: str) -> None:
            pass

        def on_close(self) -> None:
            pass

        def on_event(self, message: str) -> None:
            pass

    cb = Callback(audio_queue)
    synthesizer = SpeechSynthesizer(
        model="cosyvoice-v3-flash",
        voice="longanhuan",
        format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        callback=cb,
    )
    while True:
        try:
            seg = segment_queue.get(timeout=30)
        except queue.Empty:
            continue
        if seg is None:
            break
        cleaned = _strip_parenthetical_for_tts(seg)
        if not cleaned:
            continue
        try:
            synthesizer.streaming_call(cleaned)
        except Exception:
            pass
    try:
        synthesizer.streaming_complete()
    except Exception:
        pass
    audio_queue.put(None)


def _tts_sync_one_shot(text: str, api_key: str) -> bool:
    """
    同步合成整段文本为语音，写入 TTS_OUTPUT_WAV。
    流式无数据时用作备用。返回 True 表示成功写入。
    """
    if not text or not text.strip():
        return False
    text = _strip_parenthetical_for_tts(text)
    if not text or len(text) > TTS_MAX_SEGMENT_CHARS:
        text = text[:TTS_MAX_SEGMENT_CHARS] if text else ""
    if not text:
        return False
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    try:
        from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer
    except ImportError:
        return False
    try:
        syn = SpeechSynthesizer(
            model="cosyvoice-v3-flash",
            voice="longanhuan",
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        )
        result = syn.call(text)
        if result is None:
            return False
        if hasattr(result, "get_audio_frame"):
            raw = result.get_audio_frame()
        elif isinstance(result, bytes):
            raw = result
        else:
            raw = getattr(result, "output", None) or getattr(result, "data", None)
            if isinstance(raw, bytes):
                pass
            elif raw is not None and hasattr(raw, "read"):
                raw = raw.read()
            else:
                raw = None
        if not raw or not isinstance(raw, bytes):
            return False
        wav = _pcm_to_wav(raw, TTS_SAMPLE_RATE)
        with open(TTS_OUTPUT_WAV, "wb") as f:
            f.write(wav)
        _play_wav_file(TTS_OUTPUT_WAV)
        return True
    except Exception:
        return False


def _start_tts_session(api_key: str, tts_backend: str = "dashscope", lan_tts_url: str = "", lan_tts_voice: str = "", lan_tts_instruction: str = ""):
    """
    Start TTS for one reply: playback thread + sender thread (or LAN sender only).
    Returns (push_segment_fn, finish_fn). finish_fn() returns path to WAV (or None).
    """
    if tts_backend == "lan" and ((lan_tts_url or "").strip() or LAN_TTS_DEFAULT_URL):
        url = (lan_tts_url or "").strip() or LAN_TTS_DEFAULT_URL
        segment_queue = queue.Queue()
        sender = threading.Thread(
            target=_tts_lan_sender_worker,
            args=(segment_queue, url, (lan_tts_voice or "").strip(), (lan_tts_instruction or "").strip()),
            daemon=True,
        )
        sender.start()

        def push_segment(text: str) -> None:
            if text and text.strip():
                segment_queue.put(text.strip())

        def finish() -> str | None:
            segment_queue.put(None)
            sender.join(timeout=60)
            return None

        return push_segment, finish

    segment_queue = queue.Queue()
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    playback = threading.Thread(
        target=_tts_playback_worker,
        args=(audio_queue, stop_event, TTS_OUTPUT_WAV),
        daemon=True,
    )
    sender = threading.Thread(
        target=_tts_sender_worker,
        args=(api_key, segment_queue, audio_queue),
        daemon=True,
    )
    playback.start()
    sender.start()

    def push_segment(text: str) -> None:
        if text and text.strip():
            segment_queue.put(text.strip())

    def finish() -> str | None:
        segment_queue.put(None)
        sender.join(timeout=60)
        audio_queue.put(None)
        # 等播放线程取完队列并播完再返回，不要先 set stop_event 否则会半路退出
        playback.join(timeout=300)
        stop_event.set()
        if os.path.exists(TTS_OUTPUT_WAV) and os.path.getsize(TTS_OUTPUT_WAV) > 500:
            return TTS_OUTPUT_WAV
        return None

    return push_segment, finish


def _history_to_tuples(history: list) -> list:
    """Convert Gradio 4 message list to (user, assistant) pairs for internal use."""
    if not history:
        return []
    if isinstance(history[0], dict):
        pairs = []
        i = 0
        while i < len(history):
            role = history[i].get("role", "")
            content = history[i].get("content", "") or ""
            if role == "user":
                asst = ""
                if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                    asst = history[i + 1].get("content", "") or ""
                    i += 1
                pairs.append((content, asst))
            i += 1
        return pairs
    return list(history)


def _tuples_to_messages(history: list) -> list:
    """Convert (user, assistant) pairs to Gradio 4 message format."""
    out = []
    for user, assistant in history:
        if user:
            out.append({"role": "user", "content": user})
        if assistant:
            out.append({"role": "assistant", "content": assistant})
    return out


def build_messages(
    character_prompt: str,
    history: list,
    user_message: str,
    memory_block: str = "",
    skills_block: str = "",
) -> list:
    """Build API messages: system (character + memory + skills), history, current user message."""
    system = (character_prompt or "").strip() or DEFAULT_SYSTEM
    if memory_block:
        system = system + memory_block
    if skills_block:
        system = system + skills_block
    messages = [{"role": "system", "content": system}]
    tuples = _history_to_tuples(history)
    for user, assistant in tuples:
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})
    return messages


TTS_CUMULATIVE_CHAR_LIMIT = 200_000  # CosyVoice streaming session limit


def stream_chat(
    character_prompt: str,
    history: list,
    message: str,
    enable_tts: bool = False,
    tts_api_key: str = "",
    tts_backend: str = "dashscope",
    lan_tts_url: str = "",
    lan_tts_voice: str = "",
    lan_tts_instruction: str = "",
    temperature: float = 0.7,
    use_memory: bool = True,
    use_vector_search: bool | None = None,
    enable_skill_exec: bool | None = None,
    chat_backend: str = "lan",
    xai_api_key: str = "",
):
    """Send request to chat API (LAN MLX or Grok) and stream reply; yield (history, '', tts_status) for Gradio."""
    global _MEMORY_FLUSH_CURSOR_TURNS
    if enable_skill_exec is None:
        enable_skill_exec = os.environ.get("SKILL_EXEC_ENABLED", "").strip().lower() in ("1", "true", "yes")
    skill_exec_timeout = int(os.environ.get("SKILL_EXEC_TIMEOUT", "300"))
    if not (message or message.strip()):
        yield _tuples_to_messages(_history_to_tuples(history)), "", "", None
        return

    # Special case: user explicitly asks for a game guide / 攻略 → call Grok instead of local MLX.
    if _maybe_is_game_guide_query(message):
        tuples = _history_to_tuples(history)
        user_msg = (message or "").strip()
        guide = _call_grok_for_game_guide(user_msg)
        full = guide or "我暂时无法获取该游戏的攻略，你可以稍后再试，或告诉我更多细节让我用已有知识帮你分析。"
        _append_conversation_log(character_prompt or "", user_msg, full)
        yield _tuples_to_messages(tuples + [[user_msg, full]]), "", "", None
        return

    tuples = _history_to_tuples(history)
    new_tuples = tuples + [[message.strip(), ""]]
    yield _tuples_to_messages(new_tuples), "", "", None

    memory_block = get_memory_block_for_turn(message.strip(), use_vector=use_vector_search) if use_memory else ""
    skills_block = get_skills_block(include_exec_instructions=enable_skill_exec)
    messages = build_messages(
        character_prompt, history, message.strip(),
        memory_block=memory_block,
        skills_block=skills_block,
    )
    payload = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "stream": True,
    }
    url, headers = _get_chat_config(chat_backend or "lan", xai_api_key or "")
    if (chat_backend or "").strip().lower() == "grok":
        payload["model"] = GROK_CHAT_MODEL
    if (chat_backend or "").strip().lower() == "grok" and not (headers.get("Authorization") or "").replace("Bearer ", "").strip():
        yield _tuples_to_messages(tuples + [[message.strip(), "Grok AI 需要设置 XAI_API_KEY（.env 或环境变量）。"]]), "", "", None
        return

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        is_grok = (chat_backend or "").strip().lower() == "grok"
        error_msg = _extract_api_error_message(e, "Grok 请求失败，详见 .cursor/debug.log 中 hypothesisId=grok_err。") if is_grok else "Could not reach MLX server. Is it running on port 8000?"
        if not is_grok and getattr(e, "response", None) is not None:
            try:
                err = e.response.json()
                error_msg = err.get("error", {}).get("message", error_msg)
            except Exception:
                pass
        yield _tuples_to_messages(tuples + [[message.strip(), error_msg]]), "", "", None
        return

    tts_push = None
    tts_finish = None
    tts_buffer = ""
    tts_inside_parens = False
    tts_total_chars = 0
    api_key = (tts_api_key or "").strip() or (os.environ.get("DASHSCOPE_API_KEY") or "").strip() or ""
    use_lan = (tts_backend or "dashscope").strip().lower() == "lan"
    lan_url = (lan_tts_url or "").strip() or LAN_TTS_DEFAULT_URL
    # #region agent log
    _dlog("TTS api_key resolution", {"enable_tts": enable_tts, "tts_backend": tts_backend, "use_lan": use_lan, "has_ui_key": bool((tts_api_key or "").strip()), "env_has_key": bool((os.environ.get("DASHSCOPE_API_KEY") or "").strip()), "api_key_non_empty": bool(api_key)}, "H5")
    # #endregion
    if enable_tts:
        if use_lan:
            try:
                tts_push, tts_finish = _start_tts_session("", tts_backend="lan", lan_tts_url=lan_url, lan_tts_voice=lan_tts_voice, lan_tts_instruction=lan_tts_instruction)
            except Exception:
                tts_push = None
                tts_finish = None
        elif api_key:
            try:
                tts_push, tts_finish = _start_tts_session(api_key, tts_backend="dashscope")
            except Exception:
                tts_push = None
                tts_finish = None

    def _tts_status() -> str:
        if enable_tts and tts_push is None:
            if use_lan:
                return "局域网 TTS 暂时不可用"
            return "TTS 未配置 Key（请设置 DASHSCOPE_API_KEY 或填写 API Key）" if not api_key else "TTS 暂时不可用"
        return "语音播放中…" if tts_push else ""

    full = ""
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload_line = line[6:].decode()
        if payload_line.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload_line)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            part = delta.get("content") or ""
            full += part
            if tts_push is not None:
                tts_buffer += part
                while True:
                    seg, remaining = _split_next_tts_segment(
                        tts_buffer,
                        LAN_TTS_MAX_SEGMENT_CHARS if use_lan else None,
                        LAN_TTS_SENTENCE_MAX_CHARS if use_lan else None,
                    )
                    if not seg:
                        break
                    tts_buffer = remaining
                    to_send, tts_inside_parens = _process_segment_for_tts(seg, tts_inside_parens)
                    if to_send and tts_total_chars <= TTS_CUMULATIVE_CHAR_LIMIT:
                        tts_total_chars += len(to_send)
                        tts_push(to_send)
                status = "语音播放中…"
            else:
                status = _tts_status()
            yield _tuples_to_messages(tuples + [[message.strip(), full]]), "", status, None
        except json.JSONDecodeError:
            continue

    # Optional: if assistant reply contains [[SKILL:...]] and execution is enabled, run script and synthesize
    if enable_skill_exec and full:
        parsed = _parse_skill_invocation(full)
        if parsed:
            skill_name, args = parsed
            print(f"[Skill] Started: {skill_name} (args: {args})", flush=True)
            result = execute_skill_script(skill_name, args, timeout_seconds=skill_exec_timeout)
            print(f"[Skill] Completed: {skill_name}", flush=True)
            follow_up = (
                "The following is the output from the skill script. "
                "Provide a concise user-facing summary or answer based on this.\n\n"
                "--- Script output ---\n"
                f"{result}\n"
                "--- End ---"
            )
            messages_after = messages + [
                {"role": "assistant", "content": full},
                {"role": "user", "content": follow_up},
            ]
            try:
                payload2 = {
                    "messages": messages_after,
                    "max_tokens": MAX_TOKENS,
                    "temperature": temperature,
                    "stream": False,
                }
                if (chat_backend or "").strip().lower() == "grok":
                    payload2["model"] = GROK_CHAT_MODEL
                resp2 = requests.post(
                    url,
                    headers=headers,
                    json=payload2,
                    timeout=90,
                )
                resp2.raise_for_status()
                data2 = resp2.json()
                choice = (data2.get("choices") or [{}])[0]
                full = (choice.get("message", {}).get("content") or "").strip() or full
            except Exception:
                full = full + "\n\n[Skill output]\n" + result

    if full.strip():
        _append_conversation_log(character_prompt or "", message.strip(), full)
        new_tuples = tuples + [[message.strip(), full]]
        wants = _user_wants_remember(message.strip())
        should = _should_flush_memory_for_history(new_tuples)
        # #region agent log
        _dlog(
            "memory_trigger_eval",
            {
                "wants_remember": bool(wants),
                "should_flush": bool(should),
                "n_turns": len(new_tuples),
                "cursor_turns": _MEMORY_FLUSH_CURSOR_TURNS,
                "delta_turns": len(new_tuples) - _MEMORY_FLUSH_CURSOR_TURNS,
                "threshold": MEMORY_FLUSH_HISTORY_TURNS,
            },
            "H1",
            location="ChatBot_OpenClaw.py:stream_chat:trigger",
        )
        # #endregion
        if wants or should:
            run_memory_extract_and_append(
                message.strip(), full, character_prompt or "",
                extra_turns=tuples,
            )
            # Mark all turns up to this point as flushed.
            _MEMORY_FLUSH_CURSOR_TURNS = len(new_tuples)
            if MEMORY_SEARCH_ENABLED:
                _build_memory_index()

    if tts_finish is not None:
        last_seg = tts_buffer.strip()
        if last_seg:
            to_send, _ = _process_segment_for_tts(last_seg, tts_inside_parens)
            if to_send and tts_total_chars <= TTS_CUMULATIVE_CHAR_LIMIT:
                tts_push(to_send)
        wav_path = tts_finish()
        if wav_path is None and full.strip() and not use_lan and api_key:
            if _tts_sync_one_shot(full.strip(), api_key):
                wav_path = TTS_OUTPUT_WAV
        yield _tuples_to_messages(tuples + [[message.strip(), full]]), "", "准备就绪。可点击下方播放语音", wav_path
    else:
        yield _tuples_to_messages(tuples + [[message.strip(), full]]), "", "", None


def load_prompt_from_file(file) -> str:
    """Load character prompt from an uploaded file. Return content or empty string."""
    if file is None:
        return ""
    path = file.name if hasattr(file, "name") else file
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def clear_chat():
    """Return empty history for Clear button."""
    global _MEMORY_FLUSH_CURSOR_TURNS
    _MEMORY_FLUSH_CURSOR_TURNS = 0
    return []


def main():
    theme = gr.themes.Soft(primary_hue="violet")
    with gr.Blocks(title="Character Chat") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">'
                    f'<span style="color:#7c3aed;font-size:1.5rem;">⚡</span>'
                    f'<span style="font-size:1.25rem;font-weight:600;color:#374151;">{APP_HEADER_TITLE}</span>'
                    f'</div>'
                )
                character_prompt = gr.Textbox(
                    label="Character Persona",
                    placeholder="e.g. You are a pirate. Speak in pirate slang. Your name is Red.",
                    lines=8,
                )
                file_input = gr.File(label="Load persona file", file_types=[".txt"], elem_classes=["load-persona"])
                file_input.change(load_prompt_from_file, inputs=[file_input], outputs=[character_prompt])
                with gr.Row():
                    enable_tts = gr.Checkbox(label="Voice (TTS)", value=False)
                    gr.HTML('<span style="color:#6b7280;font-size:0.875rem;align-self:center;">Streaming TTS</span>')
                with gr.Accordion("TTS 设置", open=False, visible=False) as tts_accordion:
                    tts_backend = gr.Radio(
                        choices=[("DashScope CosyVoice", "dashscope"), ("局域网 TTS (MLX)", "lan")],
                        value="lan",
                        label="TTS 后端",
                    )
                    tts_api_key = gr.Textbox(
                        label="DashScope API Key（仅 CosyVoice 需要）",
                        placeholder="填入阿里云 DashScope API Key；不填则使用 .env 中的 DASHSCOPE_API_KEY",
                        type="password",
                    )
                    lan_tts_url = gr.Textbox(
                        label="局域网 TTS URL（仅当选择「局域网 TTS」时使用）",
                        placeholder=LAN_TTS_DEFAULT_URL,
                        value=LAN_TTS_DEFAULT_URL,
                    )
                    lan_tts_voice = gr.Textbox(
                        label="局域网 TTS 音色（voice）",
                        placeholder="可选，如 female/male 或模型文档中的音色名",
                    )
                    lan_tts_instruction = gr.Textbox(
                        label="局域网 TTS 风格（instruction_prompt）",
                        placeholder="用活泼的年轻女性声音，语气开心带点笑意",
                    )
                enable_tts.change(
                    lambda on: gr.update(visible=on, open=on),
                    inputs=[enable_tts],
                    outputs=[tts_accordion],
                )
                creativity_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.7,
                    step=0.05,
                    label="Creativity (Temp)",
                )
                use_memory_cb = gr.Checkbox(label="Use long-term memory", value=True)
                use_vector_search_cb = gr.Checkbox(
                    label="Use vector memory search (requires OPENAI_API_KEY)",
                    value=MEMORY_SEARCH_ENABLED,
                )
                enable_skill_exec_cb = gr.Checkbox(
                    label="Allow skill script execution",
                    value=os.environ.get("SKILL_EXEC_ENABLED", "").strip().lower() in ("1", "true", "yes"),
                )
                clear_btn = gr.Button("Clear Chat", variant="secondary")

            with gr.Column(scale=2):
                gr.HTML(
                    f'<div style="margin-bottom:0.75rem;">'
                    f'<div style="display:flex;align-items:center;gap:0.5rem;">'
                    f'<span style="width:2rem;height:2rem;border-radius:50%;background:#7c3aed;display:inline-block;"></span>'
                    f'<div><div class="ui-header" style="margin:0;">{DISPLAY_MODEL_NAME}</div>'
                    f'<div class="ui-subtitle">{DISPLAY_SUBTITLE}</div></div></div></div>'
                )
                chatbot = gr.Chatbot(label="Chat", height=400, show_label=False)
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Message Nova...",
                        label="Message",
                        show_label=False,
                        scale=10,
                        container=False,
                    )
                    send_btn = gr.Button("↑", variant="primary", scale=1, min_width=48)
                adjustment_direction = gr.Textbox(
                    label="",
                    lines=6,
                    show_label=False,
                    interactive=False,
                    placeholder="点击下方按钮根据对话记录生成人设调整建议",
                )
                analyze_btn = gr.Button("分析对话并生成调整方向")

        def submit(user_msg, hist, prompt, tts_on, tts_key, tts_backend_val, lan_url, lan_voice, lan_instruction, temp, use_mem, use_vec, use_skill_exec):
            for h, m, s, audio_path in stream_chat(
                prompt, hist, user_msg,
                enable_tts=tts_on,
                tts_api_key=tts_key or "",
                tts_backend=tts_backend_val or "dashscope",
                lan_tts_url=lan_url or "",
                lan_tts_voice=lan_voice or "",
                lan_tts_instruction=lan_instruction or "",
                temperature=temp,
                use_memory=use_mem,
                use_vector_search=use_vec,
                enable_skill_exec=use_skill_exec,
            ):
                yield h, m

        submit_inputs = [
            msg, chatbot, character_prompt, enable_tts, tts_api_key, tts_backend,
            lan_tts_url, lan_tts_voice, lan_tts_instruction, creativity_slider,
            use_memory_cb, use_vector_search_cb, enable_skill_exec_cb,
        ]
        msg.submit(submit, inputs=submit_inputs, outputs=[chatbot, msg])
        send_btn.click(submit, inputs=submit_inputs, outputs=[chatbot, msg])
        clear_btn.click(clear_chat, outputs=[chatbot])

        def run_analyze():
            ideas = analyze_log_for_prompt_ideas(CONVERSATIONS_LOG)
            print("调整方向：\n" + ideas)
            return ideas

        analyze_btn.click(run_analyze, inputs=[], outputs=[adjustment_direction])

    demo.launch(theme=theme, css=UI_CSS)


if __name__ == "__main__":
    main()

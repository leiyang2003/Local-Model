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
def _dlog(msg, data=None, hypothesisId=None):
    try:
        import time as _t
        o = {"timestamp": int(_t.time() * 1000), "location": "local-ml.py", "message": msg, "runId": "run1"}
        if data is not None: o["data"] = data
        if hypothesisId is not None: o["hypothesisId"] = hypothesisId
        with open("/Users/leiyang/Desktop/Coding/.cursor/debug.log", "a", encoding="utf-8") as f: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    except Exception: pass
_env_file = _env_dir / ".env"
_dlog("after load_dotenv", {"script_dir": str(_env_dir), "env_file_exists": _env_file.exists(), "cwd": str(Path.cwd()), "cwd_env_exists": (Path.cwd() / ".env").exists(), "DASHSCOPE_API_KEY_set": bool((os.environ.get("DASHSCOPE_API_KEY") or "").strip())}, "H1")
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
LAN_TTS_DEFAULT_URL = "http://192.168.31.134:8000/v1/audio/speech"
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


def build_messages(character_prompt: str, history: list, user_message: str) -> list:
    """Build API messages: system (character), history, current user message."""
    system = (character_prompt or "").strip() or DEFAULT_SYSTEM
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
):
    """Send request to MLX server and stream reply; yield (history, '', tts_status) for Gradio."""
    if not (message or message.strip()):
        yield _tuples_to_messages(_history_to_tuples(history)), "", "", None
        return

    tuples = _history_to_tuples(history)
    new_tuples = tuples + [[message.strip(), ""]]
    yield _tuples_to_messages(new_tuples), "", "", None

    messages = build_messages(character_prompt, history, message.strip())
    payload = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "stream": True,
    }

    try:
        resp = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        error_msg = "Could not reach MLX server. Is it running on port 8000?"
        if getattr(e, "response", None) is not None:
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

    if full.strip():
        _append_conversation_log(character_prompt or "", message.strip(), full)

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
    return []


def main():
    theme = gr.themes.Soft(primary_hue="violet")
    with gr.Blocks(title="Character Chat", theme=theme, css=UI_CSS) as demo:
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

        def submit(user_msg, hist, prompt, tts_on, tts_key, tts_backend_val, lan_url, lan_voice, lan_instruction, temp):
            for h, m, s, audio_path in stream_chat(
                prompt, hist, user_msg,
                enable_tts=tts_on,
                tts_api_key=tts_key or "",
                tts_backend=tts_backend_val or "dashscope",
                lan_tts_url=lan_url or "",
                lan_tts_voice=lan_voice or "",
                lan_tts_instruction=lan_instruction or "",
                temperature=temp,
            ):
                yield h, m

        submit_inputs = [
            msg, chatbot, character_prompt, enable_tts, tts_api_key, tts_backend,
            lan_tts_url, lan_tts_voice, lan_tts_instruction, creativity_slider,
        ]
        msg.submit(submit, inputs=submit_inputs, outputs=[chatbot, msg])
        send_btn.click(submit, inputs=submit_inputs, outputs=[chatbot, msg])
        clear_btn.click(clear_chat, outputs=[chatbot])

        def run_analyze():
            ideas = analyze_log_for_prompt_ideas(CONVERSATIONS_LOG)
            print("调整方向：\n" + ideas)
            return ideas

        analyze_btn.click(run_analyze, inputs=[], outputs=[adjustment_direction])

    demo.launch()


if __name__ == "__main__":
    main()

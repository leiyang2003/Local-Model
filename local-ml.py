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

# --- TTS (DashScope CosyVoice) ---
TTS_MAX_SEGMENT_CHARS = 2000   # CosyVoice per-call limit
TTS_SENTENCE_MAX_CHARS = 200   # max chars per segment when no sentence end found
TTS_SAMPLE_RATE = 22050        # PCM playback rate for CosyVoice default
_SCRIPT_DIR = Path(__file__).resolve().parent
TTS_OUTPUT_WAV = str(_SCRIPT_DIR / "last_tts.wav")  # 固定路径，界面可点击播放


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


def _split_next_tts_segment(buffer: str) -> tuple[str, str]:
    """
    从 buffer 中取出下一段用于 TTS 的文本（到句号/问号/感叹号/换行或长度上限），
    返回 (segment, remaining)。单段不超过 TTS_MAX_SEGMENT_CHARS。
    """
    if not buffer or not buffer.strip():
        return "", ""
    s = buffer.strip()
    if len(s) <= TTS_MAX_SEGMENT_CHARS:
        # 找最后一个句末标点或换行
        for sep in ("。", "！", "？", "!", "?", ".", "\n"):
            i = s.rfind(sep)
            if i != -1:
                return s[: i + 1].strip(), s[i + 1 :].strip()
        # 无句末则整段送出（短时）或按长度截断
        if len(s) <= TTS_SENTENCE_MAX_CHARS:
            return s, ""
        seg = s[:TTS_SENTENCE_MAX_CHARS]
        return seg, s[TTS_SENTENCE_MAX_CHARS:].strip()
    # 超长：先截 TTS_MAX_SEGMENT_CHARS，再在截断内找最后句末
    chunk = s[:TTS_MAX_SEGMENT_CHARS]
    for sep in ("。", "！", "？", "!", "?", ".", "\n"):
        i = chunk.rfind(sep)
        if i != -1:
            return chunk[: i + 1].strip(), s[i + 1 :].strip()
    seg = chunk[:TTS_SENTENCE_MAX_CHARS]
    return seg, s[len(seg) :].strip()


def _play_wav_file(wav_path: str) -> None:
    """Play a WAV file with system player (afplay on macOS, aplay on Linux)."""
    import sys
    try:
        if sys.platform == "darwin":
            subprocess.run(["afplay", wav_path], check=True, timeout=300, capture_output=True)
        elif sys.platform.startswith("linux"):
            subprocess.run(["aplay", "-q", wav_path], check=True, timeout=300, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass


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


def _start_tts_session(api_key: str):
    """
    Start TTS for one reply: playback thread + sender thread.
    Returns (push_segment_fn, finish_fn). finish_fn() returns path to WAV (or None).
    """
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
        "temperature": TEMPERATURE,
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
    # #region agent log
    _dlog("TTS api_key resolution", {"enable_tts": enable_tts, "has_ui_key": bool((tts_api_key or "").strip()), "env_has_key": bool((os.environ.get("DASHSCOPE_API_KEY") or "").strip()), "api_key_non_empty": bool(api_key)}, "H5")
    # #endregion
    if enable_tts and api_key:
        try:
            tts_push, tts_finish = _start_tts_session(api_key)
        except Exception:
            tts_push = None
            tts_finish = None

    def _tts_status() -> str:
        if enable_tts and tts_push is None:
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
                    seg, remaining = _split_next_tts_segment(tts_buffer)
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

    if tts_finish is not None:
        last_seg = tts_buffer.strip()
        if last_seg:
            to_send, _ = _process_segment_for_tts(last_seg, tts_inside_parens)
            if to_send and tts_total_chars <= TTS_CUMULATIVE_CHAR_LIMIT:
                tts_push(to_send)
        wav_path = tts_finish()
        if wav_path is None and full.strip() and api_key:
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
    with gr.Blocks(title="Character Chat") as demo:
        gr.Markdown("## Character Chat — MLX server")
        gr.Markdown("Set a character prompt below, then chat. Server: `http://localhost:8000`")

        with gr.Row():
            with gr.Column(scale=1):
                character_prompt = gr.Textbox(
                    label="Character prompt",
                    placeholder="e.g. You are a pirate. Speak in pirate slang. Your name is Red.",
                    lines=8,
                )
                file_input = gr.File(label="Load from file (.txt)", file_types=[".txt"])
                file_input.change(load_prompt_from_file, inputs=[file_input], outputs=[character_prompt])
                enable_tts = gr.Checkbox(label="Enable TTS (streaming voice)", value=False)
                tts_api_key = gr.Textbox(
                    label="DashScope API Key（TTS 语音）",
                    placeholder="填入阿里云 DashScope API Key 即可使用 TTS；不填则使用 .env 中的 DASHSCOPE_API_KEY",
                    type="password",
                )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=400)
                msg = gr.Textbox(placeholder="Type a message...", label="Message", show_label=False)
                clear_btn = gr.Button("Clear chat")
                tts_status = gr.Textbox(label="TTS status", value="", interactive=False, visible=True)
                tts_audio = gr.Audio(label="播放语音", type="filepath", visible=True)

        def submit(user_msg, hist, prompt, tts_on, tts_key):
            for h, m, s, audio_path in stream_chat(prompt, hist, user_msg, enable_tts=tts_on, tts_api_key=tts_key or ""):
                yield h, m, s, audio_path

        msg.submit(
            submit,
            inputs=[msg, chatbot, character_prompt, enable_tts, tts_api_key],
            outputs=[chatbot, msg, tts_status, tts_audio],
        )
        clear_btn.click(clear_chat, outputs=[chatbot])

    demo.launch()


if __name__ == "__main__":
    main()

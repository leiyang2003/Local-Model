# Local-Model

Character chat web UI that talks to a **local MLX LLM** (OpenAI-compatible API) with optional **TTS** via Alibaba DashScope CosyVoice.

## Features

- **Chat UI**: Gradio interface to chat with a character driven by your local model
- **Local LLM**: Uses [MLX LM](https://github.com/ml-explore/mlx-examples) server (e.g. Qwen3-4B-Instruct-4bit) — no cloud LLM required for dialogue
- **TTS (optional)**: Streaming speech synthesis with DashScope CosyVoice; supports playback (e.g. `afplay` on macOS, or PyAudio for real-time)

## Prerequisites

1. **MLX LM server** running locally (OpenAI-compatible `/v1/chat/completions`):
   ```bash
   mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-4bit --port 8000
   ```
2. **DashScope API key** (for TTS): get from [Alibaba Cloud](https://dashscope.aliyun.com/), set in `.env`.

## Setup

```bash
# Clone and enter project
git clone https://github.com/leiyang2003/Local-Model.git
cd Local-Model

# Create virtualenv (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

(You can also enter the API key in the UI if not set in `.env`.)

## Run

1. Start the MLX server (in a separate terminal):
   ```bash
   mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-4bit --port 8000
   ```

2. Start the chat UI:
   ```bash
   python ChatBot_OpenClaw.py
   ```
   (Or `python local-ml.py` for the minimal UI without memory/skills.)

3. Open the Gradio URL (e.g. `http://127.0.0.1:7860`), load a character prompt (e.g. `Ani.txt`) if needed, and chat.

## Character prompts

Place system/persona text in `.txt` files (e.g. `Ani.txt`). The UI lets you select a file to use as the system prompt for the chat.

## Long-term memory and skills (OpenClaw-style)

- **Workspace**: Memory and skills live under `workspace/` (or set `CHATBOT_WORKSPACE` to another path).
- **Memory (read)**: Each turn injects into context either:
  - **Full load**: `workspace/MEMORY.md` plus today’s and yesterday’s `workspace/memory/YYYY-MM-DD.md`, truncated to `MEMORY_MAX_CONTEXT_CHARS` (default 4000).
  - **Vector search** (optional): If “Use vector memory search” is on and `OPENAI_API_KEY` is set, the app embeds memory chunks and the current message, then injects only the top-k relevant snippets. Requires `openai` and an index (built on first use or after memory writes).
- **Memory (write)**: Automatically when (1) the user says something like “记住…” / “remember this”, or (2) history length exceeds `MEMORY_FLUSH_HISTORY_TURNS` (default 20) or estimated tokens exceed `MEMORY_FLUSH_ESTIMATED_TOKENS` (default 8000). A non-streaming call extracts content and appends to `MEMORY.md` and/or `memory/YYYY-MM-DD.md`.
- **Skills**: Put skill folders under `workspace/skills/`; each skill is a directory with `SKILL.md` (YAML frontmatter: `name`, `description`; optional `metadata.openclaw.requires.env` for gating). The app injects an “Available skills” list into the system prompt. Example: `workspace/skills/reminder/SKILL.md`.

**Skill script execution (optional)**  
When enabled (UI checkbox “Allow skill script execution” or `SKILL_EXEC_ENABLED=1`), the model can request running a skill script by outputting `[[SKILL:<name>]]` … `[[/SKILL]]` with `key=value` args. Only skills under `workspace/skills/` that declare a `script` in `SKILL.md` are run (allowlist). Script run timeout: `SKILL_EXEC_TIMEOUT` (seconds, default 300).

**Env (optional)**  
`CHATBOT_WORKSPACE`, `MEMORY_MAX_CONTEXT_CHARS`, `MEMORY_FLUSH_HISTORY_TURNS`, `MEMORY_FLUSH_ESTIMATED_TOKENS`, `MEMORY_SEARCH_ENABLED` (1/true/yes), `MEMORY_EMBEDDING_PROVIDER` (openai), `OPENAI_API_KEY`, `MEMORY_INDEX_PATH`, `MEMORY_MAX_RESULTS`, `MEMORY_MAX_SNIPPET_CHARS`, `SKILL_EXEC_ENABLED` (1/true/yes), `SKILL_EXEC_TIMEOUT` (seconds).

## Optional: real-time TTS playback

- **macOS**: TTS WAV is saved to `last_tts.wav` and can be played with `afplay` when the segment finishes.
- **PyAudio**: For real-time playback, install PortAudio then PyAudio:
  ```bash
  brew install portaudio   # macOS
  pip install pyaudio
  ```
  Uncomment `pyaudio` in `requirements.txt` if you use it.

## License

MIT (or as you prefer).

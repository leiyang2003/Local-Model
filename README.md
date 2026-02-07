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
   python local-ml.py
   ```

3. Open the Gradio URL (e.g. `http://127.0.0.1:7860`), load a character prompt (e.g. `Ani.txt`) if needed, and chat.

## Character prompts

Place system/persona text in `.txt` files (e.g. `Ani.txt`). The UI lets you select a file to use as the system prompt for the chat.

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

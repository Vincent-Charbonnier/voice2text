# meeting_summarizer_whisper_openai.py
import os
import tempfile
import base64
import json
import requests
import gradio as gr
import inspect
import subprocess
from datetime import datetime, timedelta
from typing import Optional

# -------------------------
# Config (env) - defaults can be empty; UI will let you set them at runtime
# -------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LOCAL_WHISPER_URL = os.getenv("LOCAL_WHISPER_URL", "")
HF_WHISPER_MODEL = os.getenv("HF_WHISPER_MODEL", "openai/whisper-large-v3")
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")

# Path to store model config (persisted JSON)
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/model_settings.json")

# Runtime-configurable (via the Model Config tab)
WHISPER_API_URL: str = ""
WHISPER_API_TOKEN: str = ""
WHISPER_MODEL_NAME: str = ""

SUMMARIZER_API_URL: str = ""
SUMMARIZER_API_TOKEN: str = ""
SUMMARIZER_MODEL_NAME: str = ""

# -------------------------
# Utility: robust component creation for varying gradio versions
# -------------------------
def make_component(component_cls, **kwargs):
    try:
        sig = inspect.signature(component_cls)
        params = sig.parameters
        mapped = dict(kwargs)

        if "source" in mapped and "source" not in params:
            if "input" in params:
                mapped["input"] = mapped.pop("source")
            else:
                mapped.pop("source", None)

        label_val = None
        if "label" in mapped:
            label_val = mapped.pop("label")
            if "label" in params:
                mapped["label"] = label_val
            elif "value" in params:
                mapped["value"] = label_val
            elif "title" in params:
                mapped["title"] = label_val
            elif "name" in params:
                mapped["name"] = label_val

        filtered = {k: v for k, v in mapped.items() if k in params}

        comp_name = getattr(component_cls, "__name__", "").lower()
        if comp_name == "button":
            if label_val is not None and "label" not in filtered and "value" not in filtered and "title" not in filtered and "name" not in filtered:
                try:
                    return component_cls(label_val, **filtered)
                except TypeError:
                    pass

        return component_cls(**filtered)
    except Exception:
        return component_cls(**kwargs)

# -------------------------
# Helpers
# -------------------------
def save_bytes_to_tempfile(bytes_data, suffix=".wav"):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(bytes_data)
    tmp.flush()
    tmp.close()
    return tmp.name

def encode_file_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# -------------------------
# Persistency: load/save JSON settings
# -------------------------
def load_model_settings(path: str = MODEL_CONFIG_PATH) -> dict:
    global WHISPER_API_URL, WHISPER_API_TOKEN, WHISPER_MODEL_NAME
    global SUMMARIZER_API_URL, SUMMARIZER_API_TOKEN, SUMMARIZER_MODEL_NAME

    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            conf = json.load(f)
        WHISPER_API_URL = conf.get("WHISPER_API_URL", WHISPER_API_URL or "")
        WHISPER_API_TOKEN = conf.get("WHISPER_API_TOKEN", WHISPER_API_TOKEN or "")
        WHISPER_MODEL_NAME = conf.get("WHISPER_MODEL_NAME", WHISPER_MODEL_NAME or "")
        SUMMARIZER_API_URL = conf.get("SUMMARIZER_API_URL", SUMMARIZER_API_URL or "")
        SUMMARIZER_API_TOKEN = conf.get("SUMMARIZER_API_TOKEN", SUMMARIZER_API_TOKEN or "")
        SUMMARIZER_MODEL_NAME = conf.get("SUMMARIZER_MODEL_NAME", SUMMARIZER_MODEL_NAME or "")
        return conf
    except Exception as e:
        print(f"Failed to load model settings from {path}: {e}")
        return {}

def save_model_settings(path: str = MODEL_CONFIG_PATH) -> bool:
    conf = {
        "WHISPER_API_URL": WHISPER_API_URL or "",
        "WHISPER_API_TOKEN": WHISPER_API_TOKEN or "",
        "WHISPER_MODEL_NAME": WHISPER_MODEL_NAME or "",
        "SUMMARIZER_API_URL": SUMMARIZER_API_URL or "",
        "SUMMARIZER_API_TOKEN": SUMMARIZER_API_TOKEN or "",
        "SUMMARIZER_MODEL_NAME": SUMMARIZER_MODEL_NAME or "",
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save model settings to {path}: {e}")
        return False

# load saved settings at startup
_load_conf = load_model_settings()

# -------------------------
# Model config helpers (UI actions)
# -------------------------
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_whisper_connection(api_url: str, api_token: str, timeout: float = 1.5) -> str:
    if not api_url:
        return "❌ Please provide an API URL."

    base = api_url.rstrip("/")
    if "/v1/" in base:
        base = base.split("/v1/")[0]

    headers = {"User-Agent": "whisper-connection-test/fast"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    def try_get(path: str):
        url = f"{base}{path}"
        try:
            r = requests.get(url, headers=headers, timeout=(timeout, timeout), verify=False)
            return r.status_code, url
        except Exception:
            return None, url

    code, url = try_get("/health")
    if code == 200:
        return f"✅ Fast connection OK via {url}"
    if code in (401, 403):
        return f"❌ Unauthorized ({code}) — invalid token at {url}"

    code, url = try_get("/v1/models")
    if code == 200:
        return f"✅ Connected (authorized) via {url}"
    if code in (401, 403):
        return f"❌ Unauthorized ({code}) — invalid token at {url}"

    code, url = try_get("/version")
    if code == 200:
        return f"⚠️ Connected via {url} — service reachable, health/models unavailable."

    code, url = try_get("/v1/audio/transcriptions")
    if code == 405:
        return f"✅ Transcription endpoint detected at {url} (POST only)."
    if code == 404:
        return f"❌ Transcription endpoint not found at {url}"

    if code is None:
        return "❌ Unable to reach service (timeout)."

    return f"❌ No valid response (HTTP {code}) from {url}"

def test_summarizer_connection(api_url: str, api_token: str, payload: dict | None = None) -> str:
    if not api_url:
        return "Please provide a summarizer endpoint URL."

    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    payload = payload or {
        "model": SUMMARIZER_MODEL_NAME or "default",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 5,
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=10, verify=False)
        if resp.status_code == 200:
            return "✅ Summarizer endpoint reachable (OpenAI-compatible)."
        return f"❌ Received status {resp.status_code}: {resp.text[:500]}"
    except Exception as e:
        return f"❌ Connection failed: {e}"

def update_model_settings(whisper_url, whisper_token, whisper_model, summarizer_url, summarizer_token, summarizer_model) -> str:
    global WHISPER_API_URL, WHISPER_API_TOKEN, WHISPER_MODEL_NAME
    global SUMMARIZER_API_URL, SUMMARIZER_API_TOKEN, SUMMARIZER_MODEL_NAME
    WHISPER_API_URL = whisper_url or ""
    WHISPER_API_TOKEN = whisper_token or ""
    WHISPER_MODEL_NAME = whisper_model or ""
    SUMMARIZER_API_URL = summarizer_url or ""
    SUMMARIZER_API_TOKEN = summarizer_token or ""
    SUMMARIZER_MODEL_NAME = summarizer_model or ""
    ok = save_model_settings()
    if ok:
        return f"Model settings updated and saved to {MODEL_CONFIG_PATH}"
    else:
        return "Model settings updated in-memory, but failed to save to disk."

def save_whisper_settings(wh_url, wh_token, wh_model):
    msg = update_model_settings(wh_url, wh_token, wh_model, SUMMARIZER_API_URL, SUMMARIZER_API_TOKEN, SUMMARIZER_MODEL_NAME)
    return "✅ Transcription model settings saved (in-memory & file). " + msg

def save_summarizer_settings(sum_url, sum_token, sum_model):
    msg = update_model_settings(WHISPER_API_URL, WHISPER_API_TOKEN, WHISPER_MODEL_NAME, sum_url, sum_token, sum_model)
    return "✅ Summarizer model settings saved (in-memory & file). " + msg

def test_whisper_current(wh_url, wh_token):
    return test_whisper_connection(wh_url, wh_token)

def test_summarizer_current(sum_url, sum_token):
    return test_summarizer_connection(sum_url, sum_token)

def clear_model_settings():
    update_model_settings("", "", "", "", "", "")
    save_model_settings()
    return f"✅ Cleared saved model settings ({MODEL_CONFIG_PATH})."

# -------------------------
# Transcription helpers + chunked transcription
# -------------------------
MAX_SINGLE_CHUNK_SEC = 30      # predictor single-clip max
DEFAULT_CHUNK_SEC = 25         # chunk size to use
DEFAULT_OVERLAP_SEC = 1.0      # overlap to avoid chopped words
CHUNKS_DIR = "/tmp/tts_chunks"  # where chunks are written

def ffmpeg_convert_to_wav(input_path: str, out_path: str, sample_rate: int = 16000):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ac", "1", "-ar", str(sample_rate),
        "-f", "wav", out_path
    ]
    subprocess.check_call(cmd)

def get_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    out = subprocess.check_output(cmd).decode().strip()
    try:
        return float(out)
    except Exception:
        return 0.0

def split_wav_to_chunks(wav_path: str, chunk_length: float = DEFAULT_CHUNK_SEC, overlap: float = DEFAULT_OVERLAP_SEC):
    total_sec = get_duration_seconds(wav_path)
    if total_sec <= 0:
        raise RuntimeError("Could not determine audio duration.")
    step = chunk_length - overlap
    if step <= 0:
        raise ValueError("chunk_length must be greater than overlap")
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    chunk_paths = []
    idx = 0
    start = 0.0
    while start < total_sec:
        out = os.path.join(CHUNKS_DIR, f"chunk_{idx:05d}.wav")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start),
            "-i", wav_path,
            "-t", str(chunk_length),
            "-ac", "1", "-ar", "16000",
            "-f", "wav", out
        ]
        subprocess.check_call(cmd)
        end = min(start + chunk_length, total_sec)
        chunk_paths.append((out, start, end))
        idx += 1
        start += step
    return chunk_paths

def transcribe_chunk_file(chunk_path: str, url: str, token: Optional[str], model: str):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    files = {"file": (os.path.basename(chunk_path), open(chunk_path, "rb"), "audio/wav")}
    data = {}
    if model:
        data["model"] = model
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=120, verify=False)
    if resp.ok:
        try:
            j = resp.json()
            text = j.get("transcription") or j.get("text") or j.get("result") or j.get("summary") or ""
            if not text and isinstance(j, dict):
                for v in j.values():
                    if isinstance(v, str) and len(v) > 0:
                        text = v
                        break
            return text, j
        except Exception:
            return resp.text, {"raw": resp.text}
    else:
        raise RuntimeError(f"Chunk transcription failed: {resp.status_code} {resp.text[:400]}")

def transcribe_long_audio(filepath: str, url: str, token: Optional[str], model: str,
                          chunk_sec: float = DEFAULT_CHUNK_SEC, overlap: float = DEFAULT_OVERLAP_SEC):
    norm_wav = filepath + ".norm.wav"
    try:
        ffmpeg_convert_to_wav(filepath, norm_wav, sample_rate=16000)
    except Exception as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e}")

    total_duration = get_duration_seconds(norm_wav)
    if total_duration <= 0:
        raise RuntimeError("Failed to compute duration after conversion.")

    chunk_infos = split_wav_to_chunks(norm_wav, chunk_length=chunk_sec, overlap=overlap)
    stitched_lines = []
    results = []

    for (chunk_path, start_sec, end_sec) in chunk_infos:
        try:
            text, raw = transcribe_chunk_file(chunk_path, url, token, model)
        except Exception as e:
            text = f"[ERROR: {e}]"
            raw = {"error": str(e)}
        text = (text or "").strip()
        ts = str(timedelta(seconds=int(start_sec)))
        stitched_lines.append(f"[{ts}] {text}")
        results.append({"start": start_sec, "end": end_sec, "text": text, "raw": raw})

    final_transcript = "\n".join(stitched_lines)
    return final_transcript, results

# -------------------------
# New transcribe_with_hf implementation
# -------------------------
def transcribe_with_hf(audio_filepath, language=None, diarization=False):
    if not audio_filepath:
        return None, None, "No audio provided", None

    def post_single_file(endpoint_url: str, token: Optional[str], source_path: str):
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with open(source_path, "rb") as f:
                files = {"file": (os.path.basename(source_path), f, "audio/wav")}
                data = {}
                if language:
                    data["language"] = language
                resp = requests.post(endpoint_url, headers=headers, files=files, data=data, timeout=120, verify=False)
            return resp
        except Exception as e:
            return e

    # 1) Try WHISPER_API_URL
    if WHISPER_API_URL:
        # generate candidate endpoints (in case user provided base URL)
        url = WHISPER_API_URL
        if url.endswith("/") or "/v1/" not in url:
            candidates = [url.rstrip("/") + p for p in ["/v1/audio/transcriptions", "/v1/audio/transcribe", "/v1/audio/transcribe/"]]
        else:
            candidates = [url]

        last_err = None
        for candidate in candidates:
            try:
                # normalize to .norm.wav to get duration
                try:
                    tmp_norm = audio_filepath + ".norm.wav"
                    ffmpeg_convert_to_wav(audio_filepath, tmp_norm, sample_rate=16000)
                    duration_sec = get_duration_seconds(tmp_norm)
                    source_for_post = tmp_norm
                except Exception:
                    duration_sec = get_duration_seconds(audio_filepath) or 0
                    source_for_post = audio_filepath

                if duration_sec > MAX_SINGLE_CHUNK_SEC:
                    try:
                        stitched, parts = transcribe_long_audio(source_for_post, candidate, WHISPER_API_TOKEN, WHISPER_MODEL_NAME or "whisper-large-v3",
                                                               chunk_sec=DEFAULT_CHUNK_SEC, overlap=DEFAULT_OVERLAP_SEC)
                        path = _save_transcript_to_temp(stitched)
                        return stitched, {"parts": parts}, f"Transcribed by chunking into {len(parts)} parts via {candidate}", path
                    except Exception as e:
                        last_err = e
                        continue
                else:
                    resp = post_single_file(candidate, WHISPER_API_TOKEN, source_for_post)
                    if isinstance(resp, Exception):
                        last_err = resp
                        continue
                    if resp.ok:
                        try:
                            j = resp.json()
                            text = j.get("transcription") or j.get("text") or j.get("result") or resp.text
                        except Exception:
                            text = resp.text
                        path = _save_transcript_to_temp(text)
                        return text, j if isinstance(j, dict) else None, f"Transcribed via {candidate}", path
                    else:
                        # If predictor complains about clip duration, fallback to chunking
                        try:
                            body = resp.text or ""
                            if "Maximum clip duration" in body or resp.status_code == 400:
                                stitched, parts = transcribe_long_audio(source_for_post, candidate, WHISPER_API_TOKEN, WHISPER_MODEL_NAME or "whisper-large-v3",
                                                                       chunk_sec=DEFAULT_CHUNK_SEC, overlap=DEFAULT_OVERLAP_SEC)
                                path = _save_transcript_to_temp(stitched)
                                return stitched, {"parts": parts}, f"Transcribed by chunking into {len(parts)} parts (fallback) via {candidate}", path
                        except Exception:
                            pass
                        last_err = f"WHISPER_API_URL error: {resp.status_code} {resp.text[:400]}"
                        continue
            except Exception as e:
                last_err = e
                continue

        return None, None, f"WHISPER_API_URL failed: {last_err}", None

    # 2) Try LOCAL_WHISPER_URL
    if LOCAL_WHISPER_URL:
        try:
            dur = get_duration_seconds(audio_filepath)
            if dur > MAX_SINGLE_CHUNK_SEC:
                stitched, parts = transcribe_long_audio(audio_filepath, LOCAL_WHISPER_URL, None, WHISPER_MODEL_NAME or "whisper-large-v3",
                                                       chunk_sec=DEFAULT_CHUNK_SEC, overlap=DEFAULT_OVERLAP_SEC)
                path = _save_transcript_to_temp(stitched)
                return stitched, {"parts": parts}, f"Transcribed by chunking into {len(parts)} parts via LOCAL_WHISPER_URL", path
            else:
                resp = post_single_file(LOCAL_WHISPER_URL, None, audio_filepath)
                if isinstance(resp, Exception):
                    return None, None, f"LOCAL_WHISPER_URL error: {resp}", None
                if resp.ok:
                    try:
                        j = resp.json()
                        text = j.get("transcription") or j.get("text") or resp.text
                    except Exception:
                        text = resp.text
                    path = _save_transcript_to_temp(text)
                    return text, j if isinstance(j, dict) else None, "Transcribed via LOCAL_WHISPER_URL", path
                else:
                    return None, None, f"LOCAL_WHISPER_URL error: {resp.status_code} {resp.text[:400]}", None
        except Exception as e:
            return None, None, f"LOCAL_WHISPER_URL exception: {e}", None

    # 3) Fallback to HF inference API
    if HF_API_TOKEN:
        hf_url = f"https://api-inference.huggingface.co/models/{HF_WHISPER_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        try:
            with open(audio_filepath, "rb") as af:
                audio_bytes = af.read()
            resp = requests.post(hf_url, headers=headers, data=audio_bytes, timeout=120, verify=False)
            if resp.ok:
                try:
                    j = resp.json()
                    text = j.get("transcription") or j.get("text") or j.get("translation") or j.get("result") or resp.text
                except Exception:
                    text = resp.text
                path = _save_transcript_to_temp(text)
                return text, j if isinstance(j, dict) else None, "Transcribed via Hugging Face inference", path
            else:
                with open(audio_filepath, "rb") as f:
                    files = {"file": (os.path.basename(audio_filepath), f, "application/octet-stream")}
                    resp2 = requests.post(hf_url, headers=headers, files=files, timeout=120, verify=False)
                if resp2.ok:
                    try:
                        j = resp2.json()
                        text = j.get("transcription") or j.get("text") or ""
                    except Exception:
                        text = resp2.text
                    path = _save_transcript_to_temp(text)
                    return text, j if isinstance(j, dict) else None, "Transcribed via Hugging Face (multipart)", path
                else:
                    return None, None, f"Hugging Face transcribe failed: {resp.status_code} {resp.text}", None
        except Exception as e:
            return None, None, f"HF transcribe error: {e}", None

    return None, None, "No transcription endpoint configured (WHISPER_API_URL, LOCAL_WHISPER_URL, or HF_API_TOKEN)", None

def _save_transcript_to_temp(text):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(tempfile.gettempdir(), f"transcript_{ts}.txt")
    with open(path, "w", encoding="utf-8") as out:
        out.write(text or "")
    return path

# -------------------------
# Summarization (Summarizer API or OpenAI Responses fallback)
# -------------------------
def summarize_with_openai(transcript_text, prompt_instruction, style="concise", length="short"):
    if not transcript_text:
        return None, "No transcript to summarize.", None

    system_prompt = (
        "You are a helpful assistant that summarizes meeting transcripts. "
        "Extract decisions, action items with owners if possible, and short bullet summaries."
    )
    user_prompt = f"{prompt_instruction or ''}\n\nTranscript:\n{transcript_text}\n\nPlease produce a {style} summary, {length} length."

    # 1) Try SUMMARIZER_API_URL (OpenAI-compatible chat completions)
    if SUMMARIZER_API_URL:
        headers = {"Content-Type": "application/json"}
        if SUMMARIZER_API_TOKEN:
            headers["Authorization"] = f"Bearer {SUMMARIZER_API_TOKEN}"

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        try:
            resp = requests.post(SUMMARIZER_API_URL, headers=headers, json=payload, timeout=120, verify=False)
            if resp.ok:
                try:
                    j = resp.json()
                    summary = None
                    if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                        choice = j["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            summary = choice["message"]["content"]
                        elif "text" in choice:
                            summary = choice["text"]

                    if not summary:
                        summary = j.get("summary") or j.get("result") or j.get("text") or resp.text

                except Exception:
                    summary = resp.text

                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                path = os.path.join(tempfile.gettempdir(), f"summary_{ts}.txt")
                with open(path, "w", encoding="utf-8") as out:
                    out.write(summary or "")
                return summary, "Summary generated via SUMMARIZER_API_URL", path
            else:
                return None, f"Summarizer API error: {resp.status_code} {resp.text[:500]}", None
        except Exception as e:
            return None, f"Error calling summarizer endpoint: {e}", None

    # 2) Fallback to OpenAI API
    if not OPENAI_API_KEY:
        return None, "No summarizer configured (SUMMARIZER_API_URL) and OPENAI_API_KEY not set.", None

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_SUMMARY_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.ok:
            j = resp.json()
            summary = j["choices"][0]["message"]["content"]
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(tempfile.gettempdir(), f"summary_{ts}.txt")
            with open(path, "w", encoding="utf-8") as out:
                out.write(summary)
            return summary, "Summary generated (OpenAI API).", path
        else:
            return None, f"OpenAI API error: {resp.status_code} {resp.text}", None
    except Exception as e:
        return None, f"OpenAI request failed: {e}", None

# -------------------------
# UI: create_app
# -------------------------
def _get_gradio_theme():
    try:
        if hasattr(gr, "themes") and hasattr(gr.themes, "Soft"):
            candidate = gr.themes.Soft(primary_hue="emerald")
            params = inspect.signature(gr.Blocks).parameters
            if "theme" in params:
                return candidate
    except Exception:
        pass
    return None

def create_app():
    load_model_settings()  # refresh before building UI

    theme = _get_gradio_theme()
    if theme is not None:
        blocks_ctx = gr.Blocks(title="Meeting Capture — Whisper + Summarizer", theme=theme)
    else:
        blocks_ctx = gr.Blocks(title="Meeting Capture — Whisper + Summarizer")

    with blocks_ctx as app:
        gr.Markdown("# Meeting Capture — Transcribe & Summarize\nUpload or record audio, edit transcript, and generate a tunable summary.")

        with gr.Tabs():
            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ## About
                    Use this app to record meetings, transcribe them, and generate tunable summaries.
                    Configure your transcription and summarizer endpoints under **Model Config**.

                    ### Endpoint Format
                    - **Transcription**: Expects multipart form-data with audio file
                    - **Summarizer**: OpenAI-compatible `/v1/chat/completions` format
                    """
                )

            with gr.Tab("Model Config"):
                gr.Markdown("### Transcription model (Whisper)")
                whisper_url_input = make_component(gr.Textbox, label="Transcription endpoint URL", placeholder="https://your-whisper-endpoint/v1/audio/transcriptions", value=WHISPER_API_URL)
                whisper_token_input = make_component(gr.Textbox, label="Transcription token (optional)", placeholder="Bearer token", type="password", value=WHISPER_API_TOKEN)
                whisper_model_input = make_component(gr.Textbox, label="Model name (optional)", placeholder="whisper-large-v3", value=WHISPER_MODEL_NAME)

                save_whisper_btn = make_component(gr.Button, label="Save Transcription Settings")
                whisper_save_status = make_component(gr.Textbox, label="Save status", interactive=False)
                test_whisper_btn = make_component(gr.Button, label="Test Transcription Connection")
                whisper_test_status = make_component(gr.Textbox, label="Test status", interactive=False)

                save_whisper_btn.click(fn=save_whisper_settings, inputs=[whisper_url_input, whisper_token_input, whisper_model_input], outputs=[whisper_save_status])
                test_whisper_btn.click(fn=test_whisper_current, inputs=[whisper_url_input, whisper_token_input], outputs=[whisper_test_status])

                gr.Markdown("### Summarizer model (OpenAI-compatible)")
                summarizer_url_input = make_component(gr.Textbox, label="Summarizer endpoint URL", placeholder="https://your-endpoint/v1/chat/completions", value=SUMMARIZER_API_URL)
                summarizer_token_input = make_component(gr.Textbox, label="Summarizer token (optional)", type="password", value=SUMMARIZER_API_TOKEN)
                summarizer_model_input = make_component(gr.Textbox, label="Summarizer model name (optional)", placeholder="(auto-detected)", value=SUMMARIZER_MODEL_NAME)

                save_summarizer_btn = make_component(gr.Button, label="Save Summarizer Settings")
                summarizer_save_status = make_component(gr.Textbox, label="Save status", interactive=False)
                test_summarizer_btn = make_component(gr.Button, label="Test Summarizer Connection")
                summarizer_test_status = make_component(gr.Textbox, label="Test status", interactive=False)

                save_summarizer_btn.click(fn=save_summarizer_settings, inputs=[summarizer_url_input, summarizer_token_input, summarizer_model_input], outputs=[summarizer_save_status])
                test_summarizer_btn.click(fn=test_summarizer_current, inputs=[summarizer_url_input, summarizer_token_input], outputs=[summarizer_test_status])

                clear_btn = make_component(gr.Button, label="Clear All Model Settings")
                clear_status = make_component(gr.Textbox, label="Clear status", interactive=False)
                clear_btn.click(fn=clear_model_settings, inputs=[], outputs=[clear_status])

            with gr.Tab("Transcribe & Summarize"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Provide audio (upload or record)")
                        audio_in = make_component(gr.Audio,
                                                 source="microphone",
                                                 type="filepath",
                                                 label="Meeting audio",
                                                 show_label=False)
                        language_hint = make_component(gr.Textbox, label="Language hint (optional)", placeholder="e.g. en")
                        diarization = make_component(gr.Checkbox, label="Enable speaker diarization (if supported)", value=False)
                        transcribe_btn = make_component(gr.Button, label="Transcribe")
                        transcribe_status = make_component(gr.Textbox, label="Transcription Status", interactive=False)
                        download_trans = make_component(gr.File, label="Download Transcript", visible=False)
                    with gr.Column(scale=1):
                        gr.Markdown("### Transcript (editable)")
                        transcript_box = make_component(gr.Textbox, label="Transcript", lines=12)
                        gr.Markdown("### Summarize")
                        prompt_box = make_component(gr.Textbox, label="Prompt (instructions for summarizer)", placeholder="E.g. 'Create 5 action items with owners.'", lines=3)
                        style_dropdown = make_component(gr.Dropdown, label="Style", choices=["concise", "bullet_points", "action_items", "detailed"], value="concise")
                        length_dropdown = make_component(gr.Dropdown, label="Length", choices=["short", "medium", "long"], value="short")
                        summarize_btn = make_component(gr.Button, label="Generate Summary")
                        summary_box = make_component(gr.Textbox, label="Generated Summary", lines=12)
                        summary_status = make_component(gr.Textbox, label="Summary Status", interactive=False)
                        download_summary = make_component(gr.File, label="Download Summary", visible=False)

                def _transcribe(audio_fp, lang, diar):
                    txt, raw, status, file_path = transcribe_with_hf(audio_fp, language=lang, diarization=diar)
                    return txt or "", status, file_path or None

                def _generate_summary(trans_text, prompt, style, length):
                    summary, status, path = summarize_with_openai(trans_text, prompt, style, length)
                    return summary or "", status, path or None

                transcribe_btn.click(fn=_transcribe, inputs=[audio_in, language_hint, diarization], outputs=[transcript_box, transcribe_status, download_trans])
                summarize_btn.click(fn=_generate_summary, inputs=[transcript_box, prompt_box, style_dropdown, length_dropdown], outputs=[summary_box, summary_status, download_summary])

        gr.Markdown("---\n*Tip:* Edit the transcript to fix ASR errors before generating the final summary.")

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
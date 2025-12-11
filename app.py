# meeting_summarizer_whisper_openai.py
import os
import tempfile
import base64
import requests
import gradio as gr
import inspect
from datetime import datetime
from typing import Optional

# -------------------------
# Config (env) - defaults can be empty; UI will let you set them at runtime
# -------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LOCAL_WHISPER_URL = os.getenv("LOCAL_WHISPER_URL", "")
HF_WHISPER_MODEL = os.getenv("HF_WHISPER_MODEL", "openai/whisper-large-v3")
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")

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
# Model config helpers (UI actions)
# -------------------------
def test_whisper_connection(api_url: str, api_token: str) -> str:
    """Test connectivity to transcription endpoint via /health or base GET."""
    if not api_url:
        return "Please provide a transcription (Whisper) endpoint URL."
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    # Try /health endpoint first
    base_url = api_url.rstrip('/').rsplit('/', 1)[0] if '/v1/' in api_url else api_url.rstrip('/')
    health_url = f"{base_url}/health"
    
    try:
        resp = requests.get(health_url, headers=headers, timeout=10, verify=False)
        if resp.status_code == 200:
            return f"✅ Whisper service reachable via /health. Transcription requires multipart/form-data POST."
        # Try base URL
        resp2 = requests.get(base_url, headers=headers, timeout=10, verify=False)
        if resp2.status_code == 200:
            return f"✅ Whisper service reachable (base URL). Transcription requires multipart/form-data POST."
        return f"❌ Health check failed: {resp.status_code}"
    except Exception as e:
        return f"❌ Connection failed: {e}"

def test_summarizer_connection(
    api_url: str,
    api_token: str,
    payload: dict | None = None
) -> str:
    """Test connectivity using an OpenAI-compatible chat completions request."""
    if not api_url:
        return "Please provide a summarizer endpoint URL."

    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # Default OpenAI-compatible payload
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
    return "Model settings updated successfully!"


def save_whisper_settings(wh_url, wh_token, wh_model):
    update_model_settings(wh_url, wh_token, wh_model, SUMMARIZER_API_URL, SUMMARIZER_API_TOKEN, SUMMARIZER_MODEL_NAME)
    return "✅ Transcription model settings saved (in-memory)."


def save_summarizer_settings(sum_url, sum_token, sum_model):
    update_model_settings(WHISPER_API_URL, WHISPER_API_TOKEN, WHISPER_MODEL_NAME, sum_url, sum_token, sum_model)
    return "✅ Summarizer model settings saved (in-memory)."


def test_whisper_current(wh_url, wh_token):
    return test_whisper_connection(wh_url, wh_token)


def test_summarizer_current(sum_url, sum_token):
    return test_summarizer_connection(sum_url, sum_token)


def clear_model_settings():
    update_model_settings("", "", "", "", "", "")
    return "✅ Cleared saved model settings."

# -------------------------
# Transcription (whisper) - uses WHISPER_API_URL first, then LOCAL_WHISPER_URL, then HF inference
# -------------------------
def transcribe_with_hf(audio_filepath, language=None, diarization=False):
    if not audio_filepath:
        return None, None, "No audio provided", None

    def post_file(endpoint_url: str, token: Optional[str]):
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with open(audio_filepath, "rb") as f:
                files = {"file": (os.path.basename(audio_filepath), f, "audio/wav")}
                data = {}
                if language:
                    data["language"] = language
                resp = requests.post(endpoint_url, headers=headers, files=files, data=data, timeout=120, verify=False) ### -F model
            return resp
        except Exception as e:
            return e

    # 1) Try user-configured WHISPER_API_URL
    if WHISPER_API_URL:
        try:
            resp = post_file(WHISPER_API_URL, WHISPER_API_TOKEN)
            if isinstance(resp, Exception):
                return None, None, f"Transcription request to WHISPER_API_URL failed: {resp}", None
            if resp.ok:
                try:
                    j = resp.json()
                    text = j.get("transcription") or j.get("text") or j.get("result") or ""
                except Exception:
                    text = resp.text
                path = _save_transcript_to_temp(text)
                return text, j if isinstance(j, dict) else None, "Transcribed via WHISPER_API_URL", path
            else:
                return None, None, f"WHISPER_API_URL error: {resp.status_code} {resp.text[:500]}", None
        except Exception as e:
            return None, None, f"WHISPER_API_URL exception: {e}", None

    # 2) LOCAL_WHISPER_URL
    if LOCAL_WHISPER_URL:
        try:
            resp = post_file(LOCAL_WHISPER_URL, None)
            if isinstance(resp, Exception):
                pass
            elif resp.ok:
                try:
                    j = resp.json()
                    text = j.get("transcription") or j.get("text") or resp.text
                except Exception:
                    text = resp.text
                path = _save_transcript_to_temp(text)
                return text, j if isinstance(j, dict) else None, "Transcribed via LOCAL_WHISPER_URL", path
        except Exception:
            pass

    # 3) HF inference fallback
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
        
        # OpenAI-compatible chat completions format (no model param needed)
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
                    # Parse OpenAI-compatible response format
                    summary = None
                    if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                        choice = j["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            summary = choice["message"]["content"]
                        elif "text" in choice:
                            summary = choice["text"]
                    
                    if not summary:
                        # Fallback: try other common response formats
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
                whisper_url_input = make_component(gr.Textbox, label="Transcription endpoint URL", placeholder="https://your-whisper-endpoint/v1/audio/transcriptions")
                whisper_token_input = make_component(gr.Textbox, label="Transcription token (optional)", placeholder="Bearer token", type="password")
                whisper_model_input = make_component(gr.Textbox, label="Model name (optional)", placeholder="whisper-large-v3")

                save_whisper_btn = make_component(gr.Button, label="Save Transcription Settings")
                whisper_save_status = make_component(gr.Textbox, label="Save status", interactive=False)
                test_whisper_btn = make_component(gr.Button, label="Test Transcription Connection")
                whisper_test_status = make_component(gr.Textbox, label="Test status", interactive=False)

                save_whisper_btn.click(fn=save_whisper_settings, inputs=[whisper_url_input, whisper_token_input, whisper_model_input], outputs=[whisper_save_status])
                test_whisper_btn.click(fn=test_whisper_current, inputs=[whisper_url_input, whisper_token_input], outputs=[whisper_test_status])

                gr.Markdown("### Summarizer model (OpenAI-compatible)")
                summarizer_url_input = make_component(gr.Textbox, label="Summarizer endpoint URL", placeholder="https://your-endpoint/v1/chat/completions")
                summarizer_token_input = make_component(gr.Textbox, label="Summarizer token (optional)", type="password")
                summarizer_model_input = make_component(gr.Textbox, label="Summarizer model name (optional)", placeholder="(auto-detected)")

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

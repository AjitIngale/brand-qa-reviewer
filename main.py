import os
import time
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai

app = FastAPI(title="Brand QA Reviewer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

SUPPORTED_MIME_TYPES = {
    # Video
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    # Images
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    # Documents
    ".pdf": "application/pdf",
}

SYSTEM_PROMPT = """You are a Brand QA and Content Review Expert.

Your job is to review uploaded collateral files such as slides, PDFs, images, and videos.

Your review must be concise, structured, practical, and easy for teams to scan quickly.

Always return your response as valid JSON in this exact structure:
{
  "overall_score": <number 1-10>,
  "overall_verdict": "<Pass | Needs Work | Fail>",
  "summary": "<2-3 sentence overall summary>",
  "checks": [
    {
      "category": "<category name>",
      "status": "<Pass | Fail | Warning>",
      "finding": "<what you found>",
      "recommendation": "<what to fix, or 'None' if passing>"
    }
  ],
  "top_issues": ["<issue 1>", "<issue 2>", "<issue 3>"],
  "quick_wins": ["<easy fix 1>", "<easy fix 2>"]
}

Check these categories:
- Color usage (does it match brand colors?)
- Typography (does it match brand fonts?)
- Logo usage (correct placement, size, clear space)
- Tone & messaging (on-brand voice?)
- Layout & spacing (clean, professional?)
- Visual consistency (consistent style throughout?)
- For videos: also check motion graphics, transitions, audio branding

Return ONLY the JSON. No preamble, no markdown backticks."""


def get_mime_type(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    return SUPPORTED_MIME_TYPES.get(ext, "application/octet-stream")


def upload_to_gemini_file_api(file_bytes: bytes, mime_type: str, display_name: str):
    """Upload file to Gemini File API and wait for it to be ACTIVE."""
    import tempfile
    import pathlib

    # Write to temp file (Gemini SDK needs a file path)
    suffix = "." + mime_type.split("/")[-1].replace("quicktime", "mov")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        uploaded = genai.upload_file(path=tmp_path, display_name=display_name, mime_type=mime_type)

        # Wait for file to be processed (important for video)
        max_wait = 120  # seconds
        waited = 0
        while uploaded.state.name == "PROCESSING":
            if waited > max_wait:
                raise TimeoutError("File processing timed out after 120s")
            time.sleep(5)
            waited += 5
            uploaded = genai.get_file(uploaded.name)

        if uploaded.state.name != "ACTIVE":
            raise ValueError(f"File in unexpected state: {uploaded.state.name}")

        return uploaded
    finally:
        os.unlink(tmp_path)


@app.post("/review")
async def review_file(
    file: UploadFile = File(...),
    brand_colors: str = Form(default=""),
    brand_fonts: str = Form(default=""),
):
    try:
        file_bytes = await file.read()
        mime_type = get_mime_type(file.filename)

        if mime_type == "application/octet-stream":
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type: {file.filename}"}
            )

        # Upload to Gemini File API
        uploaded_file = upload_to_gemini_file_api(file_bytes, mime_type, file.filename)

        # Build user prompt
        brand_context = ""
        if brand_colors:
            brand_context += f"\nBrand colors: {brand_colors}"
        if brand_fonts:
            brand_context += f"\nBrand fonts: {brand_fonts}"
        if not brand_context:
            brand_context = "\nNo brand guidelines provided — evaluate general design quality and consistency."

        user_prompt = f"Please review this file for brand QA compliance.{brand_context}\n\nReturn your findings as JSON."

        # Call Gemini
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
        )

        response = model.generate_content([uploaded_file, user_prompt])

        # Clean response and parse JSON
        import json
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        # Clean up uploaded file from Gemini
        try:
            genai.delete_file(uploaded_file.name)
        except Exception:
            pass

        return JSONResponse(content=result)

    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Gemini returned invalid JSON", "raw": response.text[:500]}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/health")
def health():
    return {"status": "ok"}

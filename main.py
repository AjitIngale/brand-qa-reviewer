import os
import time
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai

app = FastAPI(title="Brand QA Reviewer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS, DELETE, PUT",
            "Access-Control-Allow-Headers": "*",
        },
    )

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# Allowed models — frontend can select any of these
ALLOWED_MODELS = {
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-flash",
}
DEFAULT_MODEL = "gemini-2.0-flash"

SUPPORTED_MIME_TYPES = {
    ".mp4":  "video/mp4",
    ".mov":  "video/quicktime",
    ".avi":  "video/x-msvideo",
    ".mkv":  "video/x-matroska",
    ".webm": "video/webm",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".pdf":  "application/pdf",
}

SYSTEM_PROMPT = """You are a Brand QA and Content Review Expert.

Your job is to review uploaded collateral files such as slides, PDFs, images, and videos.

Your review must be concise, structured, practical, and easy for teams to scan quickly.

Always return your response as valid JSON in this exact structure:
{
  "overall_score": <number 1-10>,
  "overall_verdict": "<Pass | Needs Work | Fail>",
  "summary": "<2-3 sentence overall summary>",
  "asset_type": "<slides | video | image | pdf>",
  "checklist": {
    "meets": ["<item>"],
    "needs_changes": ["<item>"],
    "not_detected_or_not_applicable": ["<item>"]
  },
  "sections": [
    {
      "section_label": "<e.g. Slide 1 - Title>",
      "score": <number 1-10>,
      "issues": [
        {
          "type": "<e.g. Branding - Color Consistency>",
          "severity": "<High | Medium | Low>",
          "description": "<what you found>",
          "fix": "<how to fix it>"
        }
      ],
      "meets": ["<what is good on this section>"]
    }
  ],
  "top_fixes": ["<most important fix 1>", "<fix 2>", "<fix 3>"],
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"]
}

Check these categories for every file:
- Color usage (does it match brand colors?)
- Typography (does it match brand fonts?)
- Logo usage (correct placement, size, clear space)
- Tone & messaging (on-brand voice?)
- Layout & spacing (clean, professional?)
- Visual consistency (consistent style throughout?)
- Image & screenshot quality (are screenshots blurry, pixelated, or low resolution?)
- Grammar & language (spelling errors, awkward phrasing, inconsistent capitalization?)
- Screenshot relevance (do screenshots clearly show what they demonstrate?)
- For videos: also check motion graphics, transitions, audio branding
- For slides: review each slide individually in the sections array

Return ONLY the JSON. No preamble, no markdown backticks."""


def get_mime_type(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    return SUPPORTED_MIME_TYPES.get(ext, "application/octet-stream")


def upload_to_gemini(file_bytes: bytes, mime_type: str, display_name: str):
    suffix = "." + mime_type.split("/")[-1].replace("quicktime", "mov")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        uploaded = genai.upload_file(path=tmp_path, display_name=display_name, mime_type=mime_type)
        waited = 0
        max_wait = 180
        while uploaded.state.name == "PROCESSING":
            if waited >= max_wait:
                raise TimeoutError("File processing timed out after 3 minutes")
            time.sleep(5)
            waited += 5
            uploaded = genai.get_file(uploaded.name)
        if uploaded.state.name != "ACTIVE":
            raise ValueError(f"File ended in unexpected state: {uploaded.state.name}")
        return uploaded
    finally:
        os.unlink(tmp_path)


@app.post("/review")
async def review_file(
    file: UploadFile = File(...),
    brand_colors: str = Form(default=""),
    brand_fonts: str = Form(default=""),
    model_name: str = Form(default=DEFAULT_MODEL),
):
    uploaded_file = None
    # Validate model
    if model_name not in ALLOWED_MODELS:
        model_name = DEFAULT_MODEL

    try:
        file_bytes = await file.read()
        mime_type = get_mime_type(file.filename)

        if mime_type == "application/octet-stream":
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type: {file.filename}"},
            )

        uploaded_file = upload_to_gemini(file_bytes, mime_type, file.filename)

        brand_context = ""
        if brand_colors:
            brand_context += f"\nBrand colors to check against: {brand_colors}"
        if brand_fonts:
            brand_context += f"\nBrand fonts to check against: {brand_fonts}"
        if not brand_context:
            brand_context = "\nNo brand guidelines provided — evaluate general design quality."

        user_prompt = (
            f"Please review this file for brand QA compliance.{brand_context}\n\n"
            "Return your complete findings as JSON only."
        )

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content([uploaded_file, user_prompt])

        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["model_used"] = model_name
        return JSONResponse(content=result)

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "Gemini returned invalid JSON. Please try again.", "raw": response.text[:500]},
        )
    except TimeoutError as e:
        return JSONResponse(status_code=504, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception:
                pass


@app.get("/health")
def health():
    return {"status": "ok", "model": "gemini-2.0-flash"}

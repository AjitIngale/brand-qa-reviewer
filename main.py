import os
import time
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import requests as req

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
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://gulufaauvfgijhxkxwwa.supabase.co")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

genai.configure(api_key=GEMINI_API_KEY)

ALLOWED_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-3.0-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
}
DEFAULT_MODEL = "gemini-2.5-flash"

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

GUIDELINES_PATH = os.path.join(os.path.dirname(__file__), "brand_design_guidelines.txt")

def load_guidelines() -> str:
    try:
        with open(GUIDELINES_PATH, "r") as f:
            return f.read()
    except Exception:
        return ""

BASE_SYSTEM_PROMPT = """You are a Brand QA and Content Review Expert with deep knowledge of visual design principles.

Your job is to review uploaded collateral files — slides, PDFs, images, and videos — against the design guidelines provided below.

Your review must be thorough, structured, and actionable. Do NOT skip minor issues. Flag every violation you find, no matter how small.

Always return your response as valid JSON in this exact structure:
{
  "overall_score": <number 1-10>,
  "overall_verdict": "<Pass | Needs Work | Fail>",
  "summary": "<2-3 sentence overall summary>",
  "asset_type": "<slides | video | image | pdf>",
  "checklist": {
    "meets": ["<design principle that is followed>"],
    "needs_changes": ["<design principle that is violated>"],
    "not_detected_or_not_applicable": ["<principle that could not be checked or does not apply>"]
  },
  "sections": [
    {
      "section_label": "<e.g. Slide 1 - Title slide>",
      "score": <number 1-10>,
      "issues": [
        {
          "type": "<design category e.g. Typography | Alignment | Spacing | Color | Icons | Composition>",
          "severity": "<High | Medium | Low>",
          "description": "<exactly which element is affected, where it is, and what design rule it violates>",
          "fix": "<exactly how to fix it>"
        }
      ],
      "meets": ["<design principles correctly applied in this section>"]
    }
  ],
  "top_fixes": ["<most critical fix 1>", "<fix 2>", "<fix 3>", "<fix 4>", "<fix 5>"],
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"]
}

=== DESIGN GUIDELINES TO ENFORCE ===

{GUIDELINES}

=== END OF GUIDELINES ===

Additional instructions:
- Apply ALL guidelines above to every file regardless of content type.
- For slides and multi-page PDFs: review EVERY slide or page individually in the sections array.
- For each issue: name the exact element, its location on the slide, and which guideline it violates.
- List every issue as a separate entry — never bundle multiple issues into one.
- Be strict and thorough — missing issues is worse than flagging too many.

Return ONLY the JSON. No preamble, no markdown backticks."""


def get_system_prompt() -> str:
    guidelines = load_guidelines()
    if guidelines:
        return BASE_SYSTEM_PROMPT.replace("{GUIDELINES}", guidelines)
    return BASE_SYSTEM_PROMPT.replace("{GUIDELINES}", "No specific guidelines file found — apply general design best practices.")


def get_supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json"
    }


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
    user_id: str = Form(default=""),
):
    uploaded_file = None
    if model_name not in ALLOWED_MODELS:
        model_name = DEFAULT_MODEL

    # Check user credits if user_id provided
    if user_id and SUPABASE_SERVICE_KEY:
        profile_res = req.get(
            f"{SUPABASE_URL}/rest/v1/profiles?user_id=eq.{user_id}&select=*",
            headers=get_supabase_headers()
        )
        profiles = profile_res.json()
        if profiles:
            profile = profiles[0]
            if profile["reviews_used"] >= profile["reviews_limit"]:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Monthly review limit reached. Please upgrade your plan."}
                )

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
            brand_context += f"\nBrand colors (ONLY these colors are allowed): {brand_colors}"
        if brand_fonts:
            brand_context += f"\nBrand fonts (ONLY these fonts are allowed): {brand_fonts}"
        if not brand_context:
            brand_context = "\nNo brand colors or fonts specified — check general design consistency and quality."

        user_prompt = (
            f"Please review this file for brand and design QA compliance.{brand_context}\n\n"
            "Apply every guideline strictly. Flag every issue. List each issue separately. "
            "Return your complete findings as JSON only."
        )

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=get_system_prompt(),
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
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "guidelines_loaded": bool(load_guidelines()),
        "supabase_connected": bool(SUPABASE_SERVICE_KEY)
    }

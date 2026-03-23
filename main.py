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
          "type": "<e.g. Layout - Text Overflow>",
          "severity": "<High | Medium | Low>",
          "description": "<exactly what you found and where on the slide>",
          "fix": "<exactly how to fix it>"
        }
      ],
      "meets": ["<what is good on this section>"]
    }
  ],
  "top_fixes": ["<most important fix 1>", "<fix 2>", "<fix 3>"],
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"]
}

Review every file with STRICT attention to detail using universal design principles. Do NOT skip minor issues.

BRANDING:
- Color usage: check every element — background, text, icons, borders, highlights, buttons. Flag any color not matching brand colors exactly.
- Typography: check titles, subtitles, body text, captions separately. Are brand fonts used consistently throughout?
- Logo: correct placement, size, clear space, not distorted?

LAYOUT & SPACING — apply these universal design principles strictly:
- Internal padding: does every box, card, or container have equal and sufficient padding on all four sides? Flag any box where text or content is too close to or touching the edges.
- Spacing consistency: are the gaps between repeated elements (cards, rows, columns, bullets, icons) equal and consistent? Flag any uneven gaps.
- Alignment consistency: are all similar elements (titles, subtitles, body text, images, icons) aligned on the same axis throughout? Flag any element that breaks the alignment grid.
- Element size consistency: if multiple similar elements exist (cards, boxes, buttons, icons), are they all the same size? Flag any size inconsistency.
- Visual hierarchy: is there a clear and consistent size/weight difference between titles, subtitles, and body text?
- White space: is white space used consistently and intentionally? Flag crowded areas or inconsistent breathing room.
- Grid adherence: do all elements snap to a consistent underlying grid? Flag anything that appears misaligned or floating.

VISUAL ELEMENTS:
- Icon consistency: are all icons the same style (all outline OR all filled, never mixed)? Same size? Same visual weight?
- Icon clarity: are icons sharp and clear, not blurry or pixelated?
- Image quality: flag any blurry, pixelated, stretched, or low-resolution images or screenshots.
- Visual consistency: are illustrations, diagrams and graphic elements consistent in style throughout?

CONTENT QUALITY:
- Grammar: check every sentence for spelling errors, typos, missing words, wrong punctuation.
- Language: awkward phrasing, inconsistent capitalization, inconsistent terminology?
- Product and brand names: spelled correctly and consistently throughout?
- Tone: consistent and appropriate for the audience?

For slides: review EVERY slide individually in the sections array.
For each issue: state exactly WHICH element is affected, WHERE it is on the slide, and WHY it violates a design principle.
List every issue separately — do not bundle multiple issues into one.
Be strict — flag everything, even minor inconsistencies.

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
            brand_context = "\nNo brand guidelines provided — evaluate general design quality and consistency."

        user_prompt = (
            f"Please review this file for brand QA compliance.{brand_context}\n\n"
            "Be extremely strict. Flag every spacing, alignment, padding, icon, grammar, and color issue you find. "
            "List each issue separately. Return your complete findings as JSON only."
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
    return {"status": "ok", "default_model": DEFAULT_MODEL}

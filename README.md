# Brand QA Reviewer — Backend

Accepts PDF, video (MP4/MOV), and image uploads, runs them through Gemini 2.0 Flash,
and returns a structured brand QA report as JSON.

## Files

- `main.py` — FastAPI backend
- `requirements.txt` — Python dependencies
- `railway.toml` — Railway deployment config
- `index.html` — Simple frontend (open directly in browser)

## Deploy to Railway (free, ~5 min)

1. Go to https://railway.app and sign up (free tier works)
2. Click **New Project** → **Deploy from GitHub repo**
   - Or use Railway CLI: `railway init` then `railway up`
3. Add environment variable:
   - Key: `GEMINI_API_KEY`
   - Value: your Gemini API key from https://aistudio.google.com/app/apikey
4. Railway auto-deploys. Copy your public URL (e.g. `https://brand-qa.railway.app`)

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
uvicorn main:app --reload
```

Then open `index.html` in your browser (API URL = `http://localhost:8000`)

## API

### POST /review

Form data:
- `file` — the file to review (PDF, MP4, MOV, JPG, PNG, etc.)
- `brand_colors` — optional, e.g. `#df4590, #ffffff`
- `brand_fonts` — optional, e.g. `Arial, Helvetica`

Response JSON:
```json
{
  "overall_score": 7,
  "overall_verdict": "Needs Work",
  "summary": "...",
  "checks": [
    {
      "category": "Color usage",
      "status": "Pass | Fail | Warning",
      "finding": "...",
      "recommendation": "..."
    }
  ],
  "top_issues": ["...", "..."],
  "quick_wins": ["...", "..."]
}
```

## Supported file types

| Type | Extensions |
|------|-----------|
| Video | .mp4, .mov, .avi, .mkv, .webm |
| Image | .jpg, .jpeg, .png, .gif, .webp |
| Document | .pdf |

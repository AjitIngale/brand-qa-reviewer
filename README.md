# QA Buddy — Complete Project

## File Structure

### Frontend Files (goes on Cloudflare Pages)
- home.html — Landing page
- auth.html — Sign in / Sign up
- app.html — Main QA review tool
- settings.html — Account settings
- styles.css — Landing page styles
- Logo.png — Your logo (add this)
- mascot.png — Your mascot image (add this)
- banner-video.mp4 — Hero video (add this)

### Backend Files (goes on Railway)
- main.py — FastAPI Python backend
- requirements.txt — Python dependencies
- railway.toml — Railway config
- Dockerfile — Docker config
- brand_design_guidelines.txt — Design rules for AI

## Environment Variables needed on Railway
- GEMINI_API_KEY = your Gemini API key
- SUPABASE_SERVICE_KEY = your Supabase service role key

## Supabase
- Project URL: https://gulufaauvfgijhxkxwwa.supabase.co
- Tables needed: profiles, reviews (already created)

## Deploy Steps
1. Upload ALL files to one GitHub repo
2. Connect Railway to GitHub repo — deploys main.py automatically
3. Connect Cloudflare Pages to GitHub repo — serves HTML files automatically
4. Add environment variables to Railway
5. Go to Supabase → Authentication → Settings → turn OFF email confirmations for testing

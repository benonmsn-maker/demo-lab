# Streamlit Demo App

## What this folder contains
- `app.py`: Interactive Agentic RAG demo app
- `requirements.txt`: Python dependencies for deployment
- `.streamlit/secrets.toml.example`: Secrets template (do not commit real keys)

## Local run
1. Open terminal in this folder.
2. Install packages:
   - `pip install -r requirements.txt`
3. Set environment variables:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
4. Start app:
   - `streamlit run app.py`

## Streamlit Community Cloud deploy
1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Main file path: `streamlit-demo/app.py` (or `app.py` if repo root is this folder).
4. Add secrets in the app settings:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
5. Deploy and copy the public app URL.

## Hugging Face Spaces deploy (Streamlit SDK)
1. Create a new Space and choose Streamlit SDK.
2. Upload `app.py` and `requirements.txt`.
3. Add repository or Space secrets:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
4. Commit and wait for build.
5. Copy the public Space URL.

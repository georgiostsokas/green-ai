# Running and hosting the dashboard

## On your computer

**macOS:** double-click `Open_Green_AI.command`. **Windows:** double-click
`Open_Green_AI.bat`. The first launch creates a `.venv` inside the project and
installs the app dependencies (Python 3.10 or newer and an internet connection
are needed once). Later launches start straight away.

From a terminal:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[app]"
green-ai app                 # http://127.0.0.1:8501
green-ai app --port 8502     # if 8501 is taken
```

The local server listens on `127.0.0.1` only. Stop it with Ctrl+C.

## Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Create an app from this repository, branch `main`, main file
   `app/streamlit_app.py`. Dependencies come from `requirements.txt`.
3. Optional: add `AA_API_KEY` under *Advanced settings > Secrets* to preload
   Artificial Analysis data. Never commit `.streamlit/secrets.toml`.
4. Deploy, then add the app URL to the top of the README.

Theme and telemetry settings are in `.streamlit/config.toml` (usage statistics
are disabled).

## Hugging Face Spaces

Create a Space with the Streamlit SDK and push this repository to it, with this
header at the top of the Space README:

```yaml
---
title: Green AI
sdk: streamlit
app_file: app/streamlit_app.py
---
```

## Data handling

Uploaded logs and Artificial Analysis responses stay in the memory of the running
session. Nothing is written to disk and custom models do not leak between
sessions. On a hosted copy the uploaded file does reach that host, so use the
local app for logs that should not leave your machine.

Exports contain the normalised analysis columns, not the other columns of the
original log. They still contain model names and token counts, so do not commit
real logs or reports unless you mean to publish them.

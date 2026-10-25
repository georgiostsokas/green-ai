#!/bin/zsh
set -eu
cd "${0:A:h}"
print 'Green AI · starting your local preview'
if ! command -v python3 >/dev/null 2>&1; then
  print 'Python 3.10 or newer is required. Install it from https://www.python.org/downloads/'
  read '?Press Return to close.'
  exit 1
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else "Python 3.10 or newer is required")'
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import green_ai.dashboard' >/dev/null 2>&1; then
  print 'Installing app dependencies into this project (.venv). This is needed on first launch.'
  .venv/bin/python -m pip install -e '.[app]'
fi
print 'Open http://127.0.0.1:8501 · Ctrl+C stops the app. Nothing is published.'
exec .venv/bin/python -m green_ai.cli app

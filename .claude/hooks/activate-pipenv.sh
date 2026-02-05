#!/bin/bash
# Activate pipenv virtualenv and persist to CLAUDE_ENV_FILE
# so all subsequent Bash commands run inside the venv automatically.

if [ -z "$CLAUDE_ENV_FILE" ]; then
  exit 0
fi

VENV_PATH=$(pipenv --venv 2>/dev/null)

if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
  echo "export VIRTUAL_ENV='$VENV_PATH'" >> "$CLAUDE_ENV_FILE"
  echo "export PATH='$VENV_PATH/bin:\$PATH'" >> "$CLAUDE_ENV_FILE"
fi

exit 0

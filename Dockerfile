# Use Python 3.11 — mature and fully supported by all our packages
FROM python:3.11-slim

# Run as a non-root user (Hugging Face Spaces best practice)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache

# Work inside the user's home so the app can write files (database, temp files)
WORKDIR /home/user/app

# Install Python dependencies first (better build caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the app
COPY --chown=user . .

# Listen on the port the host provides (Railway sets $PORT; default 7860 otherwise)
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]

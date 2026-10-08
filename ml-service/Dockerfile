FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --system --create-home app
USER app
EXPOSE 5001
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:5001/health', timeout=2)"
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "app:create_app()"]

# Backend

This folder exposes the production FastAPI entrypoint while reusing the existing deployment code in `src/deployment`.

Run from the project root:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Primary endpoints:

- `GET /health`
- `POST /upload`
- `POST /live-detect`
- `GET /logs`
- `POST /detect` for backward compatibility

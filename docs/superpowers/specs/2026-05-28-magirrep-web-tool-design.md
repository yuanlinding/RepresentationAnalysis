# magirrep Web Tool — Design Spec

**Date:** 2026-05-28  
**Status:** Approved  

## Context

`magirrep` is a Python CLI for Bertaut representational analysis of magnetic structures. The goal is to make it accessible to lab members and collaborators without requiring a local Python install — via a simple webpage where users upload an mCIF file and see the output.

## Architecture

```
Static site (GitHub Pages / Netlify / Vercel)
  └── tools/magirrep.html   ← single static page, no build step

                    ↕  HTTP POST (multipart/form-data)

Render or Fly.io — Docker container
  └── FastAPI app
        └── POST /analyze   ← runs magirrep, returns plain text
```

Two independent deployable pieces:
- **Frontend**: a single `magirrep.html` page added to the existing static personal website under the "Tools" section.
- **Backend**: a `web/` subdirectory inside the `RepresentationAnalysis` repo, deployed as a Docker container to Render (free tier) or Fly.io.

The backend installs `magirrep` directly from GitHub at image build time:
```
pip install git+https://github.com/yuanlinding/RepresentationAnalysis
```

## Repository Layout

Changes are confined to a new `web/` subdirectory in the existing repo. No existing source files are modified.

```
RepresentationAnalysis/
  src/magirrep/          ← existing package (unchanged)
  web/
    Dockerfile           ← Python base image, installs magirrep + fastapi + uvicorn
    requirements.txt     ← fastapi, uvicorn (magirrep installed via git URL)
    app.py               ← FastAPI application (single endpoint)
  docs/
  tests/
  ...
```

The static site gets one new file wherever the "Tools" section lives:
```
personal-website/
  tools/
    magirrep.html        ← self-contained, no JS framework, no build step
```

## Frontend

A single `magirrep.html` page with no JS framework dependency:

- **File upload**: drag-and-drop zone + click-to-browse, accepts `.mcif` and `.cif`
- **Mode selector**: radio buttons — Combined (default), Magnetic only, Displacive
- **k-vector field**: text input (e.g. `0,1/2,0`), shown only when Displacive is selected (plain CIF without embedded k-vector)
- **Run button**: POSTs the file + options to the backend; disabled while a request is in flight
- **Spinner / status**: shown while waiting for the backend response
- **Output area**: monospace `<pre>` block displaying the returned plain-text report
- **Download button**: appears after output; saves result as `{filename}_magirrep.txt`

Styled with minimal inline CSS as a starting point; the user can layer their site's stylesheet on top.

## Backend API

Single FastAPI endpoint:

```
POST /analyze
Content-Type: multipart/form-data

Fields:
  file     : UploadFile   — the .mcif / .cif file
  mode     : str          — "combined" | "magnetic" | "displacive"
  kvector  : str          — optional, e.g. "0,1/2,0" (displacive mode on plain CIF only)

Responses:
  200  text/plain          — full magirrep output text
  422  application/json    — { "detail": "<validation error>" }
  500  application/json    — { "detail": "<error message>" }
```

Implementation details:
- Upload is written to a `tempfile.NamedTemporaryFile` with the correct suffix (`.mcif` / `.cif`)
- `combined` and `magnetic` modes call `pipeline.run_analysis(path, displacive_pass=(mode=="combined"))`
- `displacive` mode calls `pipeline.run_displacive_analysis(path, kvector_str=kvector)`
- Both functions write to a temp output file (via `output_file=` parameter); the file is read back and returned as the response
- A 30-second asyncio timeout guards against runaway computations
- CORS is enabled for `*` (low-traffic lab tool; tighten to specific origin if desired)
- The endpoint runs in a `ThreadPoolExecutor` so the async event loop is not blocked by the synchronous pipeline

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn \
    "git+https://github.com/yuanlinding/RepresentationAnalysis"
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Render or Fly.io builds this image on each push to `main`.

## Deployment

**Backend (Render free tier):**
- Connect Render to the `RepresentationAnalysis` GitHub repo
- Set root directory to `web/`
- Render detects the `Dockerfile` and builds automatically
- Free tier sleeps after 15 min of inactivity → ~20 second cold start on first request; acceptable for lab use

**Frontend:**
- Drop `magirrep.html` into the personal website repo
- The backend URL (Render service URL) is hardcoded in the HTML file as a JS constant

## Error Handling

| Scenario | Behaviour |
|---|---|
| Non-mCIF file uploaded | Frontend checks extension before submitting; backend returns 422 |
| Malformed mCIF | Backend catches exception, returns 500 with error message displayed in output area |
| Timeout (>30s) | Backend returns 500 `"Analysis timed out"`; frontend shows message |
| Backend cold start | Frontend shows spinner with note: "First request may take ~20 seconds" |
| CORS issue | CORS middleware set to allow all origins |

## Out of Scope

- `--distort` mode (generates file downloads — deferred; adds complexity)
- Authentication / rate limiting (lab-only, low traffic)
- Job queue / async job IDs (synchronous request is sufficient for this use case)
- WebAssembly / browser-only execution

# magirrep Web Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static HTML frontend + Dockerized FastAPI backend that lets users upload an mCIF file and see `magirrep` output in a browser, without installing anything locally.

**Architecture:** A single `magirrep.html` page (dropped into the user's existing static personal website) POSTs the uploaded file and mode selection to a FastAPI backend. The backend runs the `magirrep` pipeline directly and returns plain text. The backend is Dockerized and deployed to Render (free tier).

**Tech Stack:** Python 3.11, FastAPI, uvicorn, Docker, vanilla HTML/CSS/JS (no framework)

---

## File Map

| File | Purpose |
|---|---|
| `web/requirements.txt` | Production dependencies for the Docker image |
| `web/app.py` | FastAPI app — health check + POST /analyze endpoint |
| `web/test_app.py` | pytest integration tests for the endpoint |
| `web/Dockerfile` | Docker image for Render deployment |
| `magirrep.html` | Self-contained frontend page (copy into personal website) |

No existing source files are modified.

---

## Task 1: Backend scaffold — health endpoint

**Files:**
- Create: `web/requirements.txt`
- Create: `web/app.py` (health endpoint only)
- Create: `web/test_app.py` (health test only)

- [ ] **Step 1.1: Create `web/requirements.txt`**

```
fastapi
uvicorn[standard]
```

Note: `magirrep` is NOT listed here — in Docker it is installed from GitHub (see Dockerfile task). For local dev testing it is already installed in your repo venv (`pip install -e .` from repo root).

- [ ] **Step 1.2: Create `web/app.py` with empty app (no routes yet)**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

- [ ] **Step 1.3: Write the failing health test in `web/test_app.py`**

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

- [ ] **Step 1.4: Install test dependencies and run — expect FAIL**

```bash
pip install "fastapi" "uvicorn[standard]" httpx pytest
pytest web/test_app.py::test_health -v
```

Expected output: `FAILED web/test_app.py::test_health - assert 404 == 200`

- [ ] **Step 1.5: Add the health endpoint to `web/app.py`**

Add after the `app.add_middleware(...)` block:

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 1.6: Run test — expect PASS**

```bash
pytest web/test_app.py::test_health -v
```

Expected output: `PASSED`

- [ ] **Step 1.7: Commit**

```bash
git add web/requirements.txt web/app.py web/test_app.py
git commit -m "feat(web): scaffold FastAPI backend with health endpoint"
```

---

## Task 2: /analyze endpoint

**Files:**
- Modify: `web/app.py` (add endpoint + helper)
- Modify: `web/test_app.py` (add analyze tests)

- [ ] **Step 2.1: Add the failing analyze tests to `web/test_app.py`**

Append to the existing file (`Path` is already imported from Task 1):

```python
DATA = Path(__file__).parent.parent / "tests" / "data"
NIO_MCIF = DATA / "1.6_NiO.mcif"


def test_analyze_bad_extension():
    r = client.post(
        "/analyze",
        data={"mode": "combined"},
        files={"file": ("data.txt", b"dummy content", "text/plain")},
    )
    assert r.status_code == 422
    assert "detail" in r.json()


def test_analyze_bad_mode():
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            data={"mode": "notamode"},
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r.status_code == 422


def test_analyze_magnetic():
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            data={"mode": "magnetic"},
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert len(r.text) > 200


def test_analyze_combined():
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            data={"mode": "combined"},
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r.status_code == 200
    assert len(r.text) > 200
```

- [ ] **Step 2.2: Run new tests — expect FAIL**

```bash
pytest web/test_app.py -v -k "analyze"
```

Expected output: all four `test_analyze_*` tests FAIL with `404`.

- [ ] **Step 2.3: Replace `web/app.py` with the complete implementation**

```python
import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from magirrep.pipeline import run_analysis, run_displacive_analysis

# Single worker serialises requests — avoids sys.stdout patching races.
_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_class=PlainTextResponse)
async def analyze(
    file: UploadFile,
    mode: str = Form("combined"),
    kvector: Optional[str] = Form(None),
):
    if mode not in ("combined", "magnetic", "displacive"):
        raise HTTPException(status_code=422, detail=f"Invalid mode: {mode!r}")

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".mcif", ".cif"):
        raise HTTPException(status_code=422, detail="File must be .mcif or .cif")

    content = await file.read()

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _run, content, suffix, mode, kvector),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="Analysis timed out (>30 s)")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return result


def _run(content: bytes, suffix: str, mode: str, kvector: Optional[str]) -> str:
    fin = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    fout = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    fin_path, fout_path = fin.name, fout.name
    fin.close()
    fout.close()
    try:
        with open(fin_path, "wb") as f:
            f.write(content)
        if mode == "displacive":
            run_displacive_analysis(fin_path, kvector_str=kvector, output_file=fout_path)
        else:
            run_analysis(fin_path, displacive_pass=(mode == "combined"), output_file=fout_path)
        with open(fout_path, encoding="utf-8") as f:
            return f.read()
    finally:
        for p in (fin_path, fout_path):
            try:
                os.unlink(p)
            except OSError:
                pass
```

- [ ] **Step 2.4: Run all tests — expect PASS**

```bash
pytest web/test_app.py -v
```

Expected output: all 5 tests PASS. The `test_analyze_*` tests will take a few seconds each (real pipeline run).

- [ ] **Step 2.5: Commit**

```bash
git add web/app.py web/test_app.py
git commit -m "feat(web): add POST /analyze endpoint with mode + kvector support"
```

---

## Task 3: Dockerfile

**Files:**
- Create: `web/Dockerfile`

- [ ] **Step 3.1: Create `web/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# gcc/g++ needed only if any dependency lacks a pre-built wheel
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    "git+https://github.com/yuanlinding/RepresentationAnalysis"

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3.2: Build the image locally to verify no errors**

```bash
docker build -t magirrep-web web/
```

Expected: build succeeds. This will take several minutes on first run (downloading pymatgen, gemmi, etc.). If build fails with a missing C compiler error, the `gcc g++` line in the Dockerfile is the fix.

- [ ] **Step 3.3: Run the container and smoke-test the health endpoint**

```bash
docker run --rm -d -p 8000:8000 --name magirrep-test magirrep-web
sleep 5
curl http://localhost:8000/health
docker stop magirrep-test
```

Expected output from curl: `{"status":"ok"}`

- [ ] **Step 3.4: Commit**

```bash
git add web/Dockerfile
git commit -m "feat(web): add Dockerfile for Render deployment"
```

---

## Task 4: Frontend HTML

**Files:**
- Create: `magirrep.html` (repo root; copy to personal website later)

- [ ] **Step 4.1: Create `magirrep.html`**

Replace `YOUR_RENDER_URL` with your Render service URL after deployment (e.g. `https://magirrep-web.onrender.com`).

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>magirrep — Magnetic Irrep Analysis</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; color: #222; }
    h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
    .subtitle { color: #555; margin-bottom: 1.5rem; font-size: 0.95rem; }
    .drop-zone {
      border: 2px dashed #aaa; border-radius: 6px; padding: 2rem;
      text-align: center; cursor: pointer; transition: background 0.2s;
      background: #fafafa;
    }
    .drop-zone.drag-over { background: #eef4ff; border-color: #4a90e2; }
    .drop-zone input[type=file] { display: none; }
    .drop-zone .filename { font-size: 0.9rem; color: #333; margin-top: 0.5rem; }
    .modes { margin: 1rem 0; display: flex; gap: 1.5rem; flex-wrap: wrap; }
    .modes label { display: flex; align-items: center; gap: 0.4rem; cursor: pointer; }
    .kvector-row { margin: 0.5rem 0 1rem; display: none; align-items: center; gap: 0.5rem; }
    .kvector-row input { border: 1px solid #ccc; border-radius: 4px; padding: 0.3rem 0.5rem; width: 140px; font-family: monospace; }
    button#run-btn {
      background: #2c5f8a; color: #fff; border: none; border-radius: 4px;
      padding: 0.6rem 1.4rem; font-size: 1rem; cursor: pointer;
    }
    button#run-btn:disabled { background: #aaa; cursor: not-allowed; }
    .status { margin-top: 0.75rem; font-size: 0.9rem; color: #555; min-height: 1.2em; }
    pre#output {
      background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;
      padding: 1rem; overflow-x: auto; white-space: pre-wrap; font-size: 0.82rem;
      margin-top: 1rem; display: none;
    }
    button#dl-btn {
      display: none; margin-top: 0.5rem; background: #fff;
      border: 1px solid #2c5f8a; color: #2c5f8a; border-radius: 4px;
      padding: 0.4rem 1rem; cursor: pointer; font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <h1>magirrep</h1>
  <p class="subtitle">Bertaut representational analysis for magnetic structures.
    Upload an mCIF file from <a href="https://www.cryst.ehu.es/magndata/" target="_blank">Bilbao MAGNDATA</a>
    or a hand-crafted mCIF.</p>

  <div class="drop-zone" id="drop-zone">
    <input type="file" id="file-input" accept=".mcif,.cif">
    <div>Drag &amp; drop an <strong>.mcif</strong> or <strong>.cif</strong> file here,<br>or <u>click to browse</u></div>
    <div class="filename" id="filename-label"></div>
  </div>

  <div class="modes">
    <label><input type="radio" name="mode" value="combined" checked> Combined (magnetic + displacive)</label>
    <label><input type="radio" name="mode" value="magnetic"> Magnetic only (faster)</label>
    <label><input type="radio" name="mode" value="displacive"> Displacive</label>
  </div>

  <div class="kvector-row" id="kvector-row">
    <label for="kvector">k-vector:</label>
    <input type="text" id="kvector" placeholder="e.g. 0,1/2,0" value="">
    <span style="font-size:0.85rem;color:#777">(leave blank to read from file)</span>
  </div>

  <button id="run-btn">Run Analysis</button>
  <div class="status" id="status"></div>
  <pre id="output"></pre>
  <button id="dl-btn">Download result (.txt)</button>

  <script>
    const API = 'YOUR_RENDER_URL';  // e.g. https://magirrep-web.onrender.com

    const dropZone   = document.getElementById('drop-zone');
    const fileInput  = document.getElementById('file-input');
    const filenameLabel = document.getElementById('filename-label');
    const runBtn     = document.getElementById('run-btn');
    const statusEl   = document.getElementById('status');
    const outputEl   = document.getElementById('output');
    const dlBtn      = document.getElementById('dl-btn');
    const kvectorRow = document.getElementById('kvector-row');
    const kvectorIn  = document.getElementById('kvector');

    let selectedFile = null;

    // --- file selection ---
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      setFile(e.dataTransfer.files[0]);
    });

    function setFile(f) {
      if (!f) return;
      const ext = f.name.split('.').pop().toLowerCase();
      if (!['mcif', 'cif'].includes(ext)) {
        statusEl.textContent = 'Please upload a .mcif or .cif file.';
        return;
      }
      selectedFile = f;
      filenameLabel.textContent = f.name;
      statusEl.textContent = '';
    }

    // --- mode toggle ---
    document.querySelectorAll('input[name=mode]').forEach(r => {
      r.addEventListener('change', () => {
        kvectorRow.style.display = r.value === 'displacive' ? 'flex' : 'none';
      });
    });

    // --- run ---
    runBtn.addEventListener('click', async () => {
      if (!selectedFile) { statusEl.textContent = 'Please select a file first.'; return; }

      const mode    = document.querySelector('input[name=mode]:checked').value;
      const kvector = kvectorIn.value.trim();

      const fd = new FormData();
      fd.append('file', selectedFile);
      fd.append('mode', mode);
      if (kvector) fd.append('kvector', kvector);

      runBtn.disabled = true;
      outputEl.style.display = 'none';
      dlBtn.style.display = 'none';
      statusEl.textContent = 'Running… (first request may take ~20 s if the server is cold)';

      try {
        const resp = await fetch(`${API}/analyze`, { method: 'POST', body: fd });
        const text = await resp.text();
        if (!resp.ok) {
          let msg = text;
          try { msg = JSON.parse(text).detail; } catch (_) {}
          statusEl.textContent = `Error: ${msg}`;
        } else {
          outputEl.textContent = text;
          outputEl.style.display = 'block';
          dlBtn.style.display = 'inline-block';
          statusEl.textContent = 'Done.';
          // store for download
          dlBtn._text = text;
          dlBtn._name = selectedFile.name.replace(/\.[^.]+$/, '') + '_magirrep.txt';
        }
      } catch (err) {
        statusEl.textContent = `Network error: ${err.message}`;
      } finally {
        runBtn.disabled = false;
      }
    });

    // --- download ---
    dlBtn.addEventListener('click', () => {
      const blob = new Blob([dlBtn._text], { type: 'text/plain' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = dlBtn._name;
      a.click();
      URL.revokeObjectURL(a.href);
    });
  </script>
</body>
</html>
```

- [ ] **Step 4.2: Open the file in a browser and verify the UI renders correctly**

```bash
# macOS
open magirrep.html
# Linux
xdg-open magirrep.html
```

Check:
- Drop zone visible, click opens file picker
- Three mode radio buttons present
- k-vector row hidden by default; appears when "Displacive" is selected
- Run button present and styled

- [ ] **Step 4.3: Commit**

```bash
git add magirrep.html
git commit -m "feat(web): add self-contained frontend HTML page"
```

---

## Task 5: Deploy to Render and wire up the URL

This task is manual (browser-based). No code to write.

- [ ] **Step 5.1: Push all changes to GitHub**

```bash
git push origin main
```

- [ ] **Step 5.2: Create a new Render Web Service**

1. Go to [https://render.com](https://render.com) and sign in (free account is sufficient).
2. Click **New → Web Service**.
3. Connect your GitHub account and select the `RepresentationAnalysis` repository.
4. Configure:
   - **Name**: `magirrep-web` (or any name you like)
   - **Root Directory**: `web`
   - **Environment**: `Docker`
   - **Instance Type**: Free
5. Click **Create Web Service**. Render will detect the `Dockerfile` in `web/` and build it. The first build takes several minutes.

- [ ] **Step 5.3: Copy the Render service URL**

After deployment, Render shows a URL like `https://magirrep-web.onrender.com`. Copy it.

- [ ] **Step 5.4: Update the API constant in `magirrep.html`**

In `magirrep.html`, replace:
```javascript
const API = 'YOUR_RENDER_URL';
```
with:
```javascript
const API = 'https://magirrep-web.onrender.com';  // ← your actual URL
```

- [ ] **Step 5.5: Commit and push**

```bash
git add magirrep.html
git commit -m "feat(web): set Render backend URL in frontend"
git push origin main
```

- [ ] **Step 5.6: Smoke-test the live endpoint**

```bash
curl -X POST https://magirrep-web.onrender.com/analyze \
  -F "mode=magnetic" \
  -F "file=@tests/data/1.6_NiO.mcif" \
  --max-time 60
```

Expected: plain text output beginning with the magirrep report header.

- [ ] **Step 5.7: Copy `magirrep.html` into your personal website repo**

Place it wherever your "Tools" section is (e.g. `tools/magirrep.html`). Add a link to it from your Tools index page. Commit and push that repo.

---

## Verification

End-to-end test after deployment:

1. Open the live page in a browser.
2. Upload `tests/data/1.6_NiO.mcif`.
3. Leave mode as "Combined", click **Run Analysis**.
4. Confirm the output block appears with the magirrep report text.
5. Click **Download result** and verify the `.txt` file is saved.
6. Switch to "Displacive" mode, confirm the k-vector field appears.
7. Run again with Displacive mode — confirm a different (displacive) report appears.

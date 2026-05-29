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

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in (".mcif", ".cif"):
        raise HTTPException(status_code=422, detail="File must be .mcif or .cif")

    content = await file.read()

    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _run, content, suffix, mode, kvector),
            timeout=180.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="Analysis timed out (>3 min)")
    except (Exception, SystemExit) as exc:
        msg = str(exc)
        if not msg or msg == "1":
            msg = "Analysis failed (check that the file contains magnetic atoms and a valid propagation vector)"
        raise HTTPException(status_code=500, detail=msg)

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

import asyncio
import multiprocessing
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from multiprocessing import connection as mp_connection
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from magirrep.pipeline import run_analysis, run_displacive_analysis

# Single worker serialises requests.  Each analysis runs in a child process
# (see _run_with_timeout) that is killed on timeout, so a slow request can
# never leave a zombie computation blocking the worker.
_executor = ThreadPoolExecutor(max_workers=1)
_ANALYSIS_TIMEOUT = 180.0

# fork: the child does pure CPU work and exits — no re-import cost, and the
# analysis modules are already loaded.  (Linux-only service.)
_MP = multiprocessing.get_context("fork")


class AnalysisTimeout(Exception):
    pass


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
    mode: str = Form("magnetic"),
    kvector: Optional[str] = Form(None),
):
    if mode not in ("magnetic", "displacive"):
        raise HTTPException(status_code=422, detail=f"Invalid mode: {mode!r}")

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in (".mcif", ".cif"):
        raise HTTPException(status_code=422, detail="File must be .mcif or .cif")

    content = await file.read()

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _run_with_timeout, content, suffix, mode, kvector,
            _ANALYSIS_TIMEOUT,
        )
    except AnalysisTimeout as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        msg = str(exc)
        if not msg or msg == "1":
            msg = "Analysis failed (check that the file contains magnetic atoms and a valid propagation vector)"
        raise HTTPException(status_code=500, detail=msg)

    return result


def _child(conn, content: bytes, suffix: str, mode: str, kvector: Optional[str]):
    """Child-process entry point: run the analysis, ship result or error back."""
    try:
        conn.send(("ok", _run(content, suffix, mode, kvector)))
    except BaseException as exc:   # includes SystemExit raised by the pipeline
        conn.send(("err", str(exc)))
    finally:
        conn.close()


def _run_with_timeout(content: bytes, suffix: str, mode: str,
                      kvector: Optional[str], timeout: float) -> str:
    """Run the analysis in a child process, killing it if *timeout* elapses."""
    recv, send = _MP.Pipe(duplex=False)
    proc = _MP.Process(target=_child, args=(send, content, suffix, mode, kvector),
                       daemon=True)
    proc.start()
    send.close()
    try:
        ready = mp_connection.wait([recv, proc.sentinel], timeout)
        if recv in ready:
            try:
                status, payload = recv.recv()
            except EOFError:
                status, payload = "err", "Analysis process crashed unexpectedly"
        elif ready:   # process died without sending a result
            status, payload = "err", "Analysis process crashed unexpectedly"
        else:
            raise AnalysisTimeout(f"Analysis timed out (>{int(timeout)} s)")
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        recv.close()
    if status == "err":
        raise RuntimeError(payload)
    return payload


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
            run_analysis(fin_path, displacive_pass=False, output_file=fout_path)
        with open(fout_path, encoding="utf-8") as f:
            return f.read()
    finally:
        for p in (fin_path, fout_path):
            try:
                os.unlink(p)
            except OSError:
                pass

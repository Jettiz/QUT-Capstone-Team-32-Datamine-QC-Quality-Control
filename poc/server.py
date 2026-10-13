"""
Local server for the QC proof of concept (standard library only).

Serves the static POC pages in this folder and runs the real Python QC
analysis for every method registered in src/qc_report/registry.py:

    python poc/server.py            -> http://localhost:8000
    python poc/server.py --port 8080

API (JSON unless noted):
    POST /api/analyse?history=on|off&filename=<name>     body = the raw file
        -> run summary {run_id, file_name, history_mode, n_rows, methods: {id: summary}}
    GET  /api/runs/<run_id>                               -> run summary
    GET  /api/runs/<run_id>/methods/<method>/jobs/<key>   -> one job with its items
    GET  /api/runs/<run_id>/methods/<method>/charts/<item>.png  -> chart (PNG),
         rendered on first request and cached for the run

Opening a job also starts rendering that job's charts in the background
(worst first), so switching between analytes is instant. Runs are kept in
memory (the last MAX_RUNS); restarting the server means re-uploading the file.
"""

import argparse
import atexit
import json
import logging
import re
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

POC_DIR = Path(__file__).resolve().parent
REPO_ROOT = POC_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_qc_data  # noqa: E402
from src.qc_report import QC_METHODS  # noqa: E402

COLUMN_CONFIG = REPO_ROOT / "config" / "column_config.csv"
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_RUNS = 5

log = logging.getLogger("poc.server")

# The browser closed the connection (navigated away, or a newer chart request
# replaced this one) -- normal, nothing to report or answer.
_CLIENT_GONE = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)

# matplotlib is not guaranteed thread-safe, so charts are drawn one at a
# time (each takes ~0.1 s).
_RENDER_LOCK = threading.Lock()


class UserError(Exception):
    """A problem with the request/file that the user can fix (HTTP 400)."""


class Run:
    def __init__(self, run_id, file_name, history_mode, n_rows, methods, work_dir):
        self.run_id = run_id
        self.file_name = file_name
        self.history_mode = history_mode
        self.n_rows = n_rows
        self.methods = methods          # {method_id: QCMethodResult}
        self.work_dir = work_dir        # temp folder for the upload + chart cache
        self.prefetch_generation = 0    # bumped per opened job; stale prefetches stop

    def summary(self):
        return {
            "run_id": self.run_id,
            "file_name": self.file_name,
            "history_mode": self.history_mode,
            "n_rows": self.n_rows,
            "methods": {mid: result.to_summary() for mid, result in self.methods.items()},
        }


_RUNS = OrderedDict()
_RUNS_LOCK = threading.Lock()


def _store_run(run):
    with _RUNS_LOCK:
        _RUNS[run.run_id] = run
        while len(_RUNS) > MAX_RUNS:
            _, old = _RUNS.popitem(last=False)
            shutil.rmtree(old.work_dir, ignore_errors=True)


def _get_run(run_id):
    with _RUNS_LOCK:
        run = _RUNS.get(run_id)
    if run is None:
        raise UserError("This analysis is no longer available (the server was restarted or newer files were "
                        "analysed). Please load the file again.")
    return run


@atexit.register
def _cleanup():
    with _RUNS_LOCK:
        for run in _RUNS.values():
            shutil.rmtree(run.work_dir, ignore_errors=True)


def analyse_upload(data: bytes, file_name: str, use_history: bool) -> Run:
    """Load the uploaded file with the product loader and run every registered QC method."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UserError(f"Unsupported file type '{suffix or file_name}'. Use one of: "
                        + ", ".join(sorted(SUPPORTED_SUFFIXES)))
    if not data:
        raise UserError("The uploaded file is empty.")

    work_dir = Path(tempfile.mkdtemp(prefix="qc_poc_"))
    try:
        upload_path = work_dir / f"upload{suffix}"
        upload_path.write_bytes(data)
        try:
            df = load_qc_data(data_path=upload_path, config_path=COLUMN_CONFIG)
        except (ValueError, FileNotFoundError) as exc:
            raise UserError(f"The file could not be loaded: {exc}") from exc

        methods = {}
        for method_id, method in QC_METHODS.items():
            methods[method_id] = method.run(df, use_history=use_history)

        run = Run(uuid.uuid4().hex[:12], file_name, "on" if use_history else "off", len(df), methods, work_dir)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    _store_run(run)
    return run


def _render_cached(run: Run, method_id: str, item) -> Path:
    """Render an item's chart once per run; later calls return the cached file."""
    path = run.work_dir / "charts" / f"{method_id}_{item.id}.png"
    if path.is_file():
        return path
    with _RENDER_LOCK:
        if not path.is_file():
            tmp = path.with_name(path.stem + ".tmp.png")
            QC_METHODS[method_id].render_chart(item, tmp)
            tmp.replace(path)  # atomic: a reader never sees a half-written PNG
    return path


def chart_path(run: Run, method_id: str, item_id: str) -> Path:
    result = run.methods.get(method_id)
    item = result.find_item(item_id) if result else None
    if item is None or not item.has_chart:
        raise UserError("Unknown chart.")
    return _render_cached(run, method_id, item)


def prefetch_job_charts(run: Run, method_id: str, job) -> None:
    """Render one job's charts in the background, worst first. Opening another
    job stops the previous prefetch."""
    with _RUNS_LOCK:
        run.prefetch_generation += 1
        generation = run.prefetch_generation
    items = [item for item in job.sorted_items() if item.has_chart]

    def work():
        for item in items:
            if run.prefetch_generation != generation or not run.work_dir.exists():
                return
            try:
                _render_cached(run, method_id, item)
            except Exception as exc:  # a failed chart must not stop the others
                log.warning("Chart prefetch failed for %s/%s: %s", method_id, item.id, exc)

    threading.Thread(target=work, daemon=True, name=f"prefetch-{run.run_id}").start()


_ROUTES = [
    ("GET", re.compile(r"^/api/runs/(?P<run>[0-9a-f]+)$"), "get_run"),
    ("GET", re.compile(r"^/api/runs/(?P<run>[0-9a-f]+)/methods/(?P<method>[\w-]+)/jobs/(?P<job>\d+)$"), "get_job"),
    ("GET", re.compile(r"^/api/runs/(?P<run>[0-9a-f]+)/methods/(?P<method>[\w-]+)/charts/(?P<item>\d+)\.png$"), "get_chart"),
    ("POST", re.compile(r"^/api/analyse$"), "post_analyse"),
]


class Handler(SimpleHTTPRequestHandler):
    """Static files from poc/ plus the small JSON API above."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(POC_DIR), **kwargs)

    def end_headers(self):
        # Always serve fresh files/results during development.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    # ── dispatch ──
    def do_GET(self):
        if not self._dispatch("GET"):
            super().do_GET()

    def do_POST(self):
        if not self._dispatch("POST"):
            self._send_json({"error": "Not found"}, 404)

    def _dispatch(self, verb):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            return False
        for route_verb, pattern, handler_name in _ROUTES:
            match = pattern.match(parsed.path)
            if route_verb == verb and match:
                try:
                    getattr(self, handler_name)(parse_qs(parsed.query), **match.groupdict())
                except _CLIENT_GONE:
                    log.debug("Client went away during %s", parsed.path)
                except UserError as exc:
                    self._send_json_quietly({"error": str(exc)}, 400)
                except Exception as exc:  # unexpected: log the traceback, report briefly
                    log.error("Request failed: %s\n%s", exc, traceback.format_exc())
                    self._send_json_quietly({"error": f"Analysis failed: {exc}"}, 500)
                return True
        self._send_json({"error": "Not found"}, 404)
        return True

    # ── handlers ──
    def post_analyse(self, query):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise UserError("The file is too large for this proof of concept.")
        data = self.rfile.read(length) if length else b""
        file_name = unquote((query.get("filename") or ["upload.csv"])[0])
        use_history = (query.get("history") or ["on"])[0].lower() != "off"
        run = analyse_upload(data, file_name, use_history)
        self._send_json(run.summary())

    def get_run(self, query, run):
        self._send_json(_get_run(run).summary())

    def get_job(self, query, run, method, job):
        result = _get_run(run).methods.get(method)
        if result is None:
            raise UserError(f"No results for '{method}' in this analysis.")
        jobs = {j.key: j for j in result.jobs}
        if job not in jobs:
            raise UserError("Unknown job.")
        prefetch_job_charts(_get_run(run), method, jobs[job])
        self._send_json(jobs[job].to_dict())

    def get_chart(self, query, run, method, item):
        path = chart_path(_get_run(run), method, item)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json_quietly(self, payload, status):
        """Send an error response, unless the client has already gone."""
        try:
            self._send_json(payload, status)
        except _CLIENT_GONE:
            pass


def make_server(port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description="Serve the QC proof of concept with real analysis.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    # force=True: src/data_loader.py configures logging on import with its own name baked in.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s", force=True)
    server = make_server(args.port, args.host)
    print(f"QC proof of concept running at http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""
Smoke test for poc/server.py: upload a small CCLAS-style CSV through the
real HTTP API (History off, so nothing is written to data/processed), then
fetch the run summary, one job's items and a chart.
"""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_server_module():
    spec = importlib.util.spec_from_file_location("poc_server", REPO_ROOT / "poc" / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_row(job, day, analyte, value, lot="Sample"):
    return {
        "ANALYTICAL_TYPE": "Standard", "STD_LOT_CODE": lot, "STD_CODE": "OREAS_502C", "JOB_CODE": job,
        "NUMERIC_FINAL_VALUE": value, "ANALYSED_DATE": f"2024-01-{day:02d} 10:00:00",
        "SCHEME_CODE": "GE_ICP40Q12", "ANALYTE_CODE": analyte, "STANDARD_STATUS": "Pass",
        "PRECISION_STATUS": "", "INTERNAL_MIN_VALUE": 90.0, "INTERNAL_MAX_VALUE": 110.0,
        "INTERNAL_MIN_INCLUSIVE": "Y", "INTERNAL_MAX_INCLUSIVE": "Y",
        "INTERNAL_MAX_WARNING_VALUE": 106.0, "INTERNAL_MIN_WARNING_VALUE": 94.0,
        "INTERNAL_MIN_WARNING_INCLUSIVE": "Y", "INTERNAL_MAX_WARNING_INCLUSIVE": "Y",
        "INTERNAL_TARGET_VALUE": 100.0, "UNIT_CODE": "MG_KG", "SPECIFICATION_CODE": "OREAS_502C",
        "INSTRUMENT_ID": "",
    }


@pytest.fixture()
def server():
    module = _load_server_module()
    httpd = module.make_server(0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, data):
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
    return json.load(urllib.request.urlopen(request))


def test_analyse_summary_job_and_chart(server, tmp_path):
    rows = [_raw_row("J1", 1, "CU", 100.0), _raw_row("J1", 1, "PB", 115.0),
            _raw_row("J2", 2, "CU", 101.0), _raw_row("J2", 2, "PB", 100.0),
            _raw_row("J2", 2, "ZN", 100.0, lot="OREAS_905")]  # SRM row: not part of LCS
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

    summary = _post(server + "/api/analyse?history=off&filename=sample.csv", csv_bytes)
    control = summary["methods"]["control"]
    assert summary["history_mode"] == "off"
    assert control["usable"] is True
    assert [job["job"] for job in control["jobs"]] == ["J1", "J2"]
    assert control["counts"] == {"FAIL": 1, "WARNING": 0, "PASS": 3}

    again = json.load(urllib.request.urlopen(f"{server}/api/runs/{summary['run_id']}"))
    assert again["methods"]["control"]["counts"] == control["counts"]

    job = json.load(urllib.request.urlopen(f"{server}/api/runs/{summary['run_id']}/methods/control/jobs/0"))
    assert job["job"] == "J1"
    assert job["instruments"] == ["Instrument unknown"]
    assert job["items"][0]["code"] == "PB" and job["items"][0]["state"] == "FAIL"

    png = urllib.request.urlopen(
        f"{server}/api/runs/{summary['run_id']}/methods/control/charts/{job['items'][0]['id']}.png").read()
    assert png[:4] == b"\x89PNG"


def test_static_pages_are_served(server):
    assert urllib.request.urlopen(server + "/index.html").status == 200
    assert urllib.request.urlopen(server + "/api.js").status == 200


@pytest.mark.parametrize("path,body", [
    ("/api/runs/0123456789ab", None),
    ("/api/analyse?filename=notes.txt", b"abc"),
    ("/api/analyse?filename=empty.csv", b""),
])
def test_errors_are_reported_as_json_400(server, path, body):
    request = urllib.request.Request(server + path, data=body, method="POST" if body is not None else "GET")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request)
    assert exc.value.code == 400
    assert "error" in json.loads(exc.value.read())

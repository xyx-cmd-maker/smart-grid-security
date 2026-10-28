"""OpenAlex fetch with status and count printed before any interpretation."""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

NODE = "node"


def _curl(url: str, timeout: int = 180) -> tuple[int, str]:
    """Return (status, body). Status 0 means the request itself failed."""
    marker = "<<<OA-SEP>>>"
    script = (
        "fetch(process.argv[1]).then(r=>r.text().then(t=>process.stdout.write(r.status+process.argv[2]+t)))"
        ".catch(e=>{process.stdout.write('0'+process.argv[2]+e.message)})"
    )
    r = subprocess.run([NODE, "-e", script, url, marker], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        return 0, r.stderr.strip()[:300]
    head, _, body = r.stdout.partition(marker)
    try:
        return int(head), body
    except ValueError:
        return 0, head[:300]


def get_json(url: str, expect_results: bool = True, timeout: int = 180) -> dict[str, Any]:
    """Fetch, print status and count, and raise on anything other than a clean 200.

    An empty result list from a 200 is returned; an empty list from a 429 raises, because the two are
    indistinguishable downstream and only one of them is a finding.
    """
    status, body = _curl(url, timeout=timeout)
    if status == 0:
        raise RuntimeError(f"request failed: {body}")
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {body[:300]}")
    try:
        d = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"unparseable body: {e}; head={body[:200]}")

    if expect_results:
        meta = d.get("meta") or {}
        count = meta.get("count")
        n = len(d.get("results") or [])
        print(f"    HTTP {status}  meta.count={count}  results={n}")
        if count is None:
            raise RuntimeError("list endpoint without meta.count; do not interpret this response")
    else:
        print(f"    HTTP {status}  keys={sorted(d)[:6]}")
    return d


def works_in_source(source_id: str, query: str, pages: int = 3, per_page: int = 100) -> list[dict]:
    """Paginate a work search, printing status and count for every page."""
    out: list[dict] = []
    import urllib.parse as up
    for page in range(1, pages + 1):
        url = (f"https://api.openalex.org/works?filter=primary_location.source.id:{source_id},"
               f"title_and_abstract.search:{up.quote(query)}&per-page={per_page}&page={page}")
        d = get_json(url)
        res = d.get("results") or []
        if not res:
            print(f"    page {page}: empty, stopping")
            break
        out.extend(res)
    return out

def self_test() -> int:
    """Six synthetic responses; returns 0 if every branch behaves as specified."""
    cases = [
        ("list without meta.count", 200, '{"results":[{"title":"x"}]}', True, True),
        ("list with count", 200, '{"meta":{"count":3},"results":[{"title":"x"}]}', True, False),
        ("unparseable body", 200, '<html>nope</html>', True, True),
        ("empty list count 0", 200, '{"meta":{"count":0},"results":[]}', True, False),
        ("http 500", 500, '{"error":"boom"}', True, True),
        ("single record", 200, '{"id":"S1"}', False, False),
    ]
    global _curl
    real = _curl
    bad = 0
    try:
        for name, status, body, expect, should_raise in cases:
            _curl = lambda url, timeout=180, _s=status, _b=body: (_s, _b)
            raised = False
            try:
                get_json("http://invalid", expect_results=expect)
            except RuntimeError:
                raised = True
            mark = "OK  " if raised == should_raise else "FAIL"
            if raised != should_raise:
                bad += 1
            print(f"  {mark} {name:<26} raised={raised} expected={should_raise}")
    finally:
        _curl = real
    print(f"  branches correct: {len(cases) - bad}/{len(cases)}")
    print("  note: this verifies the guard against SYNTHETIC responses; it does not verify API behaviour")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    if len(sys.argv) < 2:
        print("usage: python code/oa_fetch.py <url>")
        sys.exit(2)
    d = get_json(sys.argv[1])
    print(f"    first title: {((d.get('results') or [{}])[0]).get('title', '?')[:90]}")

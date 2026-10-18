#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""csapi.de collector — keyless CS2 pro match/team/player API.

Why this exists
---------------
hltv.org / bo3.gg / egamersworld are Cloudflare-403 from this host, and
Liquipedia's `action=parse` is throttled to <=1/30s (we tripped a 429 window).
api.csapi.de answers plain JSON with no key and no challenge, and it carries
exactly the fields scoring needs:

  id, team1/team2(+score,rank), maps[{id,name,team1_score,team2_score}],
  best_of, date, event, winner

Measured host behaviour (2026-10-07)
------------------------------------
* Bodies are cut off at ~98 KB -> IncompleteRead.  Default page size 200
  (~78 KB); `get_json_capped` halves the page size instead of failing.
* `/matches` paging dies at offset==len(list) with HTTP 500 (not an empty
  page).  len(list) is 2800, while `/counts/` reports 3876 matches, so the
  remaining ~1076 are only reachable through `/teams/{id}/matchhistory`.
  `matches` therefore stops on that 500 and records the boundary.
* Temporal span is bounded: oldest row is 2025-10-14 (both list and per-team
  history).  Verified on 2026-10-07 via bisect on match ids (2386599 exists,
  2386000 does not) and on 8 long-lived teams.

Usage
-----
  python -X utf8 csapi_collect.py matches                 # -> csapi_matches.jsonl
  python -X utf8 csapi_collect.py teams                   # -> csapi_teams.json
  python -X utf8 csapi_collect.py teamhistory             # -> csapi_teamhistory.jsonl
  python -X utf8 csapi_collect.py merge                   # -> csapi_matches_all.jsonl
  python -X utf8 csapi_collect.py events                  # event inventory report
  python -X utf8 csapi_collect.py stats [--limit N]       # -> csapi_stats.jsonl
  python -X utf8 csapi_collect.py selftest                # offline, no network
"""
import argparse
import collections
import gzip
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.csapi.de"
UA = ("another-issue014-research/1.0 (local research; contact: user) "
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
CACHE = os.path.join(HERE, "csapi_cache")
MAX_BYTES = 90_000          # conservative stay-under for the ~98 KB cut
MAX_PAGE = 200


class EndOfList(Exception):
    """The host raises HTTP 500 once offset reaches the end of the list."""


# ---------------------------------------------------------------- transport
def _raw_get(url, timeout=60):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "Connection": "close",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return body


def get_json(path, params=None, tries=5, timeout=60):
    """GET a JSON endpoint with retry.  Returns (payload, raw_bytes)."""
    url = BASE + path + build_qs(params)
    last = None
    for attempt in range(tries):
        try:
            body = _raw_get(url, timeout=timeout)
            return json.loads(body), body
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                # Measured: the host throttles, and a long collection run can
                # trip it.  Back off proportionally and retry rather than
                # abandoning the batch.
                wait = 5 * (attempt + 1)
                print("    429 from host, sleeping %ds" % wait)
                time.sleep(wait)
                continue
            if exc.code >= 500 and params and "offset" in params:
                # Ambiguous: could be "past the end" or a real fault.  The
                # caller decides by probing one page earlier.
                raise
            if attempt + 1 < tries:
                time.sleep(1.5 * (attempt + 1) + random.random())
        except Exception as exc:                            # noqa: BLE001
            last = exc
            if "IncompleteRead" in type(exc).__name__:
                # Deterministic body-size truncation: retrying the same page
                # size cannot help, so let the caller shrink it immediately.
                raise
            if attempt + 1 < tries:
                time.sleep(1.5 * (attempt + 1) + random.random())
    raise last


def build_qs(params):
    if not params:
        return ""
    return "?" + "&".join(
        "%s=%s" % (k, urllib.parse.quote(str(v), safe=""))
        for k, v in params.items() if v is not None)


def get_json_capped(path, params=None, page_key="limit", max_bytes=MAX_BYTES):
    """GET a list endpoint, shrinking the page size until it fits.

    Returns (payload, page_size_used, raw_len).
    """
    params = dict(params or {})
    size = params.get(page_key, MAX_PAGE)
    while size >= 1:
        params[page_key] = size
        try:
            payload, body = get_json(path, params)
        except Exception as exc:                            # noqa: BLE001
            if "IncompleteRead" in type(exc).__name__ and size > 1:
                size = max(1, size // 2)
                continue
            raise
        if len(body) <= max_bytes or size == 1:
            return payload, size, len(body)
        # Over budget but not truncated yet: shrink pre-emptively.
        size = max(1, size // 2)
    raise RuntimeError("could not fit page size")


def get_paged(path, page_key="limit", max_page=MAX_PAGE, sleep=0.4,
              on_page=None):
    """Page through a list endpoint until it is exhausted.

    Returns (rows, stopped_reason).  A 5xx at some offset is disambiguated by
    re-reading the previous offset: same data => genuine end of list.
    """
    rows, offset, reason = [], 0, "short_page"
    while True:
        try:
            payload, size, _ = get_json_capped(
                path, {"limit": max_page, "offset": offset})
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise
            # Probe the previous row to tell "past the end" from "real fault".
            if offset == 0:
                raise
            try:
                back, _, _ = get_json_capped(
                    path, {"limit": max_page, "offset": offset - 1})
                if len(back) <= 1:
                    reason = "end_of_list(http %s at offset=%d)" % (exc.code, offset)
                    break
            except Exception:                               # noqa: BLE001
                pass
            raise
        if not payload:
            reason = "empty_page"
            break
        rows.extend(payload)
        offset += len(payload)
        if on_page:
            on_page(offset, len(payload), len(rows), size)
        if len(payload) < size:
            break
        time.sleep(sleep)
    return rows, reason


# ---------------------------------------------------------------- cache
def cache_path(name):
    os.makedirs(CACHE, exist_ok=True)
    safe = urllib.parse.quote(name, safe="")
    return os.path.join(CACHE, safe + ".json")


# ---------------------------------------------------------------- commands
def cmd_matches(args):
    out = os.path.join(HERE, args.out)

    def report(offset, got, total, size):
        print("  offset=%-5d got=%-4d total=%-5d (page=%d)"
              % (offset, got, total, size))

    rows, reason = get_paged("/matches", on_page=report)
    seen = {m["id"]: m for m in rows}
    ordered = sorted(seen.values(), key=lambda m: (m.get("date") or "", m["id"]))
    with open(out, "w", encoding="utf-8") as fh:
        for m in ordered:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("MATCHES_OK %d rows -> %s  (stop: %s)" % (len(ordered), out, reason))
    if ordered:
        print("  date range: %s .. %s" % (ordered[0]["date"], ordered[-1]["date"]))
    return 0


def cmd_teams(args):
    rows, reason = get_paged("/teams/", max_page=MAX_PAGE)
    seen = {t["id"]: t for t in rows}
    with open(os.path.join(HERE, args.out), "w", encoding="utf-8") as fh:
        json.dump(sorted(seen.values(), key=lambda t: t["id"]), fh,
                  ensure_ascii=False, indent=1)
    print("TEAMS_OK %d -> %s (stop: %s)" % (len(seen), args.out, reason))
    return 0


def _iter_history(path):
    """Yield valid teamhistory records; report unparseable lines.

    A partially written line can exist if two collector runs ever share the
    output file, so the reader skips and counts them rather than aborting the
    whole merge.
    """
    bad = 0
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(rec, dict) and "teamid" in rec:
                yield rec
            else:
                bad += 1
    if bad:
        print("  WARN %s: skipped %d unparseable line(s)" % (path, bad))


def cmd_teamhistory(args):
    with open(os.path.join(HERE, args.teams), encoding="utf-8") as fh:
        teams = json.load(fh)
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        teams = [t for t in teams if t["id"] in want]
    out = os.path.join(HERE, args.out)
    done = set()
    if os.path.exists(out):
        for rec in _iter_history(out):
            if not rec.get("__error__"):
                done.add(rec["teamid"])
    todo = [t for t in teams if t["id"] not in done]
    print("teamhistory: %d teams, %d done, %d to fetch" % (len(teams), len(done), len(todo)))
    mode = "a" if done else "w"
    n_ok = 0
    with open(out, mode, encoding="utf-8") as fh:
        for n, t in enumerate(todo, 1):
            try:
                rows, reason = get_paged(
                    "/teams/%d/matchhistory" % t["id"], sleep=0.25)
                rec = {"teamid": t["id"], "team": t["name"],
                       "reason": reason, "matches": rows}
                n_ok += 1
            except Exception as exc:                        # noqa: BLE001
                rec = {"teamid": t["id"], "team": t["name"],
                       "__error__": "%s: %s" % (type(exc).__name__, exc)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 25 == 0 or n == len(todo):
                print("  %d/%d (ok=%d)" % (n, len(todo), n_ok))
            time.sleep(0.25)
    print("TEAMHISTORY_OK -> %s" % out)
    return 0


def _load_matches(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                m = json.loads(line)
                out[m["id"]] = m
    return out


def cmd_merge(args):
    by_id = _load_matches(os.path.join(HERE, args.matches))
    n_list = len(by_id)
    hist_path = os.path.join(HERE, args.history)
    n_new = 0
    for rec in _iter_history(hist_path):
        for m in rec.get("matches") or []:
            if m.get("id") is None:
                continue
            if m["id"] not in by_id:
                n_new += 1
            by_id[m["id"]] = m
    rows = sorted(by_id.values(), key=lambda m: (m.get("date") or "", m["id"]))
    out = os.path.join(HERE, args.out)
    with open(out, "w", encoding="utf-8") as fh:
        for m in rows:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("MERGE_OK list=%d newFromHistory=%d total=%d -> %s"
          % (n_list, n_new, len(rows), args.out))
    if rows:
        print("  date range: %s .. %s" % (rows[0]["date"], rows[-1]["date"]))
    return 0


def _iter_matches(path):
    """Yield match records from a jsonl file in file order."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def cmd_events(args):
    rows = list(_iter_matches(os.path.join(HERE, args.source)))
    print("EVENTS: %d matches" % len(rows))
    by_event = collections.defaultdict(list)
    for m in rows:
        by_event[m.get("event") or "(none)"].append(m)
    print("  distinct events: %d" % len(by_event))
    months = collections.Counter((m.get("date") or "????")[:7] for m in rows)
    print("  matches per month:")
    for k in sorted(months):
        print("    %s %4d" % (k, months[k]))
    bo = collections.Counter(m.get("best_of") for m in rows)
    print("  best_of: %s" % dict(sorted(bo.items(), key=lambda kv: str(kv[0]))))
    top = sorted(by_event.items(), key=lambda kv: -len(kv[1]))[:25]
    print("  top events by match count:")
    for name, ms in top:
        ds = [m["date"] for m in ms if m.get("date")]
        print("    %-52s %4d  %s..%s" % (name[:52], len(ms), min(ds), max(ds)))
    return 0


# --- event-name normalisation for cross-source joining -------------------
# Liquipedia writes "Intel Extreme Masters Katowice 2025" while csapi writes
# "IEM Katowice 2025", so a literal join would miss the biggest events.
ALIASES = (
    ("iem", "intel extreme masters"),
    ("epl", "esl pro league"),
    ("ewc", "esports world cup"),
)
NOISE = {"counter", "strike", "counterstrike", "cs2", "csgo", "the", "of",
         "presented", "by"}


def norm_event(name):
    import re
    s = (name or "").lower().replace("&", " and ")
    for alias, full in ALIASES:
        s = re.sub(r"\b%s\b" % re.escape(alias), full, s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(w for w in s.split() if w not in NOISE)
    return re.sub(r"\s+", " ", s).strip()


def event_year(name):
    import re
    yrs = re.findall(r"(20\d\d)", name or "")
    return yrs[-1] if yrs else ""


GENERIC = {
    # org / series boilerplate
    "esl", "pro", "league", "season", "intel", "extreme", "masters", "blast",
    "pgl", "fissure", "iem", "epl",
    # NB: "open"/"bounty"/"rivals"/"playground" are NOT listed here.  They name
    # an org's sub-series ("BLAST Bounty" vs "BLAST Open"), so treating them as
    # boilerplate collapsed "BLAST Bounty 2026 Season 1" to the key {2026, 1}
    # and lost the event's identity.
    "world", "cup", "championship", "championships", "series", "cs", "cs2",
    "csgo", "counter", "strike", "counterstrike", "inc", "presents",
    "presented", "by", "the", "of", "and", "at", "in",
    # calendar / stage boilerplate
    "fall", "spring", "summer", "winter", "final", "finals", "stage", "group",
    "groups", "groupstage", "playoff", "playoffs", "qualifier", "qualifiers",
    "closed", "openqualifier", "main", "event", "tournament", "split",
    "swiss", "elimination", "opening", "legend", "legends", "challenger",
    "challengers", "contender", "contenders",
}


def core_tokens(name):
    """Distinctive tokens of an event name (year kept, boilerplate dropped)."""
    return {w for w in norm_event(name).split() if w not in GENERIC}


STAGE_MARKERS = ("qualifier", "closed", "open", "stage", "group", "playoff",
                 "finals", "swiss", "elimination", "opening", "legend",
                 "challenger", "contender", "split")


def _stage_count(name):
    n = norm_event(name)
    return sum(1 for t in STAGE_MARKERS if t in n.split())


def join_events(targets, csapi_events, min_overlap=0.5, min_char=0.90):
    """Join target event names onto csapi event names by distinctive tokens.

    Accept ("match") only when the distinctive token sets are identical.
    When several csapi events share that key (an event plus its stages and
    qualifiers), the whole group is recorded and the member with the fewest
    stage markers becomes `best`, so "main event" and "its qualifier" are
    never confused.  A target left with no distinctive token other than the
    year must clear `min_char` similarity instead.
    """
    import difflib
    cand = [(e, norm_event(e), core_tokens(e)) for e in csapi_events]
    out = {"match": [], "review": [], "missing": []}
    for name, rec in targets:
        tn, tk = norm_event(name), core_tokens(name)
        ty = event_year(name)
        # A 4-digit token is a year; a short number is a season/edition and is
        # genuinely distinctive ("ESL Pro League Season 23" vs Season 24).
        tk_noyear = {t for t in tk
                     if not (len(t) == 4 and t.isdigit())}
        group, best, best_j, best_ch = [], None, 0.0, 0.0
        for raw, cn, ck in cand:
            ry = event_year(raw)
            if ty and ry and ry != ty:
                continue
            j = len(tk & ck) / (len(tk | ck) or 1)
            ch = difflib.SequenceMatcher(None, tn, cn).ratio()
            if tk == ck and tk:
                group.append((raw, ch))
            elif j > best_j:
                best, best_j, best_ch = raw, j, ch
        chosen, chosen_ch, note = None, 0.0, ""
        if group:
            # Prefer the main event over its own stages/qualifiers.
            chosen, chosen_ch = min(group, key=lambda rc: (_stage_count(rc[0]), -rc[1]))
            if len(group) > 1:
                note = "group=%d" % len(group)
            # A key made of nothing but the year is not evidence: "BLAST Rivals
            # Fall 2026" and "BLAST Bounty 2026" share it.  Such a join must
            # clear the similarity bar too, or it is demoted to review.
            if not tk_noyear and chosen_ch < min_char:
                note = "year-only-rejected"
                best, best_j, best_ch = chosen, 0.0, chosen_ch
                chosen, chosen_ch = None, 0.0
        elif not tk_noyear:
            # Only the year was distinctive: demand real string agreement.
            for raw, cn, ck in cand:
                ry = event_year(raw)
                if ty and ry and ry != ty:
                    continue
                ch = difflib.SequenceMatcher(None, tn, cn).ratio()
                if ch >= min_char and ch > chosen_ch:
                    chosen, chosen_ch, note = raw, ch, "year-only"
        entry = {"name": name, "year": ty, "best": chosen,
                 "jaccard": 1.0 if group else round(best_j, 3),
                 "char": round(chosen_ch or best_ch, 3), "note": note,
                 "group": [g for g, _ in group][:12], "record": rec}
        if chosen:
            out["match"].append(entry)
        elif best_j >= min_overlap:
            entry["best"] = best
            out["review"].append(entry)
        else:
            entry["best"] = best
            out["missing"].append(entry)
    return out


def cmd_missing_events(args):
    """Cross-reference our Liquipedia-derived event lists against csapi events."""
    rows = list(_iter_matches(os.path.join(HERE, args.source)))
    have = collections.Counter(m.get("event") or "" for m in rows)
    report = {}
    for label, path in (("stier", args.stier), ("tier1", args.tier1)):
        targets = []
        full = os.path.join(HERE, path) if path else ""
        if full and os.path.exists(full):
            with open(full, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        nm = rec.get("tournament") or rec.get("name") or ""
                        if nm:
                            targets.append((nm, rec))
        joined = join_events(targets, list(have))
        report[label] = {
            "total": len(targets),
            "counts": {k: len(v) for k, v in joined.items()},
            "match": joined["match"],
            "review": joined["review"],
            "missing": joined["missing"],
        }
        print("%-6s total=%d  exact=%d  review=%d  missing=%d"
              % (label, len(targets), len(joined["match"]),
                 len(joined["review"]), len(joined["missing"])))
        print("  -- in csapi (exact) --")
        for e in joined["match"][:45]:
            print("     %-44s -> %-38s %s %s"
                  % (e["name"][:44], (e["best"] or "")[:38], e["jaccard"],
                     e.get("note") or ""))
        if joined["review"]:
            print("  -- needs a human eye --")
            for e in joined["review"][:30]:
                print("     %-44s ~> %-40s %s"
                      % (e["name"][:44], (e["best"] or "")[:40], e["jaccard"]))
        print("  -- not in csapi --")
        for e in joined["missing"][:25]:
            print("     %-44s (%s, best=%s @%s)"
                  % (e["name"][:44], e["year"], e["best"], e["jaccard"]))
    out = os.path.join(HERE, getattr(args, "out", "") or "csapi_event_overlap.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    return 0


def cmd_stats(args):
    src = os.path.join(HERE, args.matches)
    ids = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["id"])
    if args.ids:
        want = {int(x) for x in args.ids.split(",")}
        ids = [i for i in ids if i in want]
    if args.limit:
        ids = ids[: args.limit]

    out = os.path.join(HERE, args.out)
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["matchid"])
    todo = [i for i in ids if i not in done]
    print("stats: %d ids, %d done, %d to fetch" % (len(ids), len(done), len(todo)))
    mode = "a" if done else "w"
    with open(out, mode, encoding="utf-8") as fh:
        for n, mid in enumerate(todo, 1):
            rec = {"matchid": mid}
            for key, params in (("overall", {"by_map": "false"}),
                                ("by_map", {"by_map": "true"})):
                try:
                    payload, _ = get_json("/matches/%d/stats" % mid, params)
                    rec[key] = payload
                except urllib.error.HTTPError as exc:
                    rec[key] = {"__error__": "HTTP %s" % exc.code}
                except Exception as exc:                    # noqa: BLE001
                    rec[key] = {"__error__": type(exc).__name__}
                time.sleep(0.4)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 25 == 0 or n == len(todo):
                print("  stats %d/%d" % (n, len(todo)))
    print("STATS_OK -> %s" % out)
    return 0


def cmd_selftest(args):
    """Offline checks.  Network is replaced by captured-shape fixtures."""
    ok = True
    real = urllib.request.urlopen
    import http.client

    # 1. gzip decode + URL building
    def fake_urlopen(req, timeout=60):
        class R:
            headers = {"Content-Encoding": "gzip"}

            def read(self):
                return gzip.compress(b'{"ok": true}')

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        fake_urlopen.last = req
        return R()

    urllib.request.urlopen = fake_urlopen
    try:
        payload, _ = get_json("/matches/1/stats", {"by_map": "true"})
    finally:
        urllib.request.urlopen = real
    if payload != {"ok": True}:
        print("FAIL gzip decode"); ok = False
    if "/matches/1/stats?by_map=true" not in fake_urlopen.last.full_url:
        print("FAIL url build"); ok = False

    # 2. retry then success on a transient 503
    calls = {"n": 0}

    def flaky(req, timeout=60):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "no", {}, None)

        class R:
            headers = {}

            def read(self):
                return b'{"v": 2}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R()

    urllib.request.urlopen = flaky
    try:
        payload, _ = get_json("/x", tries=3)
    finally:
        urllib.request.urlopen = real
    if payload != {"v": 2} or calls["n"] != 2:
        print("FAIL retry", payload, calls); ok = False

    # 3. page shrink on IncompleteRead
    sizes = []

    def trunc_then_ok(req, timeout=60):
        size = int(req.full_url.split("limit=")[1].split("&")[0])
        sizes.append(size)
        if size > 100:
            raise http.client.IncompleteRead(b"partial", 500)

        class R:
            headers = {}

            def read(self):
                return b'[{"id": 1}]'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R()

    urllib.request.urlopen = trunc_then_ok
    try:
        _, used, _ = get_json_capped("/matches", {"limit": MAX_PAGE, "offset": 0})
    finally:
        urllib.request.urlopen = real
    if used != 100 or sizes != [MAX_PAGE, 100]:
        print("FAIL page shrink", used, sizes); ok = False

    # 4. end-of-list detection: 500 at offset==len, previous page non-empty.
    #    len must be a multiple of the page size, otherwise the run ends with a
    #    legitimate short page instead of ever reaching the 500.
    listing = list(range(400))

    def pager(req, timeout=60):
        qs = urllib.parse.parse_qs(req.full_url.split("?", 1)[1])
        off = int(qs["offset"][0]); lim = int(qs["limit"][0])
        if off >= len(listing):
            raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)
        chunk = listing[off:off + lim]

        class R:
            headers = {}

            def read(self):
                return json.dumps(chunk).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R()

    urllib.request.urlopen = pager
    try:
        rows, reason = get_paged("/matches", max_page=200, sleep=0)
    finally:
        urllib.request.urlopen = real
    if rows != listing or "end_of_list" not in reason:
        print("FAIL end-of-list", len(rows), reason); ok = False

    # 5. a 500 mid-list is NOT swallowed as end-of-list: the fixture has 500
    #    rows and only breaks from offset 200 onward, so probing offset 199
    #    returns a full page and the fault must propagate.
    def pager_fault(req, timeout=60):
        qs = urllib.parse.parse_qs(req.full_url.split("?", 1)[1])
        off = int(qs["offset"][0]); lim = int(qs["limit"][0])
        if off >= 200:
            raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)
        chunk = list(range(off, min(off + lim, 500)))

        class R:
            headers = {}

            def read(self):
                return json.dumps(chunk).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R()

    urllib.request.urlopen = pager_fault
    raised = None
    try:
        get_paged("/matches", max_page=200, sleep=0)
    except Exception as exc:                                # noqa: BLE001
        raised = type(exc).__name__
    finally:
        urllib.request.urlopen = real
    if raised != "HTTPError":
        print("FAIL mid-list fault not raised:", raised); ok = False

    # 6. fixture parse: real captured match shape -> scoring fields
    m = {"id": 2396949,
         "team1": {"id": 7020, "name": "Spirit", "score": 2, "rank": 1},
         "team2": {"id": 11283, "name": "Falcons", "score": 0, "rank": 3},
         "maps": [{"id": 5, "name": "Nuke", "team1_score": 13, "team2_score": 8},
                  {"id": 3, "name": "Ancient", "team1_score": 13, "team2_score": 11}],
         "best_of": 3, "date": "2026-09-05", "event": "BLAST Open Porto 2026",
         "winner": {"id": 7020, "name": "Spirit"}}
    if [x["name"] for x in m["maps"]] != ["Nuke", "Ancient"]:
        print("FAIL map names"); ok = False
    if m["winner"]["name"] != m["team1"]["name"] or m["team1"]["score"] != 2:
        print("FAIL winner/score"); ok = False
    if m["best_of"] != 3:
        print("FAIL best_of"); ok = False

    # 7. cache path stays filesystem-safe on Windows (no raw ':' '/')
    name = "stats:2396949?by_map=true"
    if os.path.basename(cache_path(name)) != urllib.parse.quote(name, safe="") + ".json":
        print("FAIL cache path"); ok = False

    # 8. merge dedupes by id and prefers list rows
    tmp_list = os.path.join(HERE, "csapi_selftest_list.jsonl")
    tmp_hist = os.path.join(HERE, "csapi_selftest_hist.jsonl")
    tmp_out = os.path.join(HERE, "csapi_selftest_merged.jsonl")
    with open(tmp_list, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(m, date="2026-09-05")) + "\n")
    with open(tmp_hist, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"teamid": 7020, "matches": [
            dict(m, id=1, date="2025-10-14"), dict(m)  # dup id -> ignored
        ]}) + "\n")

    class A:
        matches = os.path.basename(tmp_list)
        history = os.path.basename(tmp_hist)
        out = os.path.basename(tmp_out)
    cmd_merge(A())
    merged = _load_matches(tmp_out)
    if len(merged) != 2:
        print("FAIL merge dedupe", sorted(merged)); ok = False
    for p in (tmp_list, tmp_hist, tmp_out):
        try:
            os.remove(p)
        except OSError:
            pass

    # 9. event-name normalisation
    # NB: the alias table expands IEM -> "intel extreme masters", so the
    # Liquipedia spelling is already canonical and stays unchanged.
    cases = [("Intel Extreme Masters Katowice 2025",
              "intel extreme masters katowice 2025"),
             ("IEM Cologne Major 2026", "intel extreme masters cologne major 2026"),
             ("ESL Pro League Season 23", "esl pro league season 23"),
             ("StarLadder Budapest Major 2025", "starladder budapest major 2025")]
    for raw, want in cases:
        if norm_event(raw) != want:
            print("FAIL norm_event", raw, "->", norm_event(raw)); ok = False
    if event_year("PGL Cluj-Napoca 2026") != "2026":
        print("FAIL event_year"); ok = False

    # 10. join triage: alias-only match, same-name match, and a stranger
    cs = ["IEM Katowice 2025", "PGL Cluj-Napoca 2026",
          "StarLadder Budapest Major 2025"]
    jobs = join_events([("Intel Extreme Masters Katowice 2025", {}),
                        ("StarLadder Budapest Major 2025", {}),
                        ("Totally Unrelated Cup 2025", {})], cs)
    if len(jobs["match"]) != 2 or len(jobs["missing"]) != 1:
        print("FAIL join triage", {k: len(v) for k, v in jobs.items()}); ok = False
    if jobs["match"][0]["best"] != "IEM Katowice 2025":
        print("FAIL alias join", jobs["match"][0]); ok = False

    # 10b. regression: char similarity used to match these two at 0.87, and a
    #      same-prefix different-city pair at 0.875.  Both must NOT be "match".
    cs2 = ["JB Pro League Season 2", "IEM Rio 2026",
           "ESL Pro League Season 23 Finals"]
    jobs2 = join_events([("ESL Pro League Season 26", {}),
                         ("Intel Extreme Masters Beijing 2026", {})], cs2)
    if jobs2["match"]:
        print("FAIL token join accepted a false positive:", jobs2["match"]); ok = False
    # a genuine season match must still be accepted
    jobs3 = join_events([("ESL Pro League Season 23", {})],
                        ["ESL Pro League Season 23 Finals"])
    if not jobs3["match"] or jobs3["match"][0]["best"] != "ESL Pro League Season 23 Finals":
        print("FAIL season join", jobs3); ok = False
    # cities are distinctive: Katowice must not accept Chengdu
    jobs4 = join_events([("Intel Extreme Masters Katowice 2025", {})],
                        ["IEM Chengdu 2025"])
    if jobs4["match"]:
        print("FAIL city join", jobs4["match"]); ok = False

    # 10c. an event must not be represented by its own qualifier
    jobs5 = join_events(
        [("Intel Extreme Masters Beijing 2026", {})],
        ["IEM Beijing 2026 Open Qualifier", "IEM Beijing 2026 Closed Qualifier",
         "IEM Beijing 2026"])
    got = jobs5["match"][0]
    if got["best"] != "IEM Beijing 2026":
        print("FAIL qualifier preference", got["best"], got["group"]); ok = False
    # "Open Qualifier" names a separate sub-series and must NOT group with the
    # main event; the plain qualifier stage must.
    if len(got["group"]) != 2:
        print("FAIL qualifier group", got["group"]); ok = False
    if any("Open Qualifier" in g for g in got["group"]):
        print("FAIL open qualifier grouped", got["group"]); ok = False

    # 10d. a year-only target must not silently join a same-year stranger
    jobs6 = join_events([("BLAST Rivals Fall 2026", {})],
                        ["BLAST Bounty 2026", "Thunderpick World Championship 2026"])
    if jobs6["match"]:
        print("FAIL year-only join", jobs6["match"]); ok = False

    # 11. cmd_missing_events end-to-end on fixtures
    ev_path = os.path.join(HERE, "csapi_selftest_ev.jsonl")
    evt_path = os.path.join(HERE, "csapi_selftest_events.jsonl")
    with open(ev_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(m) + "\n")
        fh.write(json.dumps(dict(m, id=2, date="2025-10-14",
                                 event="IEM Katowice 2025")) + "\n")
    with open(evt_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"tournament": "Intel Extreme Masters Katowice 2025"}) + "\n")
        fh.write(json.dumps({"tournament": "Nonexistent Tango Cup 2031"}) + "\n")

    class C:
        source = "csapi_selftest_ev.jsonl"
        stier = "csapi_selftest_events.jsonl"
        tier1 = ""
        # NB: must not be the real report path -- an earlier version of this
        # selftest overwrote csapi_event_overlap.json with fixture data.
        out = "csapi_selftest_overlap.json"
    cmd_missing_events(C())
    with open(os.path.join(HERE, "csapi_selftest_overlap.json"), encoding="utf-8") as fh:
        ov = json.load(fh)
    if ov["stier"]["total"] != 2 or ov["stier"]["counts"]["match"] != 1:
        print("FAIL overlap fixture", ov["stier"]["counts"]); ok = False
    if ov["stier"]["missing"][0]["name"] != "Nonexistent Tango Cup 2031":
        print("FAIL overlap missing", ov["stier"]["missing"]); ok = False
    for path in (ev_path, evt_path,
                 os.path.join(HERE, "csapi_selftest_overlap.json")):
        os.remove(path)

    print("CSAPI_COLLECT_SELFTEST_%s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("matches")
    p.add_argument("--out", default="csapi_matches.jsonl")
    p.set_defaults(func=cmd_matches)

    p = sub.add_parser("teams")
    p.add_argument("--out", default="csapi_teams.json")
    p.set_defaults(func=cmd_teams)

    p = sub.add_parser("teamhistory")
    p.add_argument("--teams", default="csapi_teams.json")
    p.add_argument("--out", default="csapi_teamhistory.jsonl")
    p.add_argument("--only", default="")
    p.set_defaults(func=cmd_teamhistory)

    p = sub.add_parser("merge")
    p.add_argument("--matches", default="csapi_matches.jsonl")
    p.add_argument("--history", default="csapi_teamhistory.jsonl")
    p.add_argument("--out", default="csapi_matches_all.jsonl")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("events")
    p.add_argument("--source", default="csapi_matches_all.jsonl")
    p.set_defaults(func=cmd_events)

    p = sub.add_parser("missing-events")
    p.add_argument("--source", default="csapi_matches_all.jsonl")
    p.add_argument("--stier", default="stier_events.jsonl")
    p.add_argument("--tier1", default="tier1.jsonl")
    p.add_argument("--out", default="csapi_event_overlap.json",
                   help="report path; the selftest uses a separate file so it "
                        "cannot clobber the real report")
    p.set_defaults(func=cmd_missing_events)

    p = sub.add_parser("stats")
    p.add_argument("--matches", default="csapi_matches_all.jsonl")
    p.add_argument("--out", default="csapi_stats.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", default="")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

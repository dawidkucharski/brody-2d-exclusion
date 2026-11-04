#!/usr/bin/env python3
"""Validate every DOI in aurora_refs.bib against Crossref (DataCite fallback).

Writes reference-validation/references_raw.json (the parsed bibliography) and
reference-validation/crossref_results.json (per-DOI verification results).
Entries without DOIs, keys cited in the manuscript but missing from the bib,
and bib entries never cited are reported.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIB = (ROOT / 'aurora_refs.bib').read_text()
TEX = (ROOT / 'aurora_manuscript.tex').read_text()
OUT = ROOT / 'reference-validation'
OUT.mkdir(exist_ok=True)


def fetch(url, timeout=40):
    request = urllib.request.Request(url, headers={'User-Agent': 'ReferenceValidation/1.0'})
    return json.load(urllib.request.urlopen(request, timeout=timeout))


entries = {}
for match in re.finditer(r'@(\w+)\{([^,]+),\s*(.*?)\n\}', BIB, re.S):
    kind, key, body = match.group(1), match.group(2), match.group(3)
    fields = {}
    for field_match in re.finditer(r'(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}', body):
        fields[field_match.group(1)] = field_match.group(2)
    entries[key] = {'type': kind, 'fields': fields}

(OUT / 'references_raw.json').write_text(json.dumps(entries, indent=2, ensure_ascii=False))

cited = set(re.findall(r'\\cite[tp]?\{([^}]+)\}', TEX))
cited_keys = set()
for group in cited:
    cited_keys.update(key.strip() for key in group.split(','))

missing = sorted(cited_keys - set(entries))
unused = sorted(set(entries) - cited_keys)
print(f'entries: {len(entries)}, cited keys: {len(cited_keys)}')
if missing:
    print('MISSING FROM BIB:', missing)
if unused:
    print('NEVER CITED:', unused)

results = {}
for key, entry in entries.items():
    doi = entry['fields'].get('doi', '')
    title = re.sub(r'\{|\}', '', entry['fields'].get('title', ''))
    year = entry['fields'].get('year', '')
    if not doi:
        results[key] = {'status': 'NO_DOI', 'howpublished': entry['fields'].get('howpublished', ''),
                        'title': title, 'year': year}
        continue
    record = {'key': key, 'doi': doi, 'title': title, 'year': year, 'status': 'OK'}
    try:
        message = fetch('https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='/'))['message']
        record['registry'] = 'crossref'
        record['registered_title'] = message.get('title', [''])[0]
        record['registered_year'] = (message.get('issued', {}).get('date-parts') or [[None]])[0][0]
        record['registered_journal'] = (message.get('container-title') or [''])[0]
    except Exception as crossref_error:
        try:
            data = fetch('https://api.datacite.org/dois/' + urllib.parse.quote(doi.lower(), safe='/'))
            attrs = data['data']['attributes']
            record['registry'] = 'datacite'
            record['registered_title'] = attrs.get('titles', [{}])[0].get('title', '')
            record['registered_year'] = attrs.get('publicationYear')
            record['registered_journal'] = attrs.get('publisher', '')
        except Exception as datacite_error:
            record['status'] = 'UNRESOLVED'
            record['crossref_error'] = str(crossref_error)
            record['datacite_error'] = str(datacite_error)
    results[key] = record
    time.sleep(0.4)

(OUT / 'crossref_results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
unresolved = [key for key, record in results.items() if record.get('status') == 'UNRESOLVED']
no_doi = [key for key, record in results.items() if record.get('status') == 'NO_DOI']
print(f'unresolved DOIs: {unresolved}')
print(f'no DOI: {no_doi}')
print(f'wrote {OUT}/references_raw.json and {OUT}/crossref_results.json')

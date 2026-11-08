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


def strip_braces(text):
    return re.sub(r'\{|\}', '', text)


def bib_authors(field):
    text = strip_braces(field)
    text = re.sub(r'\s+and\s+', ' and ', text)
    return [name.strip() for name in text.split(' and ') if name.strip()]


def canonical_surname(name):
    """Reduce a (possibly brace/initial-prefixed, accented) name to its canonical form."""
    import unicodedata
    value = name
    for latex, plain in [('{\\"a}', 'ae'), ("{\\'a}", 'a'), ("{\\'e}", 'e'),
                         ("{\\'i}", 'i'), ("{\\'o}", 'o'), ("{\\'u}", 'u'),
                         ('{\\"o}', 'oe'), ('{\\"u}', 'ue'), ("{\\'n}", 'n'),
                         ('{\\v{c}}', 'c'), ("{\\'s}", 's')]:
        value = value.replace(latex, plain)
    value = strip_braces(value)
    value = re.sub(r'^([A-Za-z]\.-?\s*)+', '', value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-z0-9]', '', value.lower())


def check_authors(bib_author_field, registered_authors):
    """Return (matched_count, total_bib, total_registered)."""
    bib_names = [canonical_surname(name) for name in bib_authors(bib_author_field)
                 if name.strip().lower() not in ('others', 'et al.')]
    registered = []
    registered_combined = []
    for author in registered_authors or []:
        family = author.get('family') or ''
        given = author.get('given') or author.get('name') or ''
        registered.append(canonical_surname(family if family else given))
        # Fallback for legacy registry records with mis-tokenised names
        # (surname stored inside the given field).
        registered_combined.append(canonical_surname(f'{given} {family}'))
    matched = 0
    for surname_value in bib_names:
        if not surname_value:
            continue
        primary = any(surname_value and
                      (surname_value == candidate or
                       surname_value.startswith(candidate) or
                       candidate.startswith(surname_value))
                      for candidate in registered)
        fallback = any(surname_value and surname_value in combined
                       for combined in registered_combined)
        if primary or fallback:
            matched += 1
    return matched, len(bib_names), len(registered)


def normalise_title(text):
    return re.sub(r'[^a-z0-9 ]', '', strip_braces(text).lower())


def title_tokens(text):
    return set(normalise_title(text).split())

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
    title = strip_braces(entry['fields'].get('title', ''))
    year = entry['fields'].get('year', '')
    volume = entry['fields'].get('volume', '')
    pages = entry['fields'].get('pages', '')
    if not doi:
        results[key] = {'status': 'NO_DOI', 'howpublished': entry['fields'].get('howpublished', ''),
                        'title': title, 'year': year}
        continue
    record = {'key': key, 'doi': doi, 'title': title, 'year': year,
              'volume': volume, 'pages': pages, 'status': 'OK'}
    try:
        message = fetch('https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='/'))['message']
        record['registry'] = 'crossref'
        record['registered_title'] = message.get('title', [''])[0]
        record['registered_year'] = (message.get('issued', {}).get('date-parts') or [[None]])[0][0]
        record['registered_journal'] = (message.get('container-title') or [''])[0]
        record['registered_volume'] = message.get('volume')
        record['registered_page'] = message.get('page') or message.get('article-number')
        record['registered_authors'] = [{'family': a.get('family', ''),
                                         'given': a.get('given', '')}
                                        for a in message.get('author', [])]
    except Exception as crossref_error:
        try:
            data = fetch('https://api.datacite.org/dois/' + urllib.parse.quote(doi.lower(), safe='/'))
            attrs = data['data']['attributes']
            record['registry'] = 'datacite'
            record['registered_title'] = attrs.get('titles', [{}])[0].get('title', '')
            record['registered_year'] = attrs.get('publicationYear')
            record['registered_journal'] = attrs.get('publisher', '')
            record['registered_volume'] = None
            record['registered_page'] = None
            record['registered_authors'] = [{'family': '', 'given': c.get('name', '')}
                                            for c in attrs.get('creators', [])]
        except Exception as datacite_error:
            record['status'] = 'UNRESOLVED'
            record['crossref_error'] = str(crossref_error)
            record['datacite_error'] = str(datacite_error)
    if record['status'] == 'OK':
        bib_tokens = title_tokens(title)
        registered_tokens = title_tokens(record.get('registered_title') or '')
        record['title_match'] = float(len(bib_tokens & registered_tokens) /
                                      max(1, min(len(bib_tokens), len(registered_tokens))))
        author_field = entry['fields'].get('author', '')
        if author_field:
            matched, bib_count, registered_count = check_authors(author_field,
                                                                 record.get('registered_authors'))
            record['author_match'] = {'matched': matched, 'in_bib': bib_count,
                                      'registered': registered_count}
        else:
            record['author_match'] = None
        record['year_match'] = bool(record.get('registered_year') is not None and
                                    year and abs(int(record['registered_year']) - int(year)) <= 1)
        if volume:
            record['volume_match'] = (str(record.get('registered_volume') or '') == volume or
                                      str(record.get('registered_volume') or '') == volume.split('--')[0])
        else:
            record['volume_match'] = None
        if pages:
            registered_page = str(record.get('registered_page') or '')
            first_page = re.split(r'--|[-–]', pages)[0]
            record['pages_match'] = bool(registered_page and
                                         (registered_page.startswith(first_page) or
                                          first_page.startswith(registered_page)))
        else:
            record['pages_match'] = None
    results[key] = record
    time.sleep(0.4)

(OUT / 'crossref_results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
unresolved = [key for key, record in results.items() if record.get('status') == 'UNRESOLVED']
no_doi = [key for key, record in results.items() if record.get('status') == 'NO_DOI']
print(f'unresolved DOIs: {unresolved}')
print(f'no DOI: {no_doi}')
print('field mismatches:')
for key, record in sorted(results.items()):
    if record.get('status') != 'OK':
        continue
    problems = []
    if record.get('title_match', 1.0) < 0.6:
        problems.append(f"title={record.get('title_match'):.2f}")
    author_match = record.get('author_match')
    if author_match and author_match['in_bib'] and author_match['matched'] < author_match['in_bib']:
        problems.append(f"authors={author_match['matched']}/{author_match['in_bib']}")
    if record.get('year_match') is False:
        problems.append(f"year bib={record.get('year')} reg={record.get('registered_year')}")
    if record.get('volume_match') is False:
        problems.append(f"volume bib={record.get('volume')} reg={record.get('registered_volume')}")
    if record.get('pages_match') is False:
        problems.append(f"pages bib={record.get('pages')} reg={record.get('registered_page')}")
    if problems:
        print(' ', key, ':', '; '.join(problems))
print(f'wrote {OUT}/references_raw.json and {OUT}/crossref_results.json')

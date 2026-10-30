#!/usr/bin/env python3
"""Verify DOIs against CrossRef and print metadata."""
import json, re, time, urllib.request, urllib.error

BIB = '/Users/dawid/Projects/interfero-Riemann/elsarticle/aurora_refs.bib'

# DOIs from the LeapSpace overview (candidates) + extras to verify
overview_dois = [
    '10.1126/science.156.3775.636',          # Mandelbrot 1967
    '10.1002/0471725218',                    # Ripley 2005 book
    '10.18637/jss.v012.i06',                 # Baddeley-Turner 2005 spatstat
    '10.1002/9780470725160',                 # Illian 2008 (already in bib)
    '10.1201/b15326',                        # Diggle 2013
    '10.1111/j.1467-8306.1987.tb00158.x',    # Goodchild-Mark 1987
    '10.1103/PhysRevE.72.045204',            # Sakhr-Nieminen (already in bib)
    '10.1115/1.4036814',                     # Chelidze 2017
    '10.1103/PhysRevE.51.2950',              # Fredkin-Rice 1995
    '10.2307/3546792',                       # Roxburgh-Matsuki 1999
    '10.1103/PhysRevE.89.022922',            # Wolf 2014
    '10.1155/2021/9956518',                  # Abdel-Mageed 2021
    '10.1086/285231',                        # Caley 1991
    '10.1016/S0309-1708(01)00053-7',         # Islam-Sivakumar 2002
    '10.3390/e22121435',                     # Inglada-Perez 2020
    '10.2307/1930001',                       # Clark-Evans 1954 (extra)
]

def crossref(doi):
    url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={
        'User-Agent': 'ManuscriptRefCheck/1.0 (mailto:dawid.kucharski@put.poznan.pl)'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        m = data['message']
        title = (m.get('title') or [''])[0]
        authors = ', '.join(
            (a.get('family', '') + ' ' + a.get('given', ''))[:30] for a in m.get('author', []))[:90]
        year = ''
        for k in ('published-print', 'published-online', 'issued', 'created'):
            if k in m and m[k].get('date-parts'):
                year = m[k]['date-parts'][0][0]
                break
        ct = m.get('container-title') or ['(book)']
        return f"OK  {doi:38s} | {year} | {title[:70]:70s} | {authors:30s} | {ct[0][:40]}"
    except urllib.error.HTTPError as e:
        return f"FAIL {doi:38s} | HTTP {e.code}"
    except Exception as e:
        return f"FAIL {doi:38s} | {type(e).__name__}: {e}"

print('=== OVERVIEW CANDIDATES ===')
for d in overview_dois:
    print(crossref(d))
    time.sleep(0.3)

print('\n=== DOIs ALREADY IN aurora_refs.bib ===')
text = open(BIB).read()
for m in re.finditer(r'doi\s*=\s*\{([^}]+)\}', text):
    print(crossref(m.group(1).strip()))
    time.sleep(0.3)

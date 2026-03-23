#!/usr/bin/env python3
"""
plant_fetch.py — Fetch geolocated occurrences from GBIF and rich descriptions from Wikipedia
for a list of plant/crop species (scientific names), with a focus on Europe/Hungary by default.

Outputs:
  1) CSV  : occurrences_{timestamp}.csv  (lat/lon + metadata per record)
  2) JSONL: species_{timestamp}.jsonl   (one entry per species with summaries + DETAILED sections)
  3) (opt) GeoJSON: occurrences_{timestamp}.geojson  (if --geojson is passed)

Key feature requested:
- If Wikipedia lookups fail ("wiki false"), automatically perform a MediaWiki **search fallback**
  in **Hungarian** (hu) and **English** (en), and store descriptions from both languages if found.

Usage examples:
  python plant_fetch.py --names "Triticum aestivum"
  python plant_fetch.py --names "Triticum aestivum,Helianthus annuus" --country HU --year-from 2015 --limit 2000 --managed --lang hu
  python plant_fetch.py --names-file species.txt --geometry "POLYGON((16 45.5, 23 45.5, 23 48.6, 16 48.6, 16 45.5))" --limit 1500 --lang en --geojson

Notes:
- This script uses GBIF REST API (species/match + occurrence/search) and Wikipedia REST + MediaWiki API.
- For large data pulls, consider GBIF Occurrence Download API (requires GBIF account).
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

GBIF_API = "https://api.gbif.org/v1"
WIKI_REST = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_MEDIAWIKI = "https://{lang}.wikipedia.org/w/api.php"

# User-Agent header required by Wikipedia API
USER_AGENT = "FloraScrapeTool/1.0 (https://github.com/yourusername/flora-scrape; your.email@example.com) Python/requests"

# --- Data classes ----------------------------------------------------------------

@dataclass
class SpeciesInfo:
    query_name: str
    taxon_key: Optional[int]
    canonical_name: Optional[str]
    rank: Optional[str]
    vernacular_names: List[Dict]
    # Base (preferred) short/long description + url (based on --lang preference + fallbacks)
    description: Optional[str]
    description_lang: Optional[str]
    wikipedia_url: Optional[str]
    long_description: Optional[str]
    long_description_lang: Optional[str]
    long_wikipedia_url: Optional[str]
    sections: Optional[List[Dict]]  # [{ "title": str|None, "text": str }]
    # NEW: language-specific extras collected via search fallback, e.g. {"hu": {...}, "en": {...}}
    extra_wiki: Optional[Dict[str, Dict]]

# --- Helpers ---------------------------------------------------------------------

def safe_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, retries: int = 3, backoff: float = 1.0):
    """GET with simple retry/backoff."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
    raise RuntimeError("safe_get fell through")

# ---------------- GBIF ------------------------------------------------------------

def gbif_match_name(name: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Return (taxonKey, canonicalName, rank) using GBIF species/match."""
    url = f"{GBIF_API}/species/match"
    r = safe_get(url, params={"name": name})
    data = r.json()
    taxon_key = data.get("usageKey") or data.get("speciesKey") or data.get("acceptedUsageKey")
    canonical = data.get("canonicalName") or data.get("scientificName")
    rank = data.get("rank")
    return taxon_key, canonical, rank

def gbif_vernacular_names(taxon_key: int) -> List[Dict]:
    """Get vernacular/common names for a taxon."""
    url = f"{GBIF_API}/species/{taxon_key}/vernacularNames"
    try:
        r = safe_get(url)
        data = r.json()
        return data.get("results", [])
    except Exception:
        return []

def gbif_occurrences(
    taxon_key: int,
    country: Optional[str] = "HU",
    geometry_wkt: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    managed_only: bool = False,
    limit_total: int = 1000,
    page_size: int = 300,
    extra_params: Optional[dict] = None,
) -> List[Dict]:
    """Stream paginated occurrence/search results for a taxonKey with filters."""
    params = {
        "taxonKey": taxon_key,
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "limit": page_size,
        "offset": 0,
    }
    if country and not geometry_wkt:
        params["country"] = country
    if geometry_wkt:
        params["geometry"] = geometry_wkt
    if year_from and year_to:
        params["year"] = f"{year_from},{year_to}"
    elif year_from and not year_to:
        params["year"] = f"{year_from},3000"
    elif year_to and not year_from:
        params["year"] = f"0,{year_to}"
    if managed_only:
        params["establishmentMeans"] = "MANAGED"
    if extra_params:
        params.update(extra_params)

    url = f"{GBIF_API}/occurrence/search"
    results: List[Dict] = []

    while len(results) < limit_total:
        params["offset"] = len(results)
        r = safe_get(url, params=params)
        data = r.json()
        batch = data.get("results", [])
        if not batch:
            break
        results.extend(batch)
        time.sleep(0.1)  # small courtesy pause
        if len(batch) < page_size:
            break

    return results[:limit_total]

# ---------------- Wikipedia (summary + full text + search fallback) ---------------

def wiki_summary_prefer_titles(titles: List[str], lang: str = "en") -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Try Wikipedia REST summaries in order of provided titles."""
    headers = {
        "accept": "application/json",
        "User-Agent": USER_AGENT
    }
    for t in titles:
        title_encoded = urllib.parse.quote(t.replace(" ", "_"))
        url = WIKI_REST.format(lang=lang, title=title_encoded)
        try:
            r = safe_get(url, headers=headers, retries=2, backoff=0.8)
            if r.status_code == 200:
                j = r.json()
                extract = j.get("extract")
                page_url = (j.get("content_urls", {}) or {}).get("desktop", {}).get("page")
                if extract:
                    return extract.strip(), lang, page_url
        except Exception:
            continue
    return None, None, None

def wiki_full_extract(title: str, lang: str = "en") -> Tuple[Optional[str], Optional[str], Optional[str], Optional[List[Dict]]]:
    """Fetch FULL plaintext of a Wikipedia page via MediaWiki API and split to sections."""
    api = WIKI_MEDIAWIKI.format(lang=lang)
    params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "format": "json", "titles": title}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = safe_get(api, params=params, headers=headers)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None, None, None, None
        page = next(iter(pages.values()))
        extract = page.get("extract")
        norm_title = page.get("title")
        page_url = f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote((norm_title or title).replace(" ", "_"))

        sections: List[Dict] = []
        if extract:
            pattern = re.compile(r"^==+\s*(.*?)\s*==+\s*$", re.MULTILINE)
            indices = [(m.start(), m.end(), m.group(1)) for m in pattern.finditer(extract)]
            if not indices:
                sections = [{"title": None, "text": extract.strip()}]
            else:
                chunks = []
                for (s, e, _), (next_s, *_rest) in zip(indices, indices[1:] + [(len(extract), None, None)]):
                    body = extract[e:next_s]
                    header = pattern.match(extract[s:e].strip())
                    header_title = header.group(1) if header else None
                    chunks.append({"title": header_title, "text": body.strip()})
                sections = chunks
        return (extract.strip() if extract else None), lang, page_url, sections
    except Exception:
        return None, None, None, None

def wiki_search_titles(query: str, lang: str, limit: int = 5) -> List[str]:
    """Search MediaWiki for likely page titles."""
    api = WIKI_MEDIAWIKI.format(lang=lang)
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = safe_get(api, params=params, headers=headers)
        data = r.json()
        hits = data.get("query", {}).get("search", [])
        return [h.get("title") for h in hits if h.get("title")]
    except Exception:
        return []

def choose_vernacular_titles(vernaculars: List[Dict], target_lang: str) -> List[str]:
    """Build candidate titles from vernacular names, prioritizing target_lang then English."""
    titles = []
    for v in vernaculars:
        if (v.get("language") or "").lower() == target_lang.lower():
            name = v.get("vernacularName")
            if name and name not in titles:
                titles.append(name)
    for v in vernaculars:
        if (v.get("language") or "").lower() == "en":
            name = v.get("vernacularName")
            if name and name not in titles:
                titles.append(name)
    return titles

# ---------------- Main gather ----------------------------------------------------

def gather_for_species(
    name: str,
    country: Optional[str],
    geometry: Optional[str],
    year_from: Optional[int],
    year_to: Optional[int],
    managed_only: bool,
    limit_occ: int,
    wiki_lang: str,
) -> Tuple[SpeciesInfo, List[Dict]]:
    """Resolve species, fetch vernaculars, short+long Wikipedia descriptions, occurrences, and search fallbacks."""
    taxon_key, canonical, rank = gbif_match_name(name)
    vernaculars: List[Dict] = []
    description = None
    description_lang = None
    wiki_url = None

    long_description = None
    long_description_lang = None
    long_wiki_url = None
    sections = None

    extra_wiki: Dict[str, Dict] = {}

    if taxon_key:
        vernaculars = gbif_vernacular_names(taxon_key)

    # Candidate titles: scientific name, then vernaculars (target lang, then EN)
    titles = [canonical or name]
    titles.extend(choose_vernacular_titles(vernaculars, wiki_lang))

    # Preferred language try, then EN fallback
    summary, sl, surl = wiki_summary_prefer_titles(titles, lang=wiki_lang or "en")
    if not summary and (wiki_lang or "en").lower() != "en":
        summary, sl, surl = wiki_summary_prefer_titles(titles, lang="en")

    # Full extract
    full_text = None
    full_url = None
    full_lang = None
    for t in titles:
        ft, flang, furl, secs = wiki_full_extract(t, lang=wiki_lang or "en")
        if ft:
            full_text, full_lang, full_url, sections = ft, flang, furl, secs
            break
    if not full_text and (wiki_lang or "en").lower() != "en":
        for t in titles:
            ft, flang, furl, secs = wiki_full_extract(t, lang="en")
            if ft:
                full_text, full_lang, full_url, sections = ft, flang, furl, secs
                break

    description, description_lang, wiki_url = summary, sl, surl
    long_description, long_description_lang, long_wiki_url = full_text, full_lang, full_url

    # ---- Search fallback if both summary and long are missing ("wiki false") ----
    if not description and not long_description:
        # Try HU and EN searches
        for lang_try in ["hu", "en"]:
            titles_from_search = wiki_search_titles(canonical or name, lang_try, limit=5)
            if not titles_from_search and vernaculars:
                # also try with the first vernacular in that language
                for v in vernaculars:
                    if (v.get("language") or "").lower() == lang_try:
                        vname = v.get("vernacularName")
                        if vname:
                            titles_from_search.append(vname)
                        break

            found_summary, _, found_url = wiki_summary_prefer_titles(titles_from_search, lang=lang_try)
            # Try full
            found_full = None; found_full_url = None; found_sections = None
            for t in titles_from_search:
                ft, _, furl, secs = wiki_full_extract(t, lang=lang_try)
                if ft:
                    found_full, found_full_url, found_sections = ft, furl, secs
                    break

            if found_summary or found_full:
                extra_wiki[lang_try] = {
                    "summary": found_summary,
                    "summary_url": found_url,
                    "long": found_full,
                    "long_url": found_full_url,
                    "sections": found_sections,
                }
                # If still missing the base description, adopt from the first language that yields content
                if not description and found_summary:
                    description, description_lang, wiki_url = found_summary, lang_try, found_url
                if not long_description and found_full:
                    long_description, long_description_lang, long_wiki_url, sections = found_full, lang_try, found_full_url, found_sections

    # Occurrences
    occs: List[Dict] = []
    if taxon_key:
        occs = gbif_occurrences(
            taxon_key=taxon_key,
            country=country,
            geometry_wkt=geometry,
            year_from=year_from,
            year_to=year_to,
            managed_only=managed_only,
            limit_total=limit_occ,
        )

    sp = SpeciesInfo(
        query_name=name,
        taxon_key=taxon_key,
        canonical_name=canonical,
        rank=rank,
        vernacular_names=vernaculars,
        description=description,
        description_lang=description_lang,
        wikipedia_url=wiki_url,
        long_description=long_description,
        long_description_lang=long_description_lang,
        long_wikipedia_url=long_wiki_url,
        sections=sections,
        extra_wiki=extra_wiki or None,
    )
    return sp, occs

# ---------------- Writers --------------------------------------------------------

def write_occurrences_csv(path: str, rows: List[Dict]):
    """Write occurrences to CSV with selected columns."""
    wanted = [
        "scientificName", "taxonKey", "decimalLatitude", "decimalLongitude",
        "eventDate", "year", "month", "day",
        "locality", "municipality", "county", "stateProvince", "countryCode",
        "establishmentMeans", "basisOfRecord", "occurrenceID", "datasetKey", "references"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(wanted)
        for r in rows:
            w.writerow([r.get(k, "") for k in wanted])

def write_occurrences_geojson(path: str, rows: List[Dict]):
    """Write occurrences as a Point GeoJSON FeatureCollection."""
    features = []
    for r in rows:
        lat = r.get("decimalLatitude")
        lon = r.get("decimalLongitude")
        if lat is None or lon is None:
            continue
        props = {
            "scientificName": r.get("scientificName"),
            "taxonKey": r.get("taxonKey"),
            "eventDate": r.get("eventDate"),
            "locality": r.get("locality"),
            "countryCode": r.get("countryCode"),
            "establishmentMeans": r.get("establishmentMeans"),
            "basisOfRecord": r.get("basisOfRecord"),
            "occurrenceID": r.get("occurrenceID"),
            "datasetKey": r.get("datasetKey"),
            "references": r.get("references"),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props
        })
    collection = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False)

def write_species_jsonl(path: str, species_infos: List[SpeciesInfo]):
    with open(path, "w", encoding="utf-8") as f:
        for sp in species_infos:
            f.write(json.dumps(asdict(sp), ensure_ascii=False) + "\n")

# ---------------- CLI ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fetch GBIF coordinates + Wikipedia summaries + DETAILED sections; HU/EN search fallback if wiki false.")
    g_in = p.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--names", type=str, help="Comma-separated scientific names, e.g., 'Triticum aestivum,Helianthus annuus'")
    g_in.add_argument("--names-file", type=str, help="Path to a file with one species name per line")

    p.add_argument("--country", type=str, default="HU", help="ISO2 country code filter (default: HU). Ignored if --geometry is set.")
    p.add_argument("--geometry", type=str, default=None, help="WKT POLYGON or MULTIPOLYGON to filter occurrences (overrides --country).")
    p.add_argument("--year-from", type=int, default=None, help="Lower bound for year filter.")
    p.add_argument("--year-to", type=int, default=None, help="Upper bound for year filter.")
    p.add_argument("--managed", action="store_true", help="If set, filter to establishmentMeans=MANAGED (cultivated/managed).")

    p.add_argument("--limit", type=int, default=1000, help="Max occurrences per species to fetch (default: 1000).")
    p.add_argument("--lang", type=str, default="en", help="Preferred Wikipedia language code (e.g., 'hu' or 'en').")

    p.add_argument("--geojson", action="store_true", help="Also export GeoJSON of occurrences.")
    p.add_argument("--out-prefix", type=str, default=None, help="Prefix for output files. Defaults to 'occurrences_{ts}' etc.")

    return p.parse_args()

def main():
    args = parse_args()

    if args.names:
        species_names = [s.strip() for s in args.names.split(",") if s.strip()]
    else:
        with open(args.names_file, "r", encoding="utf-8") as f:
            species_names = [line.strip() for line in f if line.strip()]

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or f"occurrences_{ts}"
    occ_csv = f"{prefix}.csv"
    occ_geojson = f"{prefix}.geojson"
    sp_jsonl = f"species_{ts}.jsonl"

    all_occ_rows: List[Dict] = []
    species_infos: List[SpeciesInfo] = []

    for name in species_names:
        print(f"[INFO] Processing: {name}")
        try:
            sp_info, occs = gather_for_species(
                name=name,
                country=args.country,
                geometry=args.geometry,
                year_from=args.year_from,
                year_to=args.year_to,
                managed_only=args.managed,
                limit_occ=args.limit,
                wiki_lang=args.lang,
            )
            species_infos.append(sp_info)

            for r in occs:
                r["scientificName"] = r.get("scientificName") or sp_info.canonical_name or name
                r["taxonKey"] = r.get("taxonKey") or sp_info.taxon_key
                all_occ_rows.append(r)

            extra_langs = ",".join((sp_info.extra_wiki or {}).keys()) if sp_info.extra_wiki else "-"
            print(
                f"  - taxonKey: {sp_info.taxon_key}, occ: {len(occs)}, "
                f"summary: {bool(sp_info.description)}, detailed: {bool(sp_info.long_description)}, "
                f"extras: {extra_langs}"
            )
            time.sleep(0.2)
        except Exception as e:
            print(f"[WARN] Failed for '{name}': {e}", file=sys.stderr)

    if all_occ_rows:
        write_occurrences_csv(occ_csv, all_occ_rows)
        print(f"[OK] Wrote CSV: {occ_csv}  ({len(all_occ_rows)} rows)")
        if args.geojson:
            write_occurrences_geojson(occ_geojson, all_occ_rows)
            print(f"[OK] Wrote GeoJSON: {occ_geojson}")

    if species_infos:
        write_species_jsonl(sp_jsonl, species_infos)
        print(f"[OK] Wrote JSONL: {sp_jsonl}  ({len(species_infos)} species)")

if __name__ == "__main__":
    main()

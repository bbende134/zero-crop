import requests
import time
import json
import jwt  # pip install PyJWT

# --- 1. Authentication ---
KEY_FILE_PATH = "key.json"

print("Authenticating with Copernicus...")
with open(KEY_FILE_PATH, 'rb') as f:
    service_key = json.load(f)

claim_set = {
    "iss": service_key['client_id'],
    "sub": service_key['user_id'],
    "aud": service_key['token_uri'],
    "iat": int(time.time()),
    "exp": int(time.time()) + 3600
}

private_key = service_key['private_key'].encode('utf-8')
grant = jwt.encode(claim_set, private_key, algorithm="RS256")

token_response = requests.post(
    service_key['token_uri'],
    data={
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": grant
    },
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)
token_response.raise_for_status()
access_token = token_response.json().get('access_token')
print("Authentication successful.")

HEADERS = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

BASE_URL = "https://land.copernicus.eu/api"
SEARCH_ENDPOINT = f"{BASE_URL}/@search"
DOWNLOAD_REQUEST_ENDPOINT = f"{BASE_URL}/@datarequest_post"
DOWNLOAD_STATUS_ENDPOINT = f"{BASE_URL}/@datarequest_search"

def search_dataset(keyword="land cover 100m collection 3 epoch 2018"):
    """Search for the dataset via the @search endpoint and return its metadata."""
    params = {
        "portal_type": "DataSet",
        "SearchableText": keyword,
        "metadata_fields": [
            "UID",
            "dataset_full_format",
            "dataset_download_information",
        ],
    }
    r = requests.get(SEARCH_ENDPOINT, params=params, headers=HEADERS)
    r.raise_for_status()
    results = r.json()

    items = results.get('items', [])
    print(f"  Search returned {len(items)} results")
    for item in items:
        title = item.get('title', '?')
        uid = item.get('UID', '?')
        print(f"    - {title}  (UID={uid})")
        # Print download info entries
        di = item.get('dataset_download_information', {}).get('items', [])
        for d in di:
            print(f"      download_info: {json.dumps(d, indent=8)}")

    if not items:
        raise RuntimeError(f"No datasets found for keyword: {keyword}")

    # Try to find the 2018 epoch specifically
    for item in items:
        title = item.get('title', '').lower()
        if '2018' in title:
            print(f"  Selected: {item.get('title')}")
            return item

    # Fall back to first result
    print(f"  No exact 2018 match, using first result: {items[0].get('title')}")
    return items[0]


def get_dataset_info():
    """Find dataset UID and DatasetDownloadInformationID via search."""
    dataset = search_dataset()

    dataset_id = dataset['UID']
    download_info = dataset.get('dataset_download_information', {}).get('items', [])
    if not download_info:
        # Fallback: fetch the full dataset page for download info
        dataset_url = dataset['@id']
        print(f"  Fetching full dataset page: {dataset_url}")
        r = requests.get(dataset_url, headers=HEADERS)
        r.raise_for_status()
        full_data = r.json()
        download_info = full_data.get('dataset_download_information', {}).get('items', [])

    if not download_info:
        raise RuntimeError("No dataset_download_information found.")

    # Print all download info entries for debugging
    print(f"  Found {len(download_info)} download info entries:")
    for i, entry in enumerate(download_info):
        print(f"    [{i}] {json.dumps(entry, indent=8)}")

    entry = download_info[0]
    info_id = entry['@id']
    source = entry.get('full_source', 'LANDCOVER')
    fmt = entry.get('full_format', 'Geotiff')

    print(f"\n  Using: info_id={info_id}, source={source}, format={fmt}")

    return dataset_id, info_id, source, fmt


def main():
    # --- 2. Get Dataset Info ---
    print("\nFetching dataset metadata...")
    dataset_id, info_id, source, fmt = get_dataset_info()
    print(f"Dataset UID:              {dataset_id}")
    print(f"DownloadInformationID:    {info_id}")
    print(f"Source: {source}, Format: {fmt}")

    # --- 3. Submit Download Request ---
    from datetime import datetime, timezone
    # 2018 full year temporal filter (milliseconds)
    start_ms = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(datetime(2018, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)

    # Hungary bounding box [lon_min, lat_max, lon_max, lat_min]
    BBOX_HU = [16.1138866, 48.5854071, 22.8977094, 45.7370221]

    # Try NUTS + TemporalFilter first, then BoundingBox + TemporalFilter
    temporal = {
        "StartDate": start_ms,
        "EndDate": end_ms,
    }
    base = {
        "DatasetID": dataset_id,
        "DatasetDownloadInformationID": info_id,
        "OutputFormat": fmt,
        "OutputGCS": "EPSG:4326",
        "TemporalFilter": temporal,
        "Layers": ["ALL BANDS"],
    }

    payloads = [
        ("NUTS+Temporal+Layers", {**base, "NUTS": "HU"}),
        ("BBox+Temporal+Layers", {**base, "BoundingBox": BBOX_HU}),
        ("NUTS+Temporal", {k: v for k, v in base.items() if k != "Layers"} | {"NUTS": "HU"}),
        ("BBox+Temporal", {k: v for k, v in base.items() if k != "Layers"} | {"BoundingBox": BBOX_HU}),
    ]

    task_ids = []
    for label, ds_payload in payloads:
        download_payload = {"Datasets": [ds_payload]}
        print(f"\nTrying: {label}")
        print(f"  Payload: {json.dumps(download_payload, indent=2)}")

        # Retry on 429 (rate limit)
        for attempt in range(3):
            request_response = requests.post(
                DOWNLOAD_REQUEST_ENDPOINT, headers=HEADERS, json=download_payload
            )
            if request_response.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                continue
            break

        if not request_response.ok:
            print(f"  Error {request_response.status_code}: {request_response.text}")
            continue

        resp_data = request_response.json()
        print(f"  Response: {json.dumps(resp_data, indent=2)}")

        task_ids = resp_data.get('TaskIds', [])
        if task_ids:
            print(f"  Success with {label}!")
            break
        print(f"  Empty TaskIds, trying next...")

    if not task_ids:
        print("\nAll approaches returned empty TaskIds.")
        return

    request_id = task_ids[0]
    print(f"Request ID: {request_id}")

    # --- 4. Poll for Download Readiness ---
    print("\nWaiting for the server to process the request...")
    download_url = None

    while True:
        status_response = requests.get(
            f"{DOWNLOAD_STATUS_ENDPOINT}?request_id={request_id}", headers=HEADERS
        )
        status_response.raise_for_status()
        status_data = status_response.json()

        # Response is a dict keyed by request ID
        if isinstance(status_data, dict):
            entry = status_data.get(str(request_id), {})
        else:
            entry = status_data[0] if status_data else {}

        current_status = entry.get('Status') or entry.get('status')
        print(f"Current Status: {current_status}")

        if current_status == 'Finished_ok':
            download_url = entry.get('DownloadURL') or entry.get('download_url')
            print(f"\nSuccess! Data ready at: {download_url}")
            break
        elif current_status in ['Finished_nok', 'Failed', 'Rejected']:
            print(f"Request failed: {entry.get('Message', '')}")
            return

        time.sleep(30)

    # --- 5. Download the File ---
    if download_url:
        print(f"\nDownloading from: {download_url}")
        file_response = requests.get(download_url, stream=True)
        file_response.raise_for_status()

        filename = "Hungary_GlobalLandCover_100m_2018.zip"
        with open(filename, "wb") as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Download complete! Saved as '{filename}'")


if __name__ == "__main__":
    main()

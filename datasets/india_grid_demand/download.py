"""Download all monthly files of the Mendeley Grid-India dataset (DOI 10.17632/y58jknpgs8.2)."""

import json
import pathlib
import urllib.request

DATASET_ID = "y58jknpgs8"
VERSION = 2
OUT_DIR = pathlib.Path(__file__).resolve().parent


def fetch_files(dataset_id: str, version: int) -> list[dict]:
    """Return the file manifest for a public Mendeley Data dataset."""
    url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}/files"
    params = f"?folder_id=root&version={version}"
    request = urllib.request.Request(
        url + params,
        headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def download(file_info: dict, out_dir: pathlib.Path) -> None:
    """Stream one remote file to disk."""
    name = file_info["filename"].replace("Novemeber", "November")  # sic in source data
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size == file_info["size"]:
        print(f"skip {name} (already present)")
        return
    url = file_info["content_details"]["download_url"]
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"downloading {name} ({file_info['size'] / 1e6:.1f} MB)...", flush=True)
    with urllib.request.urlopen(request) as response, dest.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = fetch_files(DATASET_ID, VERSION)
    for file_info in files:
        download(file_info, OUT_DIR)
    print(f"done: {len(files)} files -> {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Download public financial PDF reports into data/raw/."""

import logging
from pathlib import Path
from typing import Dict, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[2]
_CHUNK_SIZE = 8192  # bytes per streaming chunk

# SEC EDGAR fair-access policy requires a descriptive User-Agent
_SEC_HEADERS: Dict[str, str] = {
    "User-Agent": "finance-insight-rag research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

REPORTS: Dict[str, Dict[str, str]] = {
    "berkshire_2023": {
        "url": "https://www.berkshirehathaway.com/letters/2023ltr.pdf",
        "filename": "berkshire_2023.pdf",
    },
    "tesla_2023": {
        # Tesla 2023 Annual Report to Shareholders filed with SEC EDGAR
        "url": "https://www.sec.gov/Archives/edgar/data/1318605/000110465924053372/tm2412112d4_ars.pdf",
        "filename": "tesla_2023_10k.pdf",
    },
}


def download_report(
    report_key: str,
    headers: Optional[Dict[str, str]] = None,
) -> Path:
    """Download a known report by key and save to data/raw/.

    Skips the download if the file already exists (idempotent).
    Returns the destination path.
    """
    if report_key not in REPORTS:
        raise ValueError(f"Unknown report key '{report_key}'. Choose from: {list(REPORTS)}")

    entry = REPORTS[report_key]
    url: str = entry["url"]
    dest = _PROJECT_ROOT / "data" / "raw" / entry["filename"]

    if dest.exists():
        logger.info("File already exists, skipping download: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading [%s]: %s", report_key, url)

    with requests.get(url, stream=True, timeout=60, headers=headers or {}) as response:
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    logger.info("  %.1f%%  (%d / %d bytes)", pct, downloaded, total)

    logger.info("Saved to: %s", dest)
    return dest


if __name__ == "__main__":
    download_report("tesla_2023", headers=_SEC_HEADERS)

"""Download public financial PDF reports into data/raw/."""

import logging
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Berkshire Hathaway 2023 shareholder letter (Warren Buffett) — stable public URL
_SAMPLE_URL = "https://www.berkshirehathaway.com/letters/2023ltr.pdf"

_PROJECT_ROOT = Path(__file__).parents[2]
_DEST_PATH = _PROJECT_ROOT / "data" / "raw" / "sample_report.pdf"

_CHUNK_SIZE = 8192  # bytes per streaming chunk


def download_sample_report(
    url: str = _SAMPLE_URL,
    dest: Path = _DEST_PATH,
) -> Path:
    """Download a PDF from *url* and save it to *dest*.

    Skips the download if the file already exists (idempotent).
    Returns the destination path.
    """
    if dest.exists():
        logger.info("File already exists, skipping download: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading: %s", url)

    with requests.get(url, stream=True, timeout=30) as response:
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
    download_sample_report()

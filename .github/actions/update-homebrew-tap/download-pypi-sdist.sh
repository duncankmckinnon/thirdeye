#!/usr/bin/env bash
set -euo pipefail

url="${1:?usage: download-pypi-sdist.sh URL DESTINATION}"
destination="${2:?usage: download-pypi-sdist.sh URL DESTINATION}"
retries="${PYPI_DOWNLOAD_RETRIES:-30}"
retry_delay="${PYPI_DOWNLOAD_RETRY_DELAY:-10}"

curl \
  --fail \
  --location \
  --retry "${retries}" \
  --retry-delay "${retry_delay}" \
  --retry-all-errors \
  --output "${destination}" \
  "${url}"

if [[ ! -s "${destination}" ]]; then
  echo "Downloaded source distribution is empty: ${url}" >&2
  exit 1
fi

shasum -a 256 "${destination}" | cut -d' ' -f1

#!/usr/bin/env bash
set -euo pipefail
libraries="$(python -c "import pathlib,sysconfig; print(':'.join(str(p) for p in sorted((pathlib.Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))))")"
export LD_LIBRARY_PATH="${libraries}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"

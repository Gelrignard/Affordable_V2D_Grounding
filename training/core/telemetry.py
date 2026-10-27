"""JSON and GPU telemetry helpers; importing this module never initializes CUDA."""
import datetime
import json
from pathlib import Path
import subprocess

def utc_now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def gpu_memory():
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    used, total, utilization = (int(value.strip()) for value in line.split(","))
    return {"used_mib": used, "total_mib": total, "utilization_percent": utilization}

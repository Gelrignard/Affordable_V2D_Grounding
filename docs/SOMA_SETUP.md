# SOMA setup and decoding

| File | Purpose |
|---|---|
| `data_tools/fetch_soma_assets.py` | Download and verify the pinned SOMA-X/MHR model assets |
| `data_tools/decode_tier1.py` | Decode Tier 1 or Tier 2 SOMA parameters into the retargeter's canonical human/object NPZ |
| `data_tools/requirements-soma.txt` | Dependencies for the separate CPU Torch/SOMA environment |

SOMA and dataset tools share the `data_tools/` package, which does not shadow
the installed `soma` library.

Run from the repository root:

```text
python -m data_tools.fetch_soma_assets
python -m venv .soma-env
```

Using `.soma-env/Scripts/python.exe` on Windows or `.soma-env/bin/python` on Linux:

```text
<soma-python> -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
<soma-python> -m pip install -r data_tools/requirements-soma.txt
<soma-python> -m data_tools.decode_tier1
```

Do not recreate an already working environment. Its installed location stays at
`.soma-env/`; moving an installed virtual environment can break its entry points.
The existing model cache remains at `correction/data/soma_assets/`. Both are
local runtime artifacts excluded from Git. Source code and dependency setup are
grouped in `data_tools/` without requiring model downloads or environment reinstalls.

Fetch episode 11 with [data_tools/](../data_tools/README.md) before decoding.
The decoder defaults to `correction/outputs/tier1_book/human_motion.npz` and
preserves the existing coordinate conversion and frame-selection behavior.
Use `--data`, `--assets`, `--episode`, and `--output` for explicit paths; this
adapter supports the explicit Tier 1 and Tier 2 input contracts. For a full Tier 2 timeline, pass `--tier 2 --data correction/data/tier2 --all-frames`; observation masks and source frame indices are retained.

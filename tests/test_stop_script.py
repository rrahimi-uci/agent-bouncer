"""Guard: ./stop.sh must be able to reap every subprocess the server can spawn, so no heavy
training/eval job is left running in the background. Prevents the stop.sh patterns from
drifting out of sync with the script paths launched in api.py."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_stop_sh_covers_every_launched_script():
    api = (ROOT / "src/agent_bouncer/serving/api.py").read_text()
    stop = (ROOT / "stop.sh").read_text()
    launched = set(re.findall(r"scripts/[\w/]+\.py", api))
    assert launched, "expected api.py to launch some scripts"
    missing = sorted(s for s in launched if s not in stop)
    assert not missing, f"stop.sh must also kill these orphanable scripts: {missing}"

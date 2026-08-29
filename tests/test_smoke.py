from pathlib import Path

from kitscope.verify_main import run_verification


def test_smoke_verification_passes(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    report = run_verification(repo / "configs" / "smoke.yaml", repo / "data" / "smoke", tmp_path)
    assert report["status"] == "passed"
    assert {method["method"] for method in report["methods"]} == {"KitScope-CR", "KitScope-EV"}

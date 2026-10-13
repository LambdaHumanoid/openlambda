import json
from egoalign.replay_only import main

def test_replay_only_causal_plan(capsys):
    assert main(["--source-root", "/example/staging", "--replay-root", "/example/replay",
                 "--output-root", "/example/final", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan) == 2
    assert "--replay-episodes" in plan[0]
    assert "captured_cpp" in plan[1]
    assert "all" in plan[1]
    assert not any("retarget" in arg for cmd in plan for arg in cmd)

def test_reuse_still_exports_with_validation(capsys):
    main(["--source-root", "/example/staging", "--replay-root", "/example/replay",
          "--output-root", "/example/final", "--reuse-replay", "--dry-run"])
    assert len(json.loads(capsys.readouterr().out)) == 1

def test_reject_nested_output():
    import pytest
    with pytest.raises(SystemExit):
        main(["--source-root", "/example/staging", "--replay-root", "/example/replay",
              "--output-root", "/example/staging/final", "--dry-run"])

def test_external_wire_contract_is_not_a_processing_mode():
    from egoalign.adapters.sonic.rollout_contract import SIM_EPISODE_LIFECYCLE_CONTRACT
    from scripts.human_data.run_endpoint_retarget_dataset import ROLLOUT_CONTRACT
    # Upstream telemetry schema identifiers must remain byte-compatible.
    assert SIM_EPISODE_LIFECYCLE_CONTRACT.endswith("formal-replay:v4")
    assert ROLLOUT_CONTRACT == SIM_EPISODE_LIFECYCLE_CONTRACT

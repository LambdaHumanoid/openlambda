from pathlib import Path
import signal
from unittest.mock import Mock

from egoalign.public_runtime import commands, controller_command, runtime_environment, terminate_group


def test_replay_shutdown_triggers_nested_child_cleanup(monkeypatch):
    kill = Mock()
    monkeypatch.setattr('egoalign.public_runtime.os.killpg', kill)
    process = Mock(pid=12345)
    terminate_group(process, replay=True)
    assert kill.call_args_list[0].args == (12345, signal.SIGINT)
    process.wait.assert_called_once_with(timeout=15)


def test_telemetry_recorder_configured_without_video():
    env = runtime_environment(Path('/runtime'), Path('/brainco'), Path('/output'), 28700,
                              {'SONIC_REPLAY_VIDEO_DIR': '/stale', 'SIM_CONTROL_PORT': '5556'})
    assert env['SONIC_REPLAY_VIDEO_DIR'] == '/output'
    assert env['SONIC_REPLAY_WRITE_VIDEO'] == 'false'
    assert env['SIM_CONTROL_PORT'] == '28704'
    assert env['SIM_BRIDGE_MODE'] == 'mock'
    assert env['SONIC_BRAINCO_MODEL_DIR'] == '/brainco'


def test_controller_sim_only_and_no_overrides():
    cmd = controller_command(Path('/runtime'), 28700, Path('/runtime/controller'))
    assert cmd[1] == 'lo'
    assert cmd[cmd.index('--zmq-robot-ip')+1] == 'localhost'
    for name, value in [('--zmq-port', '28700'), ('--zmq-out-port', '28701'),
                        ('--zmq-robot-state-port', '28702'), ('--zmq-robot-cmd-port', '28703')]:
        assert cmd[cmd.index(name)+1] == value
    assert '--direct-upper-body' not in cmd
    assert '--zero-history' not in cmd
    assert cmd[cmd.index('--lock-waist-axes')+1] == 'none'


def test_lifecycle_and_recording():
    cmd = commands(Path('/runtime'), Path('/python/data'), Path('/python/sim'), Path('/controller'),
                   Path('/input'), Path('/output'), '0', 28700, Path('/sentinel'))
    supervisor, replay = cmd['supervisor'], cmd['replay']
    assert 'send_sim_control_command.py' in supervisor[supervisor.index('--prepare-command')+1]
    assert 'sim:reset' in supervisor[supervisor.index('--prepare-command')+1]
    assert supervisor[supervisor.index('--deploy-command')+1].startswith('exec ')
    assert '--no-reset-sim' in replay
    assert '--sim-prepared-before-deploy' in replay
    assert replay[replay.index('--sim-control-port')+1] == '28704'
    assert '--no-enable-onscreen' in cmd['sim']

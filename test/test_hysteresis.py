from birdrecorder.recorder import Hysteresis


def test_hysteresis_initialized_off(mocker):
    mock_recorder = mocker.Mock()
    h = Hysteresis(mock_recorder, start_delay=0, stop_delay=0)
    assert not h.state


def test_trigger_immediately(mocker):
    mock_recorder = mocker.Mock()
    h = Hysteresis(mock_recorder, start_delay=0, stop_delay=0)
    h.step(True)
    assert h.state
    h.step(False)
    assert not h.state
    mock_recorder.start.assert_called_once()
    mock_recorder.stop.assert_called_once()


def test_start_immediately_stop_after3(mocker):
    mock_recorder = mocker.Mock()
    h = Hysteresis(mock_recorder, start_delay=0, stop_delay=3)
    h.step(True)
    assert h.state
    h.step(False)
    assert h.state
    h.step(False)
    assert h.state
    h.step(False)
    assert not h.state
    mock_recorder.start.assert_called_once()
    mock_recorder.stop.assert_called_once()


def test_start_after2_stop_after3(mocker):
    mock_recorder = mocker.Mock()
    h = Hysteresis(mock_recorder, start_delay=2, stop_delay=3)
    h.step(True)
    assert not h.state
    h.step(True)
    assert h.state  # first start
    h.step(False)
    assert h.state  # glitch, should not change state
    h.step(True)
    assert h.state
    h.step(False)  # 3 consecutive negative inputs
    assert h.state
    h.step(False)
    assert h.state
    h.step(False)
    assert not h.state
    mock_recorder.start.assert_called_once()
    mock_recorder.stop.assert_called_once()

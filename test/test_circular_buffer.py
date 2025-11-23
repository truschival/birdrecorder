from birdrecorder.recorder import CircularFrameStore


def test_circular_buffer_basic():
    cb = CircularFrameStore(3)
    assert cb.read() is None
    cb.write(1)
    cb.write(2)
    assert cb.read() == 1
    cb.write(3)
    cb.write(4)  # Overwrite oldest (1)
    assert cb.read() == 2
    assert cb.read() == 3
    assert cb.read() == 4
    assert cb.read() is None
    cb.write(5)
    assert cb.read() == 5
    assert cb.read() is None


def test_circular_buffer_overwrite():
    cb = CircularFrameStore(2)
    cb.write(1)
    cb.write(2)
    cb.write(3)  # This should overwrite '1'
    # Buffer is now [3,2]
    assert cb.read() == 2
    assert cb.read() == 3
    assert cb.read() is None
    # Buffer is empty now, write index at 1, read index at 1
    cb.write(4)
    cb.write(5)
    assert cb.read() == 4
    assert cb.read() == 5
    assert cb.read() is None


def test_circular_buffer_empty_read():
    cb = CircularFrameStore(2)
    assert cb.read() is None
    cb.write(1)
    assert cb.read() == 1
    assert cb.read() is None
    assert cb.read() is None
    cb.write(2)
    assert cb.read() == 2
    assert cb.read() is None
    assert cb.read() is None
    cb.write(3)
    cb.write(4)
    assert cb.read() == 3
    assert cb.read() == 4
    assert cb.read() is None

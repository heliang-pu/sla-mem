"""MEM's packing, history and inference buffer: NumPy only, no SLA env needed.

python -m pytest tests -q
"""

import functools
import pickle

import numpy as np
import pytest

from sla_mem import mem


def test_packs_a_cameras_frames_along_the_channels_and_back():
    frames = np.random.default_rng(6).integers(0, 255, (6, 4, 5, 3), dtype=np.uint8)
    packed = mem.pack(frames)
    assert packed.shape == (4, 5, 18)
    np.testing.assert_array_equal(packed[..., -3:], frames[-1])  # the current one last
    np.testing.assert_array_equal(mem.unpack(packed), frames)
    batch = np.stack([packed, packed + 1])
    np.testing.assert_array_equal(mem.repack(mem.unpack(batch)), batch)


def test_history_is_stride_apart_and_repeats_the_first_frame():
    assert mem.offsets(6, 25) == [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]
    assert mem.offsets(3, 25, fps=50) == [-1.0, -0.5, 0.0]
    assert mem.history(list(range(200)), 6, 25) == [74, 99, 124, 149, 174, 199]
    # 40 ticks into the episode: older frames are the first one.
    assert mem.history(list(range(41)), 6, 25) == [0, 0, 0, 0, 15, 40]
    assert mem.history([0], 6, 25) == [0] * 6


def test_resizes_packed_images_frame_by_frame():
    def resize(images, height, width, return_mask=False):
        assert images.shape[-1] == 3
        out = images[..., :height, :width, :]
        mask = np.ones(out.shape[:-1], bool)
        return (out, mask) if return_mask else out

    frames = np.random.default_rng(7).integers(0, 255, (3, 8, 8, 3), dtype=np.uint8)
    resized, mask = mem.per_frame(resize)(mem.pack(frames), 4, 4, return_mask=True)
    assert resized.shape == (4, 4, 9) and mask.shape == (4, 4)
    np.testing.assert_array_equal(mem.unpack(resized), frames[:, :4, :4])


def test_the_loader_asks_for_the_cameras_history_at_the_datasets_rate():
    def timestamps(data_config, action_horizon, fps):
        return {"action": [t / fps for t in range(action_horizon)]}

    patched = mem.with_history(timestamps, 3, 25, ["observation.images.top"])
    times = patched(None, 2, 50)
    assert times == {"action": [0.0, 0.02], "observation.images.top": [-1.0, -0.5, 0.0]}
    assert patched(None, 2, fps=25)["observation.images.top"] == [-2.0, -1.0, 0.0]


def test_as_image_takes_lerobot_and_server_frames_and_packs_histories():
    rng = np.random.default_rng(8)
    server = rng.integers(0, 255, (4, 5, 3), dtype=np.uint8)
    np.testing.assert_array_equal(mem.as_image(server), server)
    lerobot = rng.random((3, 4, 5), dtype=np.float32)  # [3, h, w] in [0, 1]
    image = mem.as_image(lerobot)
    assert image.dtype == np.uint8 and image.shape == (4, 5, 3)
    history = rng.random((6, 3, 4, 5), dtype=np.float32)  # [T, 3, h, w]
    packed = mem.as_image(history)
    assert packed.shape == (4, 5, 18)
    np.testing.assert_array_equal(packed[..., -3:], mem.as_image(history[-1]))


def test_frame_history_serves_the_trained_history():
    seen = mem.FrameHistory(6, 25)
    frame = {}
    for tick in range(0, 201, 25):  # shown every stride ticks
        frame[tick] = {"top": np.full((2, 2, 3), tick % 256, np.uint8)}
        seen.see(tick, frame[tick])
    stack = seen.stack(200)
    picks = mem.history(range(201), 6, 25)
    assert stack["top"].shape == (6, 2, 2, 3)
    for got, tick in zip(stack["top"], picks, strict=True):
        np.testing.assert_array_equal(got, frame[tick]["top"])
    # Early on, the first frame stands in for the ticks before the start.
    young = mem.FrameHistory(6, 25)
    young.see(0, frame[0])
    young.see(25, frame[25])
    assert [int(f[0, 0, 0]) for f in young.stack(25)["top"]] == [0, 0, 0, 0, 0, 25]
    # Only what a history can still pick is kept, and the first frame.
    assert sorted(seen.seen) == [0, 75, 100, 125, 150, 175, 200]
    with pytest.raises(ValueError, match="ticks"):
        seen.stack(210)  # a query off the stride needs frames it wasn't shown


def test_the_spawned_workers_install_pickles():
    init = functools.partial(mem.worker_init, 6, 25, ("observation.images.top",))
    assert pickle.loads(pickle.dumps(init)).args == init.args

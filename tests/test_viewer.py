# ruff: noqa: E402
"""igsv viewer bridge (ADR-0018): embedding math, fit observer seam, live server.

The observer-seam tests run everywhere; tests that need the optional igsv
package skip themselves when it is absent.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat.config import FitConfig
from structsplat.fit import fit
from structsplat.gaussians import GaussianField


def _field(n: int = 6) -> GaussianField:
    torch.manual_seed(3)
    means = torch.rand(n, 2) * 10 + 3  # comfortably inside a 16x16 image
    scales = torch.rand(n, 2) * 1.5 + 0.8
    angles = torch.linspace(-1.0, 1.0, n)
    colors = torch.rand(n, 3)
    return GaussianField(means, torch.log(scales), angles, colors)


def _target(h: int = 16, w: int = 16) -> torch.Tensor:
    torch.manual_seed(4)
    return torch.rand(h, w, 3)


def test_fit_iteration_observer_cadence_and_final_call():
    calls = []
    cfg = FitConfig(iters=7, log_every=1, ssim_weight=0.0)
    torch.manual_seed(0)
    fit(_field(), _target(), cfg, verbose=False,
        iteration_observer=lambda f, i, loss: calls.append((i, f.n, loss)),
        observer_every=3)
    assert [c[0] for c in calls] == [3, 6, 7]  # every 3rd iteration plus the final one
    assert all(isinstance(c[2], float) for c in calls)
    assert all(c[1] > 0 for c in calls)


def test_fit_observer_off_by_default():
    calls = []
    cfg = FitConfig(iters=3, log_every=1, ssim_weight=0.0)
    torch.manual_seed(0)
    fit(_field(), _target(), cfg, verbose=False,
        iteration_observer=lambda f, i, loss: calls.append(i))  # observer_every defaults to 0
    assert calls == []


def test_observer_does_not_perturb_fit():
    cfg = FitConfig(iters=5, log_every=1, ssim_weight=0.0)
    torch.manual_seed(0)
    plain = fit(_field(), _target(), cfg, verbose=False)
    torch.manual_seed(0)
    observed = fit(_field(), _target(), cfg, verbose=False,
                   iteration_observer=lambda f, i, loss: None, observer_every=1)
    assert plain["psnr"] == pytest.approx(observed["psnr"], abs=0)
    assert plain["history"]["loss"] == observed["history"]["loss"]


def test_field_to_frame_embedding():
    pytest.importorskip("igsv")
    from structsplat.viewer import SH_C0, Z_THICKNESS_PX, field_to_frame

    theta = np.pi / 2
    field = GaussianField.from_numpy(
        means=np.array([[2.0, 3.0], [5.0, 7.0]]),
        scales=np.array([[1.0, 2.0], [0.5, 0.5]]),
        angles=np.array([theta, 0.0]),
        colors=np.array([[1.5, 0.5, -0.2], [0.25, 0.5, 0.75]]),  # clamps to [0,1]
        opacities=np.array([0.0, 2.0]),  # logits
    )
    frame = field_to_frame(field, (10, 12), name="embed")

    assert frame.n == 2 and frame.name == "embed"
    np.testing.assert_allclose(frame.positions[:, 0], [2.0, 5.0], atol=1e-6)
    np.testing.assert_allclose(frame.positions[:, 1], [9.0 - 3.0, 9.0 - 7.0], atol=1e-6)
    np.testing.assert_allclose(frame.positions[:, 2], 0.0, atol=1e-9)

    # y-flip negates the angle: rotation by -theta about +Z
    np.testing.assert_allclose(frame.quats_wxyz[0], [np.cos(theta / 2), 0, 0, -np.sin(theta / 2)],
                               atol=1e-6)
    np.testing.assert_allclose(frame.quats_wxyz[1], [1, 0, 0, 0], atol=1e-6)

    np.testing.assert_allclose(frame.scales[:, :2], [[1.0, 2.0], [0.5, 0.5]], rtol=1e-5)
    np.testing.assert_allclose(frame.scales[:, 2], Z_THICKNESS_PX, rtol=1e-6)

    np.testing.assert_allclose(frame.opacities, [0.5, 1 / (1 + np.exp(-2.0))], atol=1e-6)
    np.testing.assert_allclose(frame.sh0[0], [(1.0 - 0.5) / SH_C0, 0.0, (0.0 - 0.5) / SH_C0],
                               atol=1e-5)


def test_field_to_frame_folds_in_filter_variance():
    pytest.importorskip("igsv")
    from structsplat.viewer import field_to_frame

    field = GaussianField.from_numpy(
        means=np.array([[4.0, 4.0]]),
        scales=np.array([[1.0, 2.0]]),
        angles=np.array([0.0]),
        colors=np.array([[0.5, 0.5, 0.5]]),
        filter_variance=np.array([3.0]),  # px^2, in quadrature with RS scales
    )
    frame = field_to_frame(field, (8, 8))
    np.testing.assert_allclose(frame.scales[0, :2], [2.0, np.sqrt(7.0)], rtol=1e-5)


def test_field_without_opacities_views_as_opaque():
    pytest.importorskip("igsv")
    from structsplat.viewer import field_to_frame

    frame = field_to_frame(_field(4), (16, 16))
    np.testing.assert_allclose(frame.opacities, 1.0, atol=1e-7)


def test_live_fit_viewer_socket_roundtrip():
    pytest.importorskip("igsv")
    import asyncio
    import struct

    from structsplat.viewer import LiveFitViewer

    viewer = LiveFitViewer((16, 16), port=0, name="socket-test")
    viewer.start()
    try:
        viewer.observer(_field(8), 5, 0.25)  # the fit() seam signature
        port = viewer.server.bound_port

        async def probe():
            import aiohttp
            from igsv import protocol

            total = None
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    for _ in range(60):
                        msg = protocol.decode_message(await asyncio.wait_for(ws.receive_bytes(), 5))
                        if msg.type == protocol.MessageType.KEYFRAME_END:
                            total = struct.unpack_from("<II", msg.payload)[1]
                        elif msg.type == protocol.MessageType.STATS and total is not None:
                            return total, struct.unpack_from("<I", msg.payload)[0]
            raise AssertionError("no keyframe/stats received")

        total, step = asyncio.run(asyncio.wait_for(probe(), timeout=10))
        assert total == 8
        assert step == 5
    finally:
        viewer.stop()

import torch

from src.models.eaar_nlu import Detections, EAARNLUBridge, detections_to_tensors

B, K = 3, 8


def make_batch(n_valid=(8, 3, 0), seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(B, 197, 768, generator=g)
    xy = torch.rand(B, K, 2, generator=g) * 0.6
    wh = 0.1 + torch.rand(B, K, 2, generator=g) * 0.3
    boxes = torch.cat([xy, xy + wh], dim=-1)
    labels = torch.randint(1, 91, (B, K), generator=g)
    scores = torch.rand(B, K, generator=g)
    valid = torch.zeros(B, K, dtype=torch.bool)
    for i, n in enumerate(n_valid):
        valid[i, :n] = True
    return tokens, Detections(boxes, labels, scores, valid)


def test_shapes_and_aux():
    bridge = EAARNLUBridge()
    tokens, det = make_batch()
    out, aux = bridge(tokens, det, return_aux=True)
    assert out.shape == (B, 197, 768)
    assert aux["reliability"].shape == (B, 28, 1)
    assert aux["cross_attn"].shape == (B, 4, 197, 28)
    assert aux["slot_mask"].shape == (B, 28)
    assert torch.isfinite(out).all()


def test_identity_at_init():
    bridge = EAARNLUBridge(gate_init=0.0)
    tokens, det = make_batch()
    assert torch.allclose(bridge(tokens, det), tokens)


def test_empty_image_passes_through_even_with_open_gate():
    bridge = EAARNLUBridge(gate_init=0.5)
    tokens, det = make_batch(n_valid=(0, 0, 0))
    out = bridge(tokens, det)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, tokens)


def test_padded_entries_do_not_leak():
    bridge = EAARNLUBridge(gate_init=0.5).eval()
    tokens, det = make_batch()
    ref = bridge(tokens, det)
    junk = Detections(
        torch.where(det.valid[..., None], det.boxes, torch.rand_like(det.boxes)),
        torch.where(det.valid, det.labels, torch.randint(1, 91, det.labels.shape)),
        torch.where(det.valid, det.scores, torch.rand_like(det.scores)),
        det.valid,
    )
    assert torch.allclose(bridge(tokens, junk), ref, atol=1e-5)


def test_gradients_reach_every_layer():
    bridge = EAARNLUBridge(gate_init=0.1)
    tokens, det = make_batch()
    bridge(tokens, det).sum().backward()
    for name, p in bridge.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_param_count_report():
    bridge = EAARNLUBridge()
    total = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    print(bridge.layer_param_counts(), "total:", f"{total:,}")
    assert 4_000_000 < total < 4_800_000


def test_detection_adapter_center_crop():
    # 640x480 image -> short side 480 scaled to 224, 37.33px cropped from each side
    d = detections_to_tensors(
        xyxy=[[0, 0, 100, 100], [200, 100, 440, 380]],
        confidence=[0.9, 0.8],
        class_id=[1, 18],
        image_size=(640, 480),
    )
    assert d.valid.tolist() == [True, True] + [False] * 6
    assert d.boxes.min() >= 0 and d.boxes.max() <= 1
    # first box lies mostly in the cropped-away left strip: x1 clipped to 0
    assert d.boxes[0, 0] == 0

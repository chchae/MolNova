"""Exercise the actual optional REINVENT network when its runtime is available."""
import pytest

from molnova.reinvent_launcher import preserve_encoder_dimensions


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_encoder_preserves_shapes_values_and_gradients(monkeypatch, num_layers, batch_size):
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("reinvent.models.libinvent.models.decorator")
    encoder_class = module.Encoder
    original = encoder_class.forward
    monkeypatch.setattr(encoder_class, "forward", original)
    encoder = encoder_class(num_layers, 8, 10, 0.0)
    sequences = torch.tensor([[1, 2, 3, 4]] * batch_size)
    lengths = torch.tensor([4] * batch_size)
    # A duplicated batch gives an independent reference without the singleton
    # squeeze defect; identical rows must produce identical encoder activations.
    reference, (ref_hidden, ref_cell) = encoder(sequences.repeat(2, 1), lengths.repeat(2))
    reference = reference.reshape(batch_size * 2, 4, 8)[:batch_size]
    ref_hidden = ref_hidden.reshape(num_layers, batch_size * 2, 8)[:, :batch_size]
    ref_cell = ref_cell.reshape(num_layers, batch_size * 2, 8)[:, :batch_size]
    preserve_encoder_dimensions(encoder_class)
    encoded, (hidden, cell) = encoder(sequences, lengths)
    assert encoded.shape == (batch_size, 4, 8)
    assert hidden.shape == cell.shape == (num_layers, batch_size, 8)
    torch.testing.assert_close(encoded, reference)
    torch.testing.assert_close(hidden, ref_hidden)
    torch.testing.assert_close(cell, ref_cell)
    (encoded.sum() + hidden.sum() + cell.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_singleton_tl_batch_reaches_decoder_and_backpropagates(monkeypatch):
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("reinvent.models.libinvent.models.decorator")
    monkeypatch.setattr(module.Encoder, "forward", module.Encoder.forward)
    preserve_encoder_dimensions(module.Encoder)
    params = dict(num_layers=2, num_dimensions=8, vocabulary_size=10, dropout=0.0)
    model = module.Decorator(params, params)
    sequences = torch.tensor([[1, 2, 3, 4]])
    lengths = torch.tensor([4])
    logits = model(sequences, lengths, sequences, lengths)
    assert logits.shape == (1, 4, 10)
    logits.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

"""Standalone TL launcher, copied to SSH runs without installing MolNova there.

Some REINVENT LibInvent encoders squeeze away the batch axis for singleton
batches. Restore the documented encoder shapes before the decoder sees them.
The installed REINVENT package and the training/validation data stay untouched.
"""
from functools import wraps


def preserve_encoder_dimensions(encoder_class):
    original_forward = encoder_class.forward

    @wraps(original_forward)
    def forward(self, padded_seqs, seq_lengths):
        encoded, (hidden, cell) = original_forward(self, padded_seqs, seq_lengths)
        batch_size = padded_seqs.size(0)
        return (
            encoded.reshape(batch_size, -1, self.num_dimensions),
            (
                hidden.reshape(self.num_layers, batch_size, self.num_dimensions),
                cell.reshape(self.num_layers, batch_size, self.num_dimensions),
            ),
        )

    encoder_class.forward = forward


def main():
    from reinvent.models.libinvent.models.decorator import Encoder
    from reinvent.Reinvent import main_script

    preserve_encoder_dimensions(Encoder)
    return main_script()


if __name__ == "__main__":
    raise SystemExit(main())

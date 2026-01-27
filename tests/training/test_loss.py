import unittest
import torch
from src.training.loss import cross_entropy_with_label_smoothing

class TestLoss(unittest.TestCase):
    def test_cross_entropy_with_label_smoothing(self):
        vocab_size = 10
        batch_size = 2
        seq_len = 5
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Test without smoothing
        loss_no_smooth = cross_entropy_with_label_smoothing(logits, targets, 0.0)
        expected_no_smooth = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        self.assertAlmostEqual(loss_no_smooth.item(), expected_no_smooth.item(), places=5)

        # Test with smoothing
        loss_smooth = cross_entropy_with_label_smoothing(logits, targets, 0.1)
        self.assertGreater(loss_smooth.item(), 0)

        # Test with ignore_index
        targets[0, 0] = -100
        loss_ignore = cross_entropy_with_label_smoothing(logits, targets, 0.1, ignore_index=-100)
        self.assertGreater(loss_ignore.item(), 0)

if __name__ == "__main__":
    unittest.main()

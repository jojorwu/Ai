import unittest
import torch
from src.data.data_loader import get_batches_torch

class TestDataLoader(unittest.TestCase):
    def test_get_batches_torch(self):
        data = list(range(100))
        batch_size = 4
        seq_len = 10
        device = "cpu"

        batches = list(get_batches_torch(data, batch_size, seq_len, device))
        self.assertGreater(len(batches), 0)

        for x, y, img in batches:
            self.assertEqual(x.shape, (x.size(0), seq_len))
            self.assertEqual(y.shape, (y.size(0), seq_len))
            # Verify shifting
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
            self.assertIsNone(img)

        # Verify first batch
        x_first, y_first, _ = batches[0]
        self.assertEqual(x_first[0, 0].item(), 0)
        self.assertEqual(x_first[0, -1].item(), 9)
        self.assertEqual(y_first[0, 0].item(), 1)
        self.assertEqual(y_first[0, -1].item(), 10)

        self.assertEqual(x_first[1, 0].item(), 1)

if __name__ == "__main__":
    unittest.main()

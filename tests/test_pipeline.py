import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from dataset.storage import prepare, load_partitions, audit_partitions
from dataset.signal_generator import _generate_modulation_burst, add_awgn_noise
from models.transforms import (
    RandomPhaseRotation,
    AdditiveGaussianNoise,
    RFAugmentationPipeline,
)
from metrics import snr80, scores
from train import loader, seed_everything, state_hash
from models.network import ModulationClassifier1DCNN


class PipelineTests(unittest.TestCase):
    def test_saved_partition_balance_reproducibility_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            ma = prepare(a, (2, 2, 2), 55)
            mb = prepare(b, (2, 2, 2), 55)
            da, _ = load_partitions(a)
            db, _ = load_partitions(b)
            for name in da:
                np.testing.assert_array_equal(da[name]["x"], db[name]["x"])
                for c in range(8):
                    for snr in range(-10, 20, 2):
                        self.assertEqual(
                            int(
                                ((da[name]["y"] == c) & (da[name]["snr"] == snr)).sum()
                            ),
                            2,
                        )
            bad = {
                name: {k: v.copy() for k, v in part.items()}
                for name, part in da.items()
            }
            bad["test"]["x"][0] = bad["train"]["x"][0]
            with self.assertRaises(ValueError):
                audit_partitions(bad)
            with (a / "test.npz").open("ab") as f:
                f.write(b"corruption")
            with self.assertRaises(ValueError):
                load_partitions(a)
            with self.assertRaises(FileExistsError):
                prepare(b, (2, 2, 2), 55)

    def test_phase_preserves_energy(self):
        x = torch.randn(2, 128)
        out = RandomPhaseRotation(1)(x, torch.Generator().manual_seed(3))
        torch.testing.assert_close(out.square().sum(0), x.square().sum(0))

    def test_noise_reaches_expected_snr(self):
        n = 262144
        snr = 4.0
        rho = 10 ** (snr / 10)
        rng = np.random.default_rng(2)
        clean = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
        noisy = add_awgn_noise(clean, snr, rng)
        x = torch.tensor(np.stack((noisy.real, noisy.imag)), dtype=torch.float32)
        g = torch.Generator().manual_seed(77)
        probe = torch.Generator().manual_seed(77)
        torch.rand((), generator=probe)
        drop = 6 * torch.rand((), generator=probe).item()
        out = AdditiveGaussianNoise(1, 6)(x, snr, g).numpy()
        actual = 10 * np.log10(
            np.mean(np.abs(clean) ** 2)
            / np.mean(np.abs(out[0] + 1j * out[1] - clean) ** 2)
        )
        self.assertAlmostEqual(actual, snr - drop, delta=0.12)

    def test_private_augmentation_rng_and_no_inplace_mutation(self):
        x = torch.randn(2, 128)
        before = x.clone()
        state = torch.get_rng_state()
        a = RFAugmentationPipeline(seed=10)
        b = RFAugmentationPipeline(seed=10)
        torch.testing.assert_close(a(x, 0), b(x, 0))
        torch.testing.assert_close(x, before)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with self.assertRaises(ValueError):
            a(x)

    def test_matched_initialization_and_shuffle(self):
        seed_everything(42)
        a = state_hash(ModulationClassifier1DCNN().state_dict())
        seed_everything(42)
        self.assertEqual(a, state_hash(ModulationClassifier1DCNN().state_dict()))
        part = {
            "x": np.zeros((32, 2, 128), np.float32),
            "y": np.arange(32),
            "snr": np.zeros(32),
        }
        one = torch.cat([y for _, y, _ in loader(part, 8, True, 45)])
        torch.randn(100)
        two = torch.cat([y for _, y, _ in loader(part, 8, True, 45)])
        self.assertTrue(torch.equal(one, two))

    def test_gfsk_and_metrics(self):
        x = _generate_modulation_burst("GFSK", rng=np.random.default_rng(1))
        self.assertEqual(x.shape, (128,))
        np.testing.assert_allclose(np.abs(x), 1)
        self.assertAlmostEqual(snr80([0, 2], [0.7, 0.9])["snr_db"], 1.0)
        self.assertIsNone(snr80([0, 2], [0.3, 0.7])["snr_db"])
        self.assertEqual(snr80([0, 2], [0.9, 0.95])["status"], "at_or_below_grid_min")
        m = scores(np.array([0, 0]), np.array([0, 0]), 8)
        self.assertEqual(m["macro_f1"], 1 / 8)
        self.assertEqual(m["support"][1], 0)


if __name__ == "__main__":
    unittest.main()

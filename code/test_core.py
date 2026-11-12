"""Small deterministic checks of the scientific protocol and selector."""
import unittest
import numpy as np
import torch
from core import nearest, greedy_indices, anchored_bank, choose_validated, calibrated_threshold
from core import score_features, unanchored_bank
from run_experiment import pixel_metrics
from sklearn.metrics import roc_auc_score, average_precision_score

class ProtocolTests(unittest.TestCase):
    def test_distances_against_direct_subtraction(self):
        generator = torch.Generator().manual_seed(7)
        x, bank = torch.randn(13,5,generator=generator), torch.randn(7,5,generator=generator)
        expected = ((x[:,None]-bank[None])**2).sum(-1).min(1).values
        torch.testing.assert_close(nearest(x,bank,chunk=4), expected)

    def test_greedy_coverage_and_no_repeat(self):
        x = torch.tensor([[0.],[1.],[3.],[10.],[20.]])
        ids = greedy_indices(x,4)
        self.assertEqual(len(set(ids.tolist())),4)
        errors = [float(nearest(x,x[ids[:i]]).mean()) for i in range(1,5)]
        self.assertEqual(errors,sorted(errors,reverse=True))

    def test_anchors_budget_uniformity_and_worst_group(self):
        groups = [torch.arange(10,dtype=torch.float32)[:,None],
                  torch.arange(20,30,dtype=torch.float32)[:,None],
                  torch.arange(100,110,dtype=torch.float32)[:,None]]
        order = greedy_indices(groups[0],6)
        for adaptive in [False,True]:
            bank,counts = anchored_bank(groups,order,6,.5,adaptive)
            torch.testing.assert_close(bank[:3],groups[0][order[:3]])
            self.assertEqual(len(torch.unique(bank,dim=0)),6)
            self.assertEqual(sum(counts),6)
            if not adaptive:
                self.assertLessEqual(abs(counts[1]-counts[2]),1)
            else:
                self.assertGreaterEqual(float(bank[3]),100.)

    def test_safeguard_fallback_and_tie(self):
        baseline = torch.tensor([[0.]])
        bad = [{'bank':torch.tensor([[3.]]),'rho':.5,'coverage':[1.11,.2,.3],'allocation':[1,0,0]}]
        bank,info = choose_validated(bad,baseline,1.)
        self.assertTrue(info['fallback']); torch.testing.assert_close(bank,baseline)
        options = [{'bank':baseline,'rho':rho,'coverage':[1.05,.2,.3],'allocation':[1,0,0]} for rho in [.5,.75]]
        _,info = choose_validated(options,baseline,1.)
        self.assertEqual(info['rho'],.75)

    def test_calibration_rank_and_small_sample(self):
        threshold,rank = calibrated_threshold(np.arange(43))
        self.assertEqual(rank,42); self.assertEqual(threshold,41)
        self.assertTrue(np.isinf(calibrated_threshold(np.arange(10))[0]))

    def test_reweighting_matches_equation_seven(self):
        bank = torch.tensor([[0.,0.],[1.,0.],[2.,0.],[8.,0.]])
        features = torch.tensor([[.1,0.],[4.,1.]])
        patch, score = score_features(features,bank,3)
        direct = torch.linalg.vector_norm(features[:,None]-bank[None],dim=2)
        torch.testing.assert_close(patch,direct.min(1).values)
        query = features[1]
        support = bank[[2,1,0]]
        ds = torch.linalg.vector_norm(query-support,dim=1)
        expected = (1-torch.exp(ds[0])/torch.exp(ds).sum())*ds[0]
        self.assertAlmostEqual(score,float(expected),places=6)

    def test_pixel_metrics_with_ties_match_sklearn(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0,2,500)
        score = rng.integers(0,20,500).astype(np.float32)
        values = pixel_metrics(y,score)
        self.assertAlmostEqual(values['pixel_auroc'],roc_auc_score(y,score),places=12)
        self.assertAlmostEqual(values['pixel_ap'],average_precision_score(y,score),places=12)

    def test_unanchored_budget_and_worst_group(self):
        groups = [torch.tensor([[0.],[1.],[2.]]),torch.tensor([[20.],[21.],[22.]]),torch.tensor([[100.],[101.],[102.]])]
        bank,allocation = unanchored_bank(groups,5)
        self.assertEqual(len(torch.unique(bank,dim=0)),5)
        self.assertEqual(sum(allocation),5)
        self.assertEqual(float(bank[0]),22.)
        self.assertGreaterEqual(float(bank[1]),100.)

if __name__ == '__main__':
    unittest.main(verbosity=2)

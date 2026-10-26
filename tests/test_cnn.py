import unittest
import torch
import numpy as np
from dysphagia_wearable.timing.cnn import SwallowCNN, loss_value, evaluate
from dysphagia_wearable.timing.baseline import event_metrics


class CNNTests(unittest.TestCase):
    def test_output_timing_grid(self):
        model=SwallowCNN().eval()
        with torch.no_grad():
            x,n=model(torch.zeros(2,4,4001),torch.tensor([4001,2000]))
        self.assertEqual(x.shape,(2,101))
        self.assertEqual(n.tolist(),[101,50])

    def test_padding_does_not_change_valid_predictions(self):
        torch.manual_seed(4);model=SwallowCNN().eval()
        x=torch.randn(1,4,2001)
        with torch.no_grad():
            a,n=model(x,torch.tensor([2001]))
            b,_=model(torch.nn.functional.pad(x,(0,3000)),torch.tensor([2001]))
        torch.testing.assert_close(a,b[:,:a.shape[1]],rtol=1e-4,atol=1e-5)

    def test_loss_ignores_padding(self):
        a=loss_value(torch.tensor([[0.,0.]]),torch.tensor([[0.,1.]]),torch.tensor([2]),torch.tensor(1.))
        b=loss_value(torch.tensor([[0.,0.,100.]]),torch.tensor([[0.,1.,0.]]),torch.tensor([2]),torch.tensor(1.))
        torch.testing.assert_close(a,b)

    def test_evaluation_matches_baseline(self):
        records=[{'id':'one','participant':'P1','duration':1.,'times':[.25,.35,.45],
                  'intervals':[[.2,.4]]}]
        score=np.array([.9,.9,.1])
        a=evaluate(records,{'one':score},.5)
        b=event_metrics(records,np.array(['one']*3),score,.5)
        self.assertEqual(a,b)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()

import unittest
from validate import fold_plan


class SplitTests(unittest.TestCase):
    def test_outer_assessment_covers_each_development_person_once(self):
        original={'train':[f'P{i}' for i in range(20)],'validation':[f'P{i}' for i in range(20,30)],'test':['TEST']}
        meta=[{'participant':p,'id':p+str(j)} for p in sum(original.values(),[]) for j in range(2)]
        records,plan=fold_plan(meta,original)
        self.assertNotIn('TEST',{r['participant'] for r in records})
        assessed=[p for f in plan for p in f['assessment']]
        self.assertEqual(len(assessed),30)
        self.assertEqual(len(set(assessed)),30)
        for f in plan:
            self.assertFalse(set(f['train'])&set(f['assessment']))
            self.assertFalse(set(f['validation'])&set(f['assessment']))
            self.assertFalse(set(f['train'])&set(f['validation']))

    def test_original_leaking_split_is_rejected(self):
        with self.assertRaises(AssertionError):
            fold_plan([{'participant':'P'}],{'train':['P'],'validation':[],'test':['P']})


if __name__=='__main__':unittest.main()

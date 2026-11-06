"""Registered preparation resources, final-map geometry and replay checks."""
import copy
import math
import unittest

import numpy as np

from scripts import prepare_comparison as preparation


class PreparationProtocolTests(unittest.TestCase):
    def test_final_maps_and_equal_resources_are_registered_before_selection(self):
        plan = preparation.build_plan()
        self.assertEqual(plan['seed'],842901174)
        self.assertEqual(plan['budget_per_family_per_dimension']['transitions_total'],16*4096*8)
        for dimension in (100,500):
            cell=plan['dimensions'][str(dimension)]
            source=preparation.read(preparation.ASSETS/f'full_budget_{dimension}d.json')
            untouched=copy.deepcopy(source)
            rows=preparation.candidates(dimension,source)
            self.assertEqual(source,untouched)
            for method,family,deformation in (('DCS-BW','ball','delta'),('SCS','sphere','observer')):
                family_rows=[r for r in rows if r['method']==method]
                self.assertEqual(len(family_rows),16)
                self.assertEqual({r['R_factor'] for r in family_rows},{.5,1.,2.,4.})
                for row in family_rows:
                    self.assertEqual(row['mapping']['mu'],[0.]*dimension)
                    self.assertEqual(row['mapping'][deformation],[0.]*dimension)
                    self.assertEqual(row['R'],source['maps'][family]['scale']*row['R_factor'])
                    self.assertFalse(row['baseline_author_scs'])
                if method=='DCS-BW':
                    np.testing.assert_allclose(sorted({r['h'] for r in family_rows}),
                                               np.array([.25,.5,1.,2.])/math.sqrt(dimension))
                else:
                    self.assertEqual({r['h'] for r in family_rows},{.025,.05,.1,.2})
            original=cell['original_author_candidate']
            self.assertEqual(original['mapping'],source['maps']['sphere'])
            self.assertEqual(original['h'],.1)
            self.assertTrue(original['baseline_author_scs'])
            self.assertGreater(np.linalg.norm(original['mapping']['mu']),0.)

    def test_physical_start_cloud_is_replayable_and_holdout_is_separate(self):
        spec=preparation.target_spec(100)
        first,metadata=preparation.exact_states(spec,0,257)
        again,repeated=preparation.exact_states(spec,0,257)
        holdout,other=preparation.exact_states(spec,1,257)
        np.testing.assert_array_equal(first,again)
        self.assertEqual(metadata['physical_state_sha256'],repeated['physical_state_sha256'])
        self.assertEqual(metadata['target_seed_entropy'],[842901174,100,0,0])
        self.assertNotEqual(metadata['physical_state_sha256'],other['physical_state_sha256'])
        self.assertFalse(np.array_equal(first,holdout))
        self.assertEqual(first.shape,(257,100))
        self.assertTrue(np.isfinite(first).all())

    def test_score_selection_is_order_independent_and_does_not_reward_acceptance(self):
        source=preparation.read(preparation.ASSETS/'full_budget_100d.json')
        rows=preparation.candidates(100,source)
        for method in preparation.METHODS:
            subset=[dict(r,score=0.,acceptance=1.) for r in rows if r['method']==method]
            # All-zero scores use the documented anchor, without enlarging the grid.
            chosen=preparation.select(subset)
            self.assertEqual(chosen['R_factor'],1.)
            self.assertEqual(chosen['h'],.1)
            self.assertEqual(chosen['status'],'no_positive_pilot_score')
            subset[0].update(score=.5,acceptance=0.)
            for ordering in (subset,list(reversed(subset))):
                self.assertEqual(preparation.select(ordering)['candidate_id'],subset[0]['id'])

    def test_fit_verification_ignores_seconds_but_checks_all_parameters_and_losses(self):
        saved=preparation.read(preparation.ASSETS/'full_budget_100d.json')
        replay=copy.deepcopy(saved)
        replay['fitting']['ball']['seconds']+=123.
        self.assertTrue(preparation.compare_fits(replay,saved)['passed'])
        replay['maps']['sphere']['observer'][3]+=.001
        check=preparation.compare_fits(replay,saved)
        self.assertFalse(check['passed'])
        self.assertFalse(check['checked_fields']['sphere']['observer']['within_tolerance'])
        replay=copy.deepcopy(saved)
        replay['fitting']['ball']['loss'][500]+=1.
        self.assertFalse(preparation.compare_fits(replay,saved)['passed'])

    def test_fitting_seeds_and_resources_are_explicit(self):
        for dimension in preparation.DIMENSIONS:
            config=preparation.fit_configuration(dimension)
            self.assertEqual(config['fit_seed'],42)
            self.assertEqual(preparation.FIT_SEEDS['sphere'],0)
            self.assertEqual(config['vi_batch_policy'],'fixed')
            self.assertEqual(config['map_scale_initialization'],'unit')
            self.assertEqual(config['scs_vi'],dict(optimizer_steps=1000,optimizer_samples=2000))
            self.assertEqual(config['optimizer_steps']*config['optimizer_samples'],2_000_000)


if __name__=='__main__':
    unittest.main()

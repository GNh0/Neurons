"""Causal current effects, independent learning, restart and fertility observability."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tests.test_neurons import fixture
from neurons.simulation import Simulation
from neurons.individuals import IndividualLab
from neurons.modules import Modules
from neurons.catalog import EXTRA_SENSORS, upgrade_body
from neurons.world import fresh_body
from neurons.open_world import OpenWorld
from neurons import life


class InternalPlasticityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=fixture(self.root)
        self.a=Simulation(shared=self.base,threaded=False,plastic=True)
        self.b=Simulation(shared=self.base,threaded=False,plastic=True)

    def tearDown(self):
        self.a.close();self.b.close();self.base.close();self.temp.cleanup()

    def test_reward_polarity_changes_only_active_pair_without_changing_anatomy(self):
        for p in (self.a.plasticity,self.b.plasticity):p.HOMEOSTASIS=0
        counts=np.array([3,3,0,0,0])
        self.a.plasticity.learn(counts,10,1)
        self.b.plasticity.learn(counts,10,-1)
        self.assertGreater(self.a.plasticity.graph.data[0],40)
        self.assertLess(self.b.plasticity.graph.data[0],40)
        self.assertEqual(self.a.plasticity.graph.data[1],40)
        np.testing.assert_array_equal(self.base.graph.data,[40,40])
        self.assertFalse(np.shares_memory(self.a.plasticity.graph.data,self.b.plasticity.graph.data))

    def test_learned_internal_pair_causally_changes_postsynaptic_spike(self):
        p=self.a.plasticity;p.HOMEOSTASIS=0;p.learning_rate=.2
        for _ in range(50):p.learn(np.array([3,3,0,0,0]),10,-3)
        self.assertLess(p.graph.data[0]*self.a.gain,7)
        for brain in (self.a,self.b):
            brain.stimulate([100],duration=10)
            for _ in range(3):brain.step()
        self.assertEqual(self.a.spike_counts[1],0)
        self.assertEqual(self.b.spike_counts[1],1)
        info=self.a.neuron(100)['outgoing'][0]
        self.assertEqual(info['contacts'],40)
        self.assertLess(info['gain'],1)

    def test_learning_off_freezes_internal_weights_and_traces(self):
        p=self.a.plasticity;p.learn(np.ones(5),10,1)
        weights=p.graph.data.copy();mean=p.rate_average.copy();updates=p.updates
        for _ in range(5):p.learn(np.ones(5),10,-1,enabled=False)
        np.testing.assert_array_equal(p.graph.data,weights)
        np.testing.assert_array_equal(p.rate_average,mean)
        self.assertEqual(p.updates,updates)

    def test_bounds_preserve_transmitter_sign_and_zero_activity(self):
        p=self.a.plasticity;p.learning_rate=5
        for _ in range(50):p.learn(np.array([3,3,3,3,0]),10,3)
        self.assertTrue(np.all(p.graph.data>=20));self.assertTrue(np.all(p.graph.data<=60))
        weights=p.graph.data.copy();p.learn(np.zeros(5),10,3)
        np.testing.assert_array_equal(p.graph.data,weights)
        self.assertEqual(self.a.signs[2],-1)

    def test_snapshot_is_reused_and_restores_exact_effective_weights(self):
        p=self.a.plasticity;p.learn(np.array([3,2,1,0,0]),10,.7)
        saved=p.checkpoint(self.root);filename=saved['checkpoint']
        self.assertEqual(p.checkpoint(self.root)['checkpoint'],filename)
        self.b.plasticity.restore(saved,self.root,p.rate_average.copy())
        np.testing.assert_array_equal(self.b.plasticity.graph.data,p.graph.data)
        np.testing.assert_array_equal(self.b.plasticity.rate_average,p.rate_average)
        for core in (p,self.b.plasticity):core.learn(np.array([2,1,3,0,0]),10,-.4)
        np.testing.assert_array_equal(self.b.plasticity.graph.data,p.graph.data)
        invalid=saved | {'checkpoint':'../outside.npy'}
        with self.assertRaises(ValueError):self.b.plasticity.restore(invalid,self.root,p.rate_average)


class FertilityFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.world=OpenWorld()
        self.a=life.initialize(upgrade_body(fresh_body((4,4))),sex='female')
        self.b=life.initialize(upgrade_body(fresh_body((9,4))),sex='male')
        self.a['individual_id']='a';self.b['individual_id']='b'
        self.others=[self.a,self.b]

    def test_conditions_are_explained_and_impossible_mating_not_selected(self):
        status=self.world.reproduction_status(self.a,self.others)
        self.assertGreater(status['drive'],.5);self.assertFalse(status['ready'])
        self.assertEqual(status['nearest_distance'],5)
        self.assertFalse(self.world.allowed_actions(self.a,self.others)[11])
        self.b['position']=[5,4]
        self.assertTrue(self.world.allowed_actions(self.a,self.others)[11])
        self.b['hydration']=10
        self.assertFalse(self.world.allowed_actions(self.a,self.others)[11])
        self.a['hydration']=10
        status=self.world.reproduction_status(self.a,self.others)
        self.assertTrue(any('수분' in text for text in status['blockers']))
        self.assertEqual(status['drive'],0);self.assertGreater(status['inherited_strength'],.5)

    def test_local_mate_direction_is_sensed_but_not_global_position(self):
        start=14+EXTRA_SENSORS.index('mate_n')
        signals=self.world.sense(self.a,self.others)[start:start+4]
        self.assertGreater(signals[1],signals[3])
        self.b['position']=[100,4]
        np.testing.assert_array_equal(self.world.sense(self.a,self.others)[start:start+4],0)

    def test_approaching_a_detected_mate_is_rewarded_and_moving_away_reverses_it(self):
        toward=self.world.act(self.a,1,'live',self.others)['mate_approach_reward']
        away=self.world.act(self.a,3,'live',self.others)['mate_approach_reward']
        self.assertGreater(toward,0);self.assertAlmostEqual(toward,-away)
        self.assertFalse(self.world.allowed_actions(self.a,self.others,False)[11])
        status=self.world.reproduction_status(self.a,self.others,False)
        self.assertIn('개체 수 한도 도달',status['blockers'])
        result=self.world.act(self.a,1,'live',self.others,False)
        self.assertEqual(result['mate_approach_reward'],0)

    def test_birth_requires_real_mating_action_not_just_drive_or_approach(self):
        self.b['position']=[6,4]
        result=self.world.act(self.a,1,'live',self.others)
        self.assertNotIn('reproduction_peer',result)
        result=self.world.act(self.a,11,'live',self.others)
        self.assertEqual(result['reproduction_peer'],'b')


class InternalLabTests(unittest.TestCase):
    def test_v03_readout_migration_preserves_all_600_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);template=fixture(root);modules=Modules(root,template)
            lab=IndividualLab(root,template,modules,threaded=False)
            try:
                p=next(iter(lab.individuals.values()))
                state=p.checkpoint();path=p.directory/state['checkpoint']
                with np.load(path,allow_pickle=False) as values:data={k:values[k].copy() for k in values.files}
                old=np.arange(600,dtype=np.float32).reshape(50,12)/1000
                data['weights']=old;data['eligibility']=np.zeros_like(old)
                data.pop('pair_rate_average');state.pop('plasticity')
                state['feature_names']=state['feature_names'][:50]
                np.savez_compressed(path,**data)
                p.restore(state)
                np.testing.assert_array_equal(p.policy.weights[:50],old)
                self.assertFalse(p.policy.weights[50:].any())
                np.testing.assert_array_equal(p.brain.plasticity.graph.data,template.graph.data)
            finally:lab.close();modules.close();template.close()

    def test_full_learning_path_freeze_restart_and_offspring_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);template=fixture(root);modules=Modules(root,template)
            lab=IndividualLab(root,template,modules,threaded=False)
            try:
                a,b=lab.individuals.values()
                for _ in range(8):lab.tick()
                self.assertGreater(a.brain.plasticity.changed_pairs,0)
                lab.configure(a.id,{'learning':False})
                weights=a.brain.plasticity.graph.data.copy()
                for _ in range(4):lab.tick()
                np.testing.assert_array_equal(a.brain.plasticity.graph.data,weights)
                lab.save()
                restored=IndividualLab(root,template,modules,threaded=False)
                try:
                    lab.tick();restored.tick()
                    for identifier,p in lab.individuals.items():
                        q=restored.individuals[identifier]
                        self.assertEqual(p.body,q.body)
                        np.testing.assert_array_equal(p.brain.plasticity.graph.data,q.brain.plasticity.graph.data)
                        np.testing.assert_array_equal(p.brain.voltage,q.brain.voltage)
                    a.body.update(position=[4,4],sex='female',energy=100,hydration=100,cooldown=0)
                    b.body.update(position=[5,4],sex='male',energy=100,hydration=100,cooldown=0)
                    with patch.object(a.policy,'choose',return_value=11):lab._act(a,[a,b])
                    child=next(p for p in lab.individuals.values() if p not in (a,b))
                    self.assertEqual(child.brain.plasticity.updates,0)
                    np.testing.assert_array_equal(child.brain.plasticity.graph.data,template.graph.data)
                    self.assertFalse(child.policy.weights.any())
                finally:restored.close()
            finally:lab.close();modules.close();template.close()


if __name__=='__main__':unittest.main()

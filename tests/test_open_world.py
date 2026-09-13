"""Observable consequences, lifecycle persistence and legacy checkpoint migration."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from neurons import life
from neurons.catalog import ACTIONS, EXTRA_SENSORS, upgrade_body
from neurons.open_world import OpenWorld
from neurons.world import fresh_body
from neurons.individuals import IndividualLab
from neurons.modules import Modules
from tests.test_neurons import fixture


def body(position=(4,4),identifier='a',sex='female'):
    result=life.initialize(upgrade_body(fresh_body(position)),np.random.default_rng(1),sex=sex)
    result['individual_id']=identifier
    return result


class WorldTests(unittest.TestCase):
    def setUp(self):
        self.world=OpenWorld(1)
        # Controlled affordances; procedural generation has its own tests.
        self.objects={(4,4):'berry',(6,4):'water',(8,4):'branch'}
        self.world.tile=lambda at:(0,False,self.objects.get(tuple(at)))

    def test_inspection_and_consumption_have_distinct_persistent_effects(self):
        p=body();p['energy']=40
        self.world.act(p,4,'live')
        self.assertIn('berry',p['knowledge']);self.assertEqual(p['energy'],40)
        result=self.world.act(p,6,'live')
        self.assertTrue(result['food']);self.assertEqual(p['energy'],64)
        self.assertIsNone(self.world.object_at((4,4)))
        second=self.world.act(p,6,'live')
        self.assertFalse(second['success']);self.assertEqual(p['energy'],64)

    def test_gather_craft_tools_and_shared_structure_consume_real_resources(self):
        p=body((8,4));p['inventory']={'wood':1,'stone':1}
        self.world.act(p,7,'live')
        self.assertEqual(p['inventory']['hammer'],1);self.assertEqual(p['inventory']['stone'],0)
        self.world.act(p,5,'live')
        self.assertEqual(p['inventory']['wood'],2);self.assertEqual(p['tool_uses'],1)
        p['inventory'].update(wood=3,stone=2,fiber=2,basket=1)
        result=self.world.act(p,7,'cooperate')
        self.assertEqual(result['crafted'],'hut');self.assertEqual(p['inventory']['wood'],0)
        other=body((8,4),'b');other['fatigue']=30
        self.world.act(other,8,'live')
        self.assertLess(other['fatigue'],13)

    def test_knowledge_transfer_is_an_explicit_copy_with_provenance(self):
        a=body();b=body((5,4),'b','male')
        self.world.act(a,4,'live');self.assertNotIn('berry',b['knowledge'])
        result=self.world.act(a,9,'cooperate',[a,b])
        self.assertTrue(result['success']);self.assertEqual(b['knowledge']['berry']['teacher'],'a')
        b['knowledge']['berry']['properties']['nutrition']=0
        self.assertEqual(a['knowledge']['berry']['properties']['nutrition'],24)
        self.assertFalse(self.world.act(a,9,'cooperate',[a,b])['success'])

    def test_full_inventory_does_not_destroy_a_shared_resource(self):
        a=body();b=body((5,4),'b');a['inventory']['wood']=1;b['inventory']['wood']=99
        self.assertFalse(self.world.act(a,10,'live',[a,b])['success'])
        self.assertEqual(a['inventory']['wood'],1)
        b['inventory']['wood']=98
        self.assertTrue(self.world.act(a,10,'live',[a,b])['success'])
        self.assertEqual(a['inventory']['wood']+b['inventory']['wood'],99)

    def test_death_has_no_automatic_respawn(self):
        p=body();p.update(energy=0,hydration=0,health=.5)
        result=life.advance(p,1)
        self.assertFalse(p['alive']);self.assertEqual(p['position'],[4,4])
        self.assertEqual(result[-1]['kind'],'death')
        with self.assertRaises(ValueError):self.world.act(p,0,'live')

    def test_fertility_requires_maturity_needs_mate_and_population_space(self):
        a=body();b=body((5,4),'b','male')
        self.assertIn('reproduction_peer',self.world.act(a,11,'live',[a,b]))
        self.assertNotIn('reproduction_peer',self.world.act(a,11,'live',[a,b],False))
        b['energy']=10
        self.assertNotIn('reproduction_peer',self.world.act(a,11,'live',[a,b]))
        a['stage']='larva'
        with self.assertRaises(ValueError):self.world.act(a,11,'live',[a,b])

    def test_sensors_include_needs_and_fit_the_expanded_neural_adapter(self):
        p=body();p['hydration']=25;p['energy']=20
        sensors=self.world.sense(p)
        self.assertEqual(len(sensors),14+len(EXTRA_SENSORS))
        self.assertAlmostEqual(sensors[14+EXTRA_SENSORS.index('thirst')],.75)
        self.assertAlmostEqual(sensors[14+EXTRA_SENSORS.index('hunger')],.8)

    def test_inspected_knowledge_changes_next_sensory_input(self):
        p=body();known=14+EXTRA_SENSORS.index('known_food')
        self.assertEqual(self.world.sense(p)[known],0)
        self.world.act(p,4,'live')
        self.assertEqual(self.world.sense(p)[known],1)

    def test_lifecycle_advances_stages_and_ends_at_individual_lifespan(self):
        a=body();b=body(identifier='b',sex='male')
        egg=life.initialize(upgrade_body(fresh_body()),np.random.default_rng(2),(a,b))
        self.assertEqual(egg['stage'],'egg');self.assertEqual(egg['knowledge'],{})
        for age,stage in [(59,'larva'),(299,'pupa'),(419,'adult')]:
            egg['age']=age;life.advance(egg,age+1);self.assertEqual(egg['stage'],stage)
        egg['age']=int(egg['genome']['lifespan'])
        self.assertEqual(life.advance(egg,20000)[-1]['cause'],'노화')

    def test_procedural_chunks_restore_far_terrain_and_bound_cache(self):
        world=OpenWorld(29);far=world.tile((-80001,60005))
        for i in range(100):world.tile((i*32,i*48))
        self.assertLessEqual(len(world.cache),world.CACHE_LIMIT)
        self.assertEqual(world.tile((-80001,60005)),far)
        p=body((15,0));world.act(p,1,'explore')
        self.assertEqual(p['position'],[16,0])
        world.consume((2,4));world.structures['18,0']={'object':'hut','builder':'a','created':0}
        restored=OpenWorld();restored.restore(copy.deepcopy(world.state()))
        self.assertEqual(restored.tile((-80001,60005)),far)
        self.assertIsNone(restored.object_at((2,4)));self.assertEqual(restored.object_at((18,0)),'hut')
        self.assertLessEqual(len(world.snapshot((5000,-4000),24)['ground']),2401)
        for center,radius in [((float('nan'),0),12),((0,0),1000000)]:
            with self.assertRaises(ValueError):world.snapshot(center,radius)


class LifecycleLabTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.template=fixture(self.root);self.modules=Modules(self.root,self.template)
        self.lab=IndividualLab(self.root,self.template,self.modules,threaded=False)
        self.a,self.b=self.lab.individuals.values()

    def tearDown(self):
        self.lab.close();self.modules.close();self.template.close();self.temp.cleanup()

    def test_neural_action_can_create_offspring_without_copying_memories_or_weights(self):
        a,b=self.a,self.b
        a.body.update(position=[4,4],sex='female');b.body.update(position=[5,4],sex='male')
        a.body['knowledge']['berry']={'name':'열매'};a.policy.weights[:]=1
        with patch.object(a.policy,'choose',return_value=11):self.lab.tick()
        child=next(p for p in self.lab.individuals.values() if p not in (a,b))
        self.assertEqual(child.body['stage'],'egg');self.assertEqual(child.body['generation'],1)
        self.assertEqual(set(child.body['parents']),{a.id,b.id})
        self.assertLess(a.body['energy'],81);self.assertEqual(a.body['offspring'],1)
        self.assertEqual(child.body['knowledge'],{});self.assertFalse(child.policy.weights.any())
        self.assertIsNot(child.body['genome'],a.body['genome']);self.assertNotEqual(child.body['genome'],a.body['genome'])
        self.lab.save();saved=self.lab.store.all_states()
        self.assertEqual(saved[child.id]['body']['parents'],child.body['parents'])
        self.assertTrue(any(e['kind']=='birth' for e in self.lab.store.history(child.id)))

    def test_death_archive_survives_restart_and_does_not_create_replacement_founders(self):
        ids=list(self.lab.individuals)
        for p in self.lab.individuals.values():p.body['age']=18000
        self.lab.tick()
        self.assertEqual(len(self.lab.individuals),0);self.assertEqual(len(self.lab.archives),2)
        self.assertEqual(self.lab.state()['progress']['deaths'],2)
        self.assertTrue(self.lab.detail(ids[0])['archived'])
        restored=IndividualLab(self.root,self.template,self.modules,threaded=False)
        try:
            self.assertEqual(len(restored.individuals),0);self.assertEqual(set(restored.archives),set(ids))
            new=restored.create('다음 시작','live')
            self.assertNotIn(new['id'],ids);self.assertEqual(restored.state()['progress']['deaths'],2)
        finally:restored.close()

    def test_paused_individual_does_not_age_or_act(self):
        self.a.active=False;age=self.a.body['age'];steps=self.a.body['steps']
        for _ in range(3):self.lab.tick()
        self.assertEqual(self.a.body['age'],age);self.assertEqual(self.a.body['steps'],steps)
        self.assertEqual(self.b.body['age'],life.ADULT_AGE+3)

    def test_events_and_checkpoint_are_committed_together(self):
        self.lab.tick()
        self.assertTrue(any(e['kind']=='experience' for e in self.lab.detail(self.a.id)['events']))
        self.assertFalse(any(e['kind']=='experience' for e in self.lab.store.history(self.a.id)))
        with patch.object(self.lab.store,'put_many',side_effect=RuntimeError('transaction failed')):
            with self.assertRaises(RuntimeError):self.lab.save()
        self.assertTrue(self.lab.pending_events)
        self.lab.save()
        self.assertFalse(self.lab.pending_events)
        self.assertTrue(any(e['kind']=='experience' for e in self.lab.store.history(self.a.id)))
        self.assertEqual(self.lab.store.all_states()[self.a.id]['body']['steps'],1)

    def test_legacy_22_by_4_weights_and_neural_state_are_preserved(self):
        p=self.a;legacy=p.checkpoint();path=p.directory/legacy['checkpoint']
        with np.load(path,allow_pickle=False) as values:data={k:values[k].copy() for k in values.files}
        old=np.arange(88,dtype=np.float32).reshape(22,4)/100
        data['weights']=old;data['eligibility']=np.zeros_like(old)
        for key in ('action_spikes','policy_voltage','policy_probabilities'):data[key]=data[key][:4]
        np.savez_compressed(path,**data)
        legacy.pop('feature_names');legacy.pop('action_ids')
        for key in ('genome','alive','stage','age'):legacy['body'].pop(key)
        p.restore(legacy)
        np.testing.assert_array_equal(p.policy.weights[:22,:4],old)
        self.assertFalse(p.policy.weights[22:,:].any())
        np.testing.assert_array_equal(p.policy.probabilities[:4],data['policy_probabilities'])
        self.assertFalse(p.policy.probabilities[4:].any())
        np.testing.assert_array_equal(p.brain.voltage,data['voltage'])
        self.assertEqual(p.body['stage'],'adult')


if __name__=='__main__':unittest.main()

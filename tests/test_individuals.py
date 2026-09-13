"""Causal learning, isolation, restart and LLM-off checks in temporary worlds."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import threading
from http.server import ThreadingHTTPServer
import urllib.error
import urllib.request
import numpy as np

from tests.test_neurons import fixture
from neurons.behavior import SpikingPolicy
from neurons.individuals import IndividualLab
from neurons.modules import Modules
from neurons.storage import IndividualStore, protect
from neurons.world import Arena, fresh_body
from app import make_handler


class IndividualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.template = fixture(self.root)
        self.modules = Modules(self.root, self.template)
        self.lab = IndividualLab(self.root, self.template, self.modules, threaded=False)
        self.first, self.second = self.lab.individuals.values()

    def tearDown(self):
        self.lab.close()
        self.modules.close()
        self.template.close()
        self.temp.cleanup()

    def test_shared_anatomy_but_independent_neural_weights_and_memories(self):
        a, b = self.first, self.second
        self.assertIs(a.brain.graph, b.brain.graph)
        self.assertFalse(np.shares_memory(a.brain.voltage, b.brain.voltage))
        self.assertFalse(np.shares_memory(a.policy.weights, b.policy.weights))
        original = a.brain.graph.data.copy()
        a.memories.learn('루미의 독립된 먹이 경험', 'test')
        for _ in range(8):
            self.lab.tick()
        self.assertEqual(a.body['steps'], 4)
        self.assertEqual(b.body['steps'], 4)
        self.assertGreater(a.policy.state()['changed_synapses'], 0)
        self.assertEqual(b.memories.memories()['count'], 0)
        np.testing.assert_array_equal(original, self.template.graph.data)

    def test_offline_experience_learning_makes_no_network_requests(self):
        self.lab.running = True
        with patch('urllib.request.urlopen', side_effect=AssertionError('LLM/network must stay off')):
            for _ in range(24):
                self.lab.tick()
        self.assertEqual(self.first.policy.updates, 12)
        self.assertGreater(self.first.brain.total_spikes, 0)
        self.assertGreater(len(self.lab.store.history(self.first.id)), 0)

    def test_learning_off_keeps_weights_fixed_while_actions_continue(self):
        self.lab.configure(self.first.id, {'learning': False})
        original = self.first.policy.weights.copy()
        for _ in range(12):
            self.lab.tick()
        np.testing.assert_array_equal(original, self.first.policy.weights)
        self.assertEqual(self.first.policy.updates, 0)
        self.assertEqual(self.first.body['steps'], 6)
        self.assertEqual(self.second.policy.updates, 6)

    def test_saved_state_replays_the_same_next_action_and_neural_step(self):
        for _ in range(9):
            self.lab.tick()
        self.lab.save()
        self.lab.tick()
        expected = {p.id: (p.body.copy(), p.policy.weights.copy(), p.brain.voltage.copy(), p.brain.spike_counts.copy())
                    for p in self.lab.individuals.values()}
        # Restore the saved generation without calling close(), which saves a
        # newer generation. This imitates restarting after the last checkpoint.
        restored = IndividualLab(self.root, self.template, self.modules, threaded=False)
        try:
            restored.tick()
            for identifier, person in restored.individuals.items():
                body, weights, voltage, spikes = expected[identifier]
                self.assertEqual(person.body, body)
                np.testing.assert_array_equal(person.policy.weights, weights)
                np.testing.assert_array_equal(person.brain.voltage, voltage)
                np.testing.assert_array_equal(person.brain.spike_counts, spikes)
        finally:
            restored.close()

    def test_source_replay_uses_own_rewards_and_respects_all_gates(self):
        for _ in range(12):
            self.lab.tick()
        p=self.first
        self.lab.configure(p.id, {'llm_enabled': True})
        lesson={'focus':'odor_progress','reason':'test','memory_id':1,'goal':p.goal,'generation':p.lesson_generation}
        p.pending_lesson=lesson.copy()
        before=p.policy.updates
        self.lab._apply_lesson(p)
        self.assertGreater(p.policy.updates, before)
        p.pending_lesson=lesson.copy()
        before=p.policy.weights.copy()
        self.modules.toggle('language',False)
        self.lab._apply_lesson(p)
        np.testing.assert_array_equal(before,p.policy.weights)
        self.modules.toggle('language',True)
        p.pending_lesson=lesson.copy()
        self.lab.configure(p.id, {'goal':'explore','description':'새 지역을 탐색한다.'})
        self.assertIsNone(p.pending_lesson)

    def test_llm_disabled_preserves_learned_weights_and_blocks_chat(self):
        for _ in range(8):
            self.lab.tick()
        before=self.first.policy.weights.copy()
        self.lab.configure(self.first.id, {'llm_enabled':False})
        with self.assertRaises(ValueError):
            self.lab.chat(self.first.id, '어떤 경험을 했어?')
        np.testing.assert_array_equal(before,self.first.policy.weights)

    def test_queued_research_is_cancelled_after_module_is_switched_off(self):
        p=self.first
        self.lab.running=True
        self.lab.configure(p.id,{'llm_enabled':True})
        p.memories.request_guard()
        self.lab.configure(p.id,{'llm_enabled':False})
        with self.assertRaises(ValueError):p.memories.request_guard()
        with self.assertRaises(ValueError):p.memories.chat_guard()

    def test_http_individual_controls_scope_observation_and_reject_bad_ids(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.template,self.modules,lab=self.lab))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        endpoint=f'http://127.0.0.1:{server.server_port}'
        def request(path,body=None,origin=None):
            headers={'Content-Type':'application/json'}
            if origin:headers['Origin']=origin
            req=urllib.request.Request(endpoint+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
            return urllib.request.urlopen(req,timeout=5)
        try:
            with request('/api/individuals/control',{'action':'step'},endpoint) as response:
                self.assertEqual(json.load(response)['agents'][0]['steps'],1)
            with request('/api/state?agent='+self.first.id) as response:
                self.assertEqual(json.load(response)['sim_ms'],10)
            with request('/api/state') as response:
                self.assertEqual(json.load(response)['sim_ms'],0)
            for path,body,origin,code in [('/api/individuals/no-such-id/config',{},endpoint,400),
                  ('/api/individuals/control',{'action':'start'},'https://example.com',403),
                  ('/api/individuals/'+self.first.id+'/config',{'learning':'yes'},endpoint,400)]:
                with self.assertRaises(urllib.error.HTTPError) as error:request(path,body,origin)
                self.assertEqual(error.exception.code,code)
        finally:
            server.shutdown();server.server_close();thread.join()


class BehaviorTests(unittest.TestCase):
    def test_synaptic_weights_causally_control_actions(self):
        policy=SpikingPolicy(seed=7)
        x=np.zeros(22,dtype=np.float32);x[0]=1
        policy.weights[0,1]=8
        self.assertEqual(policy.choose(x,explore=False),1)
        policy.weights[0,1]=0;policy.weights[0,3]=8
        self.assertEqual(policy.choose(x,explore=False),3)
        self.assertEqual(policy.action_spikes.sum(),2)

    def test_reward_learning_improves_action_probability_without_llm(self):
        policy=SpikingPolicy(seed=11)
        x=np.zeros(22,dtype=np.float32);x[0]=1
        for _ in range(200):
            action=policy.choose(x)
            policy.learn(1 if action==2 else -.2)
        policy.choose(x)
        self.assertGreater(policy.probabilities[2],.85)

    def test_same_move_has_different_goal_reward(self):
        world=Arena(seed=1)
        world.food=[(3,1),(7,7),(8,7)]
        a=fresh_body((1,1));b=fresh_body((1,1))
        first=world.act(a,1,'forage')['reward']
        second=world.act(b,1,'explore')['reward']
        self.assertNotEqual(first,second)
        self.assertEqual(a['position'],b['position'])

    def test_windows_credentials_round_trip(self):
        import os
        if os.name!='nt':self.skipTest('Windows only')
        encrypted=protect('non-secret test credential')
        self.assertNotIn('non-secret',encrypted)
        self.assertEqual(protect(encrypted,True),'non-secret test credential')


if __name__ == '__main__':
    unittest.main()

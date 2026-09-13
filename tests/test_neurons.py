"""Deterministic kernel and persistence tests; never use the user's memory DB."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
import urllib.error
import urllib.request

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from app import make_handler
from neurons.modules import Module, Modules
from neurons.simulation import Simulation


def fixture(directory):
    # Independent excitation and inhibition paths, plus an isolated neuron.
    graph = csr_matrix(([40, 40], ([0, 2], [1, 3])), shape=(5, 5), dtype=np.int32)
    save_npz(directory / 'connections.npz', graph)
    np.save(directory / 'ids.npy', np.arange(100, 105, dtype=np.int64))
    np.save(directory / 'positions.npy', np.zeros((5, 3), dtype=np.float32))
    np.save(directory / 'location-kind.npy', np.ones(5, dtype=np.uint8))
    rows = [dict(index=i, id=100+i, name=f'test-{i}', type='test',
                 **{'class': 'test'}, neurotransmitter='gaba' if i == 2 else 'acetylcholine',
                 position_kind=1) for i in range(5)]
    (directory / 'neurons.json').write_text(json.dumps(rows), encoding='utf-8')
    (directory / 'manifest.json').write_text(json.dumps(dict(neuron_count=5,
        connection_count=2, synaptic_contacts=80)), encoding='utf-8')
    simulation = Simulation(directory)
    simulation.close()  # Step explicitly in tests; no timing races with its daemon.
    return simulation


class IsolatedCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.sim = fixture(self.directory)
        self.modules = Modules(self.directory, self.sim)

    def tearDown(self):
        self.modules.close()
        self.sim.close()
        self.temp.cleanup()


class KernelTests(IsolatedCase):
    def test_no_input_remains_quiet_including_isolated_neuron(self):
        for _ in range(30):
            self.sim.step()
        self.assertEqual(self.sim.total_spikes, 0)
        np.testing.assert_array_equal(self.sim.voltage, np.full(5, -52))

    def test_excitation_arrives_after_two_ms_and_uses_contact_weight(self):
        self.sim.stimulate([100], duration=10)
        np.testing.assert_array_equal(self.sim.step(), [0])
        self.assertEqual(self.sim.step().size, 0)
        np.testing.assert_array_equal(self.sim.step(), [1])
        self.assertEqual(self.sim.total_spikes, 2)
        self.assertEqual(self.sim.last_transmissions, 1)

    def test_single_contact_does_not_act_like_forty(self):
        self.sim.graph.data[0] = 1
        self.sim.stimulate([100], duration=10)
        for _ in range(3):
            self.sim.step()
        self.assertEqual(self.sim.spike_counts[1], 0)
        self.assertAlmostEqual(float(self.sim.voltage[1]), -52 + .275, places=4)

    def test_inhibition_lowers_target_voltage(self):
        self.sim.stimulate([102], duration=10)
        for _ in range(3):
            self.sim.step()
        self.assertAlmostEqual(float(self.sim.voltage[3]), -63.0, places=4)
        self.assertEqual(self.sim.spike_counts[3], 0)

    def test_refractory_enforces_three_ms_between_spikes(self):
        spike_ticks = []
        for tick in range(7):
            self.sim.voltage[4] = 0
            if 4 in self.sim.step():
                spike_ticks.append(tick)
        self.assertEqual(spike_ticks, [0, 3, 6])

    def test_modulator_and_unknown_fast_effect_are_zero(self):
        for name in ('dopamine', 'serotonin', 'octopamine', 'unclear', 'unknown'):
            self.assertEqual(self.sim.transmitter_sign(name), 0)
        self.assertEqual(self.sim.transmitter_sign('histamine'), -1)

    def test_single_step_advances_exactly_ten_ms_and_stays_paused(self):
        self.sim.control('step')
        self.assertEqual(self.sim.ticks, 10)
        self.assertFalse(self.sim.running)


class MemoryTests(IsolatedCase):
    def test_korean_recall_dedup_and_persistence(self):
        result = self.modules.learn('바나나는 노란 과일이다', 'unit-test')
        self.assertEqual(self.modules.recall('바나나는 무슨 색인가')[0]['id'], result['memory_id'])
        repeated = self.modules.learn('바나나는 노란 과일이다')
        self.assertFalse(repeated['created'])
        self.assertEqual(self.modules.memories()['count'], 1)
        self.assertFalse(result['biological_weights_changed'])
        self.modules.close()
        self.modules = Modules(self.directory, self.sim)
        memory = self.modules.memories()['items'][0]
        self.assertEqual(memory['seen'], 2)
        self.assertAlmostEqual(memory['strength'], 1.15)
        self.assertEqual(memory['source'], 'unit-test')

    def test_feedback_changes_recall_ranking(self):
        first = self.modules.learn('사과는 빨간 과일이다')['memory_id']
        second = self.modules.learn('사과는 초록 과일이다')['memory_id']
        self.assertEqual(self.modules.recall('사과')[0]['id'], second)
        self.modules.feedback(first, True)
        self.assertEqual(self.modules.recall('사과')[0]['id'], first)

    def test_growth_disabled_allows_reinforcement_but_not_new_memories(self):
        self.modules.learn('바나나는 노란 과일이다')
        self.modules.toggle('growth', False)
        self.assertFalse(self.modules.learn('바나나는 노란 과일이다')['created'])
        with self.assertRaises(ValueError):
            self.modules.learn('사과는 빨간 과일이다')

    def test_memory_and_language_disable_gates(self):
        self.modules.toggle('memory', False)
        self.assertEqual(self.modules.recall('바나나'), [])
        with self.assertRaises(ValueError):
            self.modules.execute('memory', text='바나나')
        self.modules.toggle('language', False)
        with self.assertRaises(ValueError):
            self.modules.execute('language', text='안녕')

    def test_disabled_stimulus_stays_disabled_after_restart(self):
        self.modules.toggle('periodic', True)
        self.modules.toggle('stimulus', False)
        self.modules.close()
        self.modules = Modules(self.directory, self.sim)
        self.assertFalse(self.sim.periodic_enabled)
        with self.assertRaises(ValueError):
            self.modules.execute('stimulus', identifiers=[100])

    def test_custom_module_dispatch_and_gate(self):
        self.modules.register(Module('example', '검증 모듈', '등록된 함수를 호출'), lambda value: value * 2)
        self.assertEqual(self.modules.execute('example', value=3), 6)
        self.modules.toggle('example', False)
        with self.assertRaises(ValueError):
            self.modules.execute('example', value=3)


class ApiTests(IsolatedCase):
    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.sim, self.modules))
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.endpoint = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()
        super().tearDown()

    def request(self, path, body=None, **headers):
        if body is not None:
            headers.setdefault('Content-Type', 'application/json')
        request = urllib.request.Request(self.endpoint + path,
            data=json.dumps(body).encode() if body is not None else None, headers=headers)
        return urllib.request.urlopen(request, timeout=5)

    def test_api_binary_sizes_and_same_origin_learning(self):
        for path, expected in (('/api/scene', 5*5*4), ('/api/activity', 5)):
            with self.request(path) as response:
                self.assertEqual(len(response.read()), expected)
        with self.request('/api/learn', {'text': '바나나는 노란 과일이다'}, Origin=self.endpoint) as response:
            self.assertTrue(json.load(response)['created'])
        with self.request('/api/export') as response:
            self.assertEqual(json.load(response)['memories']['count'], 1)

    def test_rejects_foreign_origin_and_host(self):
        cases = [('/api/control', {'action': 'start'}, {'Origin': 'https://example.org'}),
                 ('/api/state', None, {'Host': 'example.org'})]
        for path, body, headers in cases:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.request(path, body, **headers)
            self.assertEqual(raised.exception.code, 403)
        self.assertFalse(self.sim.running)

    def test_bad_payload_and_disabled_stimulus(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.request('/api/learn', {'text': None})
        self.assertEqual(raised.exception.code, 400)
        self.modules.toggle('stimulus', False)
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.request('/api/stimulate', {'ids': [100]})
        self.assertEqual(raised.exception.code, 400)
        self.assertFalse(self.sim.running)


if __name__ == '__main__':
    unittest.main()

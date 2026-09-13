from unittest.mock import patch
import unittest

from test_neurons import IsolatedCase
from neurons.modules import Modules
from neurons.research import Research, Cancelled, public_url


class LearningTests(IsolatedCase):
    def test_activation_propagation_retrieves_an_indirect_memory(self):
        self.modules.learn('alpha beta')
        second = self.modules.learn('beta gamma')['memory_id']
        self.assertIn(second, [m['id'] for m in self.modules.recall('alpha')])
        self.modules.toggle('plasticity', False)
        self.assertNotIn(second, [m['id'] for m in self.modules.recall('alpha')])

    def test_connection_weights_change_and_survive_reopening(self):
        memory = self.modules.learn('alpha beta')['memory_id']
        before = self.modules.learning.graph()['connections'][0][2]
        self.modules.feedback(memory, True)
        after = self.modules.learning.graph()['connections'][0][2]
        self.assertGreater(after, before)
        self.modules.close()
        self.modules = Modules(self.directory, self.sim)
        self.assertAlmostEqual(self.modules.learning.graph()['connections'][0][2], after)

    def test_growth_off_preserves_graph_topology(self):
        self.modules.learn('alpha beta')
        before = self.modules.learning.summary()
        self.modules.toggle('growth', False)
        self.modules.learn('alpha beta')
        after = self.modules.learning.summary()
        self.assertEqual((before['nodes'], before['edges']), (after['nodes'], after['edges']))


class ResearchTests(IsolatedCase):
    def setUp(self):
        super().setUp()
        self.research = Research(self.modules)
        self.source = {'title': 'Spike timing dependent plasticity experiment', 'url': 'https://example.org/paper',
                       'snippet': 'test source', 'text': 'The change depends on relative spike times.',
                       'read_level': 'abstract', 'provider': 'unit-test'}
        self.planning = {'question': 'STDP의 발화 시간차는 연결에 어떤 영향을 주는가?', 'query': 'spike timing dependent plasticity',
                         'purpose': '시간차와 연결 변화 조사'}
        self.synthesis = {'finding': '시냅스 연결 변화는 발화 시간차와 관련된다.', 'source_ids': [1],
                          'assessment': 'source_summary', 'uncertainty': '실험 조건 확인 필요',
                          'next_question': '이 규칙을 조절하는 보상 신호에는 무엇이 있는가?', 'question_answered': False}

    def cycle(self, synthesis=None, during_search=None):
        def search(query):
            if during_search:
                during_search()
            return [dict(self.source)], []
        with patch.object(self.modules, 'ollama_status', return_value={'ready': True}), \
             patch.object(self.modules, 'structured', side_effect=[self.planning, synthesis or self.synthesis]), \
             patch.object(self.research.web, 'search', side_effect=search), \
             patch.object(self.research.web, 'read', side_effect=lambda row: dict(row)):
            return self.research.cycle()

    def test_sourced_cycle_persists_memory_and_changes_learning_network(self):
        result = self.cycle()
        self.assertEqual(self.modules.memories()['count'], 1)
        self.assertGreater(self.modules.learning.summary()['edges'], 0)
        self.assertFalse(result['biological_weights_changed'])
        self.assertFalse(result['question_answered'])
        self.assertEqual(self.research.status()['history'][0]['status'], 'completed')

    def test_invalid_citation_is_not_learned(self):
        invalid = dict(self.synthesis, source_ids=[99])
        with self.assertRaises(ValueError):
            self.cycle(invalid)
        self.assertEqual(self.modules.memories()['count'], 0)
        self.assertEqual(self.research.status()['history'][0]['status'], 'failed')

    def test_insufficient_evidence_is_not_learned(self):
        result = self.cycle(dict(self.synthesis, assessment='insufficient'))
        self.assertFalse(result['learning']['stored'])
        self.assertEqual(self.modules.memories()['count'], 0)

    def test_stop_during_search_prevents_learning(self):
        with self.assertRaises(Cancelled):
            self.cycle(during_search=self.research.stop)
        self.assertEqual(self.modules.memories()['count'], 0)
        self.assertEqual(self.research.status()['history'][0]['status'], 'cancelled')

    def test_disabled_web_module_prevents_autonomous_start(self):
        self.modules.toggle('web', False)
        with self.assertRaises(ValueError):
            self.research.start()
        self.assertFalse(self.research.running)

    def test_relevant_paper_precedes_unrelated_accessible_abstract(self):
        irrelevant = {'title': 'General artificial intelligence review', 'abstract': 'A broad review of computing.'}
        self.assertEqual(Research.rank_sources([irrelevant, self.source], self.planning['query'])[0], self.source)

    def test_aggregated_or_snippet_only_sources_do_not_train_memory(self):
        self.assertFalse(Research.can_learn_source(dict(self.source, url='https://deepwiki.com/example')))
        self.assertFalse(Research.can_learn_source(dict(self.source, read_level='search_snippet_only')))
        self.assertTrue(Research.can_learn_source(self.source))


class PublicAddressTests(unittest.TestCase):
    def test_local_addresses_and_credentials_are_rejected(self):
        for url in ('file:///C:/secret', 'http://127.0.0.1/', 'http://[::1]/',
                    'http://user:password@example.org/', 'http://example.org:11434/'):
            with self.assertRaises(ValueError):
                public_url(url)


if __name__ == '__main__':
    unittest.main()

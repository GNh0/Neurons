"""Live research check with an isolated temporary memory DB; saves evidence locally."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_neurons import fixture
from neurons.modules import Modules
from neurons.research import Research


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    simulation = fixture(path)
    modules = Modules(path, simulation)
    research = Research(modules)
    results = []
    try:
        for index in range(2):
            print(f'Live research cycle {index + 1}', flush=True)
            result = research.cycle()
            results.append(result)
            print(json.dumps({'question': result['question'], 'assessment': result['assessment'],
                              'sources': len(result['sources']), 'learning': result.get('learning')}, ensure_ascii=True), flush=True)
        evidence = {'cycles': results, 'learning_graph': modules.learning.summary(),
                    'memory_count': modules.memories()['count'], 'isolated_database': True}
        assert evidence['memory_count'] > 0, 'No source-based memories were learned'
        assert evidence['learning_graph']['nodes'] > 0, 'No learning nodes were created'
        assert results[0]['question'] != results[1]['question'], 'Question did not change'
        (ROOT / 'artifacts' / 'research-live-validation.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(evidence['learning_graph']), flush=True)
    finally:
        (ROOT / 'artifacts' / 'research-live-diagnostics.json').write_text(
            json.dumps(research.status(), ensure_ascii=False, indent=2), encoding='utf-8')
        modules.close()
        simulation.close()

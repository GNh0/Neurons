"""Bounded full-connectome smoke run; not an intelligence or civilization benchmark."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neurons.individuals import IndividualLab
from neurons.simulation import Simulation
from neurons.modules import Modules

def fingerprint(graph):
    return hashlib.sha256(memoryview(graph.data)).hexdigest()

def main():
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='neurons-openworld-validation-') as temporary:
        root=Path(temporary)
        template=Simulation(ROOT/'.data'/'malecns'/'compiled',threaded=False)
        modules=Modules(root,template)
        before=fingerprint(template.graph)
        lab=IndividualLab(root,template,modules,threaded=False)
        lab.running=True
        with patch('urllib.request.urlopen',side_effect=AssertionError('Offline run must not request any network')):
            for step in range(120):
                lab.tick()
                if step%30==29:print(f'{step+1}/120 world turns',flush=True)
        lab.running=False;lab.save()
        public=lab.state()
        expected={p.id:(p.body['age'],p.body['steps'],p.policy.weights.copy()) for p in lab.individuals.values()}
        lab.close()
        restored=IndividualLab(root,template,modules,threaded=False)
        import numpy as np
        for identifier,(age,steps,weights) in expected.items():
            p=restored.individuals[identifier]
            assert p.body['age']==age and p.body['steps']==steps
            np.testing.assert_array_equal(p.policy.weights,weights)
        assert before==fingerprint(template.graph)
        report={'kind':'full_graph_integration_smoke','autonomous_intelligence_validated':False,
                'world_turns':120,'llm_enabled':False,'network_calls':0,'restart_exact':True,
                'original_weights_unchanged':True,'model':public['model'],'progress':public['progress'],
                'agents':[{k:p[k] for k in ('name','steps','places','brain_spikes','neural_learning','life','knowledge','inventory')} for p in public['agents']],
                'wall_seconds':round(time.perf_counter()-started,2)}
        restored.close();modules.close();template.close()
    target=ROOT/'artifacts'/'open-world-validation.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('kind','world_turns','restart_exact','original_weights_unchanged','progress','wall_seconds')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

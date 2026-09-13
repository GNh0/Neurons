"""Full MaleCNS offline integration; does not use or mutate the user's world."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neurons.individuals import IndividualLab
from neurons.modules import Modules
from neurons.simulation import Simulation


def main():
    started=time.perf_counter()
    template=Simulation(ROOT/'.data/malecns/compiled',threaded=False)
    original=hashlib.sha256(memoryview(template.graph.data)).hexdigest()
    with tempfile.TemporaryDirectory(prefix='neurons-internal-qa-') as directory:
        root=Path(directory);modules=Modules(root,template)
        lab=IndividualLab(root,template,modules,threaded=False)
        timings=[]
        try:
            lab.running=True
            with patch('urllib.request.urlopen',side_effect=AssertionError('Offline validation requested network')):
                for step in range(120):
                    before=time.perf_counter();lab.tick();timings.append(time.perf_counter()-before)
                    if step%20==19:print(json.dumps({'turn':step+1,'seconds':round(sum(timings),2)},ensure_ascii=False),flush=True)
            lab.running=False;lab.save()
            saved={p.id:(p.brain.plasticity.graph.data.copy(),p.brain.plasticity.rate_average.copy(),p.policy.weights.copy())
                   for p in lab.individuals.values()}
            restored=IndividualLab(root,template,modules,threaded=False)
            try:
                for identifier,(weights,mean,policy) in saved.items():
                    q=restored.individuals[identifier]
                    np.testing.assert_array_equal(q.brain.plasticity.graph.data,weights)
                    np.testing.assert_array_equal(q.brain.plasticity.rate_average,mean)
                    np.testing.assert_array_equal(q.policy.weights,policy)
                lab.tick();restored.tick()
                for identifier,p in lab.individuals.items():
                    q=restored.individuals[identifier]
                    assert p.body==q.body
                    np.testing.assert_array_equal(p.brain.plasticity.graph.data,q.brain.plasticity.graph.data)
                    np.testing.assert_array_equal(p.brain.voltage,q.brain.voltage)
                public=lab.state()
                assert all(p['internal_learning']['changed_pairs']>0 for p in public['agents'])
                assert original==hashlib.sha256(memoryview(template.graph.data)).hexdigest()
                report=dict(kind='full_connectome_internal_plasticity_integration',world_turns=121,network_calls=0,
                            restart_exact=True,next_action_exact=True,anatomy_unchanged=True,
                            progress=public['progress'],model=public['model'],
                            agents=[{k:p[k] for k in ('name','steps','brain_spikes','internal_learning','reproduction')} for p in public['agents']],
                            action_seconds_median=float(np.median(timings)),action_seconds_p95=float(np.percentile(timings,95)),
                            private_pair_array_bytes=int(template.graph.nnz*4),
                            snapshot_bytes=sum(p.stat().st_size for p in root.rglob('*.npy')),
                            autonomous_intelligence_validated=False,learning_performance_improvement_validated=False,
                            wall_seconds=round(time.perf_counter()-started,2))
            finally:restored.close()
        finally:lab.close();modules.close();template.close()
    (ROOT/'artifacts/internal-plasticity-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('agents','model')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

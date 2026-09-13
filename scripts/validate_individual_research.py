"""Bounded real local-LLM research/replay check in a disposable private world."""
import json
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neurons.simulation import Simulation
from neurons.modules import Modules
from neurons.individuals import IndividualLab

with tempfile.TemporaryDirectory(prefix='neurons-research-qa-') as temporary:
    root=Path(temporary)
    template=Simulation(ROOT/'.data'/'malecns'/'compiled',threaded=False)
    modules=Modules(root,template)
    lab=IndividualLab(root,template,modules,threaded=False,defaults=False)
    state=lab.create('검증 개체','forage',seed=11)
    p=lab.individuals[state['id']]
    for _ in range(20): lab.tick()
    lab.configure(p.id,{'llm_enabled':True})
    lab.running=True
    deadline=time.time()+240
    old_phase=''
    while time.time()<deadline:
        lab.tick()
        phase=p.research.phase+' / '+p.lesson_status
        if phase!=old_phase:
            print(phase,flush=True);old_phase=phase
        if p.replay_updates:
            break
        if p.research.session_cycles and not p.research.running and not (p.lesson_thread and p.lesson_thread.is_alive()) and not p.pending_lesson:
            break
        time.sleep(1)
    lab.running=False
    p.research.stop()
    result=lab.detail(p.id)
    (ROOT/'artifacts'/'individual-research-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'replay_updates':p.replay_updates,'memories':p.memories.memories()['count'],
                     'question':p.research.current_question,'last_error':p.research.last_error},ensure_ascii=False),flush=True)
    lab.close();modules.close();template.close()

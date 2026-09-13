"""Full-graph browser QA in an isolated temporary world and memory database."""
from pathlib import Path
import sys
import tempfile
import argparse
from http.server import ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import make_handler
from neurons.simulation import Simulation
from neurons.modules import Modules
from neurons.research import Research
from neurons.individuals import IndividualLab

parser=argparse.ArgumentParser()
parser.add_argument('--lifecycle-fixture',action='store_true',help='Explicit UI fixtures; never user data or autonomous performance evidence')
args=parser.parse_args()
with tempfile.TemporaryDirectory(prefix='neurons-world-qa-') as temporary:
    root=Path(temporary)
    template=Simulation(ROOT/'.data'/'malecns'/'compiled')
    modules=Modules(root,template)
    research=Research(modules)
    modules.research=research
    lab=IndividualLab(root,template,modules)
    if args.lifecycle_fixture:
        a,b=lab.individuals.values()
        a.body.update(position=[4,4],sex='female');b.body.update(position=[5,4],sex='male')
        a.body['inventory'].update(wood=2,stone=1,berry=2)
        a.body['knowledge']['berry']={'name':'열매','position':[2,4],'kind':'food','properties':{'nutrition':24},'learned_by':'inspection'}
        lab._birth(a,b)
        elder=lab.create('생애 기록 예시','live')
        lab.individuals[elder['id']].body['age']=18000
        lab.tick();lab.save()
        lab.last_error='UI 검증용 임시 세계 · 출생·사망 사례는 시험 조건으로 구성됨'
    server=ThreadingHTTPServer(('127.0.0.1',8878),make_handler(template,modules,research,lab))
    server.daemon_threads=True
    print('Isolated full-graph world: http://127.0.0.1:8878/',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        lab.close()
        research.stop()
        template.close()
        modules.close()
        server.server_close()

"""Local-only HTTP API and 3D monitor for the complete classified MaleCNS graph."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import sys
from urllib.parse import urlparse, parse_qs

from neurons.modules import Modules
from neurons.simulation import Simulation
from neurons.research import Research
from neurons.individuals import IndividualLab

ROOT = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent


def make_handler(simulation, modules, research=None, lab=None, lab_error=''):
    chat_gate = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if not self.path.startswith(('/api/state', '/api/activity', '/api/individuals')):
                super().log_message(format, *args)

        def send(self, payload, status=200, content_type='application/json; charset=utf-8', extra=None):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            for key, value in (extra or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.valid_host():
                return self.send({'error': '로컬 접속만 지원합니다.'}, 403)
            parsed = urlparse(self.path)
            path, query = parsed.path, parse_qs(parsed.query)
            try:
                observed = lab.brain(query['agent'][0]) if lab and query.get('agent') else simulation
                if path == '/api/health':
                    return self.send({'ready': True, 'app': 'Neurons', 'version': '0.3.0', 'neurons': simulation.n})
                if path == '/api/state':
                    payload = {**observed.state(), **modules.summary()}
                    if lab and query.get('agent'):
                        payload['running'] = lab.running and lab.individuals[query['agent'][0]].active
                    return self.send(payload)
                if path == '/api/scene':
                    return self.send(simulation.scene(), content_type='application/octet-stream')
                if path == '/api/edges':
                    return self.send(simulation.overview_edges(), content_type='application/octet-stream')
                if path == '/api/activity':
                    return self.send(observed.activity_bytes(), content_type='application/octet-stream')
                if path == '/api/search':
                    return self.send(simulation.search(query.get('q', ['DNp01'])[0][:100]))
                if path.startswith('/api/neuron/'):
                    return self.send(observed.neuron(int(path.rsplit('/', 1)[1])))
                if path.startswith('/api/index/'):
                    index = int(path.rsplit('/', 1)[1])
                    if not 0 <= index < simulation.n:
                        raise ValueError('뉴런 인덱스 범위 오류')
                    return self.send(observed.neuron(int(simulation.ids[index])))
                if path == '/api/individuals':
                    center=(float(query['x'][0]),float(query['y'][0])) if 'x' in query and 'y' in query else None
                    return self.send(lab.state(query.get('focus',[None])[0],center,int(query.get('radius',[12])[0])) if lab else {'ready': False, 'error': lab_error or '개체 세계가 준비되지 않았습니다.'})
                if path.startswith('/api/individuals/') and lab:
                    parts = path.strip('/').split('/')
                    if len(parts) == 3:
                        return self.send(lab.detail(parts[2]))
                    if len(parts) == 4 and parts[3] == 'conversation':
                        return self.send(lab.individuals[parts[2]].memories.conversation())
                if path == '/api/memories':
                    return self.send(modules.memories())
                if path == '/api/conversation':
                    return self.send(modules.conversation())
                if path == '/api/llm':
                    return self.send(modules.ollama_status())
                if path == '/api/learning-graph':
                    with modules.lock:
                        return self.send(modules.learning.graph())
                if path == '/api/research' and research:
                    return self.send(research.status())
                if path == '/api/export':
                    return self.send({'simulation': simulation.state(), 'memories': modules.memories(),
                                      'conversation': modules.conversation(),
                                      'research': research.status() if research else None},
                                     extra={'Content-Disposition': 'attachment; filename="neurons-observation.json"'})
                static_files = {'/': ('index.html', 'text/html; charset=utf-8'),
                                '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                                '/launch.js': ('launch.js', 'text/javascript; charset=utf-8'),
                                '/research.js': ('research.js', 'text/javascript; charset=utf-8'),
                                '/world.js': ('world.js', 'text/javascript; charset=utf-8'),
                                '/world.css': ('world.css', 'text/css; charset=utf-8'),
                                '/space.js': ('space.js', 'text/javascript; charset=utf-8'),
                                '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if path in static_files:
                    filename, mime = static_files[path]
                    return self.send((ROOT / 'web' / filename).read_bytes(), content_type=mime)
                return self.send({'error': '찾을 수 없는 경로'}, 404)
            except (ValueError, KeyError, IndexError) as error:
                return self.send({'error': str(error)}, 400)
            except Exception as error:
                return self.send({'error': '요청 처리 중 오류가 발생했습니다.', 'detail': str(error)}, 500)

        def do_POST(self):
            origin = self.headers.get('Origin')
            accepted_origins = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if not self.valid_host() or (origin and origin not in accepted_origins):
                return self.send({'error': '로컬 앱에서 실행해 주세요.'}, 403)
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send({'error': 'JSON 요청이 필요합니다.'}, 415)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 100000:
                    return self.send({'error': '요청 크기 제한'}, 413)
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('JSON 객체가 필요합니다.')
                path = urlparse(self.path).path
                if path.startswith('/api/individuals/'):
                    if not lab:
                        return self.send({'error': lab_error or '개체 세계가 준비되지 않았습니다.'}, 503)
                    if path == '/api/individuals/control':
                        lab.control(body.get('action'))
                        return self.send(lab.state())
                    if path == '/api/individuals/create':
                        return self.send(lab.create(body.get('name'), body.get('goal'), body.get('description', '')))
                    parts = path.strip('/').split('/')
                    if len(parts) == 4 and parts[3] == 'config':
                        return self.send(lab.configure(parts[2], body))
                    if len(parts) == 4 and parts[3] == 'chat':
                        if not chat_gate.acquire(blocking=False):
                            return self.send({'error': '이전 대화 응답을 기다려 주세요.'}, 409)
                        try:
                            return self.send(lab.chat(parts[2], body.get('text', '')))
                        finally:
                            chat_gate.release()
                if path == '/api/control':
                    simulation.control(body.get('action'))
                    return self.send({'ok': True})
                if path == '/api/research' and research:
                    if body.get('action') == 'stop':
                        research.stop()
                        return self.send(research.status())
                    if body.get('action') == 'start':
                        return self.send(research.start(body.get('goal'), body.get('limit', 3), body.get('interval', 15)))
                    raise ValueError('지원하지 않는 탐구 명령입니다.')
                if path == '/api/stimulate':
                    return self.send(modules.execute('stimulus', identifiers=body.get('ids'), cell_type=body.get('type'),
                                                         preset=body.get('preset'), amplitude=body.get('amplitude', 12),
                                                         duration=body.get('duration', 100)))
                if path == '/api/module':
                    modules.toggle(body.get('id'), body.get('enabled'))
                    if lab and body.get('id') in ('language', 'web') and not body.get('enabled'):
                        lab.stop_research()
                    return self.send(modules.summary())
                if path == '/api/learn':
                    return self.send(modules.execute('memory', text=body.get('text', ''), source=body.get('source', '직접 입력')))
                if path == '/api/feedback':
                    if not isinstance(body.get('positive'), bool):
                        raise ValueError('피드백 값을 확인해 주세요.')
                    modules.feedback(body.get('id'), body['positive'])
                    return self.send({'ok': True})
                if path == '/api/model':
                    status = modules.ollama_status()
                    if body.get('model') not in status['models']:
                        raise ValueError('설치된 로컬 모델을 선택해 주세요.')
                    with modules.lock:
                        modules.model = body['model']
                        modules.set_setting('model', modules.model)
                    return self.send({'ok': True})
                if path == '/api/chat':
                    if not chat_gate.acquire(blocking=False):
                        return self.send({'error': '이전 대화 응답을 기다려 주세요.'}, 409)
                    try:
                        return self.send(modules.execute('language', text=body.get('text', '')))
                    finally:
                        chat_gate.release()
                return self.send({'error': '찾을 수 없는 경로'}, 404)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                return self.send({'error': str(error)}, 400)
            except Exception as error:
                return self.send({'error': '요청 처리 중 오류가 발생했습니다.', 'detail': str(error)}, 500)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--configure-postgres', action='store_true')
    args = parser.parse_args()
    if args.configure_postgres:
        from scripts.configure_postgres import run
        run()
        return
    compiled = ROOT / '.data' / 'malecns' / 'compiled'
    if not (compiled / 'manifest.json').exists():
        raise SystemExit('Run scripts/fetch_malecns.py and scripts/prepare_connectome.py first.')
    print('Loading full MaleCNS graph...', flush=True)
    simulation = Simulation(compiled)
    modules = Modules(ROOT / '.data', simulation)
    research = Research(modules)
    modules.research = research
    lab, lab_error = None, ''
    try:
        lab = IndividualLab(ROOT, simulation, modules)
    except Exception as error:
        # An explicit PostgreSQL setup failing is surfaced, never replaced with
        # another store containing new blank individuals.
        lab_error = '개체 저장소를 열지 못했습니다. PostgreSQL 연결 설정과 저장 상태를 확인해 주세요. (' + type(error).__name__ + ')'
        print(lab_error, flush=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(simulation, modules, research, lab, lab_error))
    server.daemon_threads = True
    print(f'Neurons 0.3.0 | {simulation.n:,} neurons | http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        research.stop()
        if lab:
            lab.close()
        simulation.close()
        server.server_close()


if __name__ == '__main__':
    main()

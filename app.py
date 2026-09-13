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

ROOT = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent


def make_handler(simulation, modules, research=None):
    chat_gate = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if self.path not in ('/api/state', '/api/activity'):
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
                if path == '/api/health':
                    return self.send({'ready': True, 'app': 'Neurons', 'version': '0.2.0', 'neurons': simulation.n})
                if path == '/api/state':
                    return self.send({**simulation.state(), **modules.summary()})
                if path == '/api/scene':
                    return self.send(simulation.scene(), content_type='application/octet-stream')
                if path == '/api/edges':
                    return self.send(simulation.overview_edges(), content_type='application/octet-stream')
                if path == '/api/activity':
                    return self.send(simulation.activity_bytes(), content_type='application/octet-stream')
                if path == '/api/search':
                    return self.send(simulation.search(query.get('q', ['DNp01'])[0][:100]))
                if path.startswith('/api/neuron/'):
                    return self.send(simulation.neuron(int(path.rsplit('/', 1)[1])))
                if path.startswith('/api/index/'):
                    index = int(path.rsplit('/', 1)[1])
                    if not 0 <= index < simulation.n:
                        raise ValueError('뉴런 인덱스 범위 오류')
                    return self.send(simulation.neuron(int(simulation.ids[index])))
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
    args = parser.parse_args()
    compiled = ROOT / '.data' / 'malecns' / 'compiled'
    if not (compiled / 'manifest.json').exists():
        raise SystemExit('Run scripts/fetch_malecns.py and scripts/prepare_connectome.py first.')
    print('Loading full MaleCNS graph...', flush=True)
    simulation = Simulation(compiled)
    modules = Modules(ROOT / '.data', simulation)
    research = Research(modules)
    modules.research = research
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(simulation, modules, research))
    server.daemon_threads = True
    print(f'Neurons 0.2.0 | {simulation.n:,} neurons | http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        research.stop()
        simulation.close()
        server.server_close()


if __name__ == '__main__':
    main()

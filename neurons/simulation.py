"""Full classified MaleCNS graph; count-weighted current-based LIF dynamics.

Every classified neuron has independent state. Every supplied internal pair is
loaded. Pair aggregation preserves summed linear impulses, not synapse-specific
biochemistry, locations, plasticity, or a complete biological emulation.
"""
from collections import Counter, deque
import json
from pathlib import Path
import threading
import time
import numpy as np
from scipy.sparse import load_npz


class Simulation:
    def __init__(self, directory: Path = None, *, shared=None, threaded=True):
        if shared is None:
            self.manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            self.neurons = json.loads((directory / 'neurons.json').read_text(encoding='utf-8'))
            self.ids = np.load(directory / 'ids.npy')
            self.positions = np.load(directory / 'positions.npy')
            self.location_kind = np.load(directory / 'location-kind.npy')
            self.graph = load_npz(directory / 'connections.npz')  # rows pre, columns post
            self.incoming = self.graph.tocsc()
        else:
            # Only anatomy is shared. Voltage, spikes, delays and input state are
            # allocated below for every individual; the graph is never trained here.
            for name in ('manifest', 'neurons', 'ids', 'positions', 'location_kind', 'graph', 'incoming'):
                setattr(self, name, getattr(shared, name))
        self.n = len(self.ids)
        self.id_index = {int(identifier): index for index, identifier in enumerate(self.ids)}
        self.classes = sorted(set(n['class'] for n in self.neurons))
        self.class_index = np.array([self.classes.index(n['class']) for n in self.neurons], dtype=np.uint8)
        self.class_counts = Counter(n['class'] for n in self.neurons)
        self.transmitter_counts = Counter(n['neurotransmitter'] for n in self.neurons)
        self.signs = np.array([self.transmitter_sign(n['neurotransmitter']) for n in self.neurons], dtype=np.float32)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.running = False
        self.dt_ms, self.tau_ms = 1.0, 20.0
        self.rest, self.threshold, self.reset_voltage = -52.0, -45.0, -52.0
        self.refractory_steps, self.delay_steps = 3, 2
        self.gain = .275
        self.voltage = np.full(self.n, self.rest, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.spike_counts = np.zeros(self.n, dtype=np.uint64)
        self.activity = np.zeros(self.n, dtype=np.float32)
        self.delay_queue = deque([np.empty(0, dtype=np.int32) for _ in range(self.delay_steps)])
        self.pending_stimuli = []
        self.ticks = self.total_spikes = self.last_spikes = self.last_transmissions = 0
        self.step_ms = 0.0
        self.history, self.events = deque(maxlen=180), deque(maxlen=160)
        self.event_seq, self.monitored = 0, 0
        self.voltage_history = deque(maxlen=180)
        self.periodic_enabled = False
        self.sensory_indices = np.flatnonzero(np.array([n['class'] == 'ol_sensory' for n in self.neurons]))[:64]
        self.add_event('dataset', 'MaleCNS v1.0 전체 연결 로드', {
            'neurons': self.n, 'connections': int(self.graph.nnz), 'synaptic_contacts': int(self.graph.sum())})
        self.thread = None
        if threaded:
            self.thread = threading.Thread(target=self._loop, daemon=True, name='connectome-simulation')
            self.thread.start()

    @staticmethod
    def transmitter_sign(name):
        if name in ('gaba', 'glutamate', 'histamine'):
            return -1.0
        if name == 'acetylcholine':
            return 1.0
        return 0.0  # No invented fast effect for modulators / unknown identities.

    def add_event(self, kind, label, details=None):
        self.event_seq += 1
        event = {'id': self.event_seq, 'kind': kind, 'label': label, 'time': time.time(),
                 'sim_ms': self.ticks * self.dt_ms, 'details': details or {}}
        self.events.appendleft(event)
        return event

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                if not self.running:
                    self.stop_event.wait(.04)
                    continue
                started = time.perf_counter()
                with self.lock:
                    for _ in range(5):
                        if not self.running:
                            break
                        self.step()
                self.stop_event.wait(max(.001, .01 - (time.perf_counter() - started)))
        except Exception as error:
            with self.lock:
                self.running = False
                self.add_event('error', '시뮬레이션 오류로 정지', {'error': str(error)})

    def step(self):
        started = time.perf_counter()
        delayed = self.delay_queue.popleft()
        self.refractory = np.maximum(self.refractory - 1, 0)
        self.voltage += (self.rest - self.voltage) * np.float32(self.dt_ms / self.tau_ms)
        eligible = self.refractory == 0
        transmissions = 0
        if len(delayed):
            rows = self.graph[delayed]
            signs = np.repeat(self.signs[delayed], np.diff(rows.indptr))
            signal = np.bincount(rows.indices, weights=rows.data * signs * self.gain, minlength=self.n)
            self.voltage[eligible] += signal[eligible].astype(np.float32)
            transmissions = int(rows.nnz)
        for stimulus in self.pending_stimuli:
            if stimulus['remaining'] > 0 and stimulus['remaining'] % 10 == 0:
                indices = stimulus['indices']
                self.voltage[indices[eligible[indices]]] += stimulus['amplitude']
            stimulus['remaining'] -= 1
        self.pending_stimuli = [s for s in self.pending_stimuli if s['remaining'] > 0]
        if self.periodic_enabled and self.ticks % 100 == 0:
            indices = self.sensory_indices
            self.voltage[indices[eligible[indices]]] += 12.0
        np.clip(self.voltage, -100.0, 100.0, out=self.voltage)
        fired = np.flatnonzero(eligible & (self.voltage >= self.threshold)).astype(np.int32)
        self.voltage[fired] = self.reset_voltage
        self.refractory[fired] = self.refractory_steps
        self.spike_counts[fired] += 1
        self.activity *= .97
        self.activity[fired] = 1.0
        self.delay_queue.append(fired)
        self.ticks += 1
        self.total_spikes += len(fired)
        self.last_spikes, self.last_transmissions = len(fired), transmissions
        if self.ticks % 5 == 0:
            self.history.append([self.ticks * self.dt_ms, len(fired)])
            self.voltage_history.append([self.ticks * self.dt_ms, float(self.voltage[self.monitored]), bool(self.monitored in fired)])
        elapsed = (time.perf_counter() - started) * 1000
        self.step_ms = elapsed if not self.step_ms else self.step_ms * .9 + elapsed * .1
        return fired

    def control(self, action):
        with self.lock:
            if action == 'start':
                self.running = True
            elif action == 'pause':
                self.running = False
            elif action == 'step':
                self.running = False
                for _ in range(10):
                    self.step()
            else:
                raise ValueError('지원하지 않는 실행 명령입니다.')
            self.add_event('control', {'start': '시뮬레이션 실행', 'pause': '시뮬레이션 일시정지', 'step': '10 ms 단일 실행'}[action])

    def stimulate(self, identifiers=None, cell_type=None, preset=None, amplitude=12.0, duration=100):
        if not 0 < float(amplitude) <= 30 or not 10 <= int(duration) <= 2000:
            raise ValueError('자극은 0–30 mV, 지속 시간은 10–2000 ms여야 합니다.')
        with self.lock:
            if identifiers:
                indices = np.array([self.id_index[int(x)] for x in identifiers], dtype=np.int32)
            elif cell_type:
                indices = np.array([n['index'] for n in self.neurons if n['type'].lower() == cell_type.lower()], dtype=np.int32)
            elif preset == 'visual':
                indices = self.sensory_indices.copy()
            else:
                indices = np.array([n['index'] for n in self.neurons if n['type'] == 'DNp01'], dtype=np.int32)
            indices = np.unique(indices)
            if not len(indices):
                raise ValueError('자극할 뉴런을 찾지 못했습니다.')
            if len(indices) > 2048 or len(self.pending_stimuli) >= 20:
                raise ValueError('한 번에 최대 2,048개 뉴런, 최대 20개 동시 자극을 지원합니다.')
            self.pending_stimuli.append({'indices': indices, 'amplitude': float(amplitude), 'remaining': int(duration)})
            self.running = True
            return self.add_event('stimulus', f'{len(indices):,}개 뉴런에 자극', {
                'neuron_ids': self.ids[indices[:30]].tolist(), 'target_count': len(indices),
                'amplitude_mv': float(amplitude), 'duration_ms': int(duration), 'pulse_interval_ms': 10,
                'mapping': '지정 계산 뉴런에 직접 자극; 실제 감각 경험 재현을 뜻하지 않음'})

    def search(self, query, limit=35):
        needle = query.strip().lower()
        matches = []
        for n in self.neurons:
            if needle in (n['name'] + ' ' + n['type'] + ' ' + str(n['id']) + ' ' + n['class']).lower():
                matches.append(n)
                if len(matches) >= limit:
                    break
        return matches

    def _partners(self, indices, weights, direction, limit=180):
        order = np.argsort(weights)[::-1][:limit]
        return [{'index': int(indices[i]), 'id': int(self.ids[indices[i]]), 'name': self.neurons[indices[i]]['name'],
                 'contacts': int(weights[i]), 'direction': direction} for i in order]

    def neuron(self, identifier):
        index = self.id_index[int(identifier)]
        with self.lock:
            if self.monitored != index:
                self.monitored = index
                self.voltage_history.clear()
            node = dict(self.neurons[index])
            a, b = self.graph.indptr[index:index + 2]
            c, d = self.incoming.indptr[index:index + 2]
            node.update({'voltage': round(float(self.voltage[index]), 3), 'spikes': int(self.spike_counts[index]),
                         'activity': float(self.activity[index]), 'outgoing_count': int(b - a), 'incoming_count': int(d - c),
                         'outgoing_contacts': int(self.graph.data[a:b].sum()), 'incoming_contacts': int(self.incoming.data[c:d].sum()),
                         'position': self.positions[index].tolist() if self.location_kind[index] else None, 'sign': float(self.signs[index]),
                         'outgoing': self._partners(self.graph.indices[a:b], self.graph.data[a:b], 'out'),
                         'incoming': self._partners(self.incoming.indices[c:d], self.incoming.data[c:d], 'in')})
            return node

    def scene(self):
        return np.column_stack([self.positions, self.class_index, self.location_kind]).astype(np.float32).tobytes()

    def overview_edges(self, limit=16000):
        rng = np.random.default_rng(176)
        candidates = np.unique(rng.integers(0, self.graph.nnz, size=limit * 2))
        sources = np.searchsorted(self.graph.indptr, candidates, side='right') - 1
        destinations = self.graph.indices[candidates]
        valid = (self.location_kind[sources] > 0) & (self.location_kind[destinations] > 0)
        return np.column_stack([sources[valid], destinations[valid], self.graph.data[candidates[valid]]])[:limit].astype(np.uint32).tobytes()

    def activity_bytes(self):
        with self.lock:
            return (np.minimum(self.activity, 1.0) * 255).astype(np.uint8).tobytes()

    def state(self):
        with self.lock:
            return {'manifest': self.manifest, 'running': self.running, 'sim_ms': self.ticks * self.dt_ms,
                    'ticks': self.ticks, 'total_spikes': self.total_spikes, 'last_spikes': self.last_spikes,
                    'active_neurons': int((self.activity > .1).sum()), 'step_wall_ms': round(self.step_ms, 3),
                    'simulation_speed': round(self.dt_ms / self.step_ms, 3) if self.step_ms else 0,
                    'last_transmitted_pairs': self.last_transmissions, 'history': list(self.history),
                    'voltage_history': list(self.voltage_history), 'events': list(self.events)[:45],
                    'classes': self.classes, 'class_counts': dict(self.class_counts), 'nt_counts': dict(self.transmitter_counts),
                    'model': {'name': 'LIF · MaleCNS v1.0', 'dt_ms': self.dt_ms, 'rest_mv': self.rest,
                              'threshold_mv': self.threshold, 'tau_ms': self.tau_ms, 'delay_ms': self.delay_steps * self.dt_ms,
                              'refractory_ms': self.refractory_steps * self.dt_ms, 'gain_mv': self.gain,
                              'zero_fast_effect_neurons': int((self.signs == 0).sum()),
                              'assumptions': ['선형 시냅스 개수 가중치', '단일 구획 LIF', 'GABA·글루탐산·히스타민 억제 가정',
                                              '조절성 전달물질·미상은 빠른 전류 효과 0', '배경 발화 없음', '개별 시냅스 상태·수상돌기 생략']}}

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

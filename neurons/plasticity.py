"""Per-individual plastic efficacy on every existing, count-aggregated pair.

This is an experimental reward-modulated rate rule, not biological STDP.
The anatomical counts/indices stay shared and immutable. Only effective current
weights are private. No contacts, neurons or new anatomical edges are invented.
"""
from pathlib import Path
import uuid
import numpy as np
from scipy.sparse import csr_matrix


class PairPlasticity:
    VERSION = 1
    MIN_GAIN, MAX_GAIN = .5, 1.5
    TARGET_RATE = .08  # spikes per ms, measured over each actual 10 ms action
    RATE_ALPHA = .02
    HOMEOSTASIS = .01
    ANCHOR = .0005

    def __init__(self, graph, signs):
        self.base = graph
        self.signs = signs
        self.graph = csr_matrix((graph.data.astype(np.float32), graph.indices, graph.indptr),
                                shape=graph.shape, copy=False)
        self.rate_average = np.zeros(graph.shape[0], dtype=np.float32)
        self.learning_rate = .02
        self.updates = self.changed_pairs = self.pair_updates = 0
        self.change_l1 = 0.
        self.last = dict(changed=0, change_l1=0., reward_component_l1=0., homeostasis_component_l1=0.)
        self.examples = []
        self.checkpoint_name = None
        self.saved_updates = -1

    def learn(self, spike_counts, duration_ms, reward_error, enabled=True):
        """Process ALL outgoing pairs of active neurons, in bounded work buffers.

        No sampled subset or fixed 600-edge limit. Silent sources have zero local
        eligibility; unknown/modulatory transmitter sources retain their weights.
        The reward term and activity regulation are separately reported.
        """
        if not enabled:
            return dict(changed=0, change_l1=0., reward_component_l1=0., homeostasis_component_l1=0.)
        counts = np.asarray(spike_counts)
        if (counts.shape != self.rate_average.shape or duration_ms <= 0
                or not np.isfinite(counts).all() or (counts < 0).any() or not np.isfinite(reward_error)):
            raise ValueError('내부 가소성의 발화 기록과 보상을 확인해 주세요.')
        rates = (counts / duration_ms).astype(np.float32)
        error = float(np.clip(reward_error, -3, 3))
        active = np.flatnonzero((rates > 0) & (self.signs != 0))
        report = dict(changed=0, change_l1=0., reward_component_l1=0., homeostasis_component_l1=0.)
        candidates = []
        for offset in range(0, len(active), 512):
            sources = active[offset:offset+512]
            starts = self.base.indptr[sources]
            lengths = self.base.indptr[sources+1] - starts
            total = int(lengths.sum())
            if not total:
                continue
            # Map this bounded CSR row group to the immutable anatomical indices.
            shifts = starts.astype(np.int64) - (np.cumsum(lengths, dtype=np.int64)-lengths)
            edges = np.repeat(shifts, lengths) + np.arange(total, dtype=np.int64)
            posts = self.base.indices[edges]
            pre = np.repeat(rates[sources], lengths)
            sign = np.repeat(self.signs[sources], lengths)
            post = rates[posts]
            base = self.base.data[edges].astype(np.float32)
            old = self.graph.data[edges].copy()
            gain = old/base
            reward = self.learning_rate * error * pre * (post-self.rate_average[posts]) * sign
            regulation = self.HOMEOSTASIS * pre * (self.TARGET_RATE-post) * sign
            regulation -= self.ANCHOR * pre * (gain-1)
            updated = base * np.clip(gain+reward+regulation, self.MIN_GAIN, self.MAX_GAIN)
            self.graph.data[edges] = updated
            old_delta, new_delta = np.abs(old/base-1), np.abs(updated/base-1)
            self.changed_pairs += int(np.count_nonzero(new_delta > 1e-6)-np.count_nonzero(old_delta > 1e-6))
            self.change_l1 += float(new_delta.sum(dtype=np.float64)-old_delta.sum(dtype=np.float64))
            difference = np.abs((updated-old)/base)
            report['changed'] += int(np.count_nonzero(difference))
            report['change_l1'] += float(difference.sum(dtype=np.float64))
            report['reward_component_l1'] += float(np.abs(reward).sum(dtype=np.float64))
            report['homeostasis_component_l1'] += float(np.abs(regulation).sum(dtype=np.float64))
            self.pair_updates += total
            if difference.max(initial=0) > 0:
                selected = np.argpartition(difference, -min(3,total))[-3:]
                for j in selected:
                    if difference[j] > 0:
                        edge = int(edges[j])
                        source = int(np.searchsorted(self.base.indptr, edge, side='right')-1)
                        candidates.append(dict(source=source, target=int(posts[j]), contacts=int(base[j]),
                                               gain=float(updated[j]/base[j]), before_gain=float(gain[j]),
                                               change=float(difference[j])))
        self.rate_average *= 1-self.RATE_ALPHA
        self.rate_average += self.RATE_ALPHA*rates
        self.updates += 1
        self.last = report
        self.examples = sorted(candidates, key=lambda e:e['change'], reverse=True)[:8]
        return report.copy()

    def state(self):
        return dict(kind='reward_modulated_rate_plasticity', version=self.VERSION,
                    pair_capacity=int(self.base.nnz), changed_pairs=self.changed_pairs,
                    updates=self.updates, pair_updates=self.pair_updates,
                    gain_change_l1=max(0., self.change_l1), last=self.last.copy(),
                    examples=[e.copy() for e in self.examples],
                    anatomical_counts_changed=False, learning_rate=self.learning_rate,
                    min_gain=self.MIN_GAIN, max_gain=self.MAX_GAIN,
                    granularity='aggregated_neuron_pair', new_connections=0)

    def checkpoint(self, directory):
        # Large arrays use a separate, lossless .npy snapshot. Repeated saves with
        # no learning reuse it; compressing 100 MB per individual would stall UI.
        if self.updates and self.saved_updates != self.updates:
            filename = 'pair-'+uuid.uuid4().hex+'.npy'
            with (Path(directory)/filename).open('wb') as stream:
                np.save(stream, self.graph.data, allow_pickle=False)
            self.checkpoint_name = filename
            self.saved_updates = self.updates
        return self.state() | {'checkpoint':self.checkpoint_name}

    def restore(self, state, directory, rate_average):
        if not state:
            return  # v0.3: keep all existing neural/readout state, start efficacy at anatomy.
        if state['version'] != self.VERSION or state['pair_capacity'] != self.base.nnz:
            raise ValueError('내부 가소성 저장 형식 또는 연결 크기가 다릅니다.')
        filename = state.get('checkpoint')
        if filename:
            if Path(filename).name != filename or not filename.startswith('pair-') or not filename.endswith('.npy'):
                raise ValueError('잘못된 내부 시냅스 상태 경로')
            data = np.load(Path(directory)/filename, mmap_mode='r', allow_pickle=False)
            if data.shape != self.graph.data.shape or data.dtype != np.float32:
                raise ValueError('내부 시냅스 상태 크기 또는 형식 오류')
            for start in range(0, data.size, 1_000_000):
                part = data[start:start+1_000_000]
                base = self.base.data[start:start+1_000_000]
                if (not np.isfinite(part).all() or (part < base*self.MIN_GAIN-1e-5).any()
                        or (part > base*self.MAX_GAIN+1e-5).any()):
                    raise ValueError('내부 시냅스 가중치 범위 오류')
                self.graph.data[start:start+1_000_000] = part
            del data
        elif state['updates']:
            raise ValueError('내부 시냅스 체크포인트가 없습니다.')
        if rate_average.shape != self.rate_average.shape or not np.isfinite(rate_average).all():
            raise ValueError('내부 가소성 활동 평균 오류')
        self.rate_average[:] = rate_average
        self.updates = state['updates']
        self.changed_pairs, self.pair_updates = state['changed_pairs'], state['pair_updates']
        self.change_l1 = state['gain_change_l1']
        self.last, self.examples = state['last'], state['examples']
        self.checkpoint_name, self.saved_updates = filename, self.updates

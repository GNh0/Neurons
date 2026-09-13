"""Experience-driven spiking action readout; no LLM calls or action scripts.

The original connectome is a fixed LIF reservoir. Added experimental readout
synapses are trained by a three-factor reward rule. This is NOT biological STDP
or a claim that every MaleCNS synapse is plastic.
"""
import numpy as np


class SpikingPolicy:
    ACTIONS = ('north', 'east', 'south', 'west')

    def __init__(self, inputs=22, seed=0, actions=None, learning_rate=.12):
        self.rng = np.random.default_rng(seed)
        self.actions = tuple(actions or self.ACTIONS)
        count = len(self.actions)
        self.learning_rate = learning_rate
        self.weights = np.zeros((inputs, count), dtype=np.float32)
        self.initial_weights = self.weights.copy()
        self.eligibility = np.zeros_like(self.weights)
        self.baseline = 0.0
        self.updates = 0
        self.voltage = np.zeros(count, dtype=np.float32)
        self.probabilities = np.full(count, 1/count, dtype=np.float32)
        self.action_spikes = np.zeros(count, dtype=np.int64)
        self.last_error = 0.0

    def choose(self, presynaptic_activity, explore=True, allowed=None):
        x = np.asarray(presynaptic_activity, dtype=np.float32)
        if x.shape != (self.weights.shape[0],) or not np.isfinite(x).all():
            raise ValueError('행동 회로 입력을 확인해 주세요.')
        # Integrate presynaptic rates into escape-noise action neurons.
        # The first stochastic spike wins. Exponential races implement softmax.
        self.voltage = x @ self.weights
        mask = np.ones(len(self.actions),dtype=bool) if allowed is None else np.asarray(allowed,dtype=bool)
        if mask.shape != self.voltage.shape or not mask.any():
            raise ValueError('실행 가능한 행동이 필요합니다.')
        rates = np.zeros_like(self.voltage)
        rates[mask] = np.exp(self.voltage[mask] - self.voltage[mask].max()).clip(1e-8, None)
        self.probabilities = rates / rates.sum()
        indices = np.flatnonzero(mask)
        action = int(indices[np.argmin(self.rng.exponential(1/rates[mask]))]) if explore else int(self.rng.choice(np.flatnonzero(mask & (self.voltage == self.voltage[mask].max()))))
        self.action_spikes[action] += 1
        post = -self.probabilities.copy()
        post[action] += 1
        self.eligibility = np.outer(x, post)
        return action

    def learn(self, reward, enabled=True):
        if not enabled:
            return 0.0
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError('보상은 유한한 수여야 합니다.')
        error = np.clip(reward - self.baseline, -3, 3)
        before = self.weights.copy()
        # pre activity × (observed post spike - expected post spike) × reward error
        self.weights += np.float32(self.learning_rate * error) * self.eligibility
        np.clip(self.weights, -4, 4, out=self.weights)
        self.baseline += .03 * (reward-self.baseline)
        self.updates += 1
        self.last_error = float(error)
        return float(np.abs(self.weights-before).sum())

    def state(self):
        difference = self.weights - self.initial_weights
        return {'kind': 'experimental_spiking_readout', 'action_neurons': len(self.actions), 'actions': self.actions,
                'plastic_synapses': int(self.weights.size), 'changed_synapses': int(np.count_nonzero(abs(difference) > 1e-6)),
                'weight_change_l1': round(float(abs(difference).sum()), 5), 'updates': self.updates,
                'probabilities': self.probabilities.tolist(), 'voltage': self.voltage.tolist(),
                'action_spikes': self.action_spikes.tolist(), 'reward_prediction_error': self.last_error,
                'weights': self.weights.round(4).tolist(), 'base_connectome_weights_changed': False}

    def replay(self, activity, action, reward, old_probability, allowed=None):
        x = np.asarray(activity, dtype=np.float32)
        # Legacy four-action experiences keep their original feature positions.
        if x.size < self.weights.shape[0]:
            x = np.pad(x, (0,self.weights.shape[0]-x.size))
        if x.shape != (self.weights.shape[0],) or not np.isfinite(x).all() or not 0 <= int(action) < len(self.actions):
            raise ValueError('저장된 복습 경험의 회로 크기를 확인해 주세요.')
        voltage = x @ self.weights
        mask=np.ones(len(self.actions),dtype=bool) if allowed is None else np.asarray(allowed,dtype=bool)
        if mask.shape!=voltage.shape or not mask[int(action)]:raise ValueError('복습 행동 범위 오류')
        rates = np.zeros_like(voltage)
        rates[mask] = np.exp(voltage[mask]-voltage[mask].max()).clip(1e-8,None)
        probability = rates/rates.sum()
        post = -probability
        post[int(action)] += 1
        ratio = min(2, float(probability[int(action)])/max(float(old_probability), 1e-6))
        self.eligibility = np.outer(x, post)*ratio
        return self.learn(reward)


class NeuralAdapter:
    """Explicit synthetic sensor mapping onto real, independently simulated cells."""
    SENSOR_NAMES = ('smell_n','smell_e','smell_s','smell_w',
                    'free_n','free_e','free_s','free_w',
                    'novel_n','novel_e','novel_s','novel_w','energy','bias')

    def __init__(self, brain, seed=0, extra_sensors=()):
        self.brain = brain
        self.sensor_names = self.SENSOR_NAMES + tuple(extra_sensors)
        self.feature_names = self.SENSOR_NAMES + tuple('pool_'+str(i) for i in range(8)) + tuple(extra_sensors)
        count = len(self.sensor_names)
        self.rng = np.random.default_rng(seed)
        sensory = np.flatnonzero(np.array([n['class'] in ('cb_sensory', 'ol_sensory') for n in brain.neurons]))
        if len(sensory) < count*4:
            sensory = np.arange(brain.n)
        self.input_cells = np.resize(sensory, count*4).reshape(count, 4)
        self.reservoir_pools = [np.arange(i, brain.n, 8) for i in range(8)]
        self.last_input = np.zeros(len(self.feature_names), dtype=np.float32)

    def observe(self, sensors):
        stimulus = np.asarray(sensors, dtype=np.float32)
        if stimulus.shape != (len(self.sensor_names),) or not np.isfinite(stimulus).all():
            raise ValueError('감각 입력을 확인해 주세요.')
        stimulus = np.clip(stimulus, 0, 1)
        brain = self.brain
        with brain.lock:
            before = brain.spike_counts.copy()
            for _ in range(10):
                selected = self.input_cells[self.rng.random(self.input_cells.shape) < stimulus[:, None]*.5]
                selected = np.unique(selected)
                brain.voltage[selected[brain.refractory[selected] == 0]] += 12
                brain.step()
            counts = (brain.spike_counts - before).astype(np.float32)
            sensory = np.clip(counts[self.input_cells].mean(axis=1)/3, 0, 1)
            reservoir = np.array([counts[p].mean()/3 if len(p) else 0 for p in self.reservoir_pools], dtype=np.float32)
            # Opponent population coding removes common activity that otherwise
            # overwhelms directional differences. These remain measured neural
            # features; world distances and rewards never bypass the neurons.
            directional = sensory[:12].reshape(3, 4)
            directional -= directional.mean(axis=1, keepdims=True)
            sensory[12:14] *= .15
            sensory[14:] *= .5
            reservoir *= .1
            self.last_input = np.concatenate([sensory[:14], reservoir, sensory[14:]]).astype(np.float32)
        return self.last_input.copy()

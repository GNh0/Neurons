"""Integrity plus a bounded benchmark against the complete compiled dataset."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurons.simulation import Simulation


def main():
    started = time.perf_counter()
    simulation = Simulation(ROOT / '.data' / 'malecns' / 'compiled')
    simulation.close()
    graph = simulation.graph
    graph.check_format(full_check=True)
    assert simulation.n == 166700
    assert graph.nnz == 25582938
    assert int(graph.sum()) == 124177617
    assert graph.has_canonical_format
    assert np.all(graph.data > 0)
    assert np.unique(simulation.ids).size == simulation.n
    assert np.isfinite(simulation.positions).all()
    assert int((simulation.location_kind > 0).sum()) == 140638
    assert len(simulation.voltage) == simulation.n
    assert all(n['index'] == i and n['id'] == int(simulation.ids[i])
               for i, n in enumerate(simulation.neurons))
    loaded_seconds = time.perf_counter() - started
    simulation.stimulate(cell_type='DNp01', duration=100)
    started = time.perf_counter()
    for _ in range(100):
        simulation.step()
    wall_seconds = time.perf_counter() - started
    assert simulation.total_spikes > 0
    assert int((simulation.spike_counts > 0).sum()) > 2
    assert int(graph.sum()) == 124177617  # Dynamic activity leaves measured contacts unchanged.
    result = {
        'neuron_count': simulation.n, 'weighted_pairs': graph.nnz,
        'synaptic_contacts': int(graph.sum()), 'positioned_neurons': 140638,
        'zero_fast_effect_neurons': int((simulation.signs == 0).sum()),
        'graph_load_and_validation_seconds': round(loaded_seconds, 3),
        'benchmark_stimulus': 'Both DNp01 cells; 12 mV every 10 ms for 100 ms',
        'simulated_ms': simulation.ticks, 'wall_seconds': round(wall_seconds, 3),
        'total_spikes': simulation.total_spikes,
        'neurons_that_fired': int((simulation.spike_counts > 0).sum()),
        'simulation_to_wall_ratio': round(.1 / wall_seconds, 4),
        'biological_validation': False,
        'note': 'A computation smoke test, not a reproduction of experimental fly behavior. Other apps may be running.'
    }
    output = ROOT / 'artifacts' / 'full-graph-validation.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

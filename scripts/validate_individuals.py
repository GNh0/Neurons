"""Held-out arena evaluation using the full graph, without any LLM or search.

Both trained and zero-weight policies use stochastic choices on matched seeds.
Evaluation freezes learning and resets neuronal/body state in every arena.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurons.behavior import NeuralAdapter, SpikingPolicy
from neurons.simulation import Simulation
from neurons.world import Arena, fresh_body


def episode(template, goal, seed, steps, weights=None, train=False):
    brain = Simulation(shared=template, threaded=False)
    adapter, policy = NeuralAdapter(brain, seed), SpikingPolicy(seed=seed)
    if weights is not None:
        policy.weights[:] = weights
    world = Arena(seed=seed)
    body = fresh_body(world.empty_cell())
    for _ in range(steps):
        activity = adapter.observe(world.sense(body))
        action = policy.choose(activity)
        reward = world.act(body, action, goal)['reward']
        if train:
            policy.learn(reward)
    brain.close()
    return policy.weights.copy(), {k: body[k] for k in ('food','collisions','steps','return')} | {
        'places': len(body['visits']), 'spikes': brain.total_spikes,
        'changed_weights': policy.state()['changed_synapses']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=600)
    parser.add_argument('--evaluate', type=int, default=120)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    start = time.perf_counter()
    template = Simulation(ROOT/'.data'/'malecns'/'compiled', threaded=False)
    results = {'neurons': template.n, 'pairs': int(template.graph.nnz), 'llm_requests': 0,
               'train_steps_per_goal': args.train, 'evaluation_steps': args.evaluate, 'goals': {}}
    for goal in ('forage','explore'):
        weights = None
        for part in range(3):
            weights, training = episode(template, goal, 11+part, args.train//3, weights, True)
            print(json.dumps({'goal': goal, 'training_part': part+1, **training}), flush=True)
        pairs = []
        for seed in range(101, 101+args.seeds):
            _, baseline = episode(template, goal, seed, args.evaluate)
            _, trained = episode(template, goal, seed, args.evaluate, weights)
            pairs.append({'seed': seed, 'untrained': baseline, 'trained': trained})
            print(json.dumps({'goal': goal, **pairs[-1]}), flush=True)
        results['goals'][goal] = pairs
    results['seconds'] = round(time.perf_counter()-start, 2)
    directory = ROOT/'artifacts'
    directory.mkdir(exist_ok=True)
    (directory/'individual-validation.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(results), flush=True)
    template.close()


if __name__ == '__main__':
    main()

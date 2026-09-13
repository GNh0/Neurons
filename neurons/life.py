"""Explicit experimental life rules, not calibrated fruit-fly biology.

Age advances in world turns, never wall time. Genetic traits are inherited;
learned readout weights and personal knowledge are not copied into offspring.
"""
import copy
import numpy as np

STAGES = {'egg': '알', 'larva': '유충', 'pupa': '번데기', 'adult': '성체', 'dead': '사망'}
ADULT_AGE = 420
TRAITS = {
    'lifespan': (6000., 18000., 12000.),
    'metabolism': (.65, 1.4, 1.),
    'curiosity': (.3, 1.5, 1.),
    'sociality': (.3, 1.5, 1.),
    'reproductive_drive': (.3, 1.5, 1.),
    'learning_rate': (.04, .18, .12),
}


def genome(rng, parents=()):
    result = {}
    for key, (low, high, default) in TRAITS.items():
        baseline = sum(p['genome'][key] for p in parents)/len(parents) if parents else default
        variation = float(rng.normal(0, (high-low)*.025))
        result[key] = float(np.clip(baseline+variation, low, high))
    return result


def initialize(body, rng=None, parents=(), tick=0, sex=None):
    if 'genome' in body:
        return body
    rng = rng if rng is not None else np.random.default_rng(0)
    body.update(alive=True, stage='egg' if parents else 'adult',
                age=0 if parents else ADULT_AGE, born_tick=tick,
                sex=sex or ('female' if int(rng.integers(2)) == 0 else 'male'),
                generation=1+max((p['generation'] for p in parents), default=-1),
                parents=[p['individual_id'] for p in parents], genome=genome(rng, parents),
                offspring=0, cooldown=0, loneliness=0., death_tick=None, death_cause=None)
    return body


def ready(body):
    return (body.get('alive', True) and body.get('stage', 'adult') == 'adult'
            and body.get('cooldown', 0) == 0 and body['energy'] >= 60
            and body.get('hydration', 100) >= 50 and body.get('health', 100) >= 60)


def compatible(a, b):
    return a is not b and ready(a) and ready(b) and a['sex'] != b['sex']


def drives(body):
    initialize(body)
    if not body['alive']:
        return {k: 0. for k in ('hunger','thirst','rest','safety','curiosity','social','reproduction')}
    return dict(hunger=1-body['energy']/100, thirst=1-body.get('hydration',100)/100,
                rest=body.get('fatigue',0)/100, safety=1-body.get('health',100)/100,
                curiosity=min(1., body['genome']['curiosity']*.6),
                social=min(1., body.get('loneliness',0)/100*body['genome']['sociality']),
                reproduction=min(1., body['genome']['reproductive_drive']*.65) if ready(body) else 0.)


def die(body, tick, cause):
    if not body['alive']:
        return None
    body.update(alive=False, stage='dead', death_tick=tick, death_cause=cause, health=0.)
    return dict(kind='death', cause=cause, age=body['age'], generation=body['generation'],
                observation=f"사망 · {cause}", world_tick=tick)


def advance(body, tick):
    """One world turn of age and metabolism for one enabled individual."""
    initialize(body)
    if not body['alive']:
        return []
    body['age'] += 1
    body['cooldown'] = max(0, body['cooldown']-1)
    previous = body['stage']
    age = body['age']
    body['stage'] = 'egg' if age < 60 else 'larva' if age < 300 else 'pupa' if age < ADULT_AGE else 'adult'
    events = []
    if previous != body['stage']:
        events.append(dict(kind='growth', stage=body['stage'], age=age, world_tick=tick,
                           observation=STAGES[body['stage']]+' 단계로 성장'))
    # Eggs and pupae use stored reserves; larvae and adults must find food/water.
    factor = body['genome']['metabolism'] * (.08 if body['stage'] in ('egg','pupa') else 1.)
    body['energy'] = max(0., body['energy']-.06*factor)
    body['hydration'] = max(0., body.get('hydration',100)-.07*factor)
    body['fatigue'] = min(100., body.get('fatigue',0)+.04*factor)
    body['loneliness'] = min(100., body['loneliness']+.12)
    damage = (.7 if body['energy'] <= 0 else 0)+(.9 if body['hydration'] <= 0 else 0)
    damage += .12 if body['fatigue'] >= 100 else 0
    body['health'] = max(0., body.get('health',100)-damage)
    cause = None
    if age >= body['genome']['lifespan']:
        cause = '노화'
    elif body['health'] <= 0:
        cause = '탈수' if body['hydration'] <= 0 else '굶주림' if body['energy'] <= 0 else '건강 고갈'
    if cause:
        events.append(die(body, tick, cause))
    return events


def public(body):
    initialize(body)
    return {**{k: copy.deepcopy(body[k]) for k in ('alive','stage','age','born_tick','sex','generation',
             'parents','genome','offspring','cooldown','death_tick','death_cause')},
            'stage_name': STAGES[body['stage']], 'drives': drives(body),
            'lifespan': round(body['genome']['lifespan']), 'mature_age': ADULT_AGE}

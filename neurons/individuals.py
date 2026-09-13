"""Independent neuronal individuals, a shared arena and optional research support."""
from collections import deque
import copy
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from neurons.behavior import NeuralAdapter, SpikingPolicy
from neurons.modules import Modules
from neurons.research import Research
from neurons.simulation import Simulation
from neurons.storage import IndividualStore, database_dsn
from neurons.world import GOALS, fresh_body
from neurons.open_world import OpenWorld
from neurons.catalog import ACTIONS, EXTRA_SENSORS, OBJECTS, RECIPES, upgrade_body
from neurons import life


class Individual:
    def __init__(self, directory, template, identifier, name, goal, description, seed, llm_lock):
        self.id, self.name, self.goal = identifier, name, goal
        self.description = description or GOALS[goal]['description']
        self.directory = Path(directory)/identifier
        self.directory.mkdir(parents=True, exist_ok=True)
        self.brain = Simulation(shared=template, threaded=False, plastic=True)
        self.adapter = NeuralAdapter(self.brain, seed, EXTRA_SENSORS)
        self.policy = SpikingPolicy(len(self.adapter.feature_names), seed, [a['id'] for a in ACTIONS])
        self.body = life.initialize(upgrade_body(fresh_body()), np.random.default_rng(seed))
        self.body['individual_id'] = identifier
        self.policy.learning_rate = self.body['genome']['learning_rate']
        self.brain.plasticity.learning_rate = .02*self.policy.learning_rate/.12
        self.active, self.learning, self.llm_enabled = True, True, False
        self.memories = Modules(self.directory, self.brain)
        self.memories.llm_lock = llm_lock  # Share model scheduling, never memory.
        self.research = Research(self.memories, context_provider=self.context)
        self.memories.research = self.research
        self.experiences = deque(maxlen=200)
        self.recent = deque(maxlen=60)
        self.last_research = 0
        self.handled_research = 0
        self.lesson_thread = None
        self.pending_lesson = None
        self.lesson_status = '경험 학습 중 · 자료 보조 꺼짐'
        self.replay_updates = 0
        self.lesson_generation = 0
        self.memories.individual_context = self.context

    def context(self):
        measured=self.policy.state()
        return {'name': self.name, 'goal': self.description, 'feedback': GOALS[self.goal]['name'],
                'steps': self.body['steps'], 'food': self.body['food'], 'collisions': self.body['collisions'],
                'places': len(self.body['visits']), 'available_sensors': list(self.adapter.sensor_names),
                'body': life.public(self.body), 'inventory': dict(self.body['inventory']),
                'knowledge': {key:{k:v for k,v in value.items() if k in ('name','position','learned_by','teacher')}
                              for key,value in self.body['knowledge'].copy().items()},
                'available_actions': ACTIONS,
                'internal_learning': self.internal_state(),
                'learning_enabled': self.learning, 'neural_learning': {k:measured[k] for k in
                     ('kind','action_neurons','plastic_synapses','changed_synapses','weight_change_l1','updates','probabilities')},
                'recent_experiences': [{k:v for k,v in event.items() if k in
                     ('step','action_name','observation','reward','weight_change','collision','new_place')}
                     for event in list(self.recent)[-6:]],
                'environment': '절차 생성 지형과 사물, 자원, 제작, 교류, 생애 주기를 가진 실험 세계. 실제 초파리 생리와 행동의 재현은 아님.'}

    def metadata(self):
        return {'id': self.id, 'name': self.name, 'goal': self.goal, 'description': self.description,
                'active': self.active, 'learning': self.learning, 'llm_enabled': self.llm_enabled,
                'body': self.body, 'baseline': self.policy.baseline, 'updates': self.policy.updates,
                'feature_names': self.adapter.feature_names, 'action_ids': self.policy.actions,
                'reward_prediction_error': self.policy.last_error,
                'policy_rng': self.policy.rng.bit_generator.state, 'adapter_rng': self.adapter.rng.bit_generator.state,
                'ticks': self.brain.ticks, 'total_spikes': self.brain.total_spikes,
                'handled_research': self.handled_research, 'replay_updates': self.replay_updates,
                'experiences': list(self.experiences), 'recent': list(self.recent),
                'base_neurons': self.brain.n, 'base_pairs': self.brain.graph.nnz}

    def checkpoint(self):
        token = uuid.uuid4().hex
        path = self.directory/('state-'+token+'.npz')
        plasticity = self.brain.plasticity.checkpoint(self.directory)
        np.savez_compressed(path, voltage=self.brain.voltage, refractory=self.brain.refractory,
                            spike_counts=self.brain.spike_counts, activity=self.brain.activity,
                            delay0=self.brain.delay_queue[0], delay1=self.brain.delay_queue[1],
                            weights=self.policy.weights, eligibility=self.policy.eligibility,
                            action_spikes=self.policy.action_spikes, policy_voltage=self.policy.voltage,
                            policy_probabilities=self.policy.probabilities,
                            pair_rate_average=self.brain.plasticity.rate_average)
        return {**self.metadata(), 'checkpoint': path.name, 'plasticity':plasticity}

    def restore(self, state):
        if state['base_neurons'] != self.brain.n or state['base_pairs'] != self.brain.graph.nnz:
            raise ValueError('저장된 개체와 연결지도 크기가 다릅니다.')
        filename = state['checkpoint']
        if Path(filename).name != filename or not filename.startswith('state-'):
            raise ValueError('잘못된 신경 상태 경로')
        with np.load(self.directory/filename, allow_pickle=False) as data:
            for name in ('voltage', 'refractory', 'spike_counts', 'activity'):
                getattr(self.brain, name)[:] = data[name]
            self.brain.delay_queue = deque([data['delay0'].copy(), data['delay1'].copy()])
            rows = state.get('feature_names', self.adapter.feature_names[:data['weights'].shape[0]])
            actions = state.get('action_ids', self.policy.actions[:data['weights'].shape[1]])
            row_map = [self.adapter.feature_names.index(k) for k in rows]
            col_map = [self.policy.actions.index(k) for k in actions]
            for key in ('weights','eligibility'):
                getattr(self.policy,key)[np.ix_(row_map,col_map)] = data[key]
            self.policy.action_spikes[col_map] = data['action_spikes']
            self.policy.voltage[col_map] = data['policy_voltage']
            self.policy.probabilities.fill(0)
            self.policy.probabilities[col_map] = data['policy_probabilities']
            self.brain.plasticity.restore(state.get('plasticity'), self.directory,
                data['pair_rate_average'] if 'pair_rate_average' in data else np.zeros(self.brain.n,dtype=np.float32))
        self.body = life.initialize(upgrade_body(state['body']),np.random.default_rng(int(self.id[:8],16)))
        self.body['individual_id'] = self.id
        self.policy.learning_rate = self.body['genome']['learning_rate']
        self.brain.plasticity.learning_rate = .02*self.policy.learning_rate/.12
        self.active, self.learning, self.llm_enabled = (state[k] for k in ('active', 'learning', 'llm_enabled'))
        self.policy.baseline, self.policy.updates = state['baseline'], state['updates']
        self.policy.last_error = state.get('reward_prediction_error', 0)
        self.policy.rng.bit_generator.state = state['policy_rng']
        self.adapter.rng.bit_generator.state = state['adapter_rng']
        self.brain.ticks, self.brain.total_spikes = state['ticks'], state['total_spikes']
        self.handled_research = state.get('handled_research', 0)
        self.replay_updates = state.get('replay_updates', 0)
        self.experiences.extend(state.get('experiences', []))
        self.recent.extend(state.get('recent', []))

    def internal_state(self):
        state=self.brain.plasticity.state()
        for edge in state['examples']:
            edge.update(source_id=int(self.brain.ids[edge['source']]),target_id=int(self.brain.ids[edge['target']]))
        return state

    def state(self, detail=False):
        result = {'id': self.id, 'name': self.name, 'goal': self.goal, 'goal_name': GOALS[self.goal]['name'],
                  'description': self.description, 'active': self.active, 'learning': self.learning,
                  'llm_enabled': self.llm_enabled, 'position': self.body['position'],
                  'energy': round(self.body['energy'], 2), 'steps': self.body['steps'],
                  'food': self.body['food'], 'collisions': self.body['collisions'],
                  'places': len(self.body['visits']), 'return': round(self.body['return'], 3),
                  'action': self.body['last_action'], 'reward': self.body['last_reward'],
                  'brain_neurons': self.brain.n, 'brain_ms': self.brain.ticks,
                  'brain_spikes': self.brain.total_spikes, 'neural_learning': self.policy.state(),
                  'internal_learning': self.internal_state(),
                  'replay_updates': self.replay_updates, 'lesson_status': self.lesson_status,
                  'life': life.public(self.body), 'hydration': round(self.body['hydration'],2),
                  'health': round(self.body['health'],2), 'fatigue': round(self.body['fatigue'],2),
                  'inventory': self.body['inventory'], 'knowledge': self.body['knowledge'],
                  'investigations': self.body['investigations'], 'gathered': self.body['gathered'],
                  'crafted': self.body['crafted'], 'tool_uses': self.body['tool_uses'],
                  'communications': self.body['communications'], 'shared': self.body['shared'],
                  'received': self.body['received'], 'observation': self.body['last_observation'],
                  'recent': list(self.recent)}
        if detail:
            result.update(memory=self.memories.memories(), research=self.research.status(),
                          visits=self.body['visits'], sensor_cells=self.brain.ids[self.adapter.input_cells].tolist())
        return copy.deepcopy(result)


class IndividualLab:
    MAX_POPULATION = 8

    def __init__(self, root, template, modules, *, threaded=True, defaults=True, dsn=None):
        self.root, self.template, self.modules = Path(root), template, modules
        self.directory = self.root/'.data'/'individuals'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.store = IndividualStore(self.directory, dsn if dsn is not None else database_dsn(root))
        self.lock = threading.RLock()
        self.world = OpenWorld()
        self.individuals = {}
        self.archives = {}
        self.pending_archives = {}
        self.pending_events = []
        self.retired = []
        self.running, self.last_error, self.step_wall_ms = False, '', 0.0
        self.stop_event = threading.Event()
        self.thread = None
        self.cursor = 0
        self.checkpoint_steps = 0
        self.last_saved = None
        saved = self.store.all_states()
        if 'world' in saved:
            self.world.restore(saved['world'])
            self.cursor = saved['world'].get('cursor', 0)
        order = saved.get('world',{}).get('order',list(saved))
        order = list(dict.fromkeys(list(order)+list(saved)))
        for identifier in order:
            if identifier == 'world':
                continue
            state = saved[identifier]
            if state.get('archived'):
                self.archives[identifier] = state
                continue
            person = self._new(identifier, state['name'], state['goal'], state['description'], 0)
            legacy_life = 'genome' not in state['body']
            person.restore(state)
            if legacy_life:person.body['sex']='female' if len(self.individuals)%2 else 'male'
        if not saved and defaults:
            self.create('루미', 'live', seed=11)
            self.create('노바', 'explore', seed=29)
        if threaded:
            self.thread = threading.Thread(target=self.loop, daemon=True, name='individual-neural-world')
            self.thread.start()

    def _new(self, identifier, name, goal, description, seed):
        if not identifier.isalnum() or len(identifier) != 32:
            raise ValueError('개체 ID 오류')
        person = Individual(self.directory, self.template, identifier, name, goal, description, seed, self.modules.llm_lock)
        person.memories.request_guard = lambda: self._request_guard(person)
        person.memories.chat_guard = lambda: self._chat_guard(person)
        self.individuals[identifier] = person
        return person

    def create(self, name, goal, description='', seed=None):
        if goal not in GOALS or not isinstance(name, str) or not 1 <= len(name.strip()) <= 40:
            raise ValueError('이름 1–40자와 지원하는 행동 목표를 지정해 주세요.')
        if not isinstance(description, str) or len(description) > 400:
            raise ValueError('목표 설명은 400자 이내입니다.')
        with self.lock:
            if len(self.individuals) >= self.MAX_POPULATION:
                raise ValueError(f'계산 중인 개체는 알을 포함해 최대 {self.MAX_POPULATION}개입니다.')
            identifier = uuid.uuid4().hex
            person = self._new(identifier, name.strip(), goal, description, seed if seed is not None else int(identifier[:8], 16))
            person.body = life.initialize(upgrade_body(fresh_body(self.world.empty_cell())),self.world.rng,
                                          tick=self.world.ticks,sex='female' if len(self.individuals)%2 else 'male')
            person.body['individual_id'] = person.id
            person.policy.learning_rate = person.body['genome']['learning_rate']
            person.brain.plasticity.learning_rate = .02*person.policy.learning_rate/.12
            self._event(person.id,{'kind':'founder','observation':'성체 개체 생성','generation':0})
            self.save()
            return person.state()

    def configure(self, identifier, values):
        with self.lock:
            person = self.individuals[identifier]
            for key in ('active', 'learning', 'llm_enabled'):
                if key in values and not isinstance(values[key], bool):
                    raise ValueError('개체 설정에는 참·거짓 값이 필요합니다.')
            goal = values.get('goal', person.goal)
            description = values.get('description', person.description)
            if goal not in GOALS or not isinstance(description, str) or not 1 <= len(description) <= 400:
                raise ValueError('목표와 설명을 확인해 주세요.')
            if goal != person.goal or description != person.description:
                person.research.stop()
                person.last_research = 0
                person.pending_lesson = None
            person.lesson_generation += 1
            person.pending_lesson = None
            person.goal, person.description = goal, description
            for key in ('active', 'learning', 'llm_enabled'):
                if key in values:
                    setattr(person, key, values[key])
            if not person.llm_enabled or not person.active:
                person.research.stop()
                person.pending_lesson = None
                person.lesson_status = '경험 학습 중 · 자료 보조 꺼짐' if not person.llm_enabled else '개체 일시정지'
            self.save()
            return person.state()

    def control(self, action):
        if action == 'start':
            with self.lock:
                self.running, self.last_error = True, ''
        elif action in ('pause', 'save'):
            with self.lock:
                if action == 'pause':
                    self.running = False
                    for person in self.individuals.values():
                        person.research.stop()
                        person.lesson_generation += 1
                        person.pending_lesson = None
                self.save()
        elif action == 'step':
            with self.lock:
                if self.running:
                    raise ValueError('먼저 세계를 일시정지해 주세요.')
            self.tick()
            with self.lock:
                self.save()
        else:
            raise ValueError('지원하지 않는 세계 명령입니다.')

    def tick(self):
        started = time.perf_counter()
        with self.lock:
            enabled = [p for p in self.individuals.values() if p.active and p.body['alive']]
            if not enabled:
                self.running = False
                return
            self.world.advance()
            for p in enabled:
                for event in life.advance(p.body,self.world.ticks):
                    self._event(p.id,event)
            active = [p for p in enabled if p.body['alive'] and p.body['stage'] in ('larva','adult')]
            if active:
                person = active[self.cursor % len(active)]
                self.cursor += 1
                self._act(person,enabled)
            deceased = [p for p in enabled if not p.body['alive']]
            for person in deceased:
                self._archive(person)
            self.checkpoint_steps += 1
            if self.checkpoint_steps >= 25 or deceased or self.pending_archives:
                self.save()
                self.checkpoint_steps = 0
            if not self.individuals:
                self.running = False
        self.step_wall_ms = (time.perf_counter()-started)*1000

    def _act(self,person,enabled):
        others=[p.body for p in enabled if p.body['alive']]
        can_reproduce=len(self.individuals)<self.MAX_POPULATION
        neural = person.adapter.observe(self.world.sense(person.body,others,can_reproduce))
        allowed=self.world.allowed_actions(person.body,others,can_reproduce)
        action = person.policy.choose(neural,allowed=allowed)
        probability = float(person.policy.probabilities[action])
        result = self.world.act(person.body, action, person.goal, others, len(self.individuals)<self.MAX_POPULATION)
        if result.get('reproduction_peer'):
            child=self._birth(person,self.individuals[result['reproduction_peer']])
            result['child_id']=child.id
        if result.get('death'):
            self._event(person.id,result['death'])
        change = person.policy.learn(result['reward'], person.learning)
        with person.brain.lock:
            internal_change=person.brain.plasticity.learn(person.adapter.last_counts,10,
                                                         person.policy.last_error,person.learning)
        experience = {**result, 'goal': person.goal, 'action': action, 'activity': neural.tolist(),
                      'probability': probability, 'step': person.body['steps'],'allowed':allowed,'world_tick':self.world.ticks,
                      'internal_change':internal_change}
        person.experiences.append(experience)
        brief = {**result, 'step': person.body['steps'], 'reward': result['reward'], 'total_food': person.body['food'],
                 'places': len(person.body['visits']), 'weight_change': change, 'internal_change':internal_change, 'action': action,
                 'position': person.body['position'], 'kind': 'experience'}
        person.recent.append(brief)
        self._event(person.id, experience | {'kind': 'experience', 'weight_change': change})
        if result.get('shared_knowledge') or result.get('shared_resource'):
            peer=self.individuals[result['peer_id']]
            receipt={'kind':'received','from':person.id,'from_name':person.name,
                     'observation':person.name+'에게서 '+result['observation'].replace('전달','받음').replace('나눔','받음'),
                     'knowledge':result.get('shared_knowledge'),'resource':result.get('shared_resource')}
            self._event(peer.id,receipt)
        self._apply_lesson(person)
        if self.running:
            self._research_tick(person)

    def _birth(self,parent,mate):
        identifier=uuid.uuid4().hex
        child=self._new(identifier,f'새싹 {sum(p.body["offspring"] for p in self.individuals.values())//2+len(self.archives)+1}',
                        'live','',int(self.world.rng.integers(2**32)))
        child.body=life.initialize(upgrade_body(fresh_body(parent.body['position'])),self.world.rng,
                                   (parent.body,mate.body),self.world.ticks)
        child.body['individual_id']=child.id
        child.policy.learning_rate=child.body['genome']['learning_rate']
        child.brain.plasticity.learning_rate=.02*child.policy.learning_rate/.12
        for p in (parent,mate):
            p.body['energy']-=20;p.body['hydration']-=10;p.body['cooldown']=600;p.body['offspring']+=1
            self._event(p.id,{'kind':'reproduction','child_id':identifier,'parents':child.body['parents'],
                              'observation':child.name+'의 알 · '+str(child.body['generation'])+'세대'})
        self._event(child.id,{'kind':'birth','parents':child.body['parents'],'genome':child.body['genome'],
                             'generation':child.body['generation'],'observation':'유전 특성과 변이를 가진 알 · 경험 기억은 새로 시작'})
        return child

    def _archive(self,person):
        person.active=False;person.research.stop();person.lesson_generation+=1;person.pending_lesson=None
        person.lesson_status='생애 종료 · 마지막 학습 상태 보존'
        archive=person.checkpoint() | {'archived':True,'public_state':person.state()}
        self.archives[person.id]=archive;self.pending_archives[person.id]=archive
        del self.individuals[person.id]
        self.retired.append(person)

    def _event(self,identifier,payload):
        self.pending_events.append({'id':uuid.uuid4().hex,'agent_id':identifier,'created':time.time(),
                                    'payload':{'world_tick':self.world.ticks,**payload}})

    def _history(self,identifier,limit=30,only_lessons=False):
        pending=[{'created':e['created'],**e['payload']} for e in reversed(self.pending_events)
                 if e['agent_id']==identifier and (not only_lessons or e['payload']['kind']!='experience')]
        return (pending+self.store.history(identifier,limit,only_lessons))[:limit]

    def _research_tick(self, person):
        available = person.llm_enabled and self.modules.registry['language'].enabled and self.modules.registry['web'].enabled
        if not available:
            if person.research.running:
                person.research.stop()
            return
        status = person.research.status()
        if status['running']:
            person.lesson_status = '자료 보조 · ' + status['phase']
            return
        finished = next((r for r in status['history'] if r['id'] > person.handled_research and r['status'] == 'completed'), None)
        if finished:
            person.handled_research = finished['id']
            expected_goal = person.description + ' 실제 감각·행동 학습에 도움이 되는 원리를 조사한다.'
            if finished.get('learning', {}).get('memory_id') and finished.get('goal') == expected_goal:
                self._event(person.id, {'kind': 'research', 'record': finished})
                person.lesson_thread = threading.Thread(target=self._lesson_plan, args=(person, finished, person.lesson_generation), daemon=True)
                person.lesson_thread.start()
            else:
                person.lesson_status = '이번 자료는 학습 근거가 부족함 · 실제 경험 학습 계속'
        if status.get('last_error'):
            person.lesson_status = '자료 탐구 오류 · 실제 경험 학습 계속'
        if person.lesson_thread and person.lesson_thread.is_alive():
            return
        if person.body['steps'] >= 10 and time.time()-person.last_research >= 180:
            person.last_research = time.time()
            try:
                person.memories.model, person.memories.endpoint = self.modules.model, self.modules.endpoint
                person.research.start(person.description + ' 실제 감각·행동 학습에 도움이 되는 원리를 조사한다.', limit=1)
                person.lesson_status = '목표에 필요한 자료 수집 시작'
            except ValueError as error:
                person.lesson_status = str(error)[:150]

    def _lesson_plan(self, person, record, generation):
        try:
            answer = person.memories.structured(
                'Use the supplied unverified source summary to suggest ONE category of the individual\'s real experiences '
                'worth reviewing for its goal. Return focus (odor_progress, obstacle, novelty, survival, objects, social, reproduction), reason (Korean), '
                'and memory_id equal to the supplied memory ID. This only prioritizes replay; never invent rewards, '
                'change goals, choose live actions or claim to train the biological connectome.',
                {'goal': person.description, 'memory_id': record['learning']['memory_id'],
                 'finding': record.get('finding', '')[:1200], 'recent_experience': person.context()},
                schema={'type': 'object', 'additionalProperties': False, 'required': ['focus','reason','memory_id'],
                        'properties': {'focus': {'type': 'string', 'enum': ['odor_progress','obstacle','novelty','survival','objects','social','reproduction']},
                                       'reason': {'type': 'string', 'maxLength': 300},
                                       'memory_id': {'type': 'integer', 'enum': [record['learning']['memory_id']]}}})
            with self.lock:
                if self.running and person.active and self._llm_available(person) and person.lesson_generation == generation:
                    if answer.get('memory_id') != record['learning']['memory_id'] or answer.get('focus') not in ('odor_progress','obstacle','novelty','survival','objects','social','reproduction'):
                        raise ValueError('학습 제안의 기억 참조가 맞지 않습니다.')
                    person.pending_lesson = answer | {'goal': person.goal, 'generation': generation}
                    person.lesson_status = '자료와 연결된 실제 경험 복습 준비'
        except Exception:
            with self.lock:
                if person.lesson_generation == generation and self._llm_available(person):
                    person.lesson_status = '자료 기반 복습 제안을 적용하지 못함 · 경험 학습 계속'

    def _llm_available(self, person):
        return person.body['alive'] and person.llm_enabled and self.modules.registry['language'].enabled and self.modules.registry['web'].enabled

    def _request_guard(self, person):
        if not self.running or not person.active or not self._llm_available(person) or person.research.stop_event.is_set():
            raise ValueError('개체의 자료 보조 요청이 중단되었습니다.')

    def _chat_guard(self, person):
        if not person.body['alive'] or not person.llm_enabled or not self.modules.registry['language'].enabled:
            raise ValueError('이 개체의 로컬 대화가 꺼졌습니다.')

    def _apply_lesson(self, person):
        lesson = person.pending_lesson
        if not lesson:
            return
        person.pending_lesson = None
        if (not person.learning or not self._llm_available(person) or lesson['goal'] != person.goal
                or lesson['generation'] != person.lesson_generation):
            return
        predicates = {'odor_progress': lambda e: not e['collision'], 'obstacle': lambda e: e['collision'],
                      'novelty': lambda e: e['new_place'],
                      'survival': lambda e: e['action'] in (6,8),
                      'objects': lambda e: e['action'] in (4,5,7),
                      'social': lambda e: e['action'] in (9,10),
                      'reproduction': lambda e: e['action']==11}
        examples = [e for e in person.experiences if e['goal'] == person.goal and predicates[lesson['focus']](e)][-16:]
        change = 0.0
        for example in examples:
            change += person.policy.replay(example['activity'], example['action'], example['reward'], example['probability'],example.get('allowed'))
        person.replay_updates += len(examples)
        person.lesson_status = f"자료 기반 경험 복습 {len(examples)}개 · 신경 가중치 변화 {change:.3f}"
        self._event(person.id, {'kind': 'source_guided_replay', **lesson,
                                    'examples': len(examples), 'weight_change': change,
                                    'independently_verified': False})

    def save(self):
        states = {'world': self.world.state() | {'cursor': self.cursor,'order':list(self.individuals)},**self.pending_archives}
        for person in self.individuals.values():
            states[person.id] = person.checkpoint()
        self.store.put_many(states,self.pending_events)
        self.pending_events.clear();self.pending_archives.clear()
        self.last_saved = time.time()
        # Only superseded checkpoints in this generated individual's own folder.
        # The database now points at the new complete generation; retain one prior.
        for person in self.individuals.values():
            for pattern in ('state-*.npz','pair-*.npy'):
                files = sorted(person.directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                for old in files[2:]:
                    if old.resolve().parent == person.directory.resolve():
                        old.unlink()
        for person in self.retired[:]:
            if not person.research.running and not (person.lesson_thread and person.lesson_thread.is_alive()):
                person.brain.close();person.memories.close();self.retired.remove(person)

    def loop(self):
        while not self.stop_event.is_set():
            if not self.running:
                self.stop_event.wait(.1)
                continue
            try:
                self.tick()
            except Exception as error:
                with self.lock:
                    self.running = False
                    self.last_error = '세계 계산 또는 저장에 실패하여 정지했습니다. ' + type(error).__name__
                    for person in self.individuals.values():
                        person.research.stop()
                        person.lesson_generation += 1
                        person.pending_lesson = None
            self.stop_event.wait(.025)

    def state(self,focus=None,center=None,radius=12):
        with self.lock:
            if center is None:
                if focus in self.individuals:center=self.individuals[focus].body['position']
                elif focus in self.archives:center=self.archives[focus]['body']['position']
                else:center=(4,4)
            living=[p.state() for p in self.individuals.values()]
            bodies=[p.body for p in self.individuals.values() if p.active and p.body['alive']]
            for state in living:
                p=self.individuals[state['id']]
                state['reproduction']=self.world.reproduction_status(p.body,bodies,len(living)<self.MAX_POPULATION)
                if not p.active:
                    state['reproduction']['ready']=False
                    state['reproduction']['blockers'].insert(0,'개체 활동 일시정지')
            dead=[s['public_state'] for s in self.archives.values()]
            all_people=living+dead
            progress={'living':len(living),'deaths':len(dead),'births':sum(p['life']['generation']>0 for p in all_people),
                      'generation':max((p['life']['generation'] for p in all_people),default=0),
                      'experiences':sum(p['steps'] for p in all_people),
                      'discoveries':len({k for p in all_people for k in p['knowledge']}),
                      'crafts':sum(p['crafted'] for p in all_people),
                      'transfers':sum(p['communications']+p['shared'] for p in all_people)}
            return {'ready': True, 'running': self.running, 'storage': self.store.backend, 'last_error': self.last_error,
                    'last_saved': self.last_saved, 'step_wall_ms': round(self.step_wall_ms, 2),
                    'world': self.world.snapshot(center,radius),'progress':progress,
                    'goals': GOALS, 'agents': living,'ancestors':sorted(dead,key=lambda p:p['life']['death_tick'],reverse=True)[:30],
                    'catalog':OBJECTS,'recipes':RECIPES,'actions':ACTIONS,'population_limit':self.MAX_POPULATION,
                    'model': {'base_neurons_per_agent': self.template.n, 'shared_pairs': int(self.template.graph.nnz),
                              'readout_synapses_per_agent': (22+len(EXTRA_SENSORS))*len(ACTIONS), 'action_neurons_per_agent': len(ACTIONS),
                              'feature_names':list(NeuralAdapter.SENSOR_NAMES)+['pool_'+str(i) for i in range(8)]+list(EXTRA_SENSORS),
                              'sensor_mapping': '인위적으로 배정한 감각 입력', 'brain_step_ms_per_action': 10,
                              'learning': '내부 연결 효율과 행동 회로의 보상 조절 학습', 'base_weights_changed': False,
                              'internal_pair_capacity':int(self.template.graph.nnz),
                              'internal_plasticity':True,'contact_specific_plasticity':False}}

    def detail(self, identifier):
        with self.lock:
            if identifier in self.archives:
                archived=self.archives[identifier]
                public=archived['public_state'] | {'archived':True,'visits':archived['body']['visits']}
            else:
                person=self.individuals[identifier]
                public=person.state(True)
                public['reproduction']=self.world.reproduction_status(person.body,
                    [p.body for p in self.individuals.values() if p.active and p.body['alive']],
                    len(self.individuals)<self.MAX_POPULATION)
            return {**public, 'events': self._history(identifier),
                    'learning_records': self._history(identifier, 16, only_lessons=True)}

    def chat(self, identifier, text):
        with self.lock:
            person = self.individuals[identifier]
            if not person.llm_enabled or not self.modules.registry['language'].enabled:
                raise ValueError('이 개체의 LLM 보조와 로컬 대화 모듈을 켜 주세요.')
            person.memories.model, person.memories.endpoint = self.modules.model, self.modules.endpoint
        return person.memories.execute('language', text=text)

    def stop_research(self):
        with self.lock:
            for person in self.individuals.values():
                person.research.stop()
                person.lesson_generation += 1
                person.pending_lesson = None
                person.lesson_status = '자료 보조 중단 · 경험 학습은 독립적으로 동작'

    def brain(self, identifier):
        with self.lock:
            if identifier in self.archives:raise ValueError('사망한 개체의 마지막 행동 가중치와 생애 기록은 세계 패널에서 볼 수 있습니다.')
            return self.individuals[identifier].brain

    def close(self):
        self.running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=20)
        with self.lock:
            for person in self.individuals.values():
                person.research.stop()
            self.save()
            for person in self.individuals.values():
                person.brain.close()
                # Research may be inside a bounded HTTP request. Do not close its
                # connection while it can still record cancellation.
                if not person.research.running and not (person.lesson_thread and person.lesson_thread.is_alive()):
                    person.memories.close()
            self.store.close()

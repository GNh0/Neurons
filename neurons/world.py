"""Small shared, observable world with explicit goal-dependent feedback."""
import numpy as np

GOALS = {
    'live': {'name': '생존과 발견', 'description': '필요를 충족하고 새로운 사물과 행동의 결과를 배운다.'},
    'forage': {'name': '먹이 확보', 'description': '먹이를 찾아 에너지를 유지한다.'},
    'explore': {'name': '새 지역 탐색', 'description': '충돌을 피하며 방문하지 않은 지역을 탐색한다.'},
    'cooperate': {'name': '교류와 공동 생활', 'description': '발견과 자원을 나누며 함께 살아갈 방법을 배운다.'},
}
MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Arena:
    def __init__(self, seed=2026, size=11, obstacles=True):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.walls = {(0, i) for i in range(size)} | {(size-1, i) for i in range(size)}
        self.walls |= {(i, 0) for i in range(size)} | {(i, size-1) for i in range(size)}
        if obstacles:
            self.walls |= {(size//2, y) for y in range(3, size-3)}
        self.food = []
        for _ in range(3):
            self.food.append(self.empty_cell())

    def empty_cell(self):
        cells = [(x, y) for y in range(1, self.size-1) for x in range(1, self.size-1)
                 if (x, y) not in self.walls and (x, y) not in self.food]
        return cells[int(self.rng.integers(len(cells)))]

    def distance(self, position):
        return min(abs(position[0]-f[0])+abs(position[1]-f[1]) for f in self.food)

    def sense(self, body):
        x, y = body['position']
        distance = self.distance((x, y))
        smell, free, novel = [], [], []
        for dx, dy in MOVES:
            cell = (x+dx, y+dy)
            available = cell not in self.walls
            smell.append(float(np.clip(.5+.45*(distance-self.distance(cell)), 0, 1)))
            free.append(float(available))
            novel.append(1/(1+body['visits'].get(f'{cell[0]},{cell[1]}', 0)) if available else 0)
        return np.array(smell+free+novel+[body['energy']/100, 1], dtype=np.float32)

    def act(self, body, action, goal):
        previous = tuple(body['position'])
        before = self.distance(previous)
        dx, dy = MOVES[action]
        destination = (previous[0]+dx, previous[1]+dy)
        collision = destination in self.walls
        if collision:
            destination = previous
        key = f'{destination[0]},{destination[1]}'
        new_place = key not in body['visits']
        after = self.distance(destination)
        eaten = destination in self.food
        if eaten:
            self.food[self.food.index(destination)] = self.empty_cell()
        body['position'] = list(destination)
        body['visits'][key] = body['visits'].get(key, 0)+1
        body['energy'] = min(100, max(1, body['energy']-.25+20*eaten))
        body['food'] += int(eaten)
        body['collisions'] += int(collision)
        body['steps'] += 1
        # These are disclosed task definitions, not learned biological drives.
        reward = (.3*(before-after) + 2*eaten - .02) if goal == 'forage' else (.7 if new_place else -.12)
        if collision:
            reward -= .7
        body['return'] += reward
        body['last_action'] = action
        body['last_reward'] = float(reward)
        return {'reward': float(reward), 'collision': collision, 'food': eaten,
                'new_place': new_place, 'position_before': list(previous), 'position_after': list(destination)}

    def state(self):
        return {'size': self.size, 'walls': [list(p) for p in sorted(self.walls)],
                'food': [list(p) for p in self.food], 'rng': self.rng.bit_generator.state}

    def restore(self, value):
        self.size = int(value['size'])
        self.walls = {tuple(p) for p in value['walls']}
        self.food = [tuple(p) for p in value['food']]
        self.rng.bit_generator.state = value['rng']


def fresh_body(position=(1, 1)):
    return {'position': list(position), 'energy': 100.0, 'food': 0, 'collisions': 0,
            'steps': 0, 'return': 0.0, 'last_action': None, 'last_reward': 0,
            'visits': {f'{position[0]},{position[1]}': 1}}

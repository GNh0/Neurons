"""World properties and interactions, separate from neural learning rules.

These are explicit simulator rules, not generated facts about the real world.
New objects reuse material, edible, liquid, tool and shelter affordances.
"""
OBJECTS = {
    'berry': dict(name='열매', kind='food', color='#e9a878', nutrition=24, water=6, material=None),
    'apple': dict(name='사과', kind='food', color='#dd7772', nutrition=30, water=10, material=None),
    'mushroom': dict(name='버섯', kind='food', color='#d7c2a0', nutrition=14, water=0, material=None),
    'bitter_mushroom': dict(name='독버섯', kind='food', color='#af84c8', nutrition=3, damage=18, water=0, material=None),
    'water': dict(name='물웅덩이', kind='liquid', color='#75c7e9', water=36, material=None),
    'tree': dict(name='나무', kind='material', color='#709b74', material='wood'),
    'branch': dict(name='나뭇가지', kind='material', color='#b4956e', material='wood'),
    'stone': dict(name='돌', kind='material', color='#95a6b2', material='stone'),
    'grass': dict(name='섬유 풀', kind='material', color='#93b886', material='fiber'),
    'shelter': dict(name='바위 그늘', kind='shelter', color='#91a4af', material=None),
    'hammer': dict(name='돌망치', kind='tool', color='#c5b898', material='hammer'),
    'basket': dict(name='바구니', kind='tool', color='#c4a47c', material='basket'),
    'hut': dict(name='작은 쉼터', kind='shelter', color='#bba27a', material=None),
}

RECIPES = (
    dict(id='hammer', name='돌망치', needs={'wood': 1, 'stone': 1}, effect='재료 채집량 증가'),
    dict(id='basket', name='바구니', needs={'wood': 1, 'fiber': 2}, effect='먹거리 채집량 증가'),
    dict(id='hut', name='작은 쉼터', needs={'wood': 3, 'stone': 2, 'fiber': 2}, effect='모두가 사용할 수 있는 휴식 장소', building=True),
)

ACTIONS = (
    dict(id='north', name='북쪽 이동'), dict(id='east', name='동쪽 이동'),
    dict(id='south', name='남쪽 이동'), dict(id='west', name='서쪽 이동'),
    dict(id='inspect', name='주변 조사'), dict(id='gather', name='수집'),
    dict(id='use', name='사용·섭취'), dict(id='craft', name='제작'), dict(id='rest', name='휴식'),
    dict(id='communicate', name='발견 전달'), dict(id='share', name='자원 나눔'),
    dict(id='mate', name='짝짓기'),
)

EXTRA_SENSORS = ('near_food', 'near_water', 'near_material', 'near_tool', 'near_shelter',
                 'unknown_object', 'thirst', 'fatigue', 'health', 'has_wood', 'has_stone', 'has_fiber',
                 'near_peer', 'peer_in_need', 'hunger', 'safety', 'social_drive',
                 'reproduction_drive', 'compatible_peer', 'adult', 'age',
                 'known_food', 'known_hazard', 'known_water', 'has_food', 'has_hammer', 'has_basket', 'can_craft',
                 'mate_n', 'mate_e', 'mate_s', 'mate_w')


def upgrade_body(body):
    """Add bodily needs and knowledge without clearing an existing individual's life."""
    defaults = dict(health=100., hydration=100., fatigue=0., inventory={}, knowledge={},
                    investigations=0, gathered=0, crafted=0, tool_uses=0, rests=0,
                    successful_actions={}, last_observation='', heading=0)
    defaults.update(communications=0, shared=0, received=0)
    for key, value in defaults.items():
        body.setdefault(key, value)
    return body

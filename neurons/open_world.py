"""Procedural, streamed world with inspectable objects and shared interactions.

Terrain is deterministic from seed and coordinates. Only changed objects,
constructed structures and an LRU of nearby chunks occupy memory.
"""
from collections import OrderedDict
import copy
import math
import numpy as np
from neurons.catalog import OBJECTS, RECIPES, ACTIONS, EXTRA_SENSORS, upgrade_body
from neurons.world import Arena, MOVES
from neurons import life


class OpenWorld(Arena):
    CHUNK = 16
    CACHE_LIMIT = 64
    LIMIT = 1_000_000
    BIOMES = ('초원', '숲', '돌지대', '습지')
    START_OBJECTS = {(2,4):'berry',(4,3):'water',(5,4):'branch',(6,3):'stone',
                     (7,3):'grass',(3,6):'shelter',(1,5):'mushroom',(8,6):'apple'}

    def __init__(self, seed=2026):
        self.seed=int(seed)
        self.rng=np.random.default_rng(seed)
        self.ticks=0
        self.depleted={}
        self.structures={}
        self.cache=OrderedDict()

    def number(self,x,y,salt=0):
        z=(x*374761393+y*668265263+self.seed*1442695041+salt*1013904223)&0xffffffff
        z=((z^(z>>13))*1274126177)&0xffffffff
        return z^(z>>16)

    def chunk(self,cx,cy):
        key=(cx,cy)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        cells=[]
        for ly in range(self.CHUNK):
            for lx in range(self.CHUNK):
                x,y=cx*self.CHUNK+lx,cy*self.CHUNK+ly
                biome=self.number(x//32,y//32,4)%4
                n=self.number(x,y)
                blocked=n%100<6 and x%8 not in (0,1) and y%8 not in (0,1)
                if abs(x)<11 and abs(y)<11: blocked=False
                roll=(n>>8)%100
                choices=(('berry','apple','grass','branch','stone','water','shelter'),
                         ('tree','berry','mushroom','bitter_mushroom','branch','grass','water'),
                         ('stone','stone','branch','shelter','berry','water','grass'),
                         ('water','water','grass','berry','mushroom','branch','stone'))[biome]
                obj=choices[(n>>17)%len(choices)] if roll<24 and not blocked else None
                if (x,y) in self.START_OBJECTS: obj=self.START_OBJECTS[(x,y)]
                cells.append((biome,blocked,obj))
        self.cache[key]=cells
        while len(self.cache)>self.CACHE_LIMIT:self.cache.popitem(last=False)
        return cells

    def tile(self,position):
        x,y=map(int,position)
        cells=self.chunk(x//self.CHUNK,y//self.CHUNK)
        return cells[(y%self.CHUNK)*self.CHUNK+x%self.CHUNK]

    def is_blocked(self,position):
        return any(abs(v)>self.LIMIT for v in position) or self.tile(position)[1]

    def object_at(self,position):
        key=','.join(map(str,position))
        if key in self.structures:return self.structures[key]['object']
        if self.depleted.get(key,0)>self.ticks:return None
        return self.tile(position)[2]

    def consume(self,position):
        self.depleted[','.join(map(str,position))]=self.ticks+800

    def empty_cell(self):
        cells=[(x,y) for y in range(1,10) for x in range(1,10) if not self.is_blocked((x,y)) and not self.object_at((x,y))]
        return cells[int(self.rng.integers(len(cells)))]

    def distance(self,position):
        x,y=position
        best=24
        for dy in range(-8,9):
            for dx in range(-8,9):
                distance=abs(dx)+abs(dy)
                if distance>=best:continue
                obj=self.object_at((x+dx,y+dy))
                if obj and OBJECTS[obj]['kind']=='food':best=distance
        return best

    def target(self,body,kinds=None):
        x,y=body['position']
        heading=body.get('heading',0)
        directions=[(0,0),MOVES[heading]]+[m for m in MOVES if m!=MOVES[heading]]
        for dx,dy in directions:
            at=(x+dx,y+dy)
            obj=self.object_at(at)
            if obj and (kinds is None or OBJECTS[obj]['kind'] in kinds):return at,obj
        return (x,y),None

    @staticmethod
    def peers(body,others):
        x,y=body['position']
        return [p for p in others if p is not body and p.get('alive',True) and abs(p['position'][0]-x)+abs(p['position'][1]-y)<=2]

    def sense(self,body,others=()):
        upgrade_body(body)
        life.initialize(body)
        x,y=body['position'];distance=self.distance((x,y))
        smell=[];free=[];novel=[]
        for dx,dy in MOVES:
            at=(x+dx,y+dy);available=not self.is_blocked(at)
            smell.append(float(np.clip(.5+.45*(distance-self.distance(at)),0,1)))
            free.append(float(available))
            novel.append(1/(1+body['visits'].get(f'{at[0]},{at[1]}',0)) if available else 0)
        _,obj=self.target(body)
        kind=OBJECTS[obj]['kind'] if obj else ''
        peers=self.peers(body,others)
        extra=[float(self.target(body,(k,))[1] is not None) for k in ('food','liquid','material','tool','shelter')]
        extra += [float(bool(obj and obj not in body['knowledge'])),1-body['hydration']/100,
                  body['fatigue']/100,body['health']/100]
        extra += [min(1,body['inventory'].get(k,0)/3) for k in ('wood','stone','fiber')]
        extra += [float(bool(peers)),float(any(p.get('energy',100)<50 or p.get('hydration',100)<50 for p in peers))]
        drives=life.drives(body)
        extra += [drives['hunger'],drives['safety'],drives['social'],drives['reproduction'],
                  float(any(life.compatible(body,p) for p in peers)),float(body['stage']=='adult'),
                  min(1.,body['age']/body['genome']['lifespan'])]
        learned=body['knowledge'].get(obj,{}).get('properties',{})
        extra += [float(learned.get('nutrition',0)>0 and not learned.get('damage')),
                  float(learned.get('damage',0)>0),float(learned.get('water',0)>0),
                  float(any(n>0 and k in OBJECTS and OBJECTS[k]['kind']=='food' for k,n in body['inventory'].items())),
                  float(body['inventory'].get('hammer',0)>0),float(body['inventory'].get('basket',0)>0),
                  float(any(all(body['inventory'].get(k,0)>=n for k,n in r['needs'].items()) for r in RECIPES))]
        return np.array(smell+free+novel+[body['energy']/100,1]+extra,dtype=np.float32)

    def act(self,body,action,goal,others=(),can_reproduce=True):
        upgrade_body(body)
        life.initialize(body)
        if not body['alive']:raise ValueError('사망한 개체는 행동할 수 없습니다.')
        if body['stage'] in ('egg','pupa'):raise ValueError('발달 중인 개체는 아직 행동할 수 없습니다.')
        if body['stage']=='larva' and action in (7,9,10,11):raise ValueError('성체의 행동입니다.')
        if not 0<=action<len(ACTIONS):raise ValueError('세계 행동 범위 오류')
        before=list(body['position']);at,obj=self.target(body);item=OBJECTS.get(obj,{})
        if action in (5,6,8):
            kinds={5:('food','material','tool'),6:('food','liquid'),8:('shelter',)}[action]
            at,obj=self.target(body,kinds);item=OBJECTS.get(obj,{})
        # When dehydrated, drinking takes priority within the generic use action.
        if action==6 and body['hydration']<65:
            water_at,water=self.target(body,('liquid',))
            if water:at,obj,item=water_at,water,OBJECTS[water]
        old_distance=self.distance(before)
        result=dict(reward=0.,collision=False,food=False,new_place=False,position_before=before,
                    position_after=before,action_name=ACTIONS[action]['name'],observation='',success=False)
        discovery=material_gain=crafting=social=recovery=0.
        body['fatigue']=min(100,body['fatigue']+.08)
        if action<4:
            dx,dy=MOVES[action];destination=(before[0]+dx,before[1]+dy)
            result['collision']=self.is_blocked(destination);body['heading']=action
            body['collisions']+=int(result['collision'])
            if not result['collision']:
                body['position']=list(destination);result['success']=True
                key=f'{destination[0]},{destination[1]}'
                result['new_place']=key not in body['visits']
                body['visits'][key]=body['visits'].get(key,0)+1
                discovery+=.35*result['new_place']*body['genome']['curiosity']
                if len(body['visits'])>16384:body['visits'].pop(next(iter(body['visits'])))
        elif action==4 and obj:
            fresh=obj not in body['knowledge']
            note={'name':item['name'],'kind':item['kind'],'position':list(at),'properties':dict(item),
                  'learned_by':'inspection','origin':body.get('individual_id'),'learned_tick':self.ticks}
            body['knowledge'][obj]=note;body['investigations']+=1
            result['observation']=f"{item['name']} 조사: 종류 {item['kind']} · 위치 {at[0]}, {at[1]}"
            result['discovered_object']={'id':obj,**note};discovery+=1.5*fresh*body['genome']['curiosity']
            result['first_discovery']=fresh
            result['success']=True
        elif action==5 and obj and item['kind'] in ('food','material','tool'):
            key=obj if item['kind']=='food' else item['material']
            amount=2 if body['inventory'].get('hammer') and item['kind']=='material' else 1
            if body['inventory'].get('basket') and item['kind']=='food':amount=2
            amount=min(amount,99-body['inventory'].get(key,0))
            body['inventory'][key]=body['inventory'].get(key,0)+amount
            body['gathered']+=amount
            if amount:self.consume(at);material_gain=.6
            if amount>1:body['tool_uses']+=1
            result.update(success=amount>0,observation=f"{item['name']} {amount}개 수집")
        elif action==6:
            used=obj
            if not used or item['kind'] not in ('food','liquid'):
                used=next((k for k,n in body['inventory'].items() if n and k in OBJECTS and OBJECTS[k]['kind']=='food'),None)
                if used:body['inventory'][used]-=1
            elif item['kind']=='food':self.consume(at)
            if used and OBJECTS[used]['kind'] in ('food','liquid'):
                food=OBJECTS[used]
                recovery=min(100-body['energy'],food.get('nutrition',0))/20+min(100-body['hydration'],food.get('water',0))/25
                body['energy']=min(100,body['energy']+food.get('nutrition',0))
                body['hydration']=min(100,body['hydration']+food.get('water',0))
                body['health']=max(0,body['health']-food.get('damage',0))
                result['food']=food['kind']=='food';body['food']+=int(result['food'])
                recovery-=food.get('damage',0)/10
                if food.get('damage'):result['injury']='독성 음식'
                result.update(success=True,observation=food['name']+' 사용·섭취')
        elif action==7:
            for recipe in RECIPES:
                if not recipe.get('building') and body['inventory'].get(recipe['id']):continue
                key=','.join(map(str,body['position']))
                if recipe.get('building') and key in self.structures:continue
                if all(body['inventory'].get(k,0)>=n for k,n in recipe['needs'].items()):
                    for k,n in recipe['needs'].items():body['inventory'][k]-=n
                    if recipe.get('building'):self.structures[key]={'object':recipe['id'],'builder':body.get('individual_id'),'created':self.ticks}
                    else:body['inventory'][recipe['id']]=1
                    body['crafted']+=1;crafting=2
                    result.update(success=True,observation=recipe['name']+' 제작 · '+recipe['effect'],crafted=recipe['id'])
                    break
        elif action==8:
            sheltered=item.get('kind')=='shelter'
            restored=min(body['fatigue'],18 if sheltered else 10)
            body['fatigue']-=restored;body['rests']+=1;recovery+=restored/18
            if sheltered:body['health']=min(100,body['health']+3)
            result.update(success=True,observation='그늘에서 휴식' if sheltered else '휴식')
        elif action in (9,10):
            nearby=self.peers(body,others)
            if nearby:
                # A peer with an unmet need / missing knowledge is eligible; no
                # transfer is invented if every peer already has the resource.
                candidates=[p for p in nearby if (any(k not in p.get('knowledge',{}) for k in body['knowledge']) if action==9
                            else any(n>0 and k not in ('hammer','basket') and p.get('inventory',{}).get(k,0)<99 for k,n in body['inventory'].items()))]
                peer=upgrade_body(candidates[0] if candidates else nearby[0]);result['peer_id']=peer.get('individual_id')
                if action==9:
                    novel=next((k for k in body['knowledge'] if k not in peer['knowledge']),None)
                    if novel:
                        peer['knowledge'][novel]=copy.deepcopy(body['knowledge'][novel])
                        peer['knowledge'][novel].update(learned_by='communication',teacher=body.get('individual_id'),learned_tick=self.ticks)
                        body['communications']+=1;peer['received']+=1;social=1.5
                        result.update(success=True,observation=body['knowledge'][novel]['name']+'의 발견 정보 전달',shared_knowledge=novel)
                else:
                    resource=next((k for k,n in body['inventory'].items() if n>0 and k not in ('hammer','basket') and peer['inventory'].get(k,0)<99),None)
                    if resource:
                        body['inventory'][resource]-=1;peer['inventory'][resource]=min(99,peer['inventory'].get(resource,0)+1)
                        body['shared']+=1;peer['received']+=1;social=.5
                        result.update(success=True,observation=resource+' 1개 나눔',shared_resource=resource)
                if result['success']:
                    body['loneliness']=max(0,body['loneliness']-30)
                    peer['loneliness']=max(0,peer.get('loneliness',0)-30)
        elif action==11:
            mate=next((p for p in self.peers(body,others) if life.compatible(body,p)),None)
            if mate and can_reproduce:
                result.update(success=True,reproduction_peer=mate['individual_id'],observation='짝짓기 · 자손의 알 생성')
                social=2*body['genome']['reproductive_drive']
            else:result['observation']='개체 수 제한으로 번식 대기' if not can_reproduce else '번식 조건 또는 가까운 짝이 없음'
        reward=discovery+material_gain+crafting+social*body['genome']['sociality']+recovery-.025-.65*result['collision']
        reward-=.12*life.drives(body)['hunger']+.12*life.drives(body)['thirst']
        if goal=='forage':reward+=.2*(old_distance-self.distance(body['position']))+int(result['food'])
        elif goal=='explore':reward+=.5*result['new_place']+.5*discovery
        elif goal=='cooperate':reward+=social+crafting
        if not result['success']:reward-=.08
        if body['health']<=0:
            reward-=4
            result['death']=life.die(body,self.ticks,result.get('injury','건강 고갈'))
        result['position_after']=list(body['position']);result['reward']=float(reward)
        body['steps']+=1;body['return']+=reward;body['last_action']=action;body['last_reward']=float(reward)
        body['last_observation']=result['observation']
        if result['success']:
            key=ACTIONS[action]['id'];body['successful_actions'][key]=body['successful_actions'].get(key,0)+1
        return result

    def advance(self):
        self.ticks+=1
        if self.ticks%100==0:self.depleted={k:v for k,v in self.depleted.items() if v>self.ticks}

    def snapshot(self,center=(0,0),radius=12):
        if len(center)!=2 or any(not math.isfinite(float(v)) or abs(float(v))>self.LIMIT for v in center):
            raise ValueError('세계 관측 좌표 범위 오류')
        if not 6<=int(radius)<=24:raise ValueError('지도 반경은 6~24칸입니다.')
        cx,cy=map(lambda v:round(float(v)),center);radius=int(radius)
        objects=[];walls=[];ground=[]
        for y in range(cy-radius,cy+radius+1):
            for x in range(cx-radius,cx+radius+1):
                biome,blocked,_=self.tile((x,y));ground.append([x,y,biome])
                if blocked:walls.append([x,y])
                obj=self.object_at((x,y))
                if obj:objects.append({'position':[x,y],'id':obj,**OBJECTS[obj]})
        return dict(kind='open_world',size=2*radius+1,center=[cx,cy],radius=radius,
                    origin=[cx-radius,cy-radius],walls=walls,objects=objects,ground=ground,
                    food=[o['position'] for o in objects if o['kind']=='food'],
                    cached_chunks=len(self.cache),chunk_limit=self.CACHE_LIMIT,chunk_size=self.CHUNK,
                    seed=self.seed,ticks=self.ticks,structures=len(self.structures),biomes=self.BIOMES)

    def state(self):
        return dict(kind='open_world',version=1,seed=self.seed,ticks=self.ticks,
                    depleted=self.depleted,structures=self.structures,rng=self.rng.bit_generator.state)

    def restore(self,value):
        self.seed=value.get('seed',2026);self.ticks=value.get('ticks',0)
        self.depleted=value.get('depleted',{});self.structures=value.get('structures',{})
        if 'rng' in value:self.rng.bit_generator.state=value['rng']
        self.cache.clear()

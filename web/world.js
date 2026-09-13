const $=id=>document.getElementById(id), fmt=n=>Number(n??0).toLocaleString('ko-KR');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors=['#7ee8c4','#bb9eeb','#ebba75','#79b9ec','#ea92ab','#c0d784','#7bd6d0','#e8cbb5'];
const driveNames={hunger:'배고픔',thirst:'갈증',rest:'피로',safety:'안전',curiosity:'호기심',social:'교류',reproduction:'번식'};
const itemNames={wood:'목재',stone:'돌',fiber:'섬유'};

export function setupWorld({api,toast,showPanel,observe}){
  let state=null,selected='',detail=null,visible=false,busy=false,chatBusy=false,polls=0,positionCache=new Map();
  let follow=true,center=null,radius=12,drag=null,requestNumber=0,projection=null,hitObjects=[];
  const canvas=$('arena'),ctx=canvas.getContext('2d');
  const people=()=>[...(state?.agents||[]),...(state?.ancestors||[])];
  const person=()=>people().find(p=>p.id===selected);
  const itemName=id=>state?.catalog?.[id]?.name||itemNames[id]||id;
  const actionName=i=>state?.actions?.[i]?.name||'—';
  const goalOptions=()=>Object.entries(state?.goals||{}).map(([id,g])=>`<option value="${esc(id)}">${esc(g.name)}</option>`).join('');
  const path=(suffix='')=>'/api/individuals/'+selected+suffix;
  async function perform(fn){try{busy=true;await fn();await refresh();}catch(e){toast(e.message,true);}finally{busy=false;}}
  function choose(id){selected=id;detail=null;follow=true;center=null;$('individualMessages').innerHTML='';$('individualChatDetails').open=false;$('individualSelect').value=id;renderSettings();render();loadDetail();refresh().catch(e=>toast(e.message,true));}
  function renderSettings(){const p=person();if(!p)return;$('individualGoal').value=p.goal;$('individualDescription').value=p.description;}
  function render(){
    if(!state?.ready){$('worldMessage').textContent=state?.error||'저장소를 확인하는 중입니다.';return;}
    $('worldStorage').textContent=state.storage+' · LOCAL';
    $('worldPlay').textContent=state.running?'Ⅱ 세계 일시정지':'▶ 세계 실행';
    $('worldStep').disabled=state.running;$('worldMessage').textContent=state.last_error||'';
    $('worldClock').textContent=`${state.running?'실행 중':'일시정지'} · ${state.step_wall_ms.toFixed(0)} ms / 행동`;
    const progress=state.progress;
    $('worldSummary').innerHTML=[['살아가는 개체',`${progress.living} / ${state.population_limit}`],['태어난 자손',fmt(progress.births)],['이어진 세대',fmt(progress.generation)],['생애를 마친 개체',fmt(progress.deaths)]].map(([t,v])=>`<div><small>${t}</small><strong>${v}</strong></div>`).join('');
    $('worldCoordinates').textContent=`${state.world.center.join(', ')} · 반경 ${state.world.radius} · 세계 ${fmt(state.world.ticks)} 턴`;
    $('worldFollow').textContent=follow?'◎ 개체 따라가는 중':'◎ 선택 개체 따라가기';
    $('worldModelNote').textContent=`지형 ${state.world.cached_chunks}/${state.world.chunk_limit} 청크 · 개체별 내부 ${fmt(state.model.internal_pair_capacity)}개 집계 연결 · 행동 ${fmt(state.model.readout_synapses_per_agent)}개 학습 연결 · 수명은 세계 턴 기준`;
    const options=people().map(p=>`<option value="${p.id}">${esc(p.name)} · ${esc(p.life.stage_name)} · ${p.life.generation}세대</option>`).join('');
    if($('individualSelect').innerHTML!==options){$('individualSelect').innerHTML=options;$('individualSelect').value=selected;}
    const observation=$('observationAgent'),old=observation.value,observedOptions='<option value="">원본 시험 시뮬레이션</option>'+state.agents.map(p=>`<option value="${p.id}">${esc(p.name)}</option>`).join('');
    if(observation.innerHTML!==observedOptions){observation.innerHTML=observedOptions;observation.value=old;if(old&&!observation.value){observation.value='';observation.dispatchEvent(new Event('change'));}}
    if($('individualGoal').innerHTML!==goalOptions()){$('individualGoal').innerHTML=goalOptions();renderSettings();}
    $('worldProgress').innerHTML=[['누적 행동',progress.experiences],['알아낸 사물 종류',progress.discoveries],['제작 성공',progress.crafts],['정보·자원 전달',progress.transfers],['공유 쉼터',state.world.structures]].map(([label,n])=>`<p class="progress-row"><span>${label}</span><b>${fmt(n)}</b></p>`).join('');
    $('worldRules').innerHTML=`<p class="caption">${state.world.biomes.join(' · ')}의 절차 생성 지형, ${Object.keys(state.catalog).length}종 사물. 화면 밖 지형은 좌표로 다시 만듭니다. 알 60턴 → 유충 240턴 → 번데기 120턴 → 성체. 활동을 끈 개체는 나이와 신경 계산도 멈춥니다.</p><p class="caption">제공된 제작법</p>`+state.recipes.map(r=>`<p class="progress-row"><span>${esc(r.name)}</span><span>${Object.entries(r.needs).map(([k,n])=>`${esc(itemName(k))} ${n}`).join(' + ')}</span></p>`).join('');
    const p=person();if(!p)return;
    const alive=p.life.alive;
    $('individualBrain').disabled=!alive;
    $('individualChatSend').disabled=chatBusy||!alive||!p.llm_enabled;
    for(const control of $('individualSettings').elements)control.disabled=!alive;
    for(const id of ['individualActive','individualLearning','individualLLM'])$(id).disabled=!alive;
    for(const [id,key] of [['individualActive','active'],['individualLearning','learning'],['individualLLM','llm_enabled']])if(!busy)$(id).checked=p[key];
    $('individualLife').innerHTML=`<div><b>${esc(p.life.stage_name)}</b><span>${p.life.sex==='female'?'♀ 암컷':'♂ 수컷'} · ${p.life.generation}세대</span></div><div class="life-bar"><i style="width:${Math.min(100,100*p.life.age/p.life.lifespan)}%"></i></div><small>나이 ${fmt(p.life.age)} / 수명 ${fmt(p.life.lifespan)} 세계 턴</small><p>${alive?`자손 ${p.life.offspring} · 번식 대기 ${p.life.cooldown}턴`:esc(p.life.death_cause)+' · 마지막 신경 가중치와 기록 보존'}</p>`;
    $('individualStats').innerHTML=[['행동 경험',fmt(p.steps)],['방문 지역',fmt(p.places)],['학습 갱신',fmt(p.neural_learning.updates)],['에너지',p.energy.toFixed(1)],['수분',p.hydration.toFixed(1)],['건강',p.health.toFixed(1)]].map(([t,v])=>`<div><small>${t}</small><b>${v}</b></div>`).join('');
    $('individualDrives').innerHTML=Object.entries(driveNames).map(([key,label])=>`<div><span>${label}</span><div><i style="width:${Math.max(0,Math.min(100,100*p.life.drives[key]))}%"></i></div><b>${Math.round(100*p.life.drives[key])}</b></div>`).join('');
    const reproduction=p.reproduction;
    $('individualReproduction').innerHTML=reproduction?`<b>${reproduction.ready?'짝짓기 가능한 상태':'현재 번식 조건'}</b><p>${reproduction.blockers.map(esc).join('<br>')||'가까운 짝이 있습니다. 행동 회로가 짝짓기를 선택하면 알이 생깁니다.'}</p><small>유전된 번식 성향 ${Math.round(reproduction.inherited_strength*100)}% · 감지한 짝 ${reproduction.sensed_mates} · 누적 시도 ${fmt(p.neural_learning.action_spikes[11])}</small><small>욕구 수치는 선택 확률과 다릅니다. 조건을 충족할 때 짝짓기가 선택 후보가 됩니다.</small>`:'<p class="caption">생애가 끝난 개체의 기록입니다.</p>';
    const internal=p.internal_learning;
    const internalExpanded=$('internalLearning').querySelector('details')?.open||false;
    $('internalLearning').innerHTML=internal?`<div class="internal-count"><b>${fmt(internal.changed_pairs)}</b><span>변화한 집계 연결 / ${fmt(internal.pair_capacity)}</span></div><p class="caption">갱신 ${fmt(internal.updates)}회 · 최근 효율 변화 합 ${internal.last.change_l1.toFixed(3)}<br>보상 항 ${internal.last.reward_component_l1.toFixed(3)} · 활동 조절 항 ${internal.last.homeostasis_component_l1.toFixed(3)}</p><p class="caption">변한 효율은 실제 뉴런 간 전류 계산에 쓰입니다. 개별 시냅스 접촉은 뉴런 쌍으로 집계하며, 새 연결 생성은 아직 포함하지 않습니다.</p>${internal.examples.length?'<details'+(internalExpanded?' open':'')+'><summary>최근 변한 연결 예시</summary>'+internal.examples.map(e=>`<p class="caption">뉴런 #${fmt(e.source_id)} → #${fmt(e.target_id)}<br>효율 ${e.before_gain.toFixed(5)} → ${e.gain.toFixed(5)}배</p>`).join('')+'</details>':''}`:'<p class="caption">이 기록은 내부 가소성 도입 전의 상태입니다.</p>';
    $('individualInventory').innerHTML=Object.entries(p.inventory).filter(([,n])=>n>0).map(([id,n])=>`<span>${esc(itemName(id))} <b>${n}</b></span>`).join('')||'<p class="caption">아직 소지품이 없습니다.</p>';
    $('individualKnowledge').innerHTML=Object.values(p.knowledge).map(k=>`<p class="knowledge-row"><b>${esc(k.name)}</b><span>${k.learned_by==='communication'?'교류로 배움':'직접 조사'} · ${k.position.join(', ')}</span></p>`).join('')||'<p class="caption">조사하거나 전달받은 사물이 기록됩니다.</p>';
    $('individualLineage').innerHTML='<p class="caption">부모 · '+(p.life.parents.map(id=>esc(people().find(a=>a.id===id)?.name||id.slice(0,8))).join(' / ')||'처음 생성한 개체')+'</p><p class="caption">유전 특성 · 대사 '+p.life.genome.metabolism.toFixed(2)+' · 호기심 '+p.life.genome.curiosity.toFixed(2)+' · 학습률 '+p.life.genome.learning_rate.toFixed(3)+'</p>';
    $('actionProbabilities').innerHTML=p.neural_learning.probabilities.map((v,i)=>`<div>${esc(actionName(i))}<div class="bar"><i style="height:${Math.max(0,Math.min(100,v*100))}%"></i></div><b>${(v*100).toFixed(1)}%</b></div>`).join('');
    if(p.steps===0)$('actionProbabilities').innerHTML='<p class="caption">아직 선택한 행동이 없습니다.</p>';
    $('individualLesson').textContent=`${p.lesson_status} · 가중치 변화 ${p.neural_learning.weight_change_l1.toFixed(3)} · 신경 발화 ${fmt(p.brain_spikes)}`;
    drawWeights(p.neural_learning.weights);
    if(detail?.id===selected)renderHistory();
  }
  function drawWeights(weights){
    const c=$('weightMap'),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;
    if(!w||!h)return;c.width=w*dpr;c.height=h*dpr;const g=c.getContext('2d');g.scale(dpr,dpr);
    const max=Math.max(.2,...weights.flat().map(Math.abs));
    const inputs=weights.length,actions=weights[0]?.length||0;
    $('weightDimensions').textContent=`${inputs} × ${actions}`;
    for(let input=0;input<inputs;input++)for(let action=0;action<actions;action++){
      const value=weights[input][action],alpha=.1+.9*Math.min(1,Math.abs(value)/max);
      g.fillStyle=value>=0?`rgba(110,235,192,${alpha})`:`rgba(185,145,228,${alpha})`;
      g.fillRect(input*w/inputs+.5,action*h/actions+.5,w/inputs-1,h/actions-1);
    }
  }
  function renderHistory(){
    $('individualHistory').innerHTML=[...(detail.learning_records||[]).filter(e=>!['birth','reproduction','growth','death'].includes(e.kind)).slice(0,8),...(detail.events||[]).filter(e=>['experience','birth','reproduction','growth','death'].includes(e.kind)).slice(0,18)].map(e=>{
      if(e.kind==='experience')return `<div class="individual-event"><small>행동 ${e.step} · ${esc(actionName(e.action))}</small><br>${esc(e.observation||(e.collision?'충돌':e.new_place?'새 지역':'이동'))}<br>보상 ${e.reward.toFixed(2)} · 행동 가중치 Δ ${e.weight_change.toFixed(3)}${e.internal_change?`<br>내부 연결 ${fmt(e.internal_change.changed)}개 갱신 · 효율 Δ ${e.internal_change.change_l1.toFixed(3)}`:''}</div>`;
      if(e.kind==='source_guided_replay')return `<div class="individual-event"><small>자료 기반 복습 · 기억 ${e.memory_id}</small><p>${esc(e.reason)}</p>실제 경험 ${e.examples}개 · 가중치 Δ ${e.weight_change.toFixed(3)}</div>`;
      if(e.kind==='research')return `<div class="individual-event"><small>자료 탐구 · 독립 검증 전</small><p>${esc(e.record.question)}</p><p>${esc(e.record.finding)}</p>${(e.record.sources||[]).filter(s=>/^https?:\/\//.test(s.url)).map(s=>`<a href="${esc(s.url)}" target="_blank" rel="noreferrer">${esc(s.title)} ↗</a>`).join('<br>')}</div>`;
      return `<div class="individual-event"><small>세계 ${fmt(e.world_tick)} 턴</small><p>${esc(e.observation||e.kind)}</p></div>`;
    }).join('')||'<p class="caption">세계를 실행하면 이 개체의 실제 행동 기록이 쌓입니다.</p>';
  }
  async function loadDetail(){
    const id=selected;if(!id)return;
    try{const next=await api('/api/individuals/'+id);if(id!==selected)return;detail=next;renderHistory();}catch(e){toast(e.message,true);}
  }
  async function refresh(){
    const number=++requestNumber,query=new URLSearchParams({radius:String(radius)});
    if(follow&&selected)query.set('focus',selected);
    else if(center){query.set('x',String(center[0]));query.set('y',String(center[1]));}
    const next=await api('/api/individuals?'+query);if(number!==requestNumber)return;state=next;
    if(state.ready&&!people().some(p=>p.id===selected)){selected=state.agents[0]?.id||state.ancestors[0]?.id||'';renderSettings();}
    render();
  }
  async function poll(){
    try{await refresh();if(visible&&++polls%3===0)await loadDetail();}catch(e){$('worldMessage').textContent='세계 서버 연결 대기';}
    setTimeout(poll,1000);
  }
  function polygon(points,fill,stroke){ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=.8;ctx.stroke();}}
  function draw(now){
    requestAnimationFrame(draw);if(!visible||!state?.ready||document.hidden)return;
    const w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(devicePixelRatio||1,2);if(!w||!h)return;
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const n=state.world.size,s=Math.min(w/(n*1.18),h/(n*.66)),cx=w/2,cy=h*.48;
    const view=(!follow&&center)||state.world.center;
    const project=(x,y,z=0)=>[cx+(x-view[0]-y+view[1])*s*.72,cy+(x+y-view[0]-view[1])*s*.39-z];
    projection={s};hitObjects=[];
    const tile=(x,y,z=0)=>[project(x-.5,y-.5,z),project(x+.5,y-.5,z),project(x+.5,y+.5,z),project(x-.5,y+.5,z)];
    const walls=new Set(state.world.walls.map(p=>p.join(','))),visits=detail?.id===selected?detail.visits:{};
    const palette=['#17332c','#142e2b','#283237','#16313d'];
    for(const [x,y,biome] of state.world.ground){
      const key=x+','+y,wall=walls.has(key),visited=visits?.[key];
      polygon(tile(x,y),wall?'#142831':visited?'#265147':palette[biome],'#2a424444');
      if(wall){const bottom=tile(x,y),top=tile(x,y,s*.34);polygon([top[1],top[2],bottom[2],bottom[1]],'#23404b','#365260');polygon([top[2],top[3],bottom[3],bottom[2]],'#192f3b','#304651');polygon(top,'#34525b','#496773');}
    }
    const p=person();if(p){ctx.beginPath();p.recent.filter(e=>e.position).slice(-22).forEach((event,i)=>{const pt=project(...event.position,2);i?ctx.lineTo(...pt):ctx.moveTo(...pt);});ctx.strokeStyle=colors[Math.max(0,state.agents.indexOf(p))%colors.length]+'90';ctx.lineWidth=2;ctx.stroke();}
    for(const object of [...state.world.objects].sort((a,b)=>a.position[0]+a.position[1]-b.position[0]-b.position[1])){
      const [x,y]=project(...object.position),r=Math.max(3,s*.19);
      hitObjects.push({x,y:y-r,object});ctx.fillStyle=object.color;
      ctx.save();ctx.translate(x,y);
      if(object.id==='tree'){
        ctx.fillStyle='#74654f';ctx.fillRect(-r*.17,-r*2,r*.34,r*2);
        ctx.fillStyle='#477959';ctx.beginPath();ctx.ellipse(-r*.4,-r*2,r,r*1.15,-.15,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='#6c9672';ctx.beginPath();ctx.ellipse(r*.3,-r*2.4,r*.9,r*1.1,.15,0,Math.PI*2);ctx.fill();
      }else if(object.kind==='liquid'){
        ctx.fillStyle='#3e8faca0';ctx.beginPath();ctx.ellipse(0,0,r*1.9,r*.7,0,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#8ddbe880';ctx.beginPath();ctx.ellipse(0,-1,r*1.1,r*.3,0,0,Math.PI);ctx.stroke();
      }else if(object.id==='hut'){
        polygon([[-r,-r],[r,-r],[r,r*.4],[-r,r*.4]],'#7c7560');polygon([[-r*1.5,-r],[0,-r*2.5],[r*1.5,-r]],'#c1a47a','#e2c18b');ctx.fillStyle='#243c38';ctx.fillRect(-r*.3,-r*.5,r*.6,r);
      }else if(object.kind==='shelter'||object.id==='stone'){
        polygon([[-r*1.3,0],[-r*.8,-r], [r*.4,-r*1.5],[r*1.2,-r*.4],[r*.9,r*.3]],object.color,'#afc3bc55');
        if(object.kind==='shelter'){ctx.fillStyle='#14272a';ctx.beginPath();ctx.ellipse(0,0,r*.6,r*.4,0,Math.PI,Math.PI*2);ctx.fill();}
      }else if(object.id==='grass'){
        ctx.strokeStyle=object.color;ctx.lineWidth=1.3;for(let j=-1;j<=1;j++){ctx.beginPath();ctx.moveTo(0,0);ctx.quadraticCurveTo(j*r*.6,-r,j*r,-r*1.6);ctx.stroke();}
      }else if(object.id==='branch'){
        ctx.strokeStyle=object.color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(-r,0);ctx.lineTo(r,-r*.5);ctx.moveTo(0,-r*.25);ctx.lineTo(r*.15,-r);ctx.stroke();
      }else{
        ctx.shadowColor=object.color;ctx.shadowBlur=7;ctx.beginPath();ctx.ellipse(0,-r*.4,r,r*.7,0,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;ctx.fillStyle='#f3e2b9';ctx.beginPath();ctx.arc(-r*.3,-r*.65,Math.max(1,r*.17),0,Math.PI*2);ctx.fill();
      }
      ctx.restore();
    }
    const shown=[...state.agents,...(p&&!p.life.alive?[p]:[])],labels=[];
    shown.forEach((agent,index)=>{
      if(Math.abs(agent.position[0]-view[0])>radius+2||Math.abs(agent.position[1]-view[1])>radius+2)return;
      const cached=positionCache.get(agent.id)||[...agent.position];cached[0]+=(agent.position[0]-cached[0])*.14;cached[1]+=(agent.position[1]-cached[1])*.14;positionCache.set(agent.id,cached);
      const [x,y]=project(...cached,12),color=agent.life.alive?colors[index%colors.length]:'#82949a',isSelected=agent.id===selected;
      hitObjects.push({x,y,agentId:agent.id});
      if(isSelected){ctx.strokeStyle=color+'90';ctx.beginPath();ctx.ellipse(x,y+11,s*.30,s*.15,0,0,Math.PI*2);ctx.stroke();}
      ctx.save();ctx.translate(x,y);ctx.rotate((agent.action<4?agent.action:0)*Math.PI/2-.55);
      const flutter=state.running&&agent.active?Math.sin(now*.025+index)*.2:0;
      if(agent.life.stage==='adult')for(const side of [-1,1]){ctx.save();ctx.rotate(side*(.45+flutter));ctx.fillStyle='#d3eef080';ctx.beginPath();ctx.ellipse(side*6,-1,5,10,side*.45,0,Math.PI*2);ctx.fill();ctx.restore();}
      ctx.fillStyle=color;ctx.shadowBlur=10;ctx.shadowColor=color;ctx.beginPath();ctx.ellipse(0,1,agent.life.stage==='egg'?2.5:3,agent.life.stage==='egg'?4:7,0,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
      if(agent.life.stage==='adult'||agent.life.stage==='larva'){ctx.beginPath();ctx.arc(0,-6,3,0,Math.PI*2);ctx.fill();}ctx.restore();
      ctx.font='11px "Malgun Gothic", sans-serif';ctx.textAlign='center';ctx.fillStyle=color;
      const labelWidth=ctx.measureText(agent.name).width+8;let labelY=y-22;
      while(labels.some(l=>Math.abs(l.x-x)<(l.w+labelWidth)/2&&Math.abs(l.y-labelY)<17))labelY-=18;
      labels.push({x,y:labelY,w:labelWidth});ctx.fillText(agent.name,x,labelY);
      if(labelY<y-22){ctx.strokeStyle=color+'60';ctx.beginPath();ctx.moveTo(x,labelY+4);ctx.lineTo(x,y-12);ctx.stroke();}
      if(isSelected){ctx.font='9px "Malgun Gothic", sans-serif';ctx.fillStyle='#9ab9bd';ctx.fillText(`${agent.life.stage_name} · ${agent.life.generation}세대`,x,y+28);}
    });
  }
  $('individualSelect').addEventListener('change',()=>choose($('individualSelect').value));
  $('worldPlay').addEventListener('click',()=>perform(()=>api('/api/individuals/control',{action:state?.running?'pause':'start'})));
  $('worldStep').addEventListener('click',()=>perform(()=>api('/api/individuals/control',{action:'step'})));
  $('worldSave').addEventListener('click',()=>perform(async()=>{await api('/api/individuals/control',{action:'save'});toast('개체들의 신경 상태와 경험을 저장했습니다.');}));
  $('individualSettings').addEventListener('submit',e=>{e.preventDefault();perform(()=>api(path('/config'),{goal:$('individualGoal').value,description:$('individualDescription').value}));});
  for(const [id,key] of [['individualActive','active'],['individualLearning','learning'],['individualLLM','llm_enabled']])$(id).addEventListener('change',()=>perform(()=>api(path('/config'),{[key]:$(id).checked})));
  $('individualBrain').addEventListener('click',()=>{if(selected){observe(selected);showPanel('inspect');}});
  $('worldFollow').addEventListener('click',()=>{follow=true;center=null;refresh().catch(e=>toast(e.message,true));});
  const zoom=delta=>{radius=Math.max(6,Math.min(24,radius+delta));refresh().catch(e=>toast(e.message,true));};
  $('worldZoomIn').onclick=()=>zoom(-3);$('worldZoomOut').onclick=()=>zoom(3);
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY>0?2:-2);},{passive:false});
  canvas.addEventListener('pointerdown',e=>{if(!state?.ready)return;drag={x:e.clientX,y:e.clientY,center:[...(center&&!follow?center:state.world.center)],moved:false};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!drag||!projection)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(dx,dy)<5&&!drag.moved)return;drag.moved=true;follow=false;const a=dx/(projection.s*.72),b=dy/(projection.s*.39);center=[drag.center[0]-(a+b)/2,drag.center[1]-(b-a)/2].map(v=>Math.max(-999970,Math.min(999970,v)));});
  canvas.addEventListener('pointerup',e=>{
    if(!drag)return;const moved=drag.moved;drag=null;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);
    if(moved){refresh().catch(e=>toast(e.message,true));return;}
    const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
    const hit=hitObjects.map(o=>({...o,d:Math.hypot(o.x-x,o.y-y)})).filter(o=>o.d<14).sort((a,b)=>a.d-b.d)[0];
    if(hit?.agentId){choose(hit.agentId);return;}
    $('worldObject').classList.toggle('hidden',!hit?.object);
    if(hit?.object){const o=hit.object;$('worldObject').innerHTML=`<b>${esc(o.name)}</b><span>${o.position.join(', ')} · ${esc(state.world.biomes[state.world.ground.find(g=>g[0]===o.position[0]&&g[1]===o.position[1])?.[2]]||'')}</span><p>${o.nutrition?'영양 +'+o.nutrition+' ':''}${o.water?'수분 +'+o.water+' ':''}${o.damage?'건강 −'+o.damage+' ':''}${o.material?'재료: '+esc(itemName(o.material)):''}</p><small>관찰자에게 공개된 세계 속성</small>`;}
  });
  canvas.addEventListener('pointercancel',()=>{drag=null;});
  $('individualNew').addEventListener('click',()=>{
    const dialog=document.createElement('dialog');dialog.className='individual-dialog';
    dialog.innerHTML=`<form><span class="eyebrow">A NEW INDIVIDUAL</span><h2>새로운 성체 개체</h2><label for="newIndividualName">이름</label><input id="newIndividualName" maxlength="40" required placeholder="개체의 이름"><label for="newIndividualGoal">행동 목표</label><select id="newIndividualGoal">${goalOptions()}</select><button class="primary full">개체 만들기</button><button type="button" class="outline full">취소</button><p class="caption">알을 포함해 최대 ${state?.population_limit||8}개체를 동시에 계산합니다. 죽은 개체의 기록은 남으며 새 개체의 자리를 차지하지 않습니다. 번식으로 생긴 자손은 알부터 시작합니다.</p></form>`;
    document.body.append(dialog);dialog.showModal();dialog.addEventListener('close',()=>dialog.remove());dialog.querySelector('[type=button]').onclick=()=>dialog.close();
    dialog.querySelector('form').onsubmit=async e=>{e.preventDefault();const button=dialog.querySelector('.primary');button.disabled=true;try{const p=await api('/api/individuals/create',{name:$('newIndividualName').value,goal:$('newIndividualGoal').value});await refresh();choose(p.id);dialog.close();}catch(err){toast(err.message,true);button.disabled=false;}};
  });
  $('individualChatDetails').addEventListener('toggle',async()=>{
    if(!$('individualChatDetails').open||!selected||!person()?.life.alive)return;const id=selected;
    try{const messages=await api(path('/conversation'));if(id===selected&&!chatBusy)$('individualMessages').innerHTML=messages.slice(-8).map(m=>`<article class="message ${m.role==='user'?'user':''}"><small>${m.role==='user'?'YOU':esc(person()?.name)}</small><p>${esc(m.text)}</p></article>`).join('');}catch(e){toast(e.message,true);}
  });
  $('individualChatForm').addEventListener('submit',async e=>{
    e.preventDefault();if(chatBusy||!selected)return;const id=selected,text=$('individualChatInput').value.trim();if(!text)return;
    chatBusy=true;$('individualChatSend').disabled=true;$('individualChatSend').textContent='로컬 모델 응답 중…';
    try{const response=await api(path('/chat'),{text});if(id===selected){$('individualMessages').innerHTML+=`<article class="message user"><small>YOU</small><p>${esc(text)}</p></article><article class="message"><small>${esc(person()?.name)}</small><p>${esc(response.text)}</p></article>`;$('individualChatInput').value='';}}
    catch(error){toast(error.message,true);}finally{chatBusy=false;$('individualChatSend').disabled=false;$('individualChatSend').textContent='메시지 보내기';}
  });
  poll();requestAnimationFrame(draw);
  return {setView(name){visible=name==='world';document.body.classList.toggle('world-mode',visible);$('worldSurface').classList.toggle('hidden',!visible);if(visible){render();loadDetail();}},select:choose};
}

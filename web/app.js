import {NeuronSpace} from './space.js';
import {setupResearch} from './research.js';
const $=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString('ko-KR');
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let state=null,space=null,selected=null,view='inspect',toastTimer,searchTimer,selectionSeq=0,chatBusy=false,lastEventId=0;

async function api(path,body){
  const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
  const res=await fetch(path,options),data=await res.json();
  if(!res.ok)throw Error(data.error||'요청에 실패했습니다.');return data;
}
function toast(text,error=false){clearTimeout(toastTimer);$('toast').textContent=text;$('toast').className='toast'+(error?' error':'');toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),6000);}
function listen(id,handler){$(id).addEventListener('click',async e=>{try{await handler(e);}catch(err){toast(err.message,true);}});}
function setView(name){
  view=name;
  for(const key of ['inspect','memory','modules','events','chat','research'])$(key+'Panel').classList.toggle('hidden',key!==name);
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===name));
  document.querySelectorAll('[data-dock]').forEach(b=>b.classList.toggle('active',b.dataset.dock===name));
  if(name==='memory')loadMemories();if(name==='modules'){renderModules();refreshModel();}if(name==='events')renderEvents();
  if(name==='chat')loadConversation();
}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
document.querySelectorAll('[data-dock]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.dock)));

function renderState(next){
  state=next;const m=next.manifest;
  $('neuronCount').textContent=fmt(m.neuron_count);$('contactCount').textContent=fmt(m.synaptic_contacts);$('edgeCount').textContent=fmt(m.connection_count);
  $('activeCount').textContent=fmt(next.active_neurons);$('spikeCount').textContent=fmt(next.total_spikes);
  $('simTime').innerHTML=`${(next.sim_ms/1000).toFixed(3)} <span>s</span>`;
  $('playBtn').innerHTML=next.running?'<span>Ⅱ</span> 일시정지':'<span>▶</span> 시뮬레이션 실행';
  $('runStatus').textContent=next.running?'전체 뉴런 계산 중':next.ticks?'시뮬레이션 일시정지':'자극을 기다리는 중';
  $('speed').textContent=next.simulation_speed?`${next.simulation_speed.toFixed(2)} ×`:'—';
  $('latency').textContent=next.step_wall_ms?`${next.step_wall_ms.toFixed(1)} ms`:'—';
  drawChart(next.history);if(view==='events'&&next.events[0]?.id!==lastEventId)renderEvents();
}
function updateRenderCount(){if(!space?.data)return;$('renderCount').textContent=`3D 표시 ${fmt(space.visibleCount())} / 계산 ${fmt(space.count)} · 표시 연결 ${fmt(space.showLinks?space.currentEdges.length/3:0)}`;}
function drawChart(data){
  const canvas=$('activityChart'),ctx=canvas.getContext('2d'),dpr=window.devicePixelRatio||1;
  canvas.width=canvas.clientWidth*dpr;canvas.height=canvas.clientHeight*dpr;ctx.scale(dpr,dpr);
  const w=canvas.clientWidth,h=canvas.clientHeight,max=Math.max(1,...data.map(x=>x[1]));
  ctx.strokeStyle='#1b303c';ctx.lineWidth=.5;for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(0,h*i/4);ctx.lineTo(w,h*i/4);ctx.stroke();}
  if(!data.length)return;ctx.beginPath();data.forEach((p,i)=>{const x=i/Math.max(data.length-1,1)*w,y=h-5-p[1]/max*(h-10);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.strokeStyle='#73d7b6';ctx.lineWidth=1.3;ctx.stroke();ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();const fill=ctx.createLinearGradient(0,0,0,h);fill.addColorStop(0,'#5fd5ae33');fill.addColorStop(1,'#5fd5ae00');ctx.fillStyle=fill;ctx.fill();
}
async function selectNeuron(identifier,byIndex=false,focus=false){
  const seq=++selectionSeq,node=await api(`/api/${byIndex?'index':'neuron'}/${identifier}`);if(seq!==selectionSeq)return;
  selected=node;space.select(node,focus);setView('inspect');renderNeuron();updateRenderCount();
}
function renderNeuron(){
  if(!selected)return;const n=selected;
  $('neuronDetail').innerHTML=`<div class="neuron-title"><div class="neuron-symbol">✣</div><div><h3>${escape(n.name)}</h3><small>NEURON ${n.id}</small></div></div><span class="tag">${escape(n.class)}</span>
  <div class="node-stats"><div><small>막전위</small><b id="nodeVoltage">${n.voltage.toFixed(1)} <span style="font-size:10px">mV</span></b></div><div><small>누적 발화</small><b id="nodeSpikes">${fmt(n.spikes)}</b></div></div>
  <dl class="node-meta"><dt>신경전달물질</dt><dd>${escape(n.neurotransmitter)}</dd><dt>계산 부호</dt><dd>${n.sign>0?'흥분성 (+)':n.sign<0?'억제성 (−)':'빠른 전류 효과 없음'}</dd><dt>위치</dt><dd>${n.position?`${n.position.map(v=>v.toFixed(1)).join(' / ')} µm`:'공개 주석에 좌표 없음'}</dd><dt>위치 유형</dt><dd>${n.position_kind===1?'세포체':n.position_kind===2?'세포체 연결부':'미제공'}</dd><dt>들어오는 연결</dt><dd>${fmt(n.incoming_count)}개 · ${fmt(n.incoming_contacts)} 접촉</dd><dt>나가는 연결</dt><dd>${fmt(n.outgoing_count)}개 · ${fmt(n.outgoing_contacts)} 접촉</dd></dl>
  <button id="stimulateSelected" class="primary full">✦ 선택 뉴런에 자극 보내기</button><button id="focusConnections" class="outline full">이 뉴런의 연결망 보기</button>
  <div class="section-title">주요 출력 연결 <b>${fmt(n.outgoing_count)}</b></div>
  ${n.outgoing.slice(0,10).map(e=>`<button class="connection-row" data-neuron="${e.id}"><span>↗ ${escape(e.name)}</span><small>${fmt(e.contacts)} 접촉</small></button>`).join('')||'<p class="caption">선택 집합 내 출력 연결이 없습니다.</p>'}
  <p class="caption">상위 10개 연결 표시. 연결망 보기에서는 입력·출력 각각 최대 180개를 표시합니다.</p>`;
  listen('stimulateSelected',async()=>{await api('/api/stimulate',{ids:[n.id]});toast(`${n.name}에 12 mV 시험 자극을 보냈습니다.`);});
  listen('focusConnections',()=>{space.select(n,true);updateRenderCount();toast('선택 뉴런의 주요 입력·출력 연결을 표시합니다.');});
  $('neuronDetail').querySelectorAll('[data-neuron]').forEach(b=>b.addEventListener('click',()=>selectNeuron(b.dataset.neuron,false,true).catch(e=>toast(e.message,true))));
}
$('neuronSearch').addEventListener('input',()=>{clearTimeout(searchTimer);const query=$('neuronSearch').value.trim();if(!query){$('searchResults').classList.add('hidden');return;}searchTimer=setTimeout(async()=>{
  try{const list=await api('/api/search?q='+encodeURIComponent(query));if($('neuronSearch').value.trim()!==query)return;
    $('searchResults').innerHTML=list.map(n=>`<button data-neuron="${n.id}">${escape(n.name)}<small>${n.id} · ${escape(n.class)}</small></button>`).join('')||'<p class="caption" style="padding:10px">일치하는 뉴런이 없습니다.</p>';
    $('searchResults').classList.remove('hidden');$('searchResults').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{selectNeuron(b.dataset.neuron).catch(e=>toast(e.message,true));$('searchResults').classList.add('hidden');}));
  }catch(e){toast(e.message,true);}},250);});

listen('playBtn',()=>api('/api/control',{action:state.running?'pause':'start'}));
listen('stepBtn',()=>api('/api/control',{action:'step'}));
listen('visualBtn',async()=>{await api('/api/stimulate',{preset:'visual'});toast('시각 감각 뉴런 64개에 시험 자극을 보냈습니다.');});
listen('rotateBtn',()=>{space.rotating=!space.rotating;$('rotateBtn').setAttribute('aria-pressed',String(space.rotating));});
listen('homeBtn',()=>{space.home();updateRenderCount();});
listen('linksBtn',()=>{space.showLinks=!space.showLinks;$('linksBtn').setAttribute('aria-pressed',String(space.showLinks));updateRenderCount();});
document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{space.filter={all:0,visual:1,vnc:2}[b.dataset.filter];document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));updateRenderCount();}));
$('missingToggle').addEventListener('change',()=>{space.showMissing=$('missingToggle').checked;$('unplacedLabel').classList.toggle('hidden',!space.showMissing);updateRenderCount();});

async function loadMemories(){try{const data=await api('/api/memories');$('memoryCount').textContent=fmt(data.count);
  $('memoryList').innerHTML=data.items.map(m=>`<article class="memory-card"><small>MEMORY ${m.id} · ${escape(m.source)}</small><p>${escape(m.text)}</p><small>${escape(m.tokens.slice(0,8).join(' · '))}</small><footer><span>연상 강도 ${m.strength.toFixed(2)} · 학습 ${m.seen}회</span><div><button data-feedback="${m.id}" data-positive="true" aria-label="기억 ${m.id} 강화">＋</button> <button data-feedback="${m.id}" data-positive="false" aria-label="기억 ${m.id} 약화">−</button></div></footer></article>`).join('')||'<div class="empty">아직 저장된 경험이 없습니다.<br>위에서 첫 기억을 가르쳐 주세요.</div>';
  $('memoryList').querySelectorAll('[data-feedback]').forEach(b=>b.addEventListener('click',async()=>{try{await api('/api/feedback',{id:Number(b.dataset.feedback),positive:b.dataset.positive==='true'});await loadMemories();toast('기억의 연상 강도가 변경됐습니다.');}catch(e){toast(e.message,true);}}));
}catch(e){toast(e.message,true);}}
$('learnForm').addEventListener('submit',async e=>{e.preventDefault();const btn=$('learnForm').querySelector('button');btn.disabled=true;try{const result=await api('/api/learn',{text:$('learnText').value,source:$('learnSource').value});$('learnText').value='';await loadMemories();toast(result.created?`새 기억 노드 #${result.memory_id}를 만들었습니다.`:'같은 기억을 다시 학습해 연결 강도를 높였습니다.');}catch(err){toast(err.message,true);}finally{btn.disabled=false;}});
function renderModules(){if(!state)return;$('moduleList').innerHTML=state.modules.map(m=>`<article class="module-card"><h3>${escape(m.name)}</h3><p>${escape(m.description)}</p><button class="switch" role="switch" aria-checked="${m.enabled}" aria-label="${escape(m.name)} 모듈" data-module="${m.id}"></button></article>`).join('');
  $('moduleList').querySelectorAll('[data-module]').forEach(b=>b.addEventListener('click',async()=>{try{b.disabled=true;const result=await api('/api/module',{id:b.dataset.module,enabled:b.getAttribute('aria-checked')!=='true'});state.modules=result.modules;renderModules();}catch(e){toast(e.message,true);b.disabled=false;}}));}
async function refreshModel(){
  const status=await api('/api/llm');$('llmDot').classList.toggle('ready',status.ready);
  $('modelStatus').textContent=status.ready?'● 연결됨 · 로컬 추론 준비':status.connected?'모델 다운로드가 필요합니다.':status.error;
  $('chatModel').textContent=status.ready?`${status.selected} · 이 PC에서 실행`:status.connected?'로컬 모델 준비 중':'로컬 모델 서버에 연결되지 않았습니다.';
  $('modelSelect').innerHTML=status.models.map(m=>`<option ${m===status.selected?'selected':''}>${escape(m)}</option>`).join('')||'<option>설치된 모델 없음</option>';
  $('modelSelect').disabled=!status.models.length;
  return status;
}
listen('modelRefresh',refreshModel);$('modelSelect').addEventListener('change',async()=>{try{await api('/api/model',{model:$('modelSelect').value});await refreshModel();toast('대화 모델을 변경했습니다.');}catch(e){toast(e.message,true);}});
function renderEvents(){if(!state)return;lastEventId=state.events[0]?.id||0;
  $('eventsList').innerHTML=state.events.map(e=>`<article class="event"><small>${new Date(e.time*1000).toLocaleTimeString('ko-KR',{hour12:false})} · SIM ${e.sim_ms.toFixed(0)} ms</small><h4>${escape(e.label)}</h4>${Object.keys(e.details).length?`<details><summary>관측 데이터</summary><pre>${escape(JSON.stringify(e.details,null,2))}</pre></details>`:''}</article>`).join('');}
function messageHtml(m){const t=m.trace||{};return `<article class="message ${m.role}"><small>${m.role==='user'?'YOU':escape(t.model||'NEURONS')}</small><p>${escape(m.text)}</p>${t.steps?`<details><summary>처리 기록 · ${t.seconds}s</summary><ol>${t.steps.map(s=>`<li>${escape(s.label)}<br>${escape(s.detail)}</li>`).join('')}</ol>${t.memories.map(x=>`<p class="reference">기억 ${x.id} · ${escape(x.source)} · 연상 점수 ${x.score}</p>`).join('')}<p class="caption">시스템의 실제 실행 기록입니다. 생물학적 뉴런의 숨은 사고를 해독한 내용은 아닙니다.</p></details>`:''}</article>`;}
async function loadConversation(){try{const items=await api('/api/conversation');if(items.length&&!chatBusy){$('chatMessages').innerHTML=items.map(messageHtml).join('');$('chatMessages').scrollTop=$('chatMessages').scrollHeight;}}catch(e){toast(e.message,true);}}
$('chatForm').addEventListener('submit',async e=>{e.preventDefault();if(chatBusy)return;const text=$('chatInput').value.trim();if(!text)return;chatBusy=true;$('chatSend').disabled=true;
  $('chatMessages').querySelector('.welcome')?.remove();$('chatMessages').insertAdjacentHTML('beforeend',messageHtml({role:'user',text})+'<article class="message pending" id="pendingChat"><small>LOCAL MODEL</small><p>기억과 관측 상태를 읽고 응답하는 중…</p></article>');$('chatMessages').scrollTop=$('chatMessages').scrollHeight;$('chatInput').value='';
  try{const result=await api('/api/chat',{text});$('pendingChat')?.remove();$('chatMessages').insertAdjacentHTML('beforeend',messageHtml({role:'assistant',text:result.text,trace:result.trace}));}
  catch(error){$('pendingChat')?.remove();$('chatMessages').insertAdjacentHTML('beforeend',`<article class="message"><small>CONNECTION</small><p>${escape(error.message)}</p></article>`);$('chatInput').value=text;toast(error.message,true);}
  finally{chatBusy=false;$('chatSend').disabled=false;$('chatMessages').scrollTop=$('chatMessages').scrollHeight;}
});
$('chatInput').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('chatForm').requestSubmit();}});
listen('chatExample',()=>{$('chatInput').value='지금 신경계 상태를 설명해줘';$('chatInput').focus();});
listen('aboutBtn',()=>{const m=state.model;$('modelFacts').innerHTML=`<b>${escape(m.name)}</b><br>시간 간격 ${m.dt_ms} ms · 전도 지연 ${m.delay_ms} ms · 불응기 ${m.refractory_ms} ms<br>임계 막전위 ${m.threshold_mv} mV · 시냅스당 가중치 ${m.gain_mv} mV<br>${m.assumptions.map(escape).join('<br>')}<br>빠른 전류 효과 0인 뉴런: ${fmt(m.zero_fast_effect_neurons)}개`;$('aboutDialog').showModal();});
listen('closeAbout',()=>$('aboutDialog').close());

async function poll(){
  try{const [next,response]=await Promise.all([api('/api/state'),fetch('/api/activity')]);renderState(next);if(response.ok)space.updateActivity(new Uint8Array(await response.arrayBuffer()));}
  catch(e){$('runStatus').textContent='서버 연결 대기';}
  setTimeout(poll,350);
}
async function pollSelection(){
  if(selected&&view==='inspect')try{const id=selected.id,n=await api('/api/neuron/'+id);if(selected?.id===id){selected=n;if($('nodeVoltage'))$('nodeVoltage').innerHTML=`${n.voltage.toFixed(1)} <span style="font-size:10px">mV</span>`;if($('nodeSpikes'))$('nodeSpikes').textContent=fmt(n.spikes);}}catch{}
  setTimeout(pollSelection,1000);
}
async function init(){
  try{
    const next=await api('/api/state');renderState(next);
    const [s,e]=await Promise.all([fetch('/api/scene'),fetch('/api/edges')]);
    if(!s.ok||!e.ok)throw Error('지도 데이터를 불러오지 못했습니다.');
    space=new NeuronSpace($('space'),index=>selectNeuron(index,true).catch(e=>toast(e.message,true)),label=>$('frameCount').textContent=label);
    space.load(new Float32Array(await s.arrayBuffer()),next.classes,new Uint32Array(await e.arrayBuffer()));
    $('loading').classList.add('hidden');updateRenderCount();await selectNeuron(10001);await refreshModel();poll();pollSelection();
    setupResearch({api,space,toast,showPanel:setView});
  }catch(error){$('loading').innerHTML=`<b>관측실을 열지 못했습니다.</b><span>${escape(error.message)}</span>`;console.error(error);}
}
init();

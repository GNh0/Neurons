const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $=id=>document.getElementById(id),fmt=n=>Number(n||0).toLocaleString('ko-KR');
export function setupResearch({api,space,toast,showPanel}){
  let initialized=false,lastGraph=-1,lastHistory='',busy=false;
  function networkMode(learning){
    space.learningMode=learning;
    $('brainMapBtn').classList.toggle('active',!learning);$('learningMapBtn').classList.toggle('active',learning);
    $('brainLegend').classList.toggle('hidden',learning);$('learningLegend').classList.toggle('hidden',!learning);
    $('learningEmpty').classList.toggle('hidden',!learning||Boolean(space.learningData?.count));
    if(learning)showPanel('research');
  }
  $('brainMapBtn').addEventListener('click',()=>networkMode(false));
  $('learningMapBtn').addEventListener('click',()=>networkMode(true));
  space.onConcept=node=>toast(`학습 개념: ${node.label} · 검색 활성 ${Number(node.activity).toFixed(3)}`);
  async function start(limit){
    try{await api('/api/research',{action:'start',goal:$('researchGoal').value,limit,interval:Number($('researchInterval').value)});await poll(true);}
    catch(error){toast(error.message,true);}
  }
  $('researchStart').addEventListener('click',()=>start(Number($('researchLimit').value)));
  $('researchOnce').addEventListener('click',()=>start(1));
  $('researchStop').addEventListener('click',async()=>{try{await api('/api/research',{action:'stop'});toast('진행 중인 요청을 정리한 뒤 탐구를 멈춥니다.');await poll(true);}catch(e){toast(e.message,true);}});
  async function poll(manual=false){
    if(busy)return;busy=true;
    try{
      const [research,graph]=await Promise.all([api('/api/research'),api('/api/learning-graph')]);
      if(!initialized){$('researchGoal').value=research.goal;initialized=true;}
      $('researchPhase').textContent=research.phase;$('researchQuestion').textContent=research.current_question||'목표를 정하면 로컬 모델이 첫 질문을 만듭니다.';
      $('researchCompleted').textContent=fmt(research.completed_cycles);$('conceptCount').textContent=fmt(graph.nodes);$('conceptEdgeCount').textContent=fmt(graph.edges);
      $('researchProgress').textContent=research.running?`${research.session_cycles} / ${research.session_limit}회 · 실행 중`:'실행 버튼으로 다음 탐구를 시작할 수 있습니다.';
      for(const id of ['researchStart','researchOnce','researchGoal','researchLimit','researchInterval'])$(id).disabled=research.running;
      $('researchStop').disabled=!research.running;
      $('researchError').textContent=research.last_error||'';
      if(graph.revision!==lastGraph){space.setLearningGraph(graph);lastGraph=graph.revision;}
      $('learningEmpty').classList.toggle('hidden',!space.learningMode||Boolean(graph.nodes));
      $('learningMapCount').textContent=`학습 노드 ${fmt(graph.nodes)} · 연결 ${fmt(graph.edges)} · 표시 ${fmt(graph.items.length)}개`;
      const key=JSON.stringify(research.history);
      if(key!==lastHistory){lastHistory=key;
        $('researchHistory').innerHTML=research.history.map(r=>`<article class="research-card">
          <small>탐구 ${r.id} · ${escape({completed:'기록 완료',running:'진행 중',failed:'실패',cancelled:'중단',interrupted:'재시작으로 중단'}[r.status]||r.status)}</small>
          <h3>${escape(r.question||'질문 생성 중')}</h3><p class="caption">${escape(r.query)}</p>
          ${r.purpose?`<p>${escape(r.purpose)}</p>`:''}${r.finding?`<p>${escape(r.finding)}</p>`:''}
          ${r.uncertainty?`<p class="caption">불확실성: ${escape(r.uncertainty)}</p>`:''}
          ${r.learning?.memory_id?`<div class="research-change">기억 #${r.learning.memory_id} · 학습 노드 +${r.learning.learning_network?.new_nodes||0} · 연결 +${r.learning.learning_network?.new_edges||0}</div>`:''}
          ${r.error?`<p class="research-error">${escape(r.error)}</p>`:''}
          ${(r.sources||[]).map(s=>`<a class="research-source" href="${escape(/^https?:\/\//.test(s.url)?s.url:'#')}" target="_blank" rel="noreferrer">[${s.id}] ${escape(s.title)} ↗<small>${escape({abstract:'논문 초록 읽음',html_excerpt:'웹 본문 일부 읽음',search_snippet_only:'검색 요약만 확보'}[s.read_level]||s.read_level)}</small></a>`).join('')}
          ${r.next_question?`<details><summary>다음에 확인할 의문</summary><p>${escape(r.next_question)}</p></details>`:''}
          ${r.steps?.length?`<details><summary>실제 실행 기록</summary><pre>${escape(JSON.stringify(r.steps,null,2))}</pre></details>`:''}
        </article>`).join('')||'<p class="empty">아직 자율 탐구 기록이 없습니다.</p>';
      }
    }catch(error){$('researchError').textContent=error.message;}finally{busy=false;if(!manual)setTimeout(poll,2500);}
  }
  poll();
}

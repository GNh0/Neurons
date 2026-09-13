const vertex = `
attribute vec3 aPosition; attribute vec3 aColor;
attribute float aActivity; attribute float aGroup; attribute float aMissing; attribute float aIndex;
uniform float uYaw; uniform float uPitch; uniform float uZoom; uniform float uAspect;
uniform float uFilter; uniform float uMissing; uniform float uSelected; uniform float uSize;
varying vec3 vColor; varying float vActivity; varying float vSelected; varying float vVisible;
void main(){
  vec3 p=aPosition;
  p=vec3(cos(uYaw)*p.x+sin(uYaw)*p.z,p.y,-sin(uYaw)*p.x+cos(uYaw)*p.z);
  p=vec3(p.x,cos(uPitch)*p.y-sin(uPitch)*p.z,sin(uPitch)*p.y+cos(uPitch)*p.z);
  float d=4.0-p.z;
  gl_Position=vec4(p.x*uZoom/d/uAspect,p.y*uZoom/d-0.17,p.z*0.08,1.0);
  vVisible=1.0;
  if(aMissing>0.5 && uMissing<0.5) vVisible=0.0;
  if(uFilter>0.5 && abs(aGroup-uFilter)>0.1) vVisible=0.0;
  vSelected=abs(aIndex-uSelected)<0.1 ? 1.0:0.0;
  vColor=aColor; vActivity=aActivity;
  gl_PointSize=clamp(uSize*(3.6/d)*(1.0+aActivity*2.0+vSelected*3.5)*sqrt(uZoom/2.5),1.0,22.0);
}`;
const pointFragment = `precision mediump float;
varying vec3 vColor; varying float vActivity; varying float vSelected; varying float vVisible;
void main(){ if(vVisible<0.5) discard; float r=length(gl_PointCoord-vec2(0.5))*2.0;
 if(r>1.0) discard;
 float glow=pow(1.0-r,1.8); vec3 c=mix(vColor,vec3(0.8,1.0,0.92),vActivity*0.9);
 c=mix(c,vec3(1.0,0.91,0.56),vSelected);
 gl_FragColor=vec4(c,glow*(0.45+vActivity*0.8+vSelected*0.4));
}`;
const lineFragment = `precision mediump float;
varying vec3 vColor; varying float vActivity; varying float vSelected; varying float vVisible;
uniform float uAlpha;
void main(){if(vVisible<0.5)discard;gl_FragColor=vec4(mix(vColor,vec3(0.65,1.0,0.84),vActivity),uAlpha+vActivity*0.2);}`;

function program(gl, fragment) {
  const compile=(type,text)=>{const shader=gl.createShader(type);gl.shaderSource(shader,text);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));return shader;};
  const p=gl.createProgram();gl.attachShader(p,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(p,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));
  const result={p,attributes:{},uniforms:{}};
  for(const k of ['aPosition','aColor','aActivity','aGroup','aMissing','aIndex']) result.attributes[k]=gl.getAttribLocation(p,k);
  for(const k of ['uYaw','uPitch','uZoom','uAspect','uFilter','uMissing','uSelected','uSize','uAlpha']) result.uniforms[k]=gl.getUniformLocation(p,k);
  return result;
}

export class NeuronSpace {
  constructor(canvas,onSelect,onFrame){
    this.canvas=canvas;this.onSelect=onSelect;this.onFrame=onFrame;
    this.gl=canvas.getContext('webgl',{antialias:true,alpha:true,powerPreference:'high-performance'});
    if(!this.gl)throw Error('WebGL을 사용할 수 없습니다. 브라우저의 하드웨어 가속을 확인해 주세요.');
    this.points=program(this.gl,pointFragment);this.lines=program(this.gl,lineFragment);
    this.yaw=-.08;this.pitch=.10;this.zoom=2.6;this.filter=0;this.showMissing=false;
    this.selected=-10;this.rotating=false;this.showLinks=true;this.data=null;this.edgeData=null;
    this.frame=0;this.frames=0;this.frameStart=performance.now();this.lastTime=performance.now();
    this.drag=null;this.handleInput();this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(canvas);
    canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();this.lost=true;this.onFrame?.('GPU 연결 끊김 · 새로고침 필요');});
    this.animate=this.animate.bind(this);requestAnimationFrame(this.animate);
  }
  makeBuffer(data){const gl=this.gl,b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);return b;}
  load(scene,classes,edgeArray){
    this.classes=classes;this.count=scene.length/5;this.positions=new Float32Array(this.count*3);
    this.colors=new Float32Array(this.count*3);this.groups=new Float32Array(this.count);
    this.missing=new Float32Array(this.count);this.activity=new Float32Array(this.count);
    this.indices=new Float32Array(this.count);this.knownCount=0;
    const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<this.count;i++)if(scene[i*5+4]>0){for(let k=0;k<3;k++){min[k]=Math.min(min[k],scene[i*5+k]);max[k]=Math.max(max[k],scene[i*5+k]);}}
    const center=min.map((v,k)=>(v+max[k])/2),span=Math.max(...max.map((v,k)=>v-min[k]));
    let missingIndex=0;
    for(let i=0;i<this.count;i++){
      const j=i*5,p=i*3,kind=scene[j+4],name=classes[scene[j+3]];
      this.indices[i]=i;this.missing[i]=kind?0:1;
      let color=[.32,.80,.76],group=3;
      if(name.includes('visual')||name.startsWith('ol_')){color=[.32,.58,.96];group=1;}
      else if(name.startsWith('vnc_')){color=[.65,.48,.91];group=2;}
      else if(/sensory|motor|descending|ascending/.test(name)){color=[.85,.70,.40];group=3;}
      this.colors.set(color,p);this.groups[i]=group;
      if(kind){
        this.knownCount++;
        this.positions[p]=(scene[j]-center[0])/span*1.85;
        this.positions[p+1]=-(scene[j+2]-center[2])/span*1.85;
        this.positions[p+2]=(scene[j+1]-center[1])/span*1.85;
      }else{
        // Explicit separate non-anatomical cluster, hidden by default.
        const t=missingIndex++*2.399963,r=.29*Math.sqrt((missingIndex%1500)/1500);
        this.positions[p]=1.35+Math.cos(t)*r;this.positions[p+1]=-.38+Math.sin(t)*r;
        this.positions[p+2]=.2*Math.sin(missingIndex*.29);this.colors.set([.58,.44,.38],p);
      }
    }
    this.data=this.createGroup(this.positions,this.colors,this.groups,this.missing,this.indices,this.activity);
    this.allEdges=edgeArray;this.setEdges(edgeArray,false);
    // Decorative background stars have no connection to simulation state.
    const n=1300,pos=new Float32Array(n*3),col=new Float32Array(n*3);
    let seed=23;const rnd=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;};
    for(let i=0;i<n;i++){pos.set([(rnd()-.5)*12,(rnd()-.5)*8,-3-rnd()*4],i*3);col.set([.22,.30,.39],i*3);}
    this.stars=this.createGroup(pos,col,new Float32Array(n),new Float32Array(n),new Float32Array(n).fill(-100),new Float32Array(n));
  }
  createGroup(pos,color,group,missing,index,activity){return{count:pos.length/3,buffers:{aPosition:this.makeBuffer(pos),aColor:this.makeBuffer(color),aGroup:this.makeBuffer(group),aMissing:this.makeBuffer(missing),aIndex:this.makeBuffer(index),aActivity:this.makeBuffer(activity)}};}
  destroyGroup(group){if(group)for(const b of Object.values(group.buffers))this.gl.deleteBuffer(b);}
  setEdges(edges,focus){
    this.destroyGroup(this.edgeData);this.focusEdges=focus;this.currentEdges=edges;const count=edges.length/3*2;
    const pos=new Float32Array(count*3),col=new Float32Array(count*3),groups=new Float32Array(count),missing=new Float32Array(count),ids=new Float32Array(count).fill(-100);
    for(let i=0;i<edges.length/3;i++)for(let end=0;end<2;end++){
      const source=edges[i*3+end],dst=i*2+end;
      pos.set(this.positions.subarray(source*3,source*3+3),dst*3);
      col.set(focus?[.47,.79,.68]:[.26,.52,.60],dst*3);groups[dst]=this.groups[source];missing[dst]=this.missing[source];
    }
    this.edgeActivity=new Float32Array(count);this.edgeData=this.createGroup(pos,col,groups,missing,ids,this.edgeActivity);
  }
  select(node,focus=false){
    this.selected=node.index;
    if(focus){const edges=[];for(const e of node.outgoing)edges.push(node.index,e.index,e.contacts);for(const e of node.incoming)edges.push(e.index,node.index,e.contacts);this.setEdges(new Uint32Array(edges),true);}
  }
  overview(){if(this.allEdges)this.setEdges(this.allEdges,false);}
  updateActivity(bytes){
    if(!this.data||bytes.length!==this.count)return;
    for(let i=0;i<bytes.length;i++)this.activity[i]=bytes[i]/255;
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.data.buffers.aActivity);this.gl.bufferSubData(this.gl.ARRAY_BUFFER,0,this.activity);
    for(let i=0;i<this.currentEdges.length/3;i++)this.edgeActivity[i*2]=this.edgeActivity[i*2+1]=this.activity[this.currentEdges[i*3]];
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.edgeData.buffers.aActivity);this.gl.bufferSubData(this.gl.ARRAY_BUFFER,0,this.edgeActivity);
  }
  resize(){const dpr=Math.min(window.devicePixelRatio||1,2);this.canvas.width=Math.round(this.canvas.clientWidth*dpr);this.canvas.height=Math.round(this.canvas.clientHeight*dpr);this.gl.viewport(0,0,this.canvas.width,this.canvas.height);this.dpr=dpr;}
  draw(p,group,mode,options={}){
    if(!group)return;const gl=this.gl;gl.useProgram(p.p);
    for(const [name,buffer] of Object.entries(group.buffers)){
      const loc=p.attributes[name];if(loc<0)continue;gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,(name==='aPosition'||name==='aColor')?3:1,gl.FLOAT,false,0,0);
    }
    const uniforms={uYaw:options.stars?this.yaw*.05:this.yaw,uPitch:options.stars?0:this.pitch,uZoom:this.zoom,
      uAspect:this.canvas.width/this.canvas.height,uFilter:options.stars?0:this.filter,uMissing:this.showMissing?1:0,
      uSelected:this.selected,uSize:(options.stars?1.5:2.0)*(this.dpr||1),uAlpha:this.focusEdges?.15:.028};
    for(const [key,value] of Object.entries(uniforms))if(p.uniforms[key]!==null)gl.uniform1f(p.uniforms[key],value);
    gl.drawArrays(mode,0,group.count);
  }
  animate(now){
    if(this.lost)return;
    const gl=this.gl,delta=Math.min(100,now-this.lastTime);this.lastTime=now;
    if(this.rotating&&!this.drag)this.yaw+=delta*.000045;
    gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE);
    this.draw(this.points,this.stars,gl.POINTS,{stars:true});
    if(this.showLinks)this.draw(this.lines,this.edgeData,gl.LINES);
    this.draw(this.points,this.data,gl.POINTS);
    this.frames++;if(now-this.frameStart>1500){this.onFrame?.(`${Math.round(this.frames*1000/(now-this.frameStart))} FPS · WebGL`);this.frames=0;this.frameStart=now;}
    requestAnimationFrame(this.animate);
  }
  projected(index){
    const p=this.positions.subarray(index*3,index*3+3);const x=Math.cos(this.yaw)*p[0]+Math.sin(this.yaw)*p[2],z=-Math.sin(this.yaw)*p[0]+Math.cos(this.yaw)*p[2];
    const y=Math.cos(this.pitch)*p[1]-Math.sin(this.pitch)*z,rz=Math.sin(this.pitch)*p[1]+Math.cos(this.pitch)*z;
    const aspect=this.canvas.clientWidth/this.canvas.clientHeight;
    return[(x*this.zoom/(4-rz)/aspect+1)*.5*this.canvas.clientWidth,(1-(y*this.zoom/(4-rz)-.17))*.5*this.canvas.clientHeight];
  }
  pick(x,y){
    if(!this.data)return;let best=-1,dist=100;
    for(let i=0;i<this.count;i++){
      if((!this.showMissing&&this.missing[i])||(this.filter&&this.groups[i]!==this.filter))continue;
      const p=this.projected(i),d=(p[0]-x)**2+(p[1]-y)**2;if(d<dist){dist=d;best=i;}
    }
    if(best>=0)this.onSelect(best);
  }
  handleInput(){
    this.canvas.addEventListener('pointerdown',e=>{this.drag={x:e.clientX,y:e.clientY,sx:e.clientX,sy:e.clientY};this.canvas.setPointerCapture(e.pointerId);});
    this.canvas.addEventListener('pointermove',e=>{if(!this.drag)return;this.yaw+=(e.clientX-this.drag.x)*.006;this.pitch=Math.max(-1.45,Math.min(1.45,this.pitch+(e.clientY-this.drag.y)*.006));this.drag.x=e.clientX;this.drag.y=e.clientY;});
    this.canvas.addEventListener('pointerup',e=>{if(this.drag&&Math.hypot(e.clientX-this.drag.sx,e.clientY-this.drag.sy)<5){const r=this.canvas.getBoundingClientRect();this.pick(e.clientX-r.left,e.clientY-r.top);}this.drag=null;});
    this.canvas.addEventListener('pointercancel',()=>this.drag=null);
    this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom=Math.max(.7,Math.min(20,this.zoom*Math.exp(-e.deltaY*.001)));},{passive:false});
  }
  home(){this.yaw=-.08;this.pitch=.1;this.zoom=2.6;this.overview();}
  visibleCount(){let n=0;for(let i=0;i<this.count;i++)if((this.showMissing||!this.missing[i])&&(!this.filter||this.groups[i]===this.filter))n++;return n;}
}

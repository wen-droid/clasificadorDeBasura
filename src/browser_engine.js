/* Inferencia de la CNN propia. No contiene aprendizaje ni pesos de terceros.
   Orden de tensores CHW; kernels OIHW, igual que PyTorch. */
function decodeWeights(encoded) {
  const raw=atob(encoded),bytes=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++) bytes[i]=raw.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
function convRelu(x,ci,co,h,w,weights,bias,stride=1) {
  const ph=h+2,pw=w+2,padded=new Float32Array(ci*ph*pw);
  for(let c=0;c<ci;c++)for(let y=0;y<h;y++)
    padded.set(x.subarray(c*h*w+y*w,c*h*w+(y+1)*w),c*ph*pw+(y+1)*pw+1);
  const oh=Math.ceil(h/stride),ow=Math.ceil(w/stride),out=new Float32Array(co*oh*ow);
  for(let o=0;o<co;o++)for(let y=0;y<oh;y++)for(let xx=0;xx<ow;xx++){
    let sum=bias[o],wi=o*ci*9;
    for(let c=0;c<ci;c++){
      let pi=c*ph*pw+(y*pw+xx)*stride;
      for(let ky=0;ky<3;ky++,pi+=pw){
        sum+=padded[pi]*weights[wi++]+padded[pi+1]*weights[wi++]+padded[pi+2]*weights[wi++];
      }
    }
    out[o*oh*ow+y*ow+xx]=Math.max(0,sum);
  }
  return out;
}
function maxPool(x,c,h,w){
  const oh=h/2,ow=w/2,out=new Float32Array(c*oh*ow);
  for(let ch=0;ch<c;ch++)for(let y=0;y<oh;y++)for(let xx=0;xx<ow;xx++){
    const i=ch*h*w+y*2*w+xx*2;
    out[ch*oh*ow+y*ow+xx]=Math.max(x[i],x[i+1],x[i+w],x[i+w+1]);
  }
  return out;
}
function averagePool3(x){
  const out=new Float32Array(96*16);
  for(let c=0;c<96;c++)for(let y=0;y<4;y++)for(let xx=0;xx<4;xx++){
    let s=0;
    for(let dy=0;dy<3;dy++)for(let dx=0;dx<3;dx++)s+=x[c*144+(y*3+dy)*12+xx*3+dx];
    out[c*16+y*4+xx]=s/9;
  }
  return out;
}
function dense(x,outputs,w,b,relu=false){
  const out=new Float32Array(outputs);
  for(let o=0;o<outputs;o++){
    let sum=b[o];for(let i=0;i<x.length;i++)sum+=w[o*x.length+i]*x[i];
    out[o]=relu?Math.max(0,sum):sum;
  }
  return out;
}
function materialScores(input, weights) {
  // Promedio 2x2 de la imagen de entrada, seguido de cuatro bloques conv/ReLU/max-pool.
  let side = 48;
  let x = new Float32Array(3 * side * side);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < side; y++) {
      for (let col = 0; col < side; col++) {
        const i = c * 96 * 96 + 2 * y * 96 + 2 * col;
        x[c * side * side + y * side + col] =
          (input[i] + input[i+1] + input[i+96] + input[i+97]) / 4;
      }
    }
  }
  let channels = 3;
  for (let block = 0; block < 4; block++) {
    const bias = weights['block' + block + '.bias'];
    x = convRelu(x, channels, bias.length, side, side, weights['block' + block + '.weight'], bias);
    x = maxPool(x, bias.length, side, side);
    channels = bias.length;
    side /= 2;
  }
  const logits = dense(x, 6, weights['head.weight'], weights['head.bias']);
  const max = Math.max(...logits);
  const exp = Array.from(logits, value => Math.exp(value - max));
  const sum = exp.reduce((a,b) => a+b, 0);
  return exp.map(value => value/sum);
}
function objectScores(input, weights) {
  let x = input, side = 96, channels = 3;
  for (let i = 0; i < 4; i++) {
    const key = 'stage' + i;
    const bias = weights[key + '.bias'];
    const stride = i === 0 ? 2 : 1;
    x = convRelu(x, channels, bias.length, side, side, weights[key + '.weight'], bias, stride);
    channels = bias.length;
    side /= stride;
    if (i >= 2) {
      x = convRelu(x, channels, channels, side, side, weights[key + '.extra.weight'], weights[key + '.extra.bias']);
    }
    x = maxPool(x, channels, side, side);
    side /= 2;
  }
  const logits = dense(x, weights['head.bias'].length, weights['head.weight'], weights['head.bias']);
  const maximum = Math.max(...logits);
  const exp = Array.from(logits, v => Math.exp(v - maximum));
  const total = exp.reduce((a,b) => a+b,0);
  return exp.map(v => v/total);
}
function predictCNN(input,weights){
  if (weights['stage0.weight']) {
    const mirror = new Float32Array(input.length);
    for (let row = 0; row < input.length; row += 96)
      for (let i = 0; i < 96; i++) mirror[row+i] = input[row+95-i];
    const a = objectScores(input, weights), b = objectScores(mirror, weights);
    return a.map((v,i) => (v+b[i])/2);
  }
  if (weights['head.weight']) return materialScores(input, weights);
  const w=weights;
  let x=convRelu(input,3,24,96,96,w['features.0.weight'],w['features.0.bias']);
  x=maxPool(x,24,96,96);
  x=convRelu(x,24,48,48,48,w['features.3.weight'],w['features.3.bias']);
  x=maxPool(x,48,48,48);
  x=convRelu(x,48,96,24,24,w['features.6.weight'],w['features.6.bias']);
  x=averagePool3(maxPool(x,96,24,24));
  x=dense(x,64,w['classifier.1.weight'],w['classifier.1.bias'],true);
  x=dense(x,w['classifier.4.bias'].length,w['classifier.4.weight'],w['classifier.4.bias']);
  const maximum=Math.max(...x),exp=Array.from(x,v=>Math.exp(v-maximum)),total=exp.reduce((a,b)=>a+b,0);
  return exp.map(v=>v/total);
}
// Redondeo al par, como round de Python/Pillow para tamaños y centrado.
function pyRound(v){const f=Math.floor(v),d=v-f;return d===.5?(f%2?f+1:f):Math.round(v)}
function coefficients(input,output){
  const scale=input/output,filter=Math.max(1,scale),precision=1<<22;
  return Array.from({length:output},(_,i)=>{
    const center=(i+.5)*scale;
    const start=Math.max(0,Math.floor(center-filter+.5));
    const end=Math.min(input,Math.floor(center+filter+.5));
    const values=[];let total=0;
    for(let j=start;j<end;j++){const weight=Math.max(0,1-Math.abs((j-center+.5)/filter));values.push(weight);total+=weight}
    return {start,values:values.map(v=>Math.floor(v/total*precision+.5))};
  });
}
function preprocessRGBA(rgba,width,height){
  let nw=96,nh=96;
  if(width>height)nh=Math.max(1,pyRound(height/width*96));
  else if(height>width)nw=Math.max(1,pyRound(width/height*96));
  const cx=coefficients(width,nw),cy=coefficients(height,nh),temp=new Uint8Array(nw*height*3);
  const precision=1<<22,half=1<<21;
  // Interpolación bilineal separable con soporte ampliado al reducir, como Pillow.
  for(let y=0;y<height;y++)for(let x=0;x<nw;x++)for(let c=0;c<3;c++){
    let sum=half;const coeff=cx[x];
    for(let k=0;k<coeff.values.length;k++)sum+=rgba[(y*width+coeff.start+k)*4+c]*coeff.values[k];
    temp[(y*nw+x)*3+c]=Math.min(255,Math.max(0,Math.floor(sum/precision)));
  }
  const out=new Float32Array(3*96*96);out.fill(128/255);
  const ox=pyRound((96-nw)/2),oy=pyRound((96-nh)/2);
  for(let y=0;y<nh;y++)for(let x=0;x<nw;x++)for(let c=0;c<3;c++){
    let sum=half;const coeff=cy[y];
    for(let k=0;k<coeff.values.length;k++)sum+=temp[((coeff.start+k)*nw+x)*3+c]*coeff.values[k];
    const value=Math.min(255,Math.max(0,Math.floor(sum/precision)));
    out[c*96*96+(y+oy)*96+x+ox]=value/255;
  }
  return out;
}
if(typeof module!=='undefined')module.exports={decodeWeights,predictCNN,preprocessRGBA};

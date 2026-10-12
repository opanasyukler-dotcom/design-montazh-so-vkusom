import numpy as np, wave, librosa, cv2
SR=48000; DUR=44.0; M0=7.0
src=open("../v14/audio.py").read(); exec(src[src.index("rng=np.random"):src.index("n=int(DUR*SR)")])
def impact():
    k=int(0.9*SR); x=np.arange(k)/SR; b=np.sin(2*np.pi*np.cumsum(45+120*np.exp(-x*22))/SR)*np.exp(-x*5)
    nz=lp(rng.standard_normal(k),np.full(k,3000.0))*np.exp(-x*14)*0.6; o=b+nz; return o/np.abs(o).max()
def riser(d=1.2):
    k=int(d*SR); x_=np.linspace(0,1,k); x=rng.standard_normal(k); o=lp(x,300+8000*x_**2)-lp(x,150+2000*x_**2); o*=x_**2.2; return o/np.abs(o).max()
n=int(DUR*SR)
SEGS=[(6.50,8.48,0.35),(10.38,13.96,2.75),(17.55,22.70,6.6),(23.12,25.72,11.95),(25.80,32.40,14.65),(42.85,45.05,21.6),
      (71.48,75.08,24.1),(130.38,131.98,28.0),(132.88,133.47,29.75),(134.17,137.88,30.34),(36.45,37.19,34.35)]
vs,_=librosa.load("voice_src.wav",sr=SR,mono=True); vo=np.zeros(n)
for si,so,t0 in SEGS:
    s=vs[int(si*SR):int(so*SR)].copy(); f=int(0.012*SR); s[:f]*=np.linspace(0,1,f); s[-f:]*=np.linspace(1,0,f)
    i=int(t0*SR); vo[i:i+len(s)]+=s[:n-i]
vo/=np.sqrt(np.mean(vo[np.abs(vo)>1e-4]**2))*8
y,_=librosa.load("mus.wav",sr=SR,mono=False,offset=M0,duration=DUR); mus=y.T[:n].copy(); mus/=np.abs(mus).max()
tt=np.arange(len(mus))/SR
act=np.zeros(n)
for si,so,t0 in SEGS: act[int((t0-0.05)*SR):int((t0+so-si+0.05)*SR)]=1
d=cv2.GaussianBlur(act.astype(np.float32).reshape(1,-1),(0,0),sigmaX=SR*0.12).ravel()
fade=np.clip((DUR-0.3-tt)/5.5,0,1)**1.6                     # long fade-out over the collages
mus*=((0.36-0.27*np.clip(d,0,1))[:len(mus)]*np.minimum(1,tt/0.4)*fade)[:,None]
sfx=np.zeros(n+SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]*g
WB,PP,IM,RS=whoosh(),pop(),impact(),riser(); KEYS=[key() for _ in range(8)]
for c in [2.0,2.75,6.45,8.85,11.85,12.75,13.7,14.6,19.6,23.9,26.7,28.0,31.2,34.3,35.4,38.3,41.2]: put(WB,c-0.25,0.14)
put(IM,2.0,0.35); put(IM,26.7,0.35); put(RS,15.7-1.1,0.18); put(IM,0.0,0.25)
TYPED=[(0.2,24,"ЖИЗНЬ РАЗДЕЛИЛАСЬНА ДО И ПОСЛЕ"),(3.45,30,"УВЕРЕННОСТЬ В СЕБЕэто дорогого стоит"),(12.0,34,"ВСЁ ВРЕМЯПРИКРЫВАЛАСЬ ОДЕЖДОЙ"),
       (17.9,30,"УДОВОЛЬСТВИЕОТ СВОЕГО ТЕЛА"),(21.75,32,"ПРИЕХАЛА ИЗДАЛЕКАиз Саратовской области"),(24.3,34,"ОЧЕНЬ СЛОЖНАЯИ ДЛИННАЯ ОПЕРАЦИЯ"),
       (26.75,40,"РЕЗУЛЬТАТПОТРЯСАЮЩИЙ"),(34.35,34,"ОЧЕНЬ ДОВОЛЬНА")]
for t0,cps,txt in TYPED:
    for i,ch in enumerate(txt):
        if ch!=" ": put(KEYS[rng.integers(8)],t0+i/cps+rng.uniform(-0.006,0.006),0.2*rng.uniform(0.7,1))
for t in [9.0,28.35,30.6,35.65,38.55,41.45]: put(PP,t,0.16)
mix=mus+vo[:,None]+sfx[:n,None]*0.5
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

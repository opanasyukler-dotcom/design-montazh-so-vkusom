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
def shutter():
    k=int(0.12*SR); o=np.zeros(k)
    for st,g in [(0,1.0),(int(0.045*SR),0.7)]:
        m=int(0.02*SR); nz=rng.standard_normal(m); nz=nz-lp(nz,np.full(m,2500.0)); o[st:st+m]+=nz*np.exp(-np.arange(m)/SR*260)*g
    return o/np.abs(o).max()
def boom():
    k=int(0.5*SR); x=np.arange(k)/SR; o=np.sin(2*np.pi*np.cumsum(38+60*np.exp(-x*18))/SR)*np.exp(-x*7); return o/np.abs(o).max()
def shimmer(d=1.6):
    k=int(d*SR); x=np.arange(k)/SR; o=np.zeros(k)
    for f in rng.uniform(2200,6500,14): o+=np.sin(2*np.pi*f*x+rng.uniform(0,6))*np.exp(-x*rng.uniform(2,5))*(1+0.5*np.sin(2*np.pi*rng.uniform(5,11)*x))
    o*=np.minimum(1,x/0.02); return o/np.abs(o).max()
def ding():
    k=int(1.4*SR); x=np.arange(k)/SR
    o=sum(a_*np.sin(2*np.pi*f*x)*np.exp(-x*kk) for f,a_,kk in [(1318.5,1,3),(1975.5,.5,4),(2637,.3,6),(659.25,.4,2.5)]); return o*np.minimum(1,x/0.004)/np.abs(o).max()
SH,BM,SM,DG=shutter(),boom(),shimmer(),ding()
sfx=np.zeros(n+SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]*g
WB,PP,IM,RS=whoosh(),pop(),impact(),riser(); KEYS=[key() for _ in range(8)]
for c in [2.0,2.75,6.45,8.85,11.85,12.75,13.7,14.6,19.6,23.9,26.2,30.2,32.2,34.3,35.4,38.3,41.2]: put(WB,c-0.25,0.24); put(BM,c,0.22)
put(IM,2.0,0.2); put(RS,26.7-1.1,0.2); put(IM,26.75,0.3); put(IM,0.0,0.25)
SW=whoosh(0.28,800,9000,0.5)
TITLES=[(0.2,2),(3.45,2),(9.0,2),(12.0,2),(17.9,2),(21.75,2),(24.2,2),(27.8,2),(34.35,1)]
for t0,nl in TITLES:
    for k in range(nl): put(SW,t0+k*0.16-0.05,0.13)
    put(PP,t0+0.05,0.12)
for t in [9.0,30.45,31.0,35.65,38.55,41.45]: put(PP,t,0.2)
for t in [0.02,2.0,11.85,12.75,13.7,23.9,30.2,32.2,35.4,38.3,41.2]: put(SH,t+0.02,0.22)      # camera shutter on photo cuts
put(SM,26.75,0.2); put(SM,35.45,0.14); put(DG,34.45,0.12); put(RS,35.4-1.0,0.14)
mix=mus+vo[:,None]+sfx[:n,None]*0.5
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

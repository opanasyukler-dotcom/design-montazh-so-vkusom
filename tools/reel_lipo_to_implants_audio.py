import numpy as np, wave, librosa, cv2
SR=48000; DUR=41.0; M0=7.0
src=open("../v14/audio.py").read(); exec(src[src.index("rng=np.random"):src.index("n=int(DUR*SR)")])
s2=open("../v15/audio.py").read()
exec(s2[s2.index("def impact"):s2.index("n=int(DUR*SR)", s2.index("def impact"))])
exec(s2[s2.index("def shutter"):s2.index("SH,BM,SM,DG=")])
n=int(DUR*SR)
SEGS=[("IMG_3119",0.00,3.16,0.30),("IMG_3119",5.25,8.58,3.50),("IMG_3119",8.81,10.80,6.95),("IMG_3116",3.70,5.40,9.20),("IMG_3116",5.69,9.05,10.95),
      ("IMG_3119",18.24,20.75,14.55),("IMG_3144",1.81,2.36,18.55),("IMG_3144",4.03,5.17,19.20),("IMG_3138",5.33,7.30,22.30),
      ("IMG_3138",11.29,13.69,24.50),("IMG_3138",14.10,15.92,26.95),("IMG_1360",2.50,4.84,29.00)]
src_a={}; vo=np.zeros(n)
for s,si,so,t0 in SEGS:
    if s not in src_a:
        x,_=librosa.load(f"a/{s}_48.wav",sr=SR,mono=True); src_a[s]=x/ (np.sqrt(np.mean(x[np.abs(x)>1e-3]**2))+1e-6)
    seg=src_a[s][int(si*SR):int(so*SR)].copy()*(1.6 if s in ('IMG_3144','IMG_1360') else 1.0); f=int(0.012*SR); seg[:f]*=np.linspace(0,1,f); seg[-f:]*=np.linspace(1,0,f)
    i=int(t0*SR); vo[i:i+len(seg)]+=seg[:n-i]
vo*=0.125
y,_=librosa.load("mus.wav",sr=SR,mono=False,offset=M0,duration=DUR); mus=y.T[:n].copy(); mus/=np.abs(mus).max()
tt=np.arange(len(mus))/SR; act=np.zeros(n)
for s,si,so,t0 in SEGS: act[int((t0-0.05)*SR):int((t0+so-si+0.05)*SR)]=1
d=cv2.GaussianBlur(act.astype(np.float32).reshape(1,-1),(0,0),sigmaX=SR*0.12).ravel()
fade=np.clip((DUR-0.3-tt)/4.5,0,1)**1.6
mus*=((0.34-0.30*np.clip(d,0,1))[:len(mus)]*np.minimum(1,tt/0.3)*fade)[:,None]
sfx=np.zeros(n+SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]*g
WB,PP,IM,RS=whoosh(),pop(),impact(),riser(); SH,BM,SM,DG=shutter(),boom(),shimmer(),ding(); SW=whoosh(0.28,800,9000,0.5)
for c in [3.5,6.8,9.1,14.4,17.06,18.5,20.6,21.15,21.7,22.3,24.4,28.9,31.5,33.75,36.0]: put(WB,c-0.25,0.22); put(BM,c,0.2)
for t in [0.0,6.8,7.15,22.3,24.4,31.5,33.75]: put(SH,t+0.02,0.2)
for t in [17.6,17.85,18.1,36.7,36.9,37.1,14.4]: put(PP,t,0.22)
put(RS,24.9-1.0,0.16); put(SM,25.0,0.18); put(IM,19.85,0.3); put(SM,19.9,0.14); put(IM,20.6,0.25); put(IM,22.3,0.2)
for t0,nl in [(0.2,2),(6.95,2),(10.0,2),(12.7,1),(14.8,2),(17.2,2),(19.85,1),(22.5,2),(26.0,2),(29.1,1),(36.1,3)]:
    for k in range(nl): put(SW,t0+k*0.16-0.05,0.13)
for t in [37.5,37.8,38.1]: put(PP,t,0.16)
put(DG,29.15,0.1)
mix=mus+vo[:,None]+sfx[:n,None]*0.5
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

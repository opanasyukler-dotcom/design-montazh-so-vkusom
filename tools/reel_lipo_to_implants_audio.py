import numpy as np, wave, librosa, cv2
SR=48000; DUR=48.6; M0=6.5
src=open("../v14/audio.py").read(); exec(src[src.index("rng=np.random"):src.index("n=int(DUR*SR)")])
s2=open("../v15/audio.py").read()
exec(s2[s2.index("def impact"):s2.index("n=int(DUR*SR)", s2.index("def impact"))])
exec(s2[s2.index("def shutter"):s2.index("SH,BM,SM,DG=")])
n=int(DUR*SR)
SEGS=[("IMG_3119",0.00,3.35,0.30),("IMG_3119",5.25,8.62,4.00),("IMG_3119",8.81,10.90,7.85),("IMG_3116",3.70,5.50,10.85),("IMG_3116",5.69,9.15,12.95),
      ("IMG_3119",18.24,20.80,17.45),("IMG_3144",1.81,2.40,22.15),("IMG_3144",4.03,6.72,22.95),("IMG_3138",5.33,7.30,29.05),
      ("IMG_3138",11.29,13.69,31.70),("IMG_3138",14.10,16.00,34.35),("IMG_1360",2.50,4.90,36.85)]
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
for c in [3.9,7.6,10.6,17.0,20.4,22.0,26.6,27.3,28.0,28.8,31.4,36.6,39.8,42.6,45.4]: put(WB,c-0.3,0.2); put(BM,c,0.18)
for t in [0.0,7.6,7.95,28.8,31.4,39.8,42.6]: put(SH,t+0.02,0.2)
for t in [21.2,21.65,22.1,17.0]: put(PP,t,0.22)
put(RS,32.2-1.0,0.16); put(SM,32.3,0.18); put(IM,25.05,0.3); put(SM,25.1,0.14); put(IM,26.6,0.25); put(IM,28.8,0.2)
for t0,nl in [(0.3,2),(7.8,2),(11.3,2),(14.9,1),(17.45,2),(20.6,2),(25.2,1),(29.0,2),(33.4,2),(36.85,1),(45.5,3)]:
    for k in range(nl): put(SW,t0+k*0.24-0.05,0.13)
put(DG,36.9,0.1)
mix=mus+vo[:,None]+sfx[:n,None]*0.5
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

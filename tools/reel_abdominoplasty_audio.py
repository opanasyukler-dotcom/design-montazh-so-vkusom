import numpy as np, wave, librosa, cv2
SR=48000; DUR=47.6; M0=17.44
src=open("../v14/audio.py").read(); exec(src[src.index("rng=np.random"):src.index("n=int(DUR*SR)")])
s2=open("../v15/audio.py").read()
exec(s2[s2.index("def impact"):s2.index("n=int(DUR*SR)", s2.index("def impact"))])
exec(s2[s2.index("def shutter"):s2.index("SH,BM,SM,DG=")])
n=int(DUR*SR)
SEGS=[("iv",3.10,5.64,0.40),("iv",7.28,11.66,3.20),("surg",3.58,6.00,8.20),("surg",9.44,10.95,10.80),("surg",12.18,15.80,12.85),
      ("surg",24.55,26.40,16.75),("surg",27.05,31.22,18.75),("surg",47.25,48.70,23.25),("iv",12.06,16.70,25.00),
      ("iv",19.92,24.62,30.20),("iv",26.10,30.40,35.30)]
src_a={}; vo=np.zeros(n)
for s,si,so,t0 in SEGS:
    if s not in src_a:
        x,_=librosa.load(f"a/{s}_48.wav",sr=SR,mono=True); src_a[s]=x/(np.sqrt(np.mean(x[np.abs(x)>1e-3]**2))+1e-6)
    seg=src_a[s][int(si*SR):int(so*SR)].copy(); f=int(0.015*SR); seg[:f]*=np.linspace(0,1,f); seg[-f:]*=np.linspace(1,0,f)
    i=int(t0*SR); vo[i:i+len(seg)]+=seg[:n-i]
vo*=0.125
y,_=librosa.load("mus.wav",sr=SR,mono=False,offset=M0,duration=DUR); mus=y.T[:n].copy(); mus/=np.abs(mus).max()
if len(mus)<n: mus=np.pad(mus,((0,n-len(mus)),(0,0)))
tt=np.arange(n)/SR; act=np.zeros(n)
for s,si,so,t0 in SEGS: act[int((t0-0.05)*SR):int((t0+so-si+0.05)*SR)]=1
d=cv2.GaussianBlur(act.astype(np.float32).reshape(1,-1),(0,0),sigmaX=SR*0.12).ravel()
fade=np.clip((DUR-0.3-tt)/4.5,0,1)**1.6
mus*=((0.34-0.30*np.clip(d,0,1))*np.minimum(1,tt/0.3)*fade)[:,None]
sfx=np.zeros(n+SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]*g
WB,PP,IM,RS=whoosh(),pop(),impact(),riser(); SH,BM,SM,DG=shutter(),boom(),shimmer(),ding(); SW=whoosh(0.28,800,9000,0.5)
for c in [8.0,12.6,17.0,20.2,23.0,24.8,30.0,35.1,40.0,41.85,43.7,45.55]: put(WB,c-0.3,0.2); put(BM,c,0.18)
for t in [9.5,11.0,40.0,41.85,43.7,45.55]: put(SH,t+0.02,0.2)
put(SH,0.35,0.18); put(SM,0.1,0.16)
for t in [3.9,4.6,5.3,6.0,17.3,17.75,18.2,18.65]: put(PP,t,0.2)
put(RS,13.1-1.0,0.16); put(SM,13.2,0.18); put(IM,14.2,0.22); put(IM,20.55,0.25); put(SM,26.7,0.14); put(IM,37.5,0.22)
for t0,nl in [(0.9,3),(8.3,2),(14.2,3),(20.55,1),(26.7,2),(30.6,2),(37.5,2)]:
    for k in range(nl): put(SW,t0+k*0.24-0.05,0.13)
put(DG,40.1,0.1)
mix=mus+vo[:,None]+sfx[:n,None]*0.5
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

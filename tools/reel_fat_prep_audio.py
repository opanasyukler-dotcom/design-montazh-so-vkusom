import numpy as np, wave, librosa, cv2
SR=48000; DUR=40.4; M0=0.86
src=open("../v14/audio.py").read(); exec(src[src.index("rng=np.random"):src.index("n=int(DUR*SR)")])
s2=open("../v15/audio.py").read()
exec(s2[s2.index("def impact"):s2.index("n=int(DUR*SR)", s2.index("def impact"))])
exec(s2[s2.index("def shutter"):s2.index("SH,BM,SM,DG=")])
n=int(DUR*SR)
SEGS=[("surg",1.42,7.60,0.30),("surg",7.62,15.36,6.75),("surg",15.38,21.92,14.80),("surg",24.55,29.98,21.45),("surg",30.05,35.36,27.10),("surg",35.38,38.34,32.65)]
x,_=librosa.load("a/surg_48.wav",sr=SR,mono=True); x=x/(np.sqrt(np.mean(x[np.abs(x)>1e-3]**2))+1e-6)
vo=np.zeros(n)
for s,si,so,t0 in SEGS:
    seg=x[int(si*SR):int(so*SR)].copy(); f=int(0.015*SR); seg[:f]*=np.linspace(0,1,f); seg[-f:]*=np.linspace(1,0,f)
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
for c in [3.4,6.75,10.8,14.6,21.5,27.0,30.0,35.9]: put(WB,c-0.3,0.2); put(BM,c,0.16)
put(SM,0.05,0.16); put(RS,3.55-0.9,0.12); put(SW,3.55,0.18)                      # focus pull, pull-back into the card
for t in [11.1,27.0]: put(SH,t,0.18)                                             # punch-in "camera"
for t,_ in [(21.85,0),(23.0,0),(24.3,0),(25.8,0)]: put(PP,t,0.22); put(DG,t+0.02,0.05)   # scheme steps
put(PP,30.3,0.18); put(PP,30.45,0.18); put(SM,30.5,0.14); put(IM,33.4,0.2); put(SM,33.45,0.12)   # gauges fill / asymmetry
for t0,nl in [(0.35,2),(7.0,2),(15.1,3),(30.2,2),(36.2,3)]:
    for k in range(nl): put(SW,t0+k*0.24-0.05,0.13)
mix=mus+vo[:,None]+sfx[:n,None]*0.5
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

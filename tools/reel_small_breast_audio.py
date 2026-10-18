import numpy as np, wave, librosa, cv2
SR=48000; DUR=37.6                                    # no music: voice + sound effects only
src=open("../v14/audio.py").read(); exec(src[src.index("rng=np.random"):src.index("n=int(DUR*SR)")])
s2=open("../v15/audio.py").read()
exec(s2[s2.index("def impact"):s2.index("n=int(DUR*SR)", s2.index("def impact"))])
exec(s2[s2.index("def shutter"):s2.index("SH,BM,SM,DG=")])
n=int(DUR*SR)
SEGS=[("surg",12.48,17.45,0.30),("surg",0.00,5.14,5.60),("surg",23.65,35.72,13.60),("surg",37.02,43.82,25.90),("surg",53.88,55.36,33.00)]
x,_=librosa.load("a/surg_48.wav",sr=SR,mono=True); x=x/(np.sqrt(np.mean(x[np.abs(x)>1e-3]**2))+1e-6)
vo=np.zeros(n)
for s,si,so,t0 in SEGS:
    seg=x[int(si*SR):int(so*SR)].copy(); f=int(0.015*SR); seg[:f]*=np.linspace(0,1,f); seg[-f:]*=np.linspace(1,0,f)
    i=int(t0*SR); vo[i:i+len(seg)]+=seg[:n-i]
vo*=0.14
sfx=np.zeros(n+SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]*g
WB,PP,IM,RS=whoosh(),pop(),impact(),riser(); SH,BM,SM,DG=shutter(),boom(),shimmer(),ding(); SW=whoosh(0.28,800,9000,0.5)
def tick():                                           # short click for the counter
    k=np.arange(int(0.025*SR))/SR; return np.sin(2*np.pi*2400*k)*np.exp(-k*260)
TK=tick()
# sound effects kept few and soft: a light whoosh on the cuts and on the schematics, one soft pop where 295 lands
for c in [1.6,5.4,11.0,13.6,20.0,22.6,25.7,32.7]: put(WB,c-0.3,0.11)
put(SW,6.45,0.1); put(PP,9.45,0.12)
M0=7.6                                                # the track's drop lands on the ДО -> ПОСЛЕ transition (6.4 s)
y,_=librosa.load("mus.wav",sr=SR,mono=False,offset=M0,duration=DUR); mus=y.T[:n].copy(); mus/=np.abs(mus).max()
if len(mus)<n: mus=np.pad(mus,((0,n-len(mus)),(0,0)))
tt=np.arange(n)/SR; act=np.zeros(n)
for s,si,so,t0 in SEGS: act[int((t0-0.05)*SR):int((t0+so-si+0.05)*SR)]=1
d=cv2.GaussianBlur(act.astype(np.float32).reshape(1,-1),(0,0),sigmaX=SR*0.12).ravel()
fade=np.clip((DUR-0.3-tt)/4.5,0,1)**1.6
mus*=((0.16-0.12*np.clip(d,0,1))*np.minimum(1,tt/0.3)*fade)[:,None]           # quiet bed, ducked under the voice
mix=mus+vo[:,None]+sfx[:n,None]*0.55
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

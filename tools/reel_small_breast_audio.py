import numpy as np, wave, librosa, cv2
SR=48000; DUR=40.8                                    # no music: voice + sound effects only
src=open("../v14/audio.py").read(); exec(src[src.index("rng=np.random"):src.index("n=int(DUR*SR)")])
s2=open("../v15/audio.py").read()
exec(s2[s2.index("def impact"):s2.index("n=int(DUR*SR)", s2.index("def impact"))])
exec(s2[s2.index("def shutter"):s2.index("SH,BM,SM,DG=")])
n=int(DUR*SR)
SEGS=[("surg",5.16,8.72,0.30),("surg",12.48,17.45,4.20),("surg",0.00,5.14,9.50),("surg",23.65,35.72,17.60),("surg",37.02,43.82,29.90),("surg",53.88,55.36,37.00)]
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
for c in [4.0,9.4,14.8,17.6,25.6,29.7,36.7]: put(WB,c-0.3,0.24); put(BM,c,0.2)
put(SM,0.05,0.2); put(SH,0.35,0.22)                                             # focus pull + viewfinder
put(PP,1.0,0.22); put(PP,1.24,0.22)                                             # "почти не было своего объёма"
put(SW,4.05,0.2); put(SH,4.1,0.18); put(SW,4.6,0.14)                            # ДО card flies in, quote frame opens
put(RS,9.5-0.6,0.12); put(SW,9.55,0.2)                                          # pull back into the card
for k in range(24): put(TK,12.2+1.2*(1-(1-k/24)**0.5)*1.0,0.18)                 # 295 counting up
put(IM,13.4,0.28); put(DG,13.42,0.14)                                           # lands on 295
put(RS,14.8-1.0,0.16); put(IM,14.8,0.22); put(SM,14.85,0.16)                    # the turn
for k in range(3): put(SW,15.05+k*0.24,0.15)
put(BM,21.3,0.3); put(PP,21.3,0.22); put(PP,21.6,0.22); put(PP,21.9,0.26); put(SM,21.95,0.18)   # "преимущество"
put(SH,25.6,0.16)
for k in range(4): put(SW,30.3+k*0.28,0.14)
put(SM,36.75,0.2); put(PP,37.2,0.2); put(PP,37.4,0.2); put(DG,37.8,0.1)          # collage
mix=vo[:,None]+sfx[:n,None]*0.55
mix=np.repeat(mix,2,1); mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

import numpy as np, wave, librosa, cv2
SR=48000; DUR=45.2; M0=9.6
exec(open("audio.py").read().split("sfx=np.zeros")[0].split("rng=np.random")[1].join(["rng=np.random",""]) if False else "")
src=open("audio.py").read(); fns=src[src.index("rng=np.random"):src.index("sfx=np.zeros")]
exec(fns)
n=int(DUR*SR)
SEGS=[(7.84,14.25,0.6),(14.62,24.12,7.06),(24.62,29.95,16.71),(30.40,33.66,25.0),(35.16,38.45,28.46),(38.90,46.24,32.0),(49.95,52.70,41.1)]
vs,_=librosa.load("../v13/voice_src.wav",sr=SR,mono=True)
vo=np.zeros(n)
for si,so,t0 in SEGS:
    s=vs[int(si*SR):int(so*SR)].copy(); f=int(0.015*SR); s[:f]*=np.linspace(0,1,f); s[-f:]*=np.linspace(1,0,f)
    i=int(t0*SR); vo[i:i+len(s)]+=s[:n-i]
vo=librosa.effects.preemphasis(vo,coef=0.0) if False else vo
vo/=np.sqrt(np.mean(vo[vo!=0]**2))*8   # rms ~ -18 dBFS
y,_=librosa.load("mus.wav",sr=SR,mono=False,offset=M0,duration=DUR); mus=y.T[:n].copy()
tt=np.arange(len(mus))/SR
mus*=(np.minimum(1,tt/0.3)*np.clip((DUR-tt)/1.4,0,1))[:,None]
mus/=np.abs(mus).max()
act=np.zeros(n)
for si,so,t0 in SEGS: act[int((t0-0.05)*SR):int((t0+so-si+0.05)*SR)]=1
d=cv2.GaussianBlur(act.astype(np.float32).reshape(1,-1),(0,0),sigmaX=SR*0.12).ravel()
mus*=(0.32-0.24*np.clip(d,0,1))[:len(mus),None]      # ~-12 dB under the voice
sfx=np.zeros(n+SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]*g
WB,RS,IM,PP,DG,CK,TK=whoosh(),riser(),impact(),pop(),ding(),click(),tick()
for c in [4.6,7.05,11.65,13.6,16.98,18.66,20.34,22.02,27.0,31.0,33.8,36.6,38.6]: put(WB,c-0.28,0.22)
put(RS,23.7-1.4,0.3); put(RS,40.9-1.4,0.28)
for c in [23.7,40.9]: put(IM,c,0.5)
def key():
    k=int(0.06*SR); x=np.arange(k)/SR
    nz=rng.standard_normal(k); hi=nz-lp(nz,np.full(k,1800.0+rng.uniform(0,1500)))
    o=hi*np.exp(-x*rng.uniform(350,600))+0.6*np.sin(2*np.pi*rng.uniform(170,260)*x)*np.exp(-x*90)
    r=int(rng.uniform(0.025,0.04)*SR); o[r:]+=0.35*hi[:k-r]*np.exp(-x[:k-r]*700)
    return o/np.abs(o).max()
KEYS=[key() for _ in range(8)]
TYPED=[(0.45,24,"ПОСЛЕ ДВУХ РОДОВГРУДЬ СТАЛАСОВСЕМ НЕ ТОЙ"),(7.2,28,"ЗНАКОМОЕ ОЩУЩЕНИЕ?"),
       (24.7,32,"ПОСЛЕ ЛИПОФИЛИНГА ГРУДИсобственная жировая ткань, без имплантов"),
       (13.75,32,"ВЫПОЛНИМ УВЕЛИЧЕНИЕ ГРУДИБЕЗ ИМПЛАНТОВ"),(27.3,32,"ПЕРВАЯ ОПЕРАЦИЯИ, КОНЕЧНО, БЫЛО ВОЛНИТЕЛЬНО"),(36.8,32,"2,5 МЕСЯЦА ПОСЛЕ ОПЕРАЦИИ")]
for t0,cps,txt in TYPED:
    for i,ch in enumerate(txt):
        if ch!=" ": put(KEYS[rng.integers(8)],t0+i/cps+rng.uniform(-0.006,0.006),0.38*rng.uniform(0.7,1))
for t in [7.9,8.95,10.0,36.7,36.85]: put(PP,t,0.16)
for t in [17.28,33.9,38.9]: put(TK,t,0.12)
put(PP,41.15,0.18); put(PP,41.75,0.18)
mix=mus+vo[:,None]+sfx[:n,None]*0.5
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix2.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

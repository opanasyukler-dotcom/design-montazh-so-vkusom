import numpy as np, wave, librosa
SR=48000; DUR=32.5; M0=17.44
rng=np.random.default_rng(7)
def lp(x,cut):
    out=np.zeros_like(x); s=0.0; a=1-np.exp(-2*np.pi*cut/SR)
    for i in range(len(x)): s+=a[i]*(x[i]-s); out[i]=s
    return out
def whoosh(d=0.4,lo=300,hi=6000,pk=0.6):
    k=int(d*SR); x_=np.linspace(0,1,k); e=np.where(x_<pk,(x_/pk)**2,((1-x_)/(1-pk))**1.5)
    cut=lo+(hi-lo)*np.sin(np.pi*np.clip(x_/(pk*2),0,1))**2; x=rng.standard_normal(k); o=(lp(x,cut)-lp(x,cut*0.25))*e; return o/np.abs(o).max()
def pop():
    k=int(0.09*SR); x=np.arange(k)/SR; o=np.sin(2*np.pi*np.cumsum(260+700*np.exp(-x*60))/SR)*np.exp(-x*45); return o/np.abs(o).max()
def key():
    k=int(0.06*SR); x=np.arange(k)/SR
    nz=rng.standard_normal(k); hi=nz-lp(nz,np.full(k,1800.0+rng.uniform(0,1500)))
    o=hi*np.exp(-x*rng.uniform(350,600))+0.6*np.sin(2*np.pi*rng.uniform(170,260)*x)*np.exp(-x*90)
    r=int(rng.uniform(0.025,0.04)*SR); o[r:]+=0.35*hi[:k-r]*np.exp(-x[:k-r]*700)
    return o/np.abs(o).max()
n=int(DUR*SR)
y,_=librosa.load("mus.wav",sr=SR,mono=False,offset=M0,duration=DUR); mus=y.T[:n].copy()
tt=np.arange(len(mus))/SR
mus*=(np.minimum(1,tt/0.05)*np.clip((DUR-tt)/1.6,0,1))[:,None]
mus/=np.abs(mus).max(); mus*=0.8
sfx=np.zeros(n+SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]*g
WB,PP=whoosh(),pop(); KEYS=[key() for _ in range(8)]
for c in [1.95,3.27,7.43,10.54,11.19,12.9,15.42,17.94,19.44,23.45,26.96,29.72]: put(WB,c-0.25,0.16)
TYPED=[(0.12,30,"ЗДЕСЬ НУЖНАПОДТЯЖКА?или можно обойтись имплантами?"),(3.37,34,"СМОТРИМ НА НАШЕ ДО"),(6.05,32,"ПОДТЯЖКУНЕ ПЛАНИРУЕМ"),
       (7.51,30,"ЗАДАЧА"),(12.05,34,"ПЕРВЫЙ РЕЗУЛЬТАТувеличение груди имплантами"),(20.04,34,"ИМПЛАНТЫ ИЛИ ПОДТЯЖКА?решаем только после оценки вашего ДО"),
       (23.65,30,"НЕ КАЖДОЙ ГРУДИПОСЛЕ ПОТЕРИ ОБЪЁМАНУЖНА ПОДТЯЖКА")]
for t0,cps,txt in TYPED:
    for i,ch in enumerate(txt):
        if ch!=" ": put(KEYS[rng.integers(8)],t0+i/cps+rng.uniform(-0.006,0.006),0.22*rng.uniform(0.7,1))
for t in [4.27,4.87,5.47,7.98,8.58,9.18,16.42,16.92,17.42,27.3,30.05]: put(PP,t,0.18)
mix=mus+sfx[:n,None]*0.6
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

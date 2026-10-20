import numpy as np, wave, librosa, cv2
SR=48000; DUR=44.0; M0=3.7; n=int(DUR*SR)
SEGS=[(7.10,14.40,0.30),(14.62,16.30,7.75),(23.28,27.62,9.80),(27.45,32.10,14.45),(32.02,38.60,19.35),(90.88,97.60,26.30),(127.38,130.95,33.40),(131.12,135.45,37.30)]
x,_=librosa.load("a/8296_48.wav",sr=SR,mono=True); x=x/(np.sqrt(np.mean(x[np.abs(x)>1e-3]**2))+1e-6)
vo=np.zeros(n)
for si,so,t0 in SEGS:
    seg=x[int(si*SR):int(so*SR)].copy(); f=int(0.015*SR); seg[:f]*=np.linspace(0,1,f); seg[-f:]*=np.linspace(1,0,f)
    i=int(t0*SR); vo[i:i+len(seg)]+=seg[:n-i]
vo*=0.14
y,_=librosa.load("mus.wav",sr=SR,mono=False,offset=M0,duration=DUR); mus=y.T[:n].copy(); mus/=np.abs(mus).max()
if len(mus)<n: mus=np.pad(mus,((0,n-len(mus)),(0,0)))
tt=np.arange(n)/SR; act=np.zeros(n)
for si,so,t0 in SEGS: act[int((t0-0.05)*SR):int((t0+so-si+0.05)*SR)]=1
d=cv2.GaussianBlur(act.astype(np.float32).reshape(1,-1),(0,0),sigmaX=SR*0.12).ravel()
fade=np.clip((DUR-0.3-tt)/4.5,0,1)**1.6
mus*=((0.24-0.17*np.clip(d,0,1))*np.minimum(1,tt/0.3)*fade)[:,None]
# ---- new sound palette (no whooshes / booms / shimmers): synthesized piano, wood marimba, rubber stamp, heartbeat
rng=np.random.RandomState(1)
def piano(f,dur=1.4):                      # Karplus-Strong string: soft piano-like note
    N=int(SR/f); buf=rng.uniform(-1,1,N); out=np.zeros(int(dur*SR))
    for i in range(len(out)): out[i]=buf[i%N]; buf[i%N]=0.996*0.5*(buf[i%N]+buf[(i+1)%N])
    return out*np.exp(-np.arange(len(out))/SR*2.2)
def marimba(f,dur=0.5):
    k=np.arange(int(dur*SR))/SR; return (np.sin(2*np.pi*f*k)+0.35*np.sin(2*np.pi*f*4*k)*np.exp(-k*40))*np.exp(-k*11)
def stamp():
    k=np.arange(int(0.35*SR))/SR; nz=rng.randn(len(k)); lo=np.convolve(nz,np.ones(60)/60,"same")
    return (lo*3+np.sin(2*np.pi*70*k))*np.exp(-k*16)
def beat():
    k=np.arange(int(0.25*SR))/SR; return np.sin(2*np.pi*55*k*(1-0.3*k))*np.exp(-k*22)
sfx=np.zeros(n+2*SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]/np.abs(s).max()*g
for c,fq in [(7.7,660),(9.75,587),(14.4,523),(19.3,659),(26.2,494),(33.3,587),(37.2,523)]: put(marimba(fq),c-0.02,0.10)   # soft wooden tick on cuts
put(stamp(),7.92,0.30)                                                                    # МИФ stamp
for k,t in enumerate([20.0,20.45,20.9,21.35,21.8,22.25,22.7]): put(piano([523,587,659,698,784,880,988][k]),t,0.09)   # keys = notes
put(marimba(1046,0.3),16.0,0.07); put(marimba(392,0.3),16.5,0.06)                         # ✓ / ✕ on the chairs
put(piano(784),35.0,0.07)                                                                 # «результат не потерян»
for t in (40.4,40.65,41.3,41.55,42.2,42.45): put(beat(),t,0.16)                           # heartbeat under «берегите нервы»
mix=mus+vo[:,None]+sfx[:n,None]*0.6
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

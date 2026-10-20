import numpy as np, wave, librosa, cv2
SR=48000; DUR=35.4; M0=1.0; n=int(DUR*SR)
SEGS=[(70.72,77.62,1.60),(80.55,88.62,9.10),(88.60,91.98,17.90),(101.60,107.05,21.80)]
x,_=librosa.load("a/8294_48.wav",sr=SR,mono=True); x=x/(np.sqrt(np.mean(x[np.abs(x)>1e-3]**2))+1e-6)
vo=np.zeros(n)
for si,so,t0 in SEGS:
    seg=x[int(si*SR):int(so*SR)].copy(); f=int(0.015*SR); seg[:f]*=np.linspace(0,1,f); seg[-f:]*=np.linspace(1,0,f)
    i=int(t0*SR); vo[i:i+len(seg)]+=seg[:n-i]
vo*=0.14
y,_=librosa.load("mus_scar.wav",sr=SR,mono=False,offset=M0,duration=DUR); mus=y.T[:n].copy(); mus/=np.abs(mus).max()
if len(mus)<n: mus=np.pad(mus,((0,n-len(mus)),(0,0)))
tt=np.arange(n)/SR; act=np.zeros(n)
for si,so,t0 in SEGS: act[int((t0-0.05)*SR):int((t0+so-si+0.05)*SR)]=1
d=cv2.GaussianBlur(act.astype(np.float32).reshape(1,-1),(0,0),sigmaX=SR*0.12).ravel()
fade=np.clip((DUR-0.3-tt)/4.0,0,1)**1.6
mus*=((0.24-0.17*np.clip(d,0,1))*np.minimum(1,tt/0.3)*fade)[:,None]
rng=np.random.RandomState(2)
def kalimba(f,dur=1.2):                    # soft kalimba/music-box pluck: clear month "steps"
    k=np.arange(int(dur*SR))/SR; return (np.sin(2*np.pi*f*k)+0.25*np.sin(2*np.pi*f*2.76*k)*np.exp(-k*9))*np.exp(-k*4.5)
def pencil(dur=0.9):                       # soft pencil line while the scar is drawn
    k=np.arange(int(dur*SR))/SR; nz=rng.randn(len(k)); hp=nz-np.convolve(nz,np.ones(8)/8,"same")
    return hp*np.sin(np.pi*k/dur)**2*0.6
sfx=np.zeros(n+2*SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]/np.abs(s).max()*g
put(pencil(),0.45,0.05)
for t,f in [(0.15,659),(8.6,587),(17.6,698),(21.6,784),(24.2,880)]: put(kalimba(f),t,0.11)          # one note per month on the timeline
put(kalimba(988,0.8),27.62,0.08)                                                                    # freeze
put(kalimba(523,2.0),31.45,0.09); put(kalimba(659,2.0),31.45,0.07); put(kalimba(784,2.0),31.5,0.07)  # gentle chord on the takeaway
mix=mus+vo[:,None]+sfx[:n,None]*0.6
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix_scar.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

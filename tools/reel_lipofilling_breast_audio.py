import numpy as np, wave, librosa
SR=48000; DUR=36.5; M0=0.86
y,_=librosa.load("mus.wav",sr=SR,mono=False,offset=M0,duration=DUR); mus=y.T.copy()
n=len(mus); tt=np.arange(n)/SR
mus*= (np.minimum(1,tt/0.15)*np.clip((DUR-tt)/1.3,0,1))[:,None]
mus/= np.abs(mus).max(); mus*=0.24
import os
if os.path.exists("voice.wav"):   # patient interview: duck the music under the voice
    vo,_=librosa.load("voice.wav",sr=SR,mono=True); vo=np.pad(vo,(0,max(0,n-len(vo))))[:n]; vo/=np.abs(vo).max()/0.9
    import cv2; d=cv2.GaussianBlur((np.abs(vo)>0.03).astype(np.float32).reshape(1,-1),(0,0),sigmaX=SR*0.2).ravel()
    mus*=(1-0.7*np.clip(d*3,0,1))[:,None]; mus+=vo[:,None]
rng=np.random.default_rng(5)
def lp(x,cut):
    out=np.zeros_like(x); s=0.0; a=1-np.exp(-2*np.pi*cut/SR)
    for i in range(len(x)): s+=a[i]*(x[i]-s); out[i]=s
    return out
def whoosh(d=0.45,lo=300,hi=6000,pk=0.6):
    k=int(d*SR); x_=np.linspace(0,1,k); e=np.where(x_<pk,(x_/pk)**2,((1-x_)/(1-pk))**1.5)
    cut=lo+(hi-lo)*np.sin(np.pi*np.clip(x_/(pk*2),0,1))**2; x=rng.standard_normal(k); o=(lp(x,cut)-lp(x,cut*0.25))*e; return o/np.abs(o).max()
def riser(d=1.4):
    k=int(d*SR); x_=np.linspace(0,1,k); x=rng.standard_normal(k); o=lp(x,300+8000*x_**2)-lp(x,150+2000*x_**2); o*=x_**2.2; return o/np.abs(o).max()
def impact():
    k=int(0.9*SR); x=np.arange(k)/SR; b=np.sin(2*np.pi*np.cumsum(45+120*np.exp(-x*22))/SR)*np.exp(-x*5)
    nz=lp(rng.standard_normal(k),np.full(k,3000.0))*np.exp(-x*14)*0.6; o=b+nz; return o/np.abs(o).max()
def pop():
    k=int(0.09*SR); x=np.arange(k)/SR; o=np.sin(2*np.pi*np.cumsum(260+700*np.exp(-x*60))/SR)*np.exp(-x*45); return o/np.abs(o).max()
def ding():
    k=int(1.4*SR); x=np.arange(k)/SR
    o=sum(a*np.sin(2*np.pi*f*x)*np.exp(-x*kk) for f,a,kk in [(1318.5,1,3),(1975.5,.5,4),(2637,.3,6),(659.25,.4,2.5)]); return o*np.minimum(1,x/0.004)/np.abs(o).max()
def click():
    k=int(0.012*SR); x=np.diff(rng.standard_normal(k),prepend=0)*np.exp(-np.arange(k)/SR*500); return x/np.abs(x).max()
def tick():
    k=int(0.05*SR); x=np.arange(k)/SR; o=np.sin(2*np.pi*2400*x)*np.exp(-x*90); return o
sfx=np.zeros(n+SR)
def put(s,t,g): i=int(max(0,t)*SR); seg=sfx[i:i+len(s)]; seg+=s[:len(seg)]*g
WB,RS,IM,PP,DG,CK,TK=whoosh(),riser(),impact(),pop(),ding(),click(),tick()
cuts=[4.13,8.03,9.03,10.03,11.03,15.46,19.36,23.31,27.74]
for c in cuts: put(WB,c-0.28,0.32)
put(RS,11.56-1.4,0.28); put(RS,32.44-1.4,0.3)
for c in [11.56,32.44]: put(IM,c,0.55)
put(IM,0.0,0.35)
for k in range(38): put(CK,0.35+k/26,0.05*rng.uniform(0.6,1))          # hook typewriter
for t in [4.18,5.11,6.08,7.06]: put(PP,t,0.25)                          # list items
for t in [8.28,20.0,28.55]: put(TK,t,0.18)                               # bracket opens
for t in [13.03,15.9,23.6,23.8]: put(PP,t,0.2)
put(DG,32.49,0.12); put(PP,32.97,0.25); put(DG,33.45,0.16)
mix=mus+ sfx[:n,None]*0.45
mix/=max(1,np.abs(mix).max()/0.97)
with wave.open("mix.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes())

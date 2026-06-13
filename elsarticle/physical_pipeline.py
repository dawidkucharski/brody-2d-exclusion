#!/usr/bin/env python3
"""AURORA physical pipeline v4 — int16/int32/float32 detection."""
import numpy as np, os, struct, json
from scipy import ndimage
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(42)

def read_sur(fp, max_size=600):
    sz = os.path.getsize(fp)
    with open(fp,'rb') as f: hdr = f.read(512)
    if hdr[:12]!=b'DIGITAL SURF': raise ValueError("bad magic")
    nx=struct.unpack_from('<H',hdr,108)[0]; ny=struct.unpack_from('<H',hdr,112)[0]
    dx=struct.unpack_from('<f',hdr,120)[0]/1000.; dy=struct.unpack_from('<f',hdr,124)[0]/1000.
    db=sz-512
    # Try int16, int32, float32
    chosen=None
    for dt,bp in [(np.int16,2),(np.int32,4),(np.float32,4)]:
        if abs(db-nx*ny*bp)<=bp: chosen=(dt,bp); break
    if chosen is None: raise ValueError(f"dtype? nx={nx} ny={ny} db={db}")
    dt,bp=chosen
    with open(fp,'rb') as f: f.seek(512); raw=np.fromfile(f,dtype=dt)
    raw=raw[:nx*ny]
    if dt==np.int16: Z=raw.astype(np.float64).reshape(ny,nx)/1000.
    elif dt==np.int32: Z=raw.astype(np.float64).reshape(ny,nx)/1000.
    else: Z=raw.astype(np.float64).reshape(ny,nx)
    # strip padding
    rv=np.var(Z,axis=1); cv=np.var(Z,axis=0)
    vr=rv>1e-6; vc=cv>1e-6
    if np.any(vr): a=np.argmax(vr); b=ny-1-np.argmax(vr[::-1]); Z=Z[a:b+1,:]
    if np.any(vc): a=np.argmax(vc); b=Z.shape[1]-1-np.argmax(vc[::-1]); Z=Z[:,a:b+1]
    Z=np.nan_to_num(Z,nan=0,posinf=0,neginf=0)
    H,W=Z.shape
    if max(W,H)>max_size: f=max(W,H)//max_size+1; Z=Z[::f,::f]; dx*=f; dy*=f
    return Z,dx,dy

def desc(Z):
    W,H=Z.shape; N=W*H; d={'W':W,'H':H,'N':N}
    Zf=Z.ravel(); mu=np.mean(Zf)
    d['Sa']=float(np.mean(np.abs(Zf-mu))); d['Sq']=float(np.std(Zf))
    d['Ssk']=float(np.mean((Zf-mu)**3)/(d['Sq']**3+1e-30))
    d['Sku']=float(np.mean((Zf-mu)**4)/(d['Sq']**4+1e-30))
    d['Sz']=float(np.max(Zf)-np.min(Zf))
    gy,gx=np.gradient(Z); gs=np.sum(gx**2+gy**2)
    d['sobolev_H1']=float(np.sqrt(gs)); d['total_variation']=float(np.sum(np.abs(gx)+np.abs(gy)))
    d['TV_per_pixel']=float(d['total_variation']/N)
    d['L2']=float(np.sqrt(np.sum(Z**2)))
    d['roughness_index']=float(np.sqrt(gs)/d['L2']) if d['L2']>0 else 0.
    Zn=(Z-mu)/(d['Sq']+1e-30); pk=Zn>1.; pt=Zn<-1.
    d['banach_N_peaks']=int(ndimage.label(pk)[1]) if pk.any() else 0
    d['banach_N_pits']=int(ndimage.label(pt)[1]) if pt.any() else 0
    if N>500**2: f=int(np.sqrt(N)/500)+1; Zd=Z[::f,::f]
    else: Zd=Z
    Zp=Zd-np.min(Zd)+1e-10; M=np.sum(Zp); ms=min(Zp.shape); mp=int(np.log2(ms))-1
    le,lz=[],[]
    for eps in 2**np.arange(1,mp+1):
        nyb,nxb=Zp.shape[0]//eps,Zp.shape[1]//eps
        if nyb<2 or nxb<2: continue
        boxes=Zp[:nyb*eps,:nxb*eps].reshape(nyb,eps,nxb,eps)
        mu=boxes.sum(axis=(1,3))/M; le.append(np.log(eps)); lz.append(np.log(np.sum(mu**2)))
    if len(le)>=3:
        le,lz=np.array(le),np.array(lz); s,_=np.polyfit(le,lz,1); d['D2']=float(s)
        pred=s*le+np.mean(lz)-s*np.mean(le)
        ssr=np.sum((lz-pred)**2); sst=np.sum((lz-np.mean(lz))**2)
        d['D2_R2']=float(1-ssr/(sst+1e-30))
    else: d['D2']=np.nan; d['D2_R2']=np.nan
    return d

sur_dir='/Users/dawid/Projects/interfero-Riemann/FV/SUR'
sur_files=sorted([f for f in os.listdir(sur_dir) if f.endswith('.sur')])
lm={
    'Ti6A14V_wedm_wyk':('Ti6Al4V','WEDM finish'),
    'Ti_szlifowane':('Ti6Al4V','Ground'),
    'Ti6A14V_wedm_zgru_1prz':('Ti6Al4V','WEDM rough'),
    'C45_t_zgrubne':('C45 steel','Rough turned'),
    'Graphite_oselkowane':('Graphite','Honed'),
    'Ti_szkielkowane':('Ti6Al4V','Bead blasted'),
    '1.4301-t.zgrub':('1.4301 steel','Rough turned'),
    'Mosiadz_oselkowane':('Brass','Honed'),
    'Mosiadz_szkielkowane':('Brass','Bead blasted'),
    'Mosiadz_szlifowane':('Brass','Ground'),
    'Ti_oselkowane':('Ti6Al4V','Honed'),
    '1.4301_oselkowane':('1.4301 steel','Honed'),
    '1.4301_szkielkowane':('1.4301 steel','Bead blasted'),
    '1.4301_szlifowane':('1.4301 steel','Ground'),
    '1.4301_t_wyk':('1.4301 steel','Finish turned'),
    'AL_oselkowane':('Al','Honed'),
    'Al_szkielkowane':('Al','Bead blasted'),
    'Al_szlifowane':('Al','Ground'),
    'C45_oselkowane':('C45 steel','Honed'),
    'C45_szkielkowane':('C45 steel','Bead blasted'),
    'C45_szlifowane':('C45 steel','Ground'),
    'C45_t_wyk':('C45 steel','Finish turned'),
    'MO58A_t_wyk':('MO58A brass','Finish turned'),
    'Al7075_t_wyk':('Al7075','Finish turned'),
    'ELLOR_t_wyk':('ELLOR','Finish turned'),
}
results={}
for fn in sur_files:
    fp=os.path.join(sur_dir,fn); nm=fn.replace('.sur','').replace('P1-','')
    mat,proc=lm.get(nm,(nm,'?')); lb=f"{mat} ({proc})"
    print(f"[{mat}] {proc}")
    try:
        Z,dx,dy=read_sur(fp); d=desc(Z); d['material']=mat; d['process']=proc; d['name']=lb
        results[lb]=d
        print(f"  {Z.shape[1]}x{Z.shape[0]} dx={dx:.2f}um Z=[{np.min(Z):.2f},{np.max(Z):.2f}]um")
        print(f"  Sa={d['Sa']:.2f} Sq={d['Sq']:.2f} Ssk={d['Ssk']:.2f} Sku={d['Sku']:.1f} D2={d['D2']:.4f} H1={d['sobolev_H1']:.0f} Npk={d['banach_N_peaks']}")
    except Exception as e: print(f"  FAIL: {e}")

print("\nSUMMARY")
print(f"{'Surface':<35} {'Sa':>7} {'Sq':>7} {'Sku':>7} {'D2':>7} {'H1':>10} {'TV/px':>8}")
for _,d in results.items():
    d2s=f"{d['D2']:.4f}" if not np.isnan(d['D2']) else "N/A"
    print(f"{d['name']:<35} {d['Sa']:7.2f} {d['Sq']:7.2f} {d['Sku']:7.1f} {d2s:>7} {d['sobolev_H1']:10.1f} {d['TV_per_pixel']:8.4f}")

outdir='/Users/dawid/Projects/interfero-Riemann/elsarticle'
mr={
    'Primes':{'H1':92.5,'TV':0.1687,'D2':1.687,'c':'#1f77b4','m':'o'},
    'Twin primes':{'H1':35.5,'TV':0.0247,'D2':1.230,'c':'#ff7f0e','m':'^'},
    'Square-free':{'H1':144.3,'TV':0.4118,'D2':1.854,'c':'#d62728','m':'s'},
    'Beatty phi':{'H1':185.6,'TV':0.6832,'D2':1.860,'c':'#9467bd','m':'D'},
    'Sums 2 sq.':{'H1':137.5,'TV':0.3744,'D2':1.787,'c':'#2ca02c','m':'*'},
}
pc={'WEDM finish':'#e41a1c','WEDM rough':'#e41a1c','Ground':'#377eb8',
    'Rough turned':'#4daf4a','Honed':'#984ea3','Bead blasted':'#ff7f00'}

fig,ax=plt.subplots(1,1,figsize=(10,8))
for _,d in results.items():
    if np.isnan(d['D2']): continue
    c=pc.get(d['process'],'#999')
    ax.scatter(d['sobolev_H1'],d['D2'],c=c,s=140,edgecolors='black',lw=0.8,zorder=5,marker='s')
    ax.annotate(f"{d['material']}\n({d['process']})",(d['sobolev_H1'],d['D2']),
                fontsize=6.5,xytext=(5,5),textcoords='offset points',alpha=0.85)
for n,ref in mr.items():
    ax.scatter(ref['H1'],ref['D2'],c=ref['c'],s=180,edgecolors='black',lw=1.2,zorder=6,marker=ref['m'])
    ax.annotate(n,(ref['H1'],ref['D2']),fontsize=7,fontweight='bold',xytext=(5,-8),textcoords='offset points')
ax.set_xlabel(r'$\|\nabla Z\|_2$ (Sobolev seminorm)',fontsize=12)
ax.set_ylabel(r'$D_2$ (correlation dimension)',fontsize=12)
ax.set_title('Physical (FV microscopy) vs mathematical (arithmetic) surfaces\nin the unified Banach–metrological descriptor space',
             fontsize=13,fontweight='bold')
from matplotlib.patches import Patch
lp=[Patch(facecolor=c,edgecolor='black',label=p) for p,c in pc.items()]
lmm=[plt.Line2D([0],[0],marker=ref['m'],color='w',markerfacecolor=ref['c'],
    markersize=9,markeredgecolor='black',label=n) for n,ref in mr.items()]
ax.legend(handles=lp+lmm,fontsize=6.5,loc='lower right',ncol=2,framealpha=0.85)
ax.grid(alpha=0.3,linestyle=':')
plt.tight_layout()
plt.savefig(f'{outdir}/fig_physical_vs_math.pdf',dpi=200,bbox_inches='tight',facecolor='white')
print(f"\nSaved: {outdir}/fig_physical_vs_math.pdf")
serializable={n:{k:v for k,v in d.items() if not callable(v)} for n,d in results.items()}
with open(f'{outdir}/physical_results.json','w') as f: json.dump(serializable,f,indent=2,default=str)
print(f"Saved: {outdir}/physical_results.json")

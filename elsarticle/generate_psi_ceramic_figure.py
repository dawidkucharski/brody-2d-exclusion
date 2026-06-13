#!/usr/bin/env python3
"""
Reproducibility script: fig_psi_ceramic_analysis.pdf
=====================================================
1D EFM profilometry from 24,000 PSI interferograms.
Method from Kucharski (2025, Measurement):
  - 240 px central radial aperture, B-spline smoothing
  - Continuity-aware ring tracking, EFM prominence 0.04
  - Per-revolution unwrapping with physical |Δε| < 0.1 constraint
  - LSCI + Gaussian filter (50% at 15 UPR)
  - 1D peak detection → angular NN spacings → Brody β
"""
import numpy as np, os, sys, json, time, warnings
from scipy import ndimage, signal, spatial, stats, optimize, interpolate
from scipy.special import gamma as gamma_func
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); np.random.seed(42)

PSI_DIR='/Users/dawid/Projects/interfero-Riemann/PSI'
OUTDIR='/Users/dawid/Projects/interfero-Riemann/elsarticle'
WL=532.0; HW=WL/2000.0
APERTURE=240; PROM=0.04; BSPL=20; MINR=3; R2TH=0.6
FPR=4800; DPF=400.0/FPR; NREV=5; NF=NREV*FPR
UPR=15; PKPROM=0.03; SUBSAMPLE=5

def poisson(s): return np.exp(-s)
def goe(s): return (np.pi*s/2)*np.exp(-np.pi*s*s/4)
def gue(s): return (32*s*s/np.pi/np.pi)*np.exp(-4*s*s/np.pi)
def brody(s,b):
    a=gamma_func((b+2)/(b+1))**(b+1)
    return (b+1)*a*s**b*np.exp(-a*s**(b+1))

def find_centre(img0):
    from scipy.ndimage import gaussian_filter as gf
    bg=gf(img0.astype(np.float64),30); ic=img0-bg
    bs=-1; bc=(0,0)
    for cy in range(160,340,10):
        for cx in range(250,500,10):
            Y,X=np.ogrid[:488,:648]; R=np.sqrt((X-cx)**2+(Y-cy)**2)
            mr=min(cx,cy,648-cx,488-cy,APERTURE)-5
            if mr<50: continue
            nb=200; re=np.linspace(0,mr,nb+1)
            dg=np.digitize(R.ravel(),re)-1; dg=np.clip(dg,0,nb-1)
            pf=np.bincount(dg,weights=ic.ravel(),minlength=nb)
            ct=np.bincount(dg,minlength=nb); pf=pf/np.maximum(ct,1)
            pf_dt=pf-np.convolve(pf,np.ones(30)/30,mode='same')
            sc=np.var(pf_dt[20:-20])
            if sc>bs: bs=sc; bc=(cy,cx)
    return bc

def efm_frame(img,cy,cx,prev_rings):
    from scipy.ndimage import gaussian_filter as gf
    H,W=img.shape
    bg=gf(img.astype(np.float64),30); ic=img-bg
    ic=ic/(np.std(ic)+1e-10)
    Y,X=np.ogrid[:H,:W]; R=np.sqrt((X-cx)**2+(Y-cy)**2)
    mr=min(cx,cy,W-cx,H-cy,APERTURE)-5
    if mr<30: return np.nan,0,0,None
    nb=300; re=np.linspace(0,mr,nb+1)
    dg=np.digitize(R.ravel(),re)-1; dg=np.clip(dg,0,nb-1)
    pf=np.bincount(dg,weights=ic.ravel(),minlength=nb)
    ct=np.bincount(dg,minlength=nb); pf=pf/np.maximum(ct,1)
    rc=(re[:-1]+re[1:])/2
    tr=ndimage.uniform_filter1d(pf,60); pfd=pf-tr
    try:
        spl=interpolate.UnivariateSpline(rc,pfd,k=3,s=BSPL); pfs=spl(rc)
    except: pfs=pfd
    prom=PROM*(np.max(pfs)-np.min(pfs))
    pks,_=signal.find_peaks(pfs,prominence=prom,distance=4,width=2)
    if len(pks)<MINR: return np.nan,0,0,None
    rr=rc[pks]
    if prev_rings is not None and len(prev_rings)>=MINR:
        matched=[]; mi=[]; used=set()
        for pi,pr in enumerate(prev_rings):
            ds=np.abs(rr-pr); bj=np.argmin(ds)
            if ds[bj]<15 and bj not in used:
                matched.append(rr[bj]); mi.append(pi); used.add(bj)
        if len(matched)>=MINR:
            rr=np.array(matched); rn=np.array([i+1 for i in mi],dtype=float)
        else: rn=np.arange(1,len(rr)+1,dtype=float)
    else: rn=np.arange(1,len(rr)+1,dtype=float)
    next_rings=rr.copy()
    D2=(2*rr)**2; A=np.column_stack([rn,np.ones_like(rn)])
    cf,_,_,_=np.linalg.lstsq(A,D2,rcond=None)
    sl,icpt=cf
    if abs(sl)<1e-8: return np.nan,0,0,None
    eps=(icpt/sl+1)%1
    Dp=sl*rn+icpt; r2=1-np.sum((D2-Dp)**2)/(np.sum((D2-np.mean(D2))**2)+1e-30)
    if r2<R2TH: return np.nan,r2,0,None
    return eps,r2,len(rr),next_rings

def unwrap_robust(eps,fpr_sub):
    """Unwrap ε with physical constraint: |Δε| < 0.1 per frame for FN 111"""
    n=len(eps); fo=np.zeros(n)
    for rev in range(NREV):
        s=rev*fpr_sub; e=s+fpr_sub
        if e>n: e=n
        es=eps[s:e].copy()
        gm=~np.isnan(es)
        if np.sum(gm)<10: continue
        bi=np.where(np.isnan(es))[0]
        if len(bi)>0:
            gi=np.where(gm)[0]
            es[bi]=np.interp(bi,gi,es[gi],left=es[gi[0]],right=es[gi[-1]])
        # Median filter to reject outliers
        from scipy.ndimage import median_filter
        es_smooth=median_filter(es,size=5)
        # Unwrap smoothed version, apply same corrections to original
        de=np.diff(es_smooth)
        de[de>0.5]-=1; de[de<-0.5]+=1
        # Physical constraint: |Δε| > 0.1 impossible for FN 111
        bad=np.abs(de)>0.1
        de[bad]=0  # hold previous value
        fos=np.zeros(len(es)); fos[1:]=np.cumsum(de)
        fo[s:e]=fos
    return fo

def lsci(theta_deg,h_um):
    th=np.deg2rad(theta_deg)
    X=np.column_stack([np.ones_like(th),np.cos(th),np.sin(th)])
    cf,_,_,_=np.linalg.lstsq(X,h_um,rcond=None)
    return h_um-X@cf

def gauss_filter(prof,samp_deg,cut_upr):
    n=len(prof); cd=360.0/cut_upr
    sig=np.sqrt(np.log(2)/2)*cd/(2*np.pi)/samp_deg
    if sig<1: return prof
    k=int(4*sig+1); x=np.arange(-k,k+1)
    krn=np.exp(-0.5*(x/sig)**2); krn/=np.sum(krn)
    return np.convolve(prof,krn,mode='same')

def fit_brody(s):
    def nll(b):
        if b<=-0.99: return 1e10
        a=gamma_func((b+2)/(b+1))**(b+1)
        pdf=(b+1)*a*s**b*np.exp(-a*s**(b+1))
        return -np.sum(np.log(np.maximum(pdf,1e-300)))
    r=optimize.minimize_scalar(nll,bounds=(-0.9,10),method='bounded'); beta=r.x
    nb,n=500,len(s); bb=np.zeros(nb)
    for i in range(nb):
        sb=np.random.choice(s,n,replace=True)
        try:
            rb=optimize.minimize_scalar(lambda bt:-np.sum(np.log(np.maximum(
                (bt+1)*gamma_func((bt+2)/(bt+1))**(bt+1)*sb**bt
                *np.exp(-gamma_func((bt+2)/(bt+1))**(bt+1)*sb**(bt+1)),
                1e-300))),bounds=(-0.9,10),method='bounded')
            bb[i]=rb.x
        except: bb[i]=beta
    return {'beta':beta,'ci_low':np.percentile(bb,2.5),'ci_high':np.percentile(bb,97.5)}

def main():
    print("="*70)
    print("PSI CERAMIC FIGURE — 1D EFM profilometry (with robust unwrap)")
    print(f"λ={WL}nm aperture={APERTURE}px prom={PROM} subsample={SUBSAMPLE}")
    print("="*70)
    pf=sorted([f for f in os.listdir(PSI_DIR) if f.endswith('.png')])
    print(f"Input: {len(pf)} frames")
    img0=plt.imread(os.path.join(PSI_DIR,pf[0]))
    if img0.ndim==3: img0=np.mean(img0[:,:,:3],axis=2)
    cy,cx=find_centre(img0)
    print(f"Centre: ({cx},{cy})")
    idx=np.arange(0,NF,SUBSAMPLE); np_=len(idx)
    fpr_sub=FPR//SUBSAMPLE; dpf_sub=DPF*SUBSAMPLE
    print(f"Processing {np_} frames ({np_//fpr_sub} frames/rev, {dpf_sub:.4f}°/frame)")
    eps=np.full(np_,np.nan); r2v=np.zeros(np_); nrv=np.zeros(np_,dtype=int)
    prev=None; t0=time.time(); ng=0
    for k,i in enumerate(idx):
        fp_=os.path.join(PSI_DIR,pf[i])
        img=plt.imread(fp_)
        if img.ndim==3: img=np.mean(img[:,:,:3],axis=2)
        e,r2,nr,prev=efm_frame(img,cy,cx,prev)
        if not np.isnan(e): eps[k]=e; r2v[k]=r2; nrv[k]=nr; ng+=1
        if (k+1)%500==0:
            print(f"  {k+1}/{np_} ({(time.time()-t0):.0f}s) good={ng} ({100*ng/(k+1):.0f}%)")
    dt=time.time()-t0
    print(f"  Done: {ng}/{np_} valid ({100*ng/np_:.0f}%), {dt:.0f}s")
    if ng<100: print("ERROR: too few valid"); sys.exit(1)
    print(f"  Median R²={np.median(r2v[r2v>0]):.4f}, rings={np.median(nrv[nrv>0]):.0f}")
    print("Unwrapping (robust, physical |Δε|<0.1 constraint)...")
    fo=unwrap_robust(eps,fpr_sub); hu=fo*HW
    th=idx*DPF
    print("LSCI + Gaussian filter...")
    ath=[]; ah=[]
    for rev in range(NREV):
        s=rev*fpr_sub; e=s+fpr_sub
        if e>np_: e=np_
        thr=th[s:e]; hhr=hu[s:e]
        m360=(thr%400)<360
        tu=(thr[m360]%400); hh=hhr[m360]
        if len(tu)<100: continue
        si=np.argsort(tu); ts=tu[si]; hs=hh[si]
        hl=lsci(ts,hs); hf=gauss_filter(hl,dpf_sub,UPR)
        ath.append(ts); ah.append(hf)
    tp=np.concatenate(ath); hp=np.concatenate(ah)
    si=np.argsort(tp); tf=tp[si]; hf_nm=hp[si]*1000
    rms=np.std(hf_nm); pv=np.max(hf_nm)-np.min(hf_nm)
    print(f"  Profile: {len(tf)} pts, RMS={rms:.0f}nm, PV={pv:.0f}nm")
    print("Peak detection...")
    prom_nm=PKPROM*(np.max(hf_nm)-np.min(hf_nm))
    pi,_=signal.find_peaks(hf_nm,prominence=prom_nm,distance=5)
    npk=len(pi)
    print(f"  {npk} peaks")
    if npk<5:
        pi,_=signal.find_peaks(hf_nm,prominence=prom_nm/3,distance=5); npk=len(pi)
        print(f"  {npk} peaks (lowered threshold)")
    pa=tf[pi]; ph_=hf_nm[pi]
    print("Angular NN spacings & Brody...")
    pas=np.sort(pa); ang_sp=np.diff(pas); ans=ang_sp/np.mean(ang_sp)
    fit=fit_brody(ans); beta=fit['beta']
    ks=stats.ks_1samp(ans,stats.expon.cdf)
    print(f"  β={beta:.2f}[{fit['ci_low']:.2f},{fit['ci_high']:.2f}] KS p={ks.pvalue:.4f}")
    results={'surface':'JENOPTIK FN 111','method':'EFM 1D profilometry',
        'wavelength_nm':WL,'n_frames':np_,'subsample':SUBSAMPLE,'n_good':ng,
        'n_peaks':int(npk),'n_spacings':len(ans),'best_brody_beta':float(beta),
        'beta_ci_low':float(fit['ci_low']),'beta_ci_high':float(fit['ci_high']),
        'ks_poisson_D':float(ks.statistic),'ks_poisson_p':float(ks.pvalue),
        'residual_rms_nm':float(rms),'residual_pv_nm':float(pv)}
    with open(f'{OUTDIR}/psi_ceramic_results.json','w') as f: json.dump(results,f,indent=2)
    # Figure
    print("Generating figure...")
    fig=plt.figure(figsize=(16,5.5))
    gs=fig.add_gridspec(1,3,width_ratios=[1.15,1,1],wspace=0.28)
    ax_a=fig.add_subplot(gs[0,0])
    ax_a.plot(tf,hf_nm,'b-',lw=0.5,alpha=0.8)
    ax_a.fill_between(tf,0,hf_nm,alpha=0.15,color='b')
    ax_a.axhline(0,color='gray',ls=':',alpha=0.5)
    ax_a.set_title('(a) Residual height profile (LSCI + 15 UPR filter)',
        fontsize=11,fontweight='bold',loc='left')
    ax_a.set_xlabel('Angular position [°]',fontsize=9)
    ax_a.set_ylabel('Height [nm]',fontsize=9)
    ax_a.text(0.02,0.98,f'RMS={rms:.0f} nm\nPV={pv:.0f} nm\n{npk} peaks',
        transform=ax_a.transAxes,fontsize=8,va='top',
        bbox=dict(boxstyle='round',facecolor='white',alpha=0.85))
    ax_a.set_xlim(0,360)
    ax_b=fig.add_subplot(gs[0,1])
    ax_b.plot(tf,hf_nm,'gray',lw=0.5,alpha=0.7)
    ax_b.scatter(pa,ph_,c='#d62728',s=12,alpha=0.8,edgecolors='darkred',
        linewidth=0.3,zorder=5,label=f'{npk} peaks')
    ax_b.axhline(0,color='gray',ls=':',alpha=0.5)
    ax_b.set_title('(b) Detected peaks (prominence > 3%)',
        fontsize=11,fontweight='bold',loc='left')
    ax_b.set_xlabel('Angular position [°]',fontsize=9)
    ax_b.set_ylabel('Height [nm]',fontsize=9)
    ax_b.legend(fontsize=8,loc='lower right'); ax_b.set_xlim(0,360)
    ax_c=fig.add_subplot(gs[0,2])
    ax_c.hist(ans,bins=30,density=True,alpha=0.45,color='#6c3483',
        edgecolor='black',linewidth=0.4,label=f'n={len(ans)}')
    sp=np.linspace(0.01,4.5,300)
    ax_c.plot(sp,poisson(sp),'k:',lw=2,alpha=0.8,label='Poisson (exp)')
    ax_c.plot(sp,goe(sp),'b--',lw=2,alpha=0.8,label='GOE (β=1)')
    ax_c.plot(sp,gue(sp),'r-.',lw=2,alpha=0.8,label='GUE (β=2)')
    ax_c.plot(sp,brody(sp,beta),'m-',lw=3,
        label=f'Brody β={beta:.2f}[{fit["ci_low"]:.2f},{fit["ci_high"]:.2f}]')
    ax_c.set_xlabel('Normalised angular NN spacing s',fontsize=10)
    ax_c.set_ylabel('P(s)',fontsize=10)
    ax_c.set_title('(c) NN spacing distribution',fontsize=11,fontweight='bold',loc='left')
    ax_c.legend(fontsize=7.5,loc='upper right',framealpha=0.85)
    ax_c.set_xlim(0,4); ax_c.grid(alpha=0.3,linestyle=':')
    fig.suptitle(f'Phase-extracted surface profile — FN 111\n'
        f'One 360° segment from {NREV} continuous $400^\\circ$ rotations '
        f'({NREV}$\\times${FPR} frames), EFM+tracking, '
        f'$\\lambda$={WL}nm, LSCI+{UPR}UPR Gaussian',
        fontsize=12,fontweight='bold',y=1.02)
    op=f'{OUTDIR}/fig_psi_ceramic_analysis.pdf'
    fig.savefig(op,dpi=200,facecolor='white',edgecolor='none',bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {op}")
    print(f"\n{'='*70}")
    print(f"DONE — β={beta:.2f}[{fit['ci_low']:.2f},{fit['ci_high']:.2f}], {npk} peaks, RMS={rms:.0f}nm")
    print(f"{'='*70}")

if __name__=='__main__': main()

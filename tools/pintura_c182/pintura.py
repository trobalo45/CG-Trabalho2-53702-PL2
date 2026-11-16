# =============================================================================
#  Pintura (livery) UBI do Cessna 182 - passo 2 (depois de mapa_posicoes.py)
#  Tome Pedro - 53702 - PL2
#  O esquema original (faixa, divisas das asas, leme) passa a azul UBI, sem as letras e
#  emblemas antigos; os logos e a matricula sao projetados em 3D, por isso ficam sempre
#  direitos. Correr na pasta do trabalho:  python3 tools/pintura_c182/pintura.py
#  (fontes Inter em /usr/share/fonts/opentype/inter/; mudar o caminho se for preciso)
# =============================================================================
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
N=2048
P=np.load('tools/pintura_c182/P.npy'); NM=np.load('tools/pintura_c182/NM.npy'); ok=np.isfinite(P[...,0])
X,Y,Z=[np.nan_to_num(P[...,i]) for i in range(3)]
NM=NM/np.maximum(np.linalg.norm(NM,axis=-1,keepdims=True),1e-6); nx,ny=NM[...,0],NM[...,1]
src8=np.array(Image.open('models/c182/c182_exterior_original.jpg').convert('RGB').resize((N,N),Image.LANCZOS))
src=src8.astype(np.float32)/255
mx=src.max(-1); mn=src.min(-1); S_=np.where(mx>0,(mx-mn)/np.maximum(mx,1e-6),0); lum=src.mean(-1)
sat=S_>0.3
# 1) autocolantes pequenos antigos (fora das zonas de cor): pintam-se de branco (inpaint)
lab,n=ndi.label(sat); sz=ndi.sum(np.ones_like(lab),lab,range(n+1))
small=(sz<4000)[lab]&sat
img=src8.copy()
img=cv2.inpaint(img,ndi.binary_dilation(small,iterations=3).astype(np.uint8),7,cv2.INPAINT_TELEA)
sat=sat&~small
# 2) letras e emblemas dentro das zonas de cor: buracos pequenos passam a cor
labh,nh=ndi.label(~sat); szh=ndi.sum(np.ones_like(labh),labh,range(nh+1))
holes=(szh<N*N*0.006)[labh]; holes[labh==0]=False
navy_m=sat|holes
NAVY=np.array([12,41,80],np.float32)/255; WHITE=np.array([240,242,245],np.float32)/255
# 3) sombreado: so os detalhes finos (rebites, juntas); a mediana apaga as formas antigas
med=ndi.median_filter(lum,size=11)
detail=np.clip(lum/np.maximum(med,1e-3),0.85,1.08)
cls=(sat&(src[...,0]>=src[...,2]))*1+(sat&(src[...,2]>src[...,0]))*2+(~sat)*3
edge=ndi.binary_dilation(ndi.maximum_filter(cls,3)!=ndi.minimum_filter(cls,3),iterations=3)
detail=np.where(edge|holes,1.0,detail)
big=ndi.gaussian_filter(np.where(navy_m,lum,0),25)/np.maximum(ndi.gaussian_filter(navy_m.astype(np.float32),25),1e-3)
big=np.clip(big/np.median(lum[sat]),0.85,1.12)
navy_col=NAVY*(detail*big)[...,None]
m=ndi.gaussian_filter(navy_m.astype(np.float32),0.7)[...,None]
base=img.astype(np.float32)/255
# branco mais limpo: as zonas brancas antigas (cinzento claro) passam a branco, mantendo os rebites
li=base.mean(-1); wmask=(~navy_m)&(li>0.58)&((base.max(-1)-base.min(-1))<0.12)
medi=ndi.median_filter(li,size=11)
dw=np.clip(li/np.maximum(medi,1e-3),0.85,1.06)
bw=ndi.gaussian_filter(np.where(wmask,li,0),25)/np.maximum(ndi.gaussian_filter(wmask.astype(np.float32),25),1e-3)
bw=np.clip(bw/np.median(li[wmask]),0.9,1.03)
wm=ndi.gaussian_filter(wmask.astype(np.float32),0.7)[...,None]
base=base*(1-wm)+(WHITE*(dw*bw)[...,None])*wm
col=base*(1-m)+navy_col*m

# 4) autocolantes em 3D
def stamp(im, mask, U, V):
    global col
    a=np.array(im).astype(np.float32)/255; h,w=a.shape[:2]
    ins=mask&(U>=0)&(U<=1)&(V>=0)&(V<=1)
    if not ins.any(): return
    cu=U[ins]*(w-1); cv=V[ins]*(h-1)
    smp=np.stack([ndi.map_coordinates(a[...,k],[cv,cu],order=1) for k in range(4)],-1)
    al=smp[:,3:4]; col[ins]=col[ins]*(1-al)+smp[:,:3]*al
def side(im,zc,yc,h,sides=(+1,-1),xlim=0.8):
    w=h*im.size[0]/im.size[1]
    for sg in sides:
        mk=ok&(np.sign(X)==sg)&(nx*sg>0.35)&(np.abs(X)<xlim)
        U=((zc-Z) if sg>0 else (Z-zc))/w+0.5; V=(yc-Y)/h+0.5; stamp(im,mk,U,V)
def wing(im,xc,zc,h,top=True):
    w=h*im.size[0]/im.size[1]
    mk=ok&(np.abs(X)>1.0)&(Y>1.75)&((ny>0.6) if top else (ny<-0.6))
    U=((xc-X) if top else (X-xc))/w+0.5; V=(zc-Z)/h+0.5; stamp(im,mk,U,V)
def rgba(c): return tuple(int(v*255) for v in c)+(255,)
FB='/usr/share/fonts/opentype/inter/Inter-BlackItalic.otf'
def text(s,font,size,fg,pad=(0.04,0.1)):
    f=ImageFont.truetype(font,size); bb=f.getbbox(s); px,py=int(size*pad[0]),int(size*pad[1])
    im=Image.new('RGBA',(bb[2]-bb[0]+2*px,bb[3]-bb[1]+2*py),(0,0,0,0))
    ImageDraw.Draw(im).text((px-bb[0],py-bb[1]),s,font=f,fill=fg); return im
logo=Image.open('tools/pintura_c182/ubi_logo.png').convert('RGBA')
def tint(im,c):
    a=np.array(im); a[...,:3]=np.array(rgba(c)[:3]); return Image.fromarray(a).crop(Image.fromarray(a).getbbox())
full_w=tint(logo,WHITE); sym_w=tint(logo.crop((0,0,180,190)),WHITE); txt_w=tint(logo.crop((202,0,500,190)),WHITE)
sym_n=tint(logo.crop((0,0,180,190)),NAVY)
# faixa azul lateral: centro em altura por z (para centrar os autocolantes)
band=navy_m&ok&(np.abs(nx)>0.8)&(np.abs(X)<0.8)
def band_center(z):
    mk=band&(np.abs(Z-z)<0.1)
    return (np.percentile(Y[mk],10)+np.percentile(Y[mk],90))/2 if mk.sum()>50 else None
for z in [0.9,0.3,-0.4,-1.0,-1.6,-2.4,-3.0]: print('banda z',z,band_center(z))
yc_logo=band_center(-0.4); yc_reg=band_center(-2.6)
side(full_w,-0.55,yc_logo,0.28)                     # logo UBI completo na faixa
side(text('53702',FB,300,rgba(WHITE)),-3.1,yc_reg,0.22)   # matricula
side(sym_w,-5.25,2.05,0.40,xlim=0.3)                # leme
wing(full_w,-3.78,0.02,0.52,top=True)               # asa direita, cima
wing(full_w,3.76,-0.02,0.52,top=True)               # asa esquerda, cima
wing(text('53702',FB,300,rgba(WHITE)),3.81,0.05,0.42,top=False)
wing(sym_w,-3.73,0.03,0.48,top=False)

# 5) margem a volta das pecas (o filtro da textura le os vizinhos)
far=~ok; d,(iy,ix)=ndi.distance_transform_edt(far,return_indices=True)
pad=far&(d<10); col[pad]=col[iy[pad],ix[pad]]
out=(np.clip(col,0,1)*255).astype(np.uint8)
Image.fromarray(out).save('models/c182/c182_exterior.jpg',quality=92)

print('ok')

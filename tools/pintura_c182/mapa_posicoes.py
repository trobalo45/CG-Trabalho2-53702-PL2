# =============================================================================
#  Pintura (livery) do Cessna 182 - passo 1: mapa de posicoes
#  Tome Pedro - 53702 - PL2
#  Para cada pixel da textura exterior (atlas), guarda o ponto 3D do aviao que esse
#  pixel pinta (rasterizando os triangulos no espaco uv). Correr na pasta do trabalho:
#     python3 tools/pintura_c182/mapa_posicoes.py
# =============================================================================
import numpy as np
EXT={'Material.%03d'%n for n in [2,3,4,5,7,8,9,10,11,12,14]}
V=[];VT=[];VN=[];tris=[];cur=None
for l in open('models/c182/cessna182.obj'):
    if l.startswith('v '): V.append(l.split()[1:4])
    elif l.startswith('vt '): VT.append(l.split()[1:3])
    elif l.startswith('vn '): VN.append(l.split()[1:4])
    elif l.startswith('usemtl'): cur=l.split()[1]
    elif l.startswith('f ') and cur in EXT:
        tris.append([tuple(int(x) if x else 0 for x in t.split('/')) for t in l.split()[1:4]])
V=np.array(V,float);VT=np.array(VT,float);VN=np.array(VN,float)
T=np.array(tris)-1  # (n,3,3)
print('tris',len(T))
N=2048
P=np.full((N,N,3),np.nan,np.float32); NM=np.zeros((N,N,3),np.float32); CNT=np.zeros((N,N),np.uint8)
uv=VT[T[:,:,1]]; uv[...,1]=1-uv[...,1]; uv=uv%1.0*N   # pixel coords (x, y)
pos=V[T[:,:,0]]; nor=VN[T[:,:,2]]
for k in range(len(T)):
    a,b,c=uv[k]
    x0=int(np.floor(min(a[0],b[0],c[0]))); x1=int(np.ceil(max(a[0],b[0],c[0])))
    y0=int(np.floor(min(a[1],b[1],c[1]))); y1=int(np.ceil(max(a[1],b[1],c[1])))
    x0=max(x0,0);y0=max(y0,0);x1=min(x1,N-1);y1=min(y1,N-1)
    if x1<x0 or y1<y0: continue
    xs,ys=np.meshgrid(np.arange(x0,x1+1)+0.5,np.arange(y0,y1+1)+0.5)
    d=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
    if abs(d)<1e-12: continue
    l1=((b[1]-c[1])*(xs-c[0])+(c[0]-b[0])*(ys-c[1]))/d
    l2=((c[1]-a[1])*(xs-c[0])+(a[0]-c[0])*(ys-c[1]))/d
    l3=1-l1-l2
    e=-0.02
    m=(l1>=e)&(l2>=e)&(l3>=e)
    if not m.any(): continue
    p=l1[m,None]*pos[k,0]+l2[m,None]*pos[k,1]+l3[m,None]*pos[k,2]
    n=l1[m,None]*nor[k,0]+l2[m,None]*nor[k,1]+l3[m,None]*nor[k,2]
    yy=ys[m].astype(int); xx=xs[m].astype(int)
    P[yy,xx]=p; NM[yy,xx]=n; CNT[yy,xx]=np.minimum(CNT[yy,xx].astype(int)+1,255)
np.save('tools/pintura_c182/P.npy',P); np.save('tools/pintura_c182/NM.npy',NM)
print('covered',np.isfinite(P[...,0]).mean())

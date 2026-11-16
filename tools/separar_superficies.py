# =============================================================================
#  Separa os flaps e os ailerons do Cessna 182 em objetos proprios.
#  Tome Pedro - 53702 - PL2
#  No modelo original, os flaps e os ailerons das duas asas sao um so objeto
#  ("aileron_r.001"). Para os animar com matrizes diferentes, cada face vai para
#  flap_esq / flap_dir / aileron_esq / aileron_dir conforme o x do seu centro
#  (flaps ate |x| = 2.62 m; +x = asa esquerda). Uso, na pasta do trabalho:
#     python3 tools/separar_superficies.py models/c182/cessna182.obj
# =============================================================================
import sys
path = sys.argv[1]
lines = open(path).read().split('\n')
V = [list(map(float, l.split()[1:4])) for l in lines if l.startswith('v ')]
out, buf, inside, mat = [], {}, False, None

def flush():
    for name in ('flap_esq', 'flap_dir', 'aileron_esq', 'aileron_dir'):
        if name not in buf: continue
        out.append('o ' + name)
        for m, faces in buf[name]:
            out.append('usemtl ' + m); out.extend(faces)

for l in lines:
    if l.startswith('o '):
        if inside: flush(); buf = {}
        inside = (l.split()[1] == 'aileron_r.001')
        if inside: continue
    if inside:
        if l.startswith('usemtl'): mat = l.split()[1]; continue
        if l.startswith('f '):
            ids = [int(t.split('/')[0]) - 1 for t in l.split()[1:]]
            x = sum(V[i][0] for i in ids) / len(ids)
            name = ('flap_' if abs(x) < 2.62 else 'aileron_') + ('esq' if x > 0 else 'dir')
            groups = buf.setdefault(name, [])
            if not groups or groups[-1][0] != mat: groups.append((mat, []))
            groups[-1][1].append(l)
            continue
        if l.startswith('s '): continue
    out.append(l)
if inside: flush()
open(path, 'w').write('\n'.join(out))
print('ok')

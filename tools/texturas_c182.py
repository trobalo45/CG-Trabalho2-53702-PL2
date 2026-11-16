# =============================================================================
#  Texturas e .mtl do Cessna 182 (depois de tools/converter_c182.py).
#  Tome Pedro - 53702 - PL2
#
#  Uso:  python3 tools/texturas_c182.py <pasta com as imagens PNG do .glb> models/c182
#  (as imagens foram guardadas do Blender com image.save(); as mesmas estao em
#   cessna-182-skylane/textures/)
#
#  1. Reduz as texturas (no maximo 2048 px) e grava-as com nomes simples
#     (JPG quando nao tem transparencia, PNG quando tem).
#  2. Reescreve o .mtl: Kd = 1 quando ha textura (a cor vem da imagem), map_Kd,
#     brilho maior na pintura exterior, d < 1 nos vidros.
# =============================================================================
import sys, re
from PIL import Image
src, out = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/') + '/'
IM = {'c182_sign_3': ('c182_exterior.jpg', 2048), 'black-c182_sign_1': ('c182_rodas.jpg', 512),
      'int1': ('c182_interior1.jpg', 2048), 'inxt2': ('c182_interior2.jpg', 2048),
      'gauge_c182t_1': ('c182_instrumentos.jpg', 1024), '1': ('c182_ecra1.jpg', 1024), '2': ('c182_ecra2.jpg', 1024),
      'erj_windshield': ('c182_vidros.png', 1024), 'texts_182': ('c182_letreiros.png', 1024),
      'generic_glass': ('c182_vidro.png', 64), 'lights': ('c182_luzes.jpg', 256)}
for k, (f, s) in IM.items():
    im = Image.open(src + k + '.png').convert('RGBA')
    if max(im.size) > s:
        im = im.resize((s, int(s * im.size[1] / im.size[0])), Image.LANCZOS)
    if f.endswith('.jpg'): im.convert('RGB').save(out + f, quality=88)
    else: im.save(out + f, optimize=True)

# material -> imagem (lido dos nos de textura de cada material no Blender)
MAP = {'1': '1', '2': '2', 'c182_sign_1': 'black-c182_sign_1', 'erj_windshield': 'erj_windshield',
       'gauge_c182t_1': 'gauge_c182t_1', 'generic_glass': 'generic_glass', 'int1': 'int1', 'int2': 'inxt2',
       'lights': 'lights', 'lights.001': 'lights', 'lights.002': 'lights', 'Material.028': 'texts_182'}
EXT = {'Material.%03d' % n for n in [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14]}      # pintura exterior
for m in EXT: MAP[m] = 'c182_sign_3'
for n in [18, 19, 20]: MAP['Material.%03d' % n] = 'int1'
for n in [21, 23, 24, 25, 30]: MAP['Material.%03d' % n] = 'inxt2'
D = {'generic_glass': 0.35, 'erj_windshield': 0.95}                             # vidros

mats = re.findall(r'^newmtl (\S+)', open(out + 'cessna182.mtl').read(), re.M)
L = ['# Cessna 182 Skylane (VTX, Sketchfab) - materiais simplificados para o programa',
     '# map_Kd = textura difusa; d < 1 = transparente (vidros)', '']
for m in mats:
    shiny = m in EXT or m in D or m in ('default', 'Material.015', 'Material.016', 'Material.017')
    L += ['newmtl ' + m,
          'Kd 1.0 1.0 1.0' if m in MAP else ('Kd 0.93 0.94 0.95' if m in ('default', 'Material.015', 'Material.016') else 'Kd 0.85 0.85 0.85'),
          'Ks 0.45 0.45 0.45\nNs 60' if shiny else 'Ks 0.15 0.15 0.15\nNs 20',
          'd %.2f' % D.get(m, 1.0)]
    if m in MAP: L.append('map_Kd ' + IM[MAP[m]][0])
    L.append('')
open(out + 'cessna182.mtl', 'w').write('\n'.join(L))

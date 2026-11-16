# =============================================================================
#  Converte o Cessna 182 Skylane (Sketchfab, VTX, ficheiro .glb) para OBJ.
#  Tome Pedro - 53702 - PL2
#
#  Uso (precisa do Blender instalado):
#     blender -b --python tools/converter_c182.py -- <ficheiro.glb> models/c182/cessna182.obj
#  Depois: python3 tools/texturas_c182.py (texturas e .mtl)
#
#  O que faz:
#    1. Importa o .glb (malhas + materiais + texturas).
#    2. Deixa de fora o que nao interessa a um avião em voo:
#         rbf*         capas e fitas "Remove Before Flight" (so com o avião parado)
#         Icosphere*   esferas auxiliares do modelo
#         *.col        malhas de colisao
#         interior_lit iluminacao noturna dos instrumentos (fica por cima dos normais)
#    3. Poe o modelo nas convencoes do programa:
#         origem no ponto de contacto das rodas principais;
#         nariz para +Z, Y para cima, esquerda do piloto = +X.
#       No Blender o nariz esta em +Y e Z e para cima. Roda-se 180 graus em Z
#       (nariz para -Y) e o exportador de OBJ faz (x, y, z) -> (x, z, -y).
#       Resultado: (x, y, z)_Blender -> (-x, z, y)_OBJ.
#    4. Exporta triangulado, com normais, coordenadas de textura (vt) e
#       materiais; as texturas sao copiadas para a pasta do .obj.
# =============================================================================
import sys, math
import bpy, mathutils

glb, out = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)

EXCLUIR = ('rbf', 'Icosphere', 'interior_lit')
for o in list(bpy.data.objects):
    if o.type == 'MESH' and (o.name.startswith(EXCLUIR) or '.col' in o.name):
        bpy.data.objects.remove(o, do_unlink=True)

# rodas principais (traseiras): o ponto de contacto com o chao passa a ser a origem
def bbox(o):
    return [o.matrix_world @ mathutils.Vector(v) for v in o.bound_box]
rodas = bbox(bpy.data.objects['wheel_lr.child']) + bbox(bpy.data.objects['wheel_rr.child'])
y_rodas = sum(p.y for p in rodas) / len(rodas)
z_chao = min(p.z for p in rodas)

M = mathutils.Matrix.Rotation(math.pi, 4, 'Z') @ mathutils.Matrix.Translation((0, -y_rodas, -z_chao))
for o in bpy.data.objects:
    if o.parent is None:                      # so as raizes; os filhos acompanham
        o.matrix_world = M @ o.matrix_world
bpy.context.view_layer.update()

bpy.ops.wm.obj_export(filepath=out, apply_modifiers=True, export_triangulated_mesh=True,
                      export_uv=True, export_normals=True, export_materials=True,
                      path_mode='COPY', forward_axis='NEGATIVE_Z', up_axis='Y',
                      export_selected_objects=False)
print("exportado", out, "y_rodas", y_rodas, "z_chao", z_chao)

# =============================================================================
#  Exporta a pista do professor (RUNWAY 8.blend, do link do enunciado) para OBJ.
#  Tome Pedro - 53702 - PL2
#
#  Uso (precisa do Blender instalado):
#     blender -b "RUNWAY 8.blend" --python tools/exportar_pista.py -- models/runway.obj
#
#  O que faz:
#    1. Os numeros da pista ("09" e "27") sao objetos de TEXTO no Blender; o
#       formato .obj so guarda malhas, por isso convertem-se em malha primeiro.
#    2. Exporta tudo com os modificadores aplicados (o "Mirror" duplica a pista
#       e as marcas do eixo, o "Solidify" da espessura as marcas e aos numeros),
#       triangulado (o enunciado pede malhas de triangulos), com normais e
#       materiais (.mtl).
#    3. Eixos: o Blender usa Z para cima; o OpenGL / o programa usa Y para cima.
#       up_axis='Y', forward_axis='NEGATIVE_Z' faz a conversao (x, y, z) -> (x, z, -y).
#  A geometria nao e alterada; a escala e a posicao na cena sao dadas no main.cpp
#  pela matriz model da pista.
# =============================================================================
import sys
import bpy

out = sys.argv[-1]

bpy.ops.object.select_all(action='DESELECT')
for o in list(bpy.data.objects):
    if o.type == 'FONT':
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
bpy.ops.object.convert(target='MESH')

bpy.ops.wm.obj_export(filepath=out, apply_modifiers=True, export_triangulated_mesh=True,
                      export_uv=False, export_normals=True, export_materials=True,
                      forward_axis='NEGATIVE_Z', up_axis='Y', export_selected_objects=False)
print("Pista exportada para", out)

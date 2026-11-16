#!/usr/bin/env python3
# =============================================================================
#  Gera as 6 imagens da skybox (cubemap) usada no Trabalho 2 de CG.
#  Tome Pedro - 53702 - PL2
#
#  Para cada pixel de cada face calcula-se a DIRECAO 3D que ele representa
#  (como o OpenGL faz quando le um GL_TEXTURE_CUBE_MAP) e pinta-se o ceu
#  nessa direcao:
#    - gradiente do horizonte (cor do nevoeiro do jogo) ate ao zenite (azul)
#    - sol na mesma direcao da luz pontual da cena (-250, 400, 150)
#    - nuvens: ruido fractal (varias camadas de ruido) projetado num "teto"
#      horizontal, como se as nuvens estivessem a uma altura fixa
#  Como tudo depende so da direcao, as 6 faces encaixam sem costuras.
#
#  Uso:  python3 tools/gerar_skybox.py      (escreve models/skybox/*.jpg)
#  Precisa de: numpy, pillow
# =============================================================================
import os
import numpy as np
from PIL import Image

N = 1024                                    # resolucao de cada face
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "skybox")

HORIZON = np.array([0.62, 0.78, 0.92])      # = cor do nevoeiro no main.cpp
ZENITH  = np.array([0.22, 0.45, 0.82])
SUN_DIR = np.array([-250.0, 400.0, 150.0]); SUN_DIR /= np.linalg.norm(SUN_DIR)

rng = np.random.default_rng(53702)
GRID = 256
LATTICE = rng.random((GRID, GRID))          # valores aleatorios numa grelha


def value_noise(x, y):
    """Ruido suave: interpola os valores da grelha aleatoria (repete-se a cada GRID)."""
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - xi, y - yi
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)   # suavizacao (smoothstep)
    a = LATTICE[xi % GRID, yi % GRID]
    b = LATTICE[(xi + 1) % GRID, yi % GRID]
    c = LATTICE[xi % GRID, (yi + 1) % GRID]
    d = LATTICE[(xi + 1) % GRID, (yi + 1) % GRID]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(x, y, octaves=6):
    """Ruido fractal: soma de ruidos cada vez mais finos e mais fracos."""
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    for _ in range(octaves):
        total = total + amp * value_noise(x * freq, y * freq)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


def sky_color(d):
    """Cor do ceu para um array de direcoes d (..., 3) normalizadas."""
    up = d[..., 1]
    e = np.clip(up, 0.0, 1.0)
    t = e ** 0.55
    col = HORIZON * (1 - t[..., None]) + ZENITH * t[..., None]

    # nuvens num teto a altura 1: ponto = d / d.y (so para cima do horizonte)
    safe = np.maximum(up, 0.02)
    px, pz = d[..., 0] / safe, d[..., 2] / safe
    n = fbm(px * 2.2 + 50, pz * 2.2 + 50)
    cover = np.clip((n - 0.52) / 0.18, 0, 1)                 # so os valores altos viram nuvem
    fade = np.clip((up - 0.03) / 0.25, 0, 1)                 # desaparecem perto do horizonte
    cloud = cover * fade
    shade = 0.82 + 0.18 * np.clip(fbm(px * 4 + 7, pz * 4 + 7, 4) * 1.4 - 0.2, 0, 1)
    col = col * (1 - cloud[..., None]) + (np.array([1.0, 1.0, 1.0]) * shade[..., None]) * cloud[..., None]

    # sol: disco + brilho a volta
    cosang = np.clip(d @ SUN_DIR, -1, 1)
    glow = np.clip((cosang - 0.95) / 0.05, 0, 1) ** 3 * 0.35
    disc = (cosang > 0.9993).astype(float)
    col = col + glow[..., None] * np.array([1.0, 0.95, 0.8]) + disc[..., None] * 1.0

    # abaixo do horizonte: cor do nevoeiro (o chao do jogo desaparece nela)
    below = up < 0
    col[below] = HORIZON
    return np.clip(col, 0, 1)


# Tabela das faces do cubemap (especificacao do OpenGL).
# s = coordenada horizontal da imagem, t = vertical (de cima para baixo), ambos em [-1, 1].
FACES = {
    "posx": lambda s, t: ( np.ones_like(s), -t, -s),
    "negx": lambda s, t: (-np.ones_like(s), -t,  s),
    "posy": lambda s, t: ( s,  np.ones_like(s),  t),
    "negy": lambda s, t: ( s, -np.ones_like(s), -t),
    "posz": lambda s, t: ( s, -t,  np.ones_like(s)),
    "negz": lambda s, t: (-s, -t, -np.ones_like(s)),
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    c = (np.arange(N) + 0.5) / N * 2 - 1
    s, t = np.meshgrid(c, c)                                  # t muda por linha, s por coluna
    for name, f in FACES.items():
        x, y, z = f(s, t)
        d = np.stack([x, y, z], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        img = (sky_color(d) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(OUT, name + ".jpg"), quality=90)
        print("escrito", name + ".jpg")

// =============================================================================
//  Computacao Grafica (14350) - Trabalho 2
//  Simulacao da ATERRAGEM de um avião (Exercicio 2)
//
//  Tome Pedro - 53702 - PL2
//
//  O que o programa faz:
//    1. Gera em .obj o aeroporto (edificios, bases das luzes), o terreno
//       montanhoso e as cidades
//    2. Carrega os .obj (avião com texturas, pista do professor e os gerados)
//    para a GPU
//    3. Ilumina a cena com UMA luz pontual e Phong shading (calculo por
//    fragmento)
//    4. O utilizador PILOTA um Cessna 182 com o teclado e o rato e aterra-o.
//       O avião comeca estabilizado na rampa de 3 graus, a 1.55 km da pista,
//       antes de uma serra.
//       O movimento vem de um modelo de voo simples com os dados do avião
//       (massa, area da asa, potencia, flaps, travoes).
//    5. No toque, a aterragem e avaliada (razao de descida, inclinacao, pista).
//
//  Convencoes do mundo (unidades em metros):
//    - Y aponta para cima, o chao esta em y = 0
//    - A pista vai de z = 0 (soleira da "09", onde se aterra) ate z = RWY_LEN
//    - Rumo 0 = avião a voar no sentido +Z
// =============================================================================

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// stb_easy_font (Sean Barrett, dominio publico): transforma texto em quadrados
// (sem texturas), para escrever o menu e o painel no ecra.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "stb_easy_font.h"
#pragma GCC diagnostic pop

// stb_image (Sean Barrett, dominio publico): le imagens JPG/PNG (skybox e
// texturas do avião)
#define STB_IMAGE_IMPLEMENTATION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "stb_image.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
//  Constantes da cena
// -----------------------------------------------------------------------------
// Pista (models/runway.obj, exportada de "RUNWAY 8.blend").
// No ficheiro: 120 m x 7320 m, centrada na origem, eixo ao longo de Z, a
// cabeceira "09" do lado +Z com a barra da soleira em z = +2545.6. Na cena
// aplica-se  M = T(0, 0, z0) * S(0.25) * Ry(180):
//   Ry(180) -> a "09" fica do lado -Z (aterra-se no sentido +Z, como o avião
//   voa) S(0.25) -> 30 m x 1830 m (tamanho de um aerodromo para avioes como o
//   Cessna) T       -> a soleira da "09" fica em z = 0
const float RWY_SCALE = 0.25f;
const float RWY_THRESH_OBJ = 2545.6f; // soleira no ficheiro .obj
const float RWY_LEN =
    2 * RWY_THRESH_OBJ * RWY_SCALE;             // de soleira a soleira: ~1273 m
const float RWY_WIDTH = 2 * 56.93f * RWY_SCALE; // asfalto: ~28.5 m
const float AIM_Z =
    103.0f; // ponto de mira: meio das primeiras marcas de toque (76 a 129 m)

// -----------------------------------------------------------------------------
//  Dados do Cessna 182 Skylane (valores tipicos do POH / literatura,
//  arredondados)
// -----------------------------------------------------------------------------
const float G = 9.81f;      // gravidade (m/s^2)
const float RHO = 1.225f;   // densidade do ar ao nivel do mar (kg/m^3)
const float MASS = 1406.0f; // massa maxima a descolagem (kg)
const float WING_S = 16.2f; // area da asa (m^2)
const float CL0 = 0.30f;    // coef. de sustentacao com alfa = 0 (sem flaps)
const float CL_A = 5.0f;    // declive da curva CL(alfa) (por radiano)
const float CL_MAX =
    1.75f;                // CL maximo sem flaps (a sustentacao nao passa daqui)
const float CD0 = 0.032f; // resistencia parasita
const float K_IND = 0.06f;    // resistencia induzida: CD = CD0 + K*CL^2
const float FLAP_CL = 0.50f;  // CL extra com flaps a 30 graus
const float FLAP_CD = 0.045f; // resistencia extra com flaps a 30 graus
const float POWER =
    120000.0f; // potencia util (230 hp * rendimento da helice ~0.7), W
const float T_MAX = 3300.0f;  // tracao maxima parado (N)
const float MU_ROLL = 0.02f;  // atrito de rolamento no asfalto
const float MU_GRASS = 0.08f; // atrito de rolamento na relva
const float MU_BRAKE = 0.35f; // atrito com travoes

const float KT = 0.514444f;     // 1 no = 0.514444 m/s
const float FPM = 196.85f;      // 1 m/s = 196.85 ft/min
const float V_APP = 65.0f * KT; // velocidade de aproximacao: 65 kt
const float GLIDE_DEG = 3.0f;   // rampa de aproximacao: 3 graus
const float START_Z = -1550.0f; // onde comeca a simulacao: 1.55 km antes da
                                // pista (antes da serra)

// -----------------------------------------------------------------------------
//  Janela
// -----------------------------------------------------------------------------
GLFWwindow *window;
int fbWidth = 1280,
    fbHeight = 720; // tamanho do framebuffer (pixeis reais, conta com Retina)

// =============================================================================
//  SHADERS (Phong shading: a iluminacao e calculada em cada fragmento)
// =============================================================================
const char *VERTEX_SHADER = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;      // coordenadas de textura (vt do .obj)

uniform mat4 M;              // model: coordenadas do objeto -> mundo
uniform mat4 V;              // view:  mundo -> camara
uniform mat4 P;              // projecao perspetiva
uniform mat3 normalMatrix;   // transposta da inversa de M (para as normais)

out vec3 fragPos;            // posicao do fragmento no mundo
out vec3 fragNormal;         // normal no mundo (interpolada pelo rasterizador)
out float localZ;            // z no espaco do objeto (para esconder a helice no cockpit)
out vec2 uv;

void main() {
    uv          = aUV;
    vec4 world  = M * vec4(aPos, 1.0);
    fragPos     = world.xyz;
    fragNormal  = normalMatrix * aNormal;
    localZ      = aPos.z;
    gl_Position = P * V * world;
}
)";

const char *FRAGMENT_SHADER = R"(
#version 330 core
in vec3 fragPos;
in vec3 fragNormal;
in float localZ;
in vec2 uv;
uniform float clipZ;         // descarta o que estiver a frente deste z (helice vista do cockpit)

// luz pontual
uniform vec3  lightPos;
uniform vec3  lightColor;
uniform vec3  attenuation;   // (kc, kl, kq): 1 / (kc + kl*d + kq*d^2)
uniform float ambient;       // intensidade da luz ambiente

// camara
uniform vec3 viewPos;

// material (vem do ficheiro .mtl)
uniform vec3  Kd;            // cor difusa
uniform vec3  Ks;            // cor especular
uniform float Ns;            // expoente especular (brilho)
uniform float alpha;         // opacidade do material (vidro < 1)
uniform vec3  emissive;      // luz propria (janelas acesas a noite); normalmente 0
uniform sampler2D tex;       // textura difusa (map_Kd do .mtl)
uniform bool  useTex;        // false nos materiais sem textura

// reflexo do ceu (so no avião): a cubemap da skybox, lida na direcao refletida
uniform samplerCube skyCube;
uniform float reflectivity;  // 0 = sem reflexo
uniform vec3  skyTint;       // escurece o reflexo de noite (como a skybox)

// nevoeiro (o terreno ao longe mistura-se com a cor do ceu)
uniform vec3  fogColor;
uniform float fogDensity;

out vec4 color;

void main() {
    if (localZ > clipZ) discard;
    // cor do material: Kd, multiplicada pela textura quando existe
    vec4  texel = useTex ? texture(tex, uv) : vec4(1.0);
    vec3  base  = Kd * texel.rgb;
    float a     = alpha * texel.a;
    // partes recortadas da textura (letreiros, contornos): alpha test
    if (alpha >= 1.0 && a < 0.5) discard;
    if (a < 0.02) discard;
    // A normal interpolada deixa de ter comprimento 1, por isso normaliza-se.
    vec3 N = normalize(fragNormal);
    // Superficies finas (asas, helice) veem-se dos dois lados:
    // se estamos a ver a face de tras, a normal tem de apontar para nos.
    if (!gl_FrontFacing) N = -N;

    vec3  toLight = lightPos - fragPos;
    float d = length(toLight);
    vec3  L = toLight / d;                        // direcao para a luz
    vec3  E = normalize(viewPos - fragPos);       // direcao para o olho
    vec3  R = reflect(-L, N);                     // raio refletido

    float diff = max(dot(N, L), 0.0);                                  // Lambert
    float spec = (diff > 0.0) ? pow(max(dot(R, E), 0.0), Ns) : 0.0;    // Phong
    float att  = 1.0 / (attenuation.x + attenuation.y * d + attenuation.z * d * d);

    vec3 c = ambient * lightColor * base
           + att * lightColor * (diff * base + spec * Ks)
           + emissive;

    // Reflexo do ceu: o raio do olho reflete na superficie (a mesma funcao reflect
    // do especular) e vai buscar a cor da skybox nessa direcao. So nas superficies
    // brilhantes (Ks alto: pintura e vidros). Fresnel: reflete mais quando se ve de lado.
    if (reflectivity > 0.0) {
        vec3  Rv   = reflect(-E, N);
        vec3  env  = texture(skyCube, Rv).rgb * skyTint;
        float fres = 0.04 + 0.96 * pow(1.0 - max(dot(N, E), 0.0), 5.0);
        float k    = reflectivity * smoothstep(0.3, 0.45, Ks.r) * (0.25 + 0.75 * fres);
        c = mix(c, env, k);
    }

    float dist = length(viewPos - fragPos);
    float f = exp(-pow(dist * fogDensity, 2.0));
    // o reflexo especular torna o vidro mais visivel
    color = vec4(mix(fogColor, c, f), clamp(a + spec * 0.5, 0.0, 1.0));
}
)";

GLuint compileShader(GLenum type, const char *src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(s, 1024, nullptr, log);
    std::cerr << "Erro a compilar shader:\n" << log << std::endl;
  }
  return s;
}

GLuint createProgram() {
  GLuint vs = compileShader(GL_VERTEX_SHADER, VERTEX_SHADER);
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
  GLuint p = glCreateProgram();
  glAttachShader(p, vs);
  glAttachShader(p, fs);
  glLinkProgram(p);
  GLint ok;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetProgramInfoLog(p, 1024, nullptr, log);
    std::cerr << "Erro a ligar o programa:\n" << log << std::endl;
  }
  glDeleteShader(vs);
  glDeleteShader(fs);
  return p;
}

// =============================================================================
//  LEITURA DE FICHEIROS .OBJ / .MTL
// =============================================================================
//  Suporta:  v, vn, vt, f (triangulos ou poligonos, formatos
//            v | v/vt | v//vn | v/vt/vn, indices negativos), o, mtllib, usemtl.
//  Do .mtl le: newmtl, Kd, Ks, Ns, d, map_Kd.
//
//  Os vertices sao "desindexados": cada triangulo guarda as suas 3 posicoes,
//  3 normais e 3 coordenadas de textura seguidas num unico VBO
//  [x y z nx ny nz u v | x y z nx ny nz u v | ...]  (8 floats por vertice).
//  Os triangulos sao agrupados por (objeto, material) e desenham-se com um
//  glDrawArrays por grupo. O nome do objeto (linha "o") permite dar a cada
//  peca uma matriz propria (helice, flaps, cabeca do piloto...).
// =============================================================================
struct Material {
  glm::vec3 Kd = glm::vec3(0.8f);
  glm::vec3 Ks = glm::vec3(0.3f);
  float Ns = 32.0f;
  float d = 1.0f;    // opacidade (1 = opaco; < 1 = transparente, ex.: vidro)
  std::string mapKd; // ficheiro da textura difusa (vazio = sem textura)
  GLuint tex = 0;    // textura ja na GPU
};

// -----------------------------------------------------------------------------
//  Texturas 2D (map_Kd): le a imagem com stb_image e envia-a para a GPU.
//  Cada ficheiro so e carregado uma vez (varios materiais partilham a mesma).
// -----------------------------------------------------------------------------
std::map<std::string, GLuint> textureCache;

GLuint loadTexture(const std::string &path) {
  if (textureCache.count(path))
    return textureCache[path];
  int w, h, ch;
  // no OpenGL a linha v = 0 e a de baixo; nas imagens a primeira linha e a de
  // cima
  stbi_set_flip_vertically_on_load(true);
  unsigned char *data = stbi_load(path.c_str(), &w, &h, &ch, 4);
  stbi_set_flip_vertically_on_load(false);
  if (!data) {
    std::cerr << "Aviso: nao consegui ler a textura " << path << std::endl;
    return textureCache[path] = 0;
  }
  GLuint t;
  glGenTextures(1, &t);
  glBindTexture(GL_TEXTURE_2D, t);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               data);
  glGenerateMipmap(GL_TEXTURE_2D); // versoes mais pequenas (ao longe)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                  GL_REPEAT); // uv fora de [0,1] repete
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  stbi_image_free(data);
  return textureCache[path] = t;
}

struct SubMesh {      // bocado do VBO: o mesmo objeto e o mesmo material
  std::string object; // nome do objeto no .obj (linha "o"), "" se nao houver
  std::string name;   // nome do material
  Material mat;
  GLint first;   // primeiro vertice
  GLsizei count; // numero de vertices
};

struct Model {
  GLuint vao = 0, vbo = 0;
  std::vector<SubMesh> parts;
  glm::vec3 bbMin = glm::vec3(1e30f), bbMax = glm::vec3(-1e30f);
  int numTriangles = 0;
};

std::map<std::string, Material> loadMTL(const std::string &path) {
  std::map<std::string, Material> mats;
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Aviso: nao encontrei " << path << " (uso cor por defeito)"
              << std::endl;
    return mats;
  }
  std::string line, current;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag == "newmtl") {
      ss >> current;
      mats[current] = Material();
    } else if (tag == "Kd" && !current.empty())
      ss >> mats[current].Kd.r >> mats[current].Kd.g >> mats[current].Kd.b;
    else if (tag == "Ks" && !current.empty())
      ss >> mats[current].Ks.r >> mats[current].Ks.g >> mats[current].Ks.b;
    else if (tag == "Ns" && !current.empty())
      ss >> mats[current].Ns;
    else if (tag == "d" && !current.empty())
      ss >> mats[current].d;
    else if (tag == "map_Kd" && !current.empty())
      ss >> mats[current].mapKd;
  }
  return mats;
}

// Converte um indice do .obj (comeca em 1, pode ser negativo) para indice C++.
int fixIndex(int i, int n) { return (i > 0) ? i - 1 : n + i; }

bool loadOBJ(const std::string &path, Model &model) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Erro: nao consegui abrir " << path << std::endl;
    return false;
  }
  std::string dir = path.substr(0, path.find_last_of("/\\") + 1);

  std::vector<glm::vec3> positions, normals;
  std::vector<glm::vec2> uvs;
  std::map<std::string, Material> materials;
  std::map<std::string, std::vector<float>>
      dataPerMaterial;            // vertices de cada material
  std::vector<std::string> order; // ordem em que os materiais aparecem
  std::string currentMat = "default";
  std::string currentObj = ""; // linha "o nome"

  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;

    if (tag == "v") {
      glm::vec3 p;
      ss >> p.x >> p.y >> p.z;
      positions.push_back(p);
      model.bbMin = glm::min(model.bbMin, p);
      model.bbMax = glm::max(model.bbMax, p);
    } else if (tag == "vn") {
      glm::vec3 n;
      ss >> n.x >> n.y >> n.z;
      normals.push_back(n);
    } else if (tag == "vt") {
      glm::vec2 t;
      ss >> t.x >> t.y;
      uvs.push_back(t);
    } else if (tag == "mtllib") {
      std::string file;
      ss >> file;
      materials = loadMTL(dir + file);
    } else if (tag == "o") {
      ss >> currentObj;
    } else if (tag == "usemtl") {
      ss >> currentMat;
    } else if (tag == "f") {
      // le todos os vertices da face ("v", "v/vt", "v//vn" ou "v/vt/vn")
      std::vector<int> vi, ti, ni;
      std::string tok;
      while (ss >> tok) {
        int v = 0, t = 0, n = 0;
        size_t s1 = tok.find('/');
        v = std::stoi(tok.substr(0, s1));
        if (s1 != std::string::npos) {
          size_t s2 = tok.find('/', s1 + 1);
          if (s2 != s1 + 1) // "v/vt..." (nao "v//vn")
            t = std::stoi(tok.substr(s1 + 1, s2 - s1 - 1));
          if (s2 != std::string::npos && s2 + 1 < tok.size())
            n = std::stoi(tok.substr(s2 + 1));
        }
        vi.push_back(fixIndex(v, (int)positions.size()));
        ti.push_back(t != 0 ? fixIndex(t, (int)uvs.size()) : -1);
        ni.push_back(n != 0 ? fixIndex(n, (int)normals.size()) : -1);
      }
      std::string key =
          currentObj + "|" + currentMat; // grupo = objeto + material
      if (dataPerMaterial.find(key) == dataPerMaterial.end())
        order.push_back(key);
      std::vector<float> &out = dataPerMaterial[key];

      // poligono com k vertices -> k-2 triangulos em leque (0, i, i+1)
      for (size_t i = 1; i + 1 < vi.size(); i++) {
        int tri[3] = {0, (int)i, (int)i + 1};
        glm::vec3 a = positions[vi[tri[0]]], b = positions[vi[tri[1]]],
                  c = positions[vi[tri[2]]];
        glm::vec3 faceN = glm::cross(b - a, c - a);
        faceN = (glm::length(faceN) > 0.0f) ? glm::normalize(faceN)
                                            : glm::vec3(0, 1, 0);
        for (int k : tri) {
          glm::vec3 p = positions[vi[k]];
          glm::vec3 n =
              (ni[k] >= 0) ? normals[ni[k]] : faceN; // sem vn -> normal da face
          glm::vec2 t =
              (ti[k] >= 0) ? uvs[ti[k]] : glm::vec2(0.0f); // sem vt -> (0,0)
          out.insert(out.end(), {p.x, p.y, p.z, n.x, n.y, n.z, t.x, t.y});
        }
        model.numTriangles++;
      }
    }
  }

  // junta tudo num unico array e guarda onde comeca cada material
  std::vector<float> all;
  for (const std::string &key : order) {
    SubMesh sm;
    size_t bar = key.find('|');
    sm.object = key.substr(0, bar);
    std::string name = key.substr(bar + 1);
    sm.name = name;
    sm.mat = materials.count(name) ? materials[name] : Material();
    if (!sm.mat.mapKd.empty())
      sm.mat.tex = loadTexture(dir + sm.mat.mapKd);
    sm.first = (GLint)(all.size() / 8);
    sm.count = (GLsizei)(dataPerMaterial[key].size() / 8);
    all.insert(all.end(), dataPerMaterial[key].begin(),
               dataPerMaterial[key].end());
    model.parts.push_back(sm);
  }

  // envia para a GPU
  glGenVertexArrays(1, &model.vao);
  glBindVertexArray(model.vao);
  glGenBuffers(1, &model.vbo);
  glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
  glBufferData(GL_ARRAY_BUFFER, all.size() * sizeof(float), all.data(),
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0); // posicao
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(1); // normal
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(2); // coordenadas de textura
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                        (void *)(6 * sizeof(float)));
  glBindVertexArray(0);

  std::cout << "Carregado " << path << ": " << positions.size() << " vertices, "
            << model.numTriangles << " triangulos, " << model.parts.size()
            << " grupos (objeto+material)" << std::endl;
  return true;
}

// Matrizes extra por peca (nome do objeto -> matriz em coordenadas do modelo).
// A matriz final de uma peca e M * local: primeiro a peca mexe-se dentro do
// avião (rodar a helice, baixar um flap) e depois vai com o avião para o mundo.
// Uma matriz toda a zero esconde a peca.
using PartMatrices = std::map<std::string, glm::mat4>;

// night = true acende as janelas dos edificios (materiais "vidro_ed" e
// "janela_acesa") transparent = false desenha so as partes opacas; true so as
// transparentes (d < 1). Desenham-se primeiro TODOS os opacos e so depois os
// transparentes, com blending e sem escrever no depth buffer, para se ver o que
// esta por tras do vidro.
void drawModel(const Model &model, GLuint prog, const glm::mat4 &M,
               bool night = false, bool transparent = false,
               const PartMatrices *local = nullptr) {
  glBindVertexArray(model.vao);
  for (const SubMesh &sm : model.parts) {
    if ((sm.mat.d < 1.0f) != transparent)
      continue;
    glm::mat4 Mi = M;
    if (local) {
      auto it = local->find(sm.object);
      if (it != local->end()) {
        if (it->second[3][3] == 0.0f)
          continue; // peca escondida
        Mi = M * it->second;
      }
    }
    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(Mi)));
    glUniformMatrix4fv(glGetUniformLocation(prog, "M"), 1, GL_FALSE,
                       glm::value_ptr(Mi));
    glUniformMatrix3fv(glGetUniformLocation(prog, "normalMatrix"), 1, GL_FALSE,
                       glm::value_ptr(normalMatrix));
    // Camadas no mesmo plano (a pista do professor tem a base branca, o asfalto
    // e as marcas todos em y = 0): cada camada e "puxada" um pouco mais para a
    // camara no depth buffer com glPolygonOffset, para nao haver z-fighting.
    int layer = (sm.name == "Material" || sm.name == "Material.002" ||
                 sm.name == "estrada")
                    ? 1
                : (sm.name == "white_strip" || sm.name == "amarelo" ||
                   sm.name.rfind("janela", 0) == 0)
                    ? 2
                    : 0;
    bool paint = layer > 0;
    if (paint) {
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(-1.0f * layer, -4.0f * layer);
    }
    glUniform1f(glGetUniformLocation(prog, "alpha"), sm.mat.d);
    glUniform3fv(glGetUniformLocation(prog, "Kd"), 1,
                 glm::value_ptr(sm.mat.Kd));
    glUniform3fv(glGetUniformLocation(prog, "Ks"), 1,
                 glm::value_ptr(sm.mat.Ks));
    glUniform1f(glGetUniformLocation(prog, "Ns"), sm.mat.Ns);
    glUniform1i(glGetUniformLocation(prog, "useTex"), sm.mat.tex != 0);
    if (sm.mat.tex) {
      glActiveTexture(GL_TEXTURE0); // unidade 0 <-> sampler "tex"
      glBindTexture(GL_TEXTURE_2D, sm.mat.tex);
    }
    bool lit = sm.name.rfind("vidro_ed", 0) == 0 || sm.name == "janela_acesa";
    glm::vec3 em =
        (night && lit) ? glm::vec3(0.75f, 0.62f, 0.35f) : glm::vec3(0.0f);
    glUniform3fv(glGetUniformLocation(prog, "emissive"), 1, glm::value_ptr(em));
    glDrawArrays(GL_TRIANGLES, sm.first, sm.count);
    if (paint)
      glDisable(GL_POLYGON_OFFSET_FILL);
  }
  glBindVertexArray(0);
}

// =============================================================================
//  DESENHO 2D POR CIMA DA CENA (menu e painel de instrumentos)
// =============================================================================
//  Coordenadas em pixeis do framebuffer: (0,0) no canto superior esquerdo,
//  x para a direita, y para baixo. O vertex shader converte para NDC [-1,1].
//  Desenha-se no fim, sem depth test e com transparencia (blending).
// =============================================================================
const char *OVERLAY_VS = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
uniform vec2 screen;                      // largura e altura do framebuffer
void main() {
    gl_Position = vec4(aPos.x / screen.x * 2.0 - 1.0, 1.0 - aPos.y / screen.y * 2.0, 0.0, 1.0);
}
)";

const char *OVERLAY_FS = R"(
#version 330 core
uniform vec4 color;
out vec4 fragColor;
void main() { fragColor = color; }
)";

GLuint overlayProg = 0, overlayVAO = 0, overlayVBO = 0;
float uiScale = 1.0f; // 1 num ecra normal de 720 px de altura

void initOverlay() {
  GLuint vs = compileShader(GL_VERTEX_SHADER, OVERLAY_VS);
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, OVERLAY_FS);
  overlayProg = glCreateProgram();
  glAttachShader(overlayProg, vs);
  glAttachShader(overlayProg, fs);
  glLinkProgram(overlayProg);
  glDeleteShader(vs);
  glDeleteShader(fs);
  glGenVertexArrays(1, &overlayVAO);
  glGenBuffers(1, &overlayVBO);
  glBindVertexArray(overlayVAO);
  glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glBindVertexArray(0);
}

// envia uma lista de triangulos 2D e desenha-os com uma cor
void drawTriangles2D(const std::vector<float> &xy, const glm::vec4 &color) {
  if (xy.empty())
    return;
  glUniform4fv(glGetUniformLocation(overlayProg, "color"), 1,
               glm::value_ptr(color));
  glBindVertexArray(overlayVAO);
  glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
  glBufferData(GL_ARRAY_BUFFER, xy.size() * sizeof(float), xy.data(),
               GL_DYNAMIC_DRAW);
  glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(xy.size() / 2));
}

void drawRect(float x, float y, float w, float h, const glm::vec4 &color) {
  drawTriangles2D({x, y, x + w, y, x + w, y + h, x, y, x + w, y + h, x, y + h},
                  color);
}

// largura do texto em pixeis (para centrar)
float textWidth(const std::string &s, float scale) {
  return stb_easy_font_width((char *)s.c_str()) * scale * uiScale;
}

// Escreve texto (so ASCII, sem acentos). A letra base tem ~7 px de altura.
void drawText(float x, float y, const std::string &s, float scale,
              const glm::vec4 &color) {
  static char buffer[60000]; // ~270 bytes por caracter
  int quads = stb_easy_font_print(0, 0, (char *)s.c_str(), nullptr, buffer,
                                  sizeof(buffer));
  float k = scale * uiScale;
  std::vector<float> xy;
  xy.reserve(quads * 12);
  for (int q = 0; q < quads; q++) {
    // cada vertice ocupa 16 bytes: x, y, z (floats) + cor (4 bytes)
    float *v[4];
    for (int i = 0; i < 4; i++)
      v[i] = (float *)(buffer + (q * 4 + i) * 16);
    int order[6] = {0, 1, 2, 0, 2, 3}; // quadrado -> 2 triangulos
    for (int i : order) {
      xy.push_back(x + v[i][0] * k);
      xy.push_back(y + v[i][1] * k);
    }
  }
  drawTriangles2D(xy, color);
}

void drawTextCentered(float cx, float y, const std::string &s, float scale,
                      const glm::vec4 &color) {
  drawText(cx - textWidth(s, scale) / 2, y, s, scale, color);
}

// =============================================================================
//  SKYBOX (extra): um cubo enorme a volta da camara com uma textura cubemap
// =============================================================================
//  As 6 imagens (models/skybox/*.jpg) foram geradas por tools/gerar_skybox.py.
//  Truques:
//    - usa-se so a ROTACAO da camara (mat3(V)): o cubo anda sempre com ela,
//      por isso parece infinitamente longe;
//    - gl_Position = (...).xyww  ->  depois da divisao por w a profundidade e 1
//      (o maximo), por isso o ceu fica sempre atras de tudo;
//    - desenha-se no fim da cena com glDepthFunc(GL_LEQUAL), assim so pinta os
//      pixeis onde nao ha nada (poupa trabalho).
//    - a direcao para ler a textura e a propria posicao do vertice do cubo.
// =============================================================================
const char *SKY_VS = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 VP;            // P * mat4(mat3(V))
out vec3 dir;
void main() {
    dir = aPos;
    gl_Position = (VP * vec4(aPos, 1.0)).xyww;
}
)";

const char *SKY_FS = R"(
#version 330 core
in vec3 dir;
uniform samplerCube sky;
uniform vec3 tint;          // (1,1,1) de dia, escuro de noite
out vec4 color;
void main() { color = vec4(texture(sky, dir).rgb * tint, 1.0); }
)";

GLuint skyProg = 0, skyVAO = 0, skyVBO = 0, skyTex = 0;

bool loadCubemap(const std::string &dir) {
  // ordem exigida pelo OpenGL: +X, -X, +Y, -Y, +Z, -Z
  const char *names[6] = {"posx", "negx", "posy", "negy", "posz", "negz"};
  glGenTextures(1, &skyTex);
  glBindTexture(GL_TEXTURE_CUBE_MAP, skyTex);
  for (int i = 0; i < 6; i++) {
    int w, h, ch;
    std::string path = dir + "/" + names[i] + ".jpg";
    unsigned char *data = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!data) {
      std::cerr << "Aviso: nao consegui ler " << path << " (sem skybox)"
                << std::endl;
      return false;
    }
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, w, h, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
  }
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // CLAMP_TO_EDGE para nao aparecerem linhas nas juntas das faces
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  std::cout << "Skybox carregada de " << dir << std::endl;
  return true;
}

bool initSkybox() {
  GLuint vs = compileShader(GL_VERTEX_SHADER, SKY_VS);
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, SKY_FS);
  skyProg = glCreateProgram();
  glAttachShader(skyProg, vs);
  glAttachShader(skyProg, fs);
  glLinkProgram(skyProg);
  glDeleteShader(vs);
  glDeleteShader(fs);

  // cubo de lado 2 centrado na origem: 6 faces x 2 triangulos x 3 vertices
  const float c[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                         {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
  const int f[6][4] = {{0, 1, 2, 3}, {5, 4, 7, 6}, {4, 0, 3, 7},
                       {1, 5, 6, 2}, {3, 2, 6, 7}, {4, 5, 1, 0}};
  std::vector<float> v;
  for (auto &q : f)
    for (int k : {0, 1, 2, 0, 2, 3})
      v.insert(v.end(), {c[q[k]][0], c[q[k]][1], c[q[k]][2]});
  glGenVertexArrays(1, &skyVAO);
  glGenBuffers(1, &skyVBO);
  glBindVertexArray(skyVAO);
  glBindBuffer(GL_ARRAY_BUFFER, skyVBO);
  glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(),
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
  glBindVertexArray(0);
  return loadCubemap("models/skybox");
}

void drawSkybox(const glm::mat4 &V, const glm::mat4 &P, const glm::vec3 &tint) {
  if (!skyTex)
    return;
  glm::mat4 VP = P * glm::mat4(glm::mat3(V)); // tira a translacao da camara
  glDepthFunc(GL_LEQUAL);
  glUseProgram(skyProg);
  glUniformMatrix4fv(glGetUniformLocation(skyProg, "VP"), 1, GL_FALSE,
                     glm::value_ptr(VP));
  glUniform1i(glGetUniformLocation(skyProg, "sky"), 0);
  glUniform3fv(glGetUniformLocation(skyProg, "tint"), 1, glm::value_ptr(tint));
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, skyTex);
  glBindVertexArray(skyVAO);
  glDrawArrays(GL_TRIANGLES, 0, 36);
  glBindVertexArray(0);
  glDepthFunc(GL_LESS);
}

// =============================================================================
//  ESCRITOR DE .OBJ GENERICO (aeroporto, terreno e cidades)
// =============================================================================
//  Guarda vertices, normais e faces por material, e grava .obj + .mtl.
//  As faces sao sempre escritas no sentido anti-horario visto de FORA.
// =============================================================================
struct ObjWriter {
  struct Mat {
    glm::vec3 Kd, Ks;
    float Ns, d;
  };
  std::vector<glm::vec3> verts, norms;
  std::map<std::string, std::vector<int>>
      faces; // por "objeto|material": (v, n) (v, n) (v, n) ...
  std::vector<std::string> faceOrder;
  std::map<std::string, Mat> mats;
  std::vector<std::string> order;
  std::string object; // objeto atual (vira uma linha "o" no .obj)

  void material(const std::string &n, glm::vec3 kd,
                glm::vec3 ks = glm::vec3(0.1f), float ns = 16, float d = 1.0f) {
    if (!mats.count(n))
      order.push_back(n);
    mats[n] = {kd, ks, ns, d};
  }
  std::vector<int> &
  F(const std::string &m) { // faces do objeto atual com o material m
    std::string key = object + "|" + m;
    if (!faces.count(key))
      faceOrder.push_back(key);
    return faces[key];
  }
  int v(glm::vec3 p) {
    verts.push_back(p);
    return (int)verts.size();
  }
  int n(glm::vec3 d) {
    norms.push_back(glm::normalize(d));
    return (int)norms.size();
  }

  // quadrilatero a-b-c-d (anti-horario visto do lado para onde a normal aponta)
  void quad(const std::string &m, glm::vec3 a, glm::vec3 b, glm::vec3 c,
            glm::vec3 d) {
    glm::vec3 nn = glm::cross(b - a, c - a);
    int ni = n(nn), ia = v(a), ib = v(b), ic = v(c), id = v(d);
    std::vector<int> &f = F(m);
    f.insert(f.end(), {ia, ni, ib, ni, ic, ni, ia, ni, ic, ni, id, ni});
  }
  void tri(const std::string &m, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
    int ni = n(glm::cross(b - a, c - a)), ia = v(a), ib = v(b), ic = v(c);
    std::vector<int> &f = F(m);
    f.insert(f.end(), {ia, ni, ib, ni, ic, ni});
  }

  // caixa com a base centrada em 'base', tamanho (sx, sy, sz), rodada 'yaw'
  // graus em Y
  void box(const std::string &m, glm::vec3 base, glm::vec3 size,
           float yaw = 0) {
    glm::mat4 T = glm::rotate(glm::translate(glm::mat4(1), base),
                              glm::radians(yaw), glm::vec3(0, 1, 0));
    auto P = [&](float x, float y, float z) {
      return glm::vec3(
          T * glm::vec4(x * size.x / 2, y * size.y, z * size.z / 2, 1));
    };
    glm::vec3 p000 = P(-1, 0, -1), p100 = P(1, 0, -1), p110 = P(1, 1, -1),
              p010 = P(-1, 1, -1);
    glm::vec3 p001 = P(-1, 0, 1), p101 = P(1, 0, 1), p111 = P(1, 1, 1),
              p011 = P(-1, 1, 1);
    quad(m, p001, p101, p111, p011); // frente (+z)
    quad(m, p100, p000, p010, p110); // tras  (-z)
    quad(m, p101, p100, p110, p111); // +x
    quad(m, p000, p001, p011, p010); // -x
    quad(m, p011, p111, p110, p010); // topo
    quad(m, p000, p100, p101, p001); // fundo
  }

  // cilindro vertical (com normais suaves nos lados) e tampa em cima
  void cylinder(const std::string &m, glm::vec3 base, float r, float h,
                int seg = 16, bool cap = true) {
    std::vector<int> &f = F(m);
    int top = cap ? v(base + glm::vec3(0, h, 0)) : 0,
        nUp = cap ? n(glm::vec3(0, 1, 0)) : 0;
    for (int i = 0; i < seg; i++) {
      float a0 = 2 * glm::pi<float>() * i / seg,
            a1 = 2 * glm::pi<float>() * (i + 1) / seg;
      glm::vec3 d0(std::sin(a0), 0, std::cos(a0)),
          d1(std::sin(a1), 0, std::cos(a1));
      int n0 = n(d0), n1 = n(d1);
      int b0 = v(base + r * d0), b1 = v(base + r * d1);
      int t0 = v(base + r * d0 + glm::vec3(0, h, 0)),
          t1 = v(base + r * d1 + glm::vec3(0, h, 0));
      f.insert(f.end(), {b0, n0, b1, n1, t1, n1, b0, n0, t1, n1, t0, n0});
      if (cap)
        f.insert(f.end(), {top, nUp, t0, nUp, t1, nUp});
    }
  }

  // elipsoide (esfera se os 3 raios forem iguais) com normais suaves.
  // 'pick' escolhe o material de cada bocado pela direcao a partir do centro
  // ("" = nao desenha esse bocado, ex.: a abertura do capacete para a cara).
  void sphere(const std::string &m, glm::vec3 c, glm::vec3 r, int slices = 20,
              int stacks = 14,
              std::function<std::string(glm::vec3)> pick = nullptr) {
    auto P = [&](int i, int j) {
      float th = glm::pi<float>() * j / stacks,
            ph = 2 * glm::pi<float>() * i / slices;
      return glm::vec3(std::sin(th) * std::sin(ph), std::cos(th),
                       std::sin(th) * std::cos(ph));
    };
    for (int j = 0; j < stacks; j++)
      for (int i = 0; i < slices; i++) {
        glm::vec3 d[4] = {P(i, j), P(i, j + 1), P(i + 1, j + 1), P(i + 1, j)};
        std::string mm =
            pick ? pick(glm::normalize(d[0] + d[1] + d[2] + d[3])) : m;
        if (mm.empty())
          continue;
        int vi[4], ni[4];
        for (int k = 0; k < 4; k++) {
          vi[k] = v(c + r * d[k]);
          ni[k] = n(d[k] / (r * r));
        }
        std::vector<int> &f = F(mm);
        f.insert(f.end(), {vi[0], ni[0], vi[1], ni[1], vi[2], ni[2], vi[0],
                           ni[0], vi[2], ni[2], vi[3], ni[3]});
      }
  }

  bool save(const std::string &objPath, const std::string &mtlName,
            const std::string &mtlPath, const std::string &comment) {
    std::ofstream mtl(mtlPath), obj(objPath);
    if (!mtl || !obj)
      return false;
    mtl << "# Materiais gerados por main.cpp\n";
    for (const std::string &m : order) {
      const Mat &q = mats[m];
      mtl << "\nnewmtl " << m << "\nKd " << q.Kd.r << " " << q.Kd.g << " "
          << q.Kd.b << "\nKs " << q.Ks.r << " " << q.Ks.g << " " << q.Ks.b
          << "\nNs " << q.Ns << "\nd " << q.d << "\n";
    }
    obj << "# " << comment << "\nmtllib " << mtlName << "\n";
    for (glm::vec3 &p : verts)
      obj << "v " << p.x << " " << p.y << " " << p.z << "\n";
    for (glm::vec3 &d : norms)
      obj << "vn " << d.x << " " << d.y << " " << d.z << "\n";
    std::string lastObj;
    for (const std::string &key : faceOrder) {
      size_t bar = key.find('|');
      std::string o = key.substr(0, bar), m = key.substr(bar + 1);
      if (!o.empty() && o != lastObj) {
        obj << "o " << o << "\n";
        lastObj = o;
      }
      obj << "usemtl " << m << "\n";
      std::vector<int> &f = faces[key];
      for (size_t i = 0; i < f.size(); i += 6)
        obj << "f " << f[i] << "//" << f[i + 1] << " " << f[i + 2] << "//"
            << f[i + 3] << " " << f[i + 4] << "//" << f[i + 5] << "\n";
    }
    return true;
  }
};

// =============================================================================
//  LUZES DA PISTA (extra)
// =============================================================================
//  Cada luz e um ponto com cor. Desenha-se como um "point sprite": um quadrado
//  virado para a camara (GL_POINTS + gl_PointSize) onde o fragment shader pinta
//  um disco que se desvanece para as bordas. Mistura aditiva (GL_ONE): as luzes
//  somam-se a cor que ja la esta, por isso brilham.
//  As bases fisicas das luzes (caixinhas) vao para models/airport.obj.
//
//  PAPI (Precision Approach Path Indicator): 4 luzes ao lado do ponto de mira.
//  Cada uma tem um angulo; fica BRANCA se o piloto a ve acima desse angulo e
//  VERMELHA se abaixo. Na rampa de 3 graus: 2 brancas + 2 vermelhas.
//    4 brancas = muito alto | 4 vermelhas = muito baixo
// =============================================================================
enum LightKind { L_EDGE, L_THRESHOLD, L_END, L_APPROACH, L_PAPI };

struct RunwayLight {
  glm::vec3 pos;
  glm::vec3 color;
  float size; // raio do brilho (m)
  LightKind kind;
  float param; // PAPI: angulo (graus); aproximacao: distancia a soleira
};
std::vector<RunwayLight> runwayLights;
std::vector<glm::vec3>
    cityLamps; // candeeiros das cidades (so brilham de noite; ver CIDADES)

const glm::vec3 L_WHITE(1.0f, 0.95f, 0.80f), L_GREEN(0.2f, 1.0f, 0.35f),
    L_RED(1.0f, 0.15f, 0.1f);

void buildRunwayLights() {
  runwayLights.clear();
  const float hw = RWY_WIDTH / 2;
  // bordas: brancas a cada 60 m, 2.5 m fora do asfalto
  for (float z = 0; z <= RWY_LEN + 0.1f; z += 60)
    for (float s : {-1.0f, 1.0f})
      runwayLights.push_back(
          {{s * (hw + 2.5f), 0.4f, z}, L_WHITE, 0.9f, L_EDGE, 0});
  // soleira (verde) e fim da pista (vermelho)
  for (float x = -hw; x <= hw + 0.1f; x += hw / 4) {
    runwayLights.push_back({{x, 0.3f, -3.0f}, L_GREEN, 1.0f, L_THRESHOLD, 0});
    runwayLights.push_back({{x, 0.3f, RWY_LEN + 3.0f}, L_RED, 1.0f, L_END, 0});
  }
  // luzes de aproximacao: barras de 5 luzes no eixo, de 30 em 30 m, e uma barra
  // larga aos 150 m
  for (float d = 30; d <= 300; d += 30) {
    for (float x = -2; x <= 2.01f; x += 1)
      runwayLights.push_back({{x, 0.6f, -d}, L_WHITE, 1.0f, L_APPROACH, d});
    if (d == 150)
      for (float x = -15; x <= 15.01f; x += 2.5f)
        if (std::fabs(x) > 3)
          runwayLights.push_back(
              {{x, 0.6f, -d}, L_WHITE, 1.0f, L_APPROACH, -1});
  }
  // PAPI do lado esquerdo do piloto (+X), ao nivel do ponto de mira.
  // A unidade mais perto da pista tem o angulo maior.
  const float ang[4] = {3.5f, 3.1667f, 2.8333f, 2.5f};
  for (int i = 0; i < 4; i++)
    runwayLights.push_back(
        {{hw + 15 + 9 * i, 0.8f, AIM_Z}, L_WHITE, 1.6f, L_PAPI, ang[i]});
}

// cor de cada luz neste instante (PAPI depende de onde esta o piloto;
// aproximacao pisca)
glm::vec3 lightColorNow(const RunwayLight &L, const glm::vec3 &pilotEye,
                        double time, float &sizeMul) {
  sizeMul = 1.0f;
  if (L.kind == L_PAPI) {
    glm::vec3 d = pilotEye - L.pos;
    float horiz = std::sqrt(d.x * d.x + d.z * d.z);
    float elev = glm::degrees(std::atan2(d.y, horiz));
    return (elev > L.param) ? L_WHITE : L_RED;
  }
  if (L.kind == L_APPROACH && L.param > 0) {
    // "coelho": um flash percorre as barras de longe para perto, 2 vezes por
    // segundo
    float phase = (float)std::fmod(time * 2.0, 1.0);
    float pos = 300 - phase * 300; // de 300 m ate 0
    if (std::fabs(pos - L.param) < 15) {
      sizeMul = 2.2f;
      return glm::vec3(1.2f);
    }
  }
  return L.color;
}

// bases das luzes (vao para o airport.obj)
void addLightFixtures(ObjWriter &w) {
  w.material("luz_base", glm::vec3(0.25f), glm::vec3(0.2f), 16);
  w.material("papi", glm::vec3(0.85f, 0.85f, 0.80f), glm::vec3(0.3f), 16);
  for (const RunwayLight &L : runwayLights) {
    if (L.kind == L_PAPI)
      w.box("papi", glm::vec3(L.pos.x, 0, L.pos.z),
            glm::vec3(1.6f, 0.75f, 0.8f));
    else
      w.box("luz_base", glm::vec3(L.pos.x, 0, L.pos.z),
            glm::vec3(0.2f, L.pos.y - 0.2f, 0.2f)); // abaixo do brilho
  }
}

const char *GLOW_VS = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColorSize;   // rgb = cor, a = raio do brilho em metros
uniform mat4 VP;
uniform float pxPerMeter;                  // pixeis que 1 m ocupa a distancia 1
uniform float minSize;                     // tamanho minimo em pixeis (para se ver ao longe)
uniform float maxSize;                     // tamanho maximo (para nao tapar o ecra ao perto)
out vec3 col;
void main() {
    gl_Position = VP * vec4(aPos, 1.0);
    // tamanho no ecra diminui com a distancia (w = distancia a camara)
    gl_PointSize = clamp(2.0 * aColorSize.a * pxPerMeter / gl_Position.w, minSize, maxSize);
    col = aColorSize.rgb;
}
)";

const char *GLOW_FS = R"(
#version 330 core
in vec3 col;
uniform float intensity;
out vec4 color;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;    // -1..1 dentro do quadrado do ponto
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;                 // so o disco
    float a = exp(-r2 * 4.0) + 0.6 * exp(-r2 * 30.0);    // centro forte + aureola
    color = vec4(col * a * intensity, 1.0);
}
)";

GLuint glowProg = 0, glowVAO = 0, glowVBO = 0;

void initGlow() {
  GLuint vs = compileShader(GL_VERTEX_SHADER, GLOW_VS);
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, GLOW_FS);
  glowProg = glCreateProgram();
  glAttachShader(glowProg, vs);
  glAttachShader(glowProg, fs);
  glLinkProgram(glowProg);
  glDeleteShader(vs);
  glDeleteShader(fs);
  glGenVertexArrays(1, &glowVAO);
  glGenBuffers(1, &glowVBO);
  glBindVertexArray(glowVAO);
  glBindBuffer(GL_ARRAY_BUFFER, glowVBO);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glBindVertexArray(0);
}

// junta os pontos de luz da pista (e os candeeiros das cidades a noite) ao
// vetor data: por ponto 7 floats [x y z r g b raio]
void addRunwayLights(std::vector<float> &data, const glm::vec3 &pilotEye,
                     double time, bool night) {
  for (const RunwayLight &L : runwayLights) {
    float k;
    glm::vec3 c = lightColorNow(L, pilotEye, time, k);
    data.insert(data.end(), {L.pos.x, L.pos.y, L.pos.z, c.r, c.g, c.b,
                             L.size * k * (night ? 1.6f : 1.0f)});
  }
  if (night) // candeeiros das cidades: luz amarela-alaranjada
    for (const glm::vec3 &p : cityLamps)
      data.insert(data.end(), {p.x, p.y, p.z, 1.0f, 0.72f, 0.38f, 1.6f});
}

// desenha todos os pontos de luz (pista, cidades, avião) como point sprites
// aditivos
void drawGlowPoints(const std::vector<float> &data, const glm::mat4 &VP,
                    float fovDeg, int viewH, bool night) {
  if (data.empty())
    return;
  float pxPerMeter = viewH / (2.0f * std::tan(glm::radians(fovDeg) / 2));
  glUseProgram(glowProg);
  glUniformMatrix4fv(glGetUniformLocation(glowProg, "VP"), 1, GL_FALSE,
                     glm::value_ptr(VP));
  glUniform1f(glGetUniformLocation(glowProg, "pxPerMeter"), pxPerMeter);
  glUniform1f(glGetUniformLocation(glowProg, "minSize"), 3.0f * uiScale);
  glUniform1f(glGetUniformLocation(glowProg, "maxSize"), 36.0f * uiScale);
  glUniform1f(glGetUniformLocation(glowProg, "intensity"), night ? 1.3f : 0.9f);
  glBindVertexArray(glowVAO);
  glBindBuffer(GL_ARRAY_BUFFER, glowVBO);
  glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(),
               GL_DYNAMIC_DRAW);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE); // aditivo
  glDepthMask(GL_FALSE);       // as luzes nao tapam nada
  glDrawArrays(GL_POINTS, 0, (GLsizei)(data.size() / 7));
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  glBindVertexArray(0);
}

// =============================================================================
//  EDIFICIOS DO AEROPORTO (extra)
// =============================================================================
// quadrilatero virado para 'out' (troca a ordem se for preciso)
void quadOut(ObjWriter &w, const std::string &m, glm::vec3 a, glm::vec3 b,
             glm::vec3 c, glm::vec3 d, glm::vec3 out) {
  if (glm::dot(glm::cross(b - a, c - a), out) < 0)
    w.quad(m, d, c, b, a);
  else
    w.quad(m, a, b, c, d);
}

// tronco de cone entre p0 (raio r0) e p1 (raio r1), em qualquer direcao (para a
// manga de vento)
void frustum(ObjWriter &w, const std::string &m, glm::vec3 p0, glm::vec3 p1,
             float r0, float r1, int seg = 12) {
  glm::vec3 ax = glm::normalize(p1 - p0);
  glm::vec3 u = glm::normalize(glm::cross(
      ax, std::fabs(ax.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)));
  glm::vec3 v = glm::cross(ax, u);
  for (int i = 0; i < seg; i++) {
    float a0 = 2 * glm::pi<float>() * i / seg,
          a1 = 2 * glm::pi<float>() * (i + 1) / seg;
    glm::vec3 d0 = std::cos(a0) * u + std::sin(a0) * v,
              d1 = std::cos(a1) * u + std::sin(a1) * v;
    quadOut(w, m, p0 + r0 * d0, p0 + r0 * d1, p1 + r1 * d1, p1 + r1 * d0,
            d0 + d1);
  }
}

// Hangar: paredes + telhado em arco (meio cilindro achatado) com o eixo ao
// longo de Z
void hangar(ObjWriter &w, glm::vec3 c, float sx, float wallH, float sz,
            float roofH) {
  w.box("hangar", c, glm::vec3(sx, wallH, sz));
  const int seg = 14;
  float z0 = c.z - sz / 2, z1 = c.z + sz / 2, R = sx / 2;
  for (int i = 0; i < seg; i++) {
    float t0 = glm::pi<float>() * i / seg,
          t1 = glm::pi<float>() * (i + 1) / seg;
    glm::vec3 a(c.x + R * std::cos(t0), c.y + wallH + roofH * std::sin(t0), 0);
    glm::vec3 b(c.x + R * std::cos(t1), c.y + wallH + roofH * std::sin(t1), 0);
    glm::vec3 out(std::cos((t0 + t1) / 2), std::sin((t0 + t1) / 2), 0);
    quadOut(w, "telhado", {a.x, a.y, z0}, {b.x, b.y, z0}, {b.x, b.y, z1},
            {a.x, a.y, z1}, out);
    // as duas paredes da frente/tras do arco (triangulos ate ao centro)
    glm::vec3 m0(c.x, c.y + wallH, z0), m1(c.x, c.y + wallH, z1);
    w.tri("hangar", m1, {a.x, a.y, z1}, {b.x, b.y, z1});
    w.tri("hangar", m0, {b.x, b.y, z0}, {a.x, a.y, z0});
  }
  // porta grande virada para a pista (+X)
  float xd = c.x + sx / 2 + 0.05f;
  quadOut(w, "porta", {xd, 0.0f, c.z - sz * 0.35f},
          {xd, 0.0f, c.z + sz * 0.35f}, {xd, wallH * 0.85f, c.z + sz * 0.35f},
          {xd, wallH * 0.85f, c.z - sz * 0.35f}, {1, 0, 0});
}

// Posicoes dos edificios (todos a direita do piloto, isto e, do lado -X da
// pista)
const glm::vec3 TOWER_POS(-75.0f, 0.0f, 240.0f);
const float TOWER_H = 14.0f; // altura do fuste; a cabine fica por cima
const std::vector<glm::vec3> PARKED = {
    {-62.0f, 0.0f, 360.0f}, {-62.0f, 0.0f, 385.0f}}; // avioes estacionados

void addBuildings(ObjWriter &w) {
  w.material("placa", glm::vec3(0.30f, 0.30f, 0.31f), glm::vec3(0.08f), 16);
  w.material("amarelo", glm::vec3(0.90f, 0.72f, 0.10f), glm::vec3(0.2f), 16);
  w.material("betao", glm::vec3(0.78f, 0.76f, 0.72f), glm::vec3(0.15f), 16);
  w.material("vidro_ed", glm::vec3(0.16f, 0.26f, 0.32f), glm::vec3(0.9f), 80);
  w.material("telhado", glm::vec3(0.55f, 0.57f, 0.60f), glm::vec3(0.6f), 48);
  w.material("hangar", glm::vec3(0.70f, 0.72f, 0.74f), glm::vec3(0.3f), 32);
  w.material("porta", glm::vec3(0.32f, 0.34f, 0.37f), glm::vec3(0.2f), 16);
  w.material("poste", glm::vec3(0.55f, 0.55f, 0.55f), glm::vec3(0.4f), 32);
  w.material("laranja", glm::vec3(0.95f, 0.42f, 0.08f), glm::vec3(0.2f), 16);
  w.material("branco", glm::vec3(0.92f, 0.92f, 0.92f), glm::vec3(0.2f), 16);

  // placa de estacionamento e caminho de rolagem (asfalto ao nivel da pista)
  w.quad("placa", {-120, 0, 150}, {-120, 0, 420}, {-40, 0, 420}, {-40, 0, 150});
  w.quad("placa", {-40, 0, 290}, {-40, 0, 308}, {-16, 0, 308}, {-16, 0, 290});
  w.quad("amarelo", {-55, 0.05f, 298.6f}, {-55, 0.05f, 299.4f},
         {-16, 0.05f, 299.4f}, {-16, 0.05f, 298.6f});
  w.quad("amarelo", {-55.4f, 0.05f, 299.4f}, {-55.4f, 0.05f, 400},
         {-54.6f, 0.05f, 400}, {-54.6f, 0.05f, 299.4f});

  // torre de controlo: fuste, cabine de vidro, telhado e antena
  w.cylinder("betao", TOWER_POS, 2.5f, TOWER_H, 20);
  w.cylinder("betao", TOWER_POS + glm::vec3(0, TOWER_H - 0.4f, 0), 4.2f, 0.4f,
             20); // varanda
  w.cylinder("vidro_ed", TOWER_POS + glm::vec3(0, TOWER_H, 0), 3.5f, 3.2f,
             8); // cabine (octogonal)
  w.cylinder("telhado", TOWER_POS + glm::vec3(0, TOWER_H + 3.2f, 0), 4.0f, 0.5f,
             8);
  w.box("poste", TOWER_POS + glm::vec3(0, TOWER_H + 3.7f, 0),
        glm::vec3(0.15f, 4.0f, 0.15f));

  // hangar e terminal
  hangar(w, glm::vec3(-95, 0, 190), 24, 7, 30, 5);
  w.box("betao", glm::vec3(-92, 0, 340), glm::vec3(22, 8, 60));
  w.box("telhado", glm::vec3(-92, 8, 340), glm::vec3(23, 0.6f, 61));
  float xf = -92 + 11 + 0.05f; // fachada virada para a pista
  for (float y0 : {1.2f, 4.6f})
    quadOut(w, "vidro_ed", {xf, y0, 312}, {xf, y0, 368}, {xf, y0 + 2.2f, 368},
            {xf, y0 + 2.2f, 312}, {1, 0, 0});

  // manga de vento: poste e cone as riscas laranja / branco (aponta para onde
  // vai o vento)
  glm::vec3 ws(-30, 0, 60);
  w.cylinder("poste", ws, 0.08f, 6.0f, 8);
  glm::vec3 a = ws + glm::vec3(0, 5.8f, 0),
            dir = glm::normalize(glm::vec3(0.5f, -0.15f, -1.0f));
  for (int i = 0; i < 4; i++)
    frustum(w, (i % 2) ? "branco" : "laranja", a + dir * (0.9f * i),
            a + dir * (0.9f * (i + 1)), 0.45f - 0.07f * i,
            0.45f - 0.07f * (i + 1));
}

// =============================================================================
//  CAMARA DO COCKPIT: olhos do piloto, em coordenadas do avião
//  (as do cessna182.obj: origem nas rodas principais, nariz para +Z,
//  +X e a esquerda de quem olha para a frente -> lugar do comandante).
// =============================================================================
const glm::vec3 PILOT_EYE(0.27f, 2.05f, 0.25f);
const float PROP_CLIP_Z =
    2.45f; // no cockpit esconde-se tudo a frente disto (a helice)

// =============================================================================
//  PILOTO (extra) -> models/pilot.obj (gerado pelo programa)
// =============================================================================
//  Boneco feito de elipsoides (ObjWriter::sphere), troncos de cone (frustum) e
//  caixas, sentado no lugar da esquerda, em coordenadas do avião. Tem 3 objetos
//  ("o" no .obj), cada um com a sua matriz:
//    corpo  - vai com o avião;
//    cabeca - roda a volta do pescoco (olha para dentro das curvas);
//    bracos - as maos seguem o manche quando se puxa ou empurra.
//  (a camara do cockpit, PILOT_EYE, fica um pouco acima dos olhos do boneco,
//   para se ver bem por cima do tablier)
//  Fato azul UBI, capacete branco com risca azul e viseira escura, luvas
//  pretas.
// =============================================================================
const glm::vec3 PILOT_NECK(0.27f, 1.65f, 0.13f); // pivo da cabeca

// membro: tronco de cone com "articulacoes" redondas nas pontas
void limb(ObjWriter &w, const std::string &m, glm::vec3 a, glm::vec3 b,
          float ra, float rb) {
  frustum(w, m, a, b, ra, rb, 12);
  w.sphere(m, a, glm::vec3(ra), 10, 6);
  w.sphere(m, b, glm::vec3(rb), 10, 6);
}

void generatePilotOBJ() {
  ObjWriter w;
  w.material("fato", glm::vec3(0.05f, 0.16f, 0.31f), glm::vec3(0.15f),
             12); // azul UBI
  w.material("emblema", glm::vec3(0.95f), glm::vec3(0.2f), 16);
  w.material("luvas", glm::vec3(0.06f), glm::vec3(0.3f), 24);
  w.material("botas", glm::vec3(0.08f, 0.07f, 0.06f), glm::vec3(0.3f), 24);
  w.material("pele", glm::vec3(0.86f, 0.66f, 0.52f), glm::vec3(0.15f), 16);
  w.material("capacete", glm::vec3(0.95f, 0.96f, 0.97f), glm::vec3(0.7f), 80);
  w.material("risca", glm::vec3(0.05f, 0.16f, 0.31f), glm::vec3(0.7f), 80);
  w.material("viseira", glm::vec3(0.04f, 0.05f, 0.07f), glm::vec3(0.9f), 120);
  const float x = 0.27f;

  w.object = "corpo";
  // sentado: costas no encosto do banco (z ~ -0.05), assento a y ~ 1.0
  w.sphere("fato", {x, 1.28f, 0.08f}, {0.19f, 0.25f, 0.12f}); // tronco
  w.box("emblema", {x + 0.09f, 1.34f, 0.193f},
        {0.07f, 0.06f, 0.012f}); // emblema no peito
  limb(w, "fato", {x, 1.49f, 0.10f}, PILOT_NECK + glm::vec3(0, 0.02f, 0), 0.05f,
       0.045f); // pescoco
  for (float s : {-1.0f, 1.0f}) {
    glm::vec3 hip(x + 0.10f * s, 1.07f, 0.12f),
        knee(x + 0.11f * s, 1.17f, 0.55f), ankle(x + 0.11f * s, 0.95f, 0.72f);
    limb(w, "fato", hip, knee, 0.075f, 0.06f);    // coxa
    limb(w, "fato", knee, ankle, 0.055f, 0.045f); // perna
    w.box("botas", {ankle.x, 0.88f, ankle.z + 0.06f},
          {0.10f, 0.09f, 0.24f}); // bota
  }

  w.object = "cabeca";
  glm::vec3 hc(x, 1.76f, 0.16f);
  w.sphere("pele", hc, glm::vec3(0.105f));
  // capacete: esfera maior; cada bocado escolhe o material pela direcao d
  w.sphere("capacete", hc + glm::vec3(0, 0.02f, -0.01f), glm::vec3(0.128f), 24,
           16, [](glm::vec3 d) -> std::string {
             if (d.z > 0.45f && d.y < -0.05f)
               return ""; // abertura para a cara
             if (d.z > 0.45f && d.y < 0.35f)
               return "viseira"; // viseira escura a frente dos olhos
             if (std::fabs(d.x) < 0.13f && d.y > 0.0f)
               return "risca"; // risca azul a meio
             return "capacete";
           });

  w.object = "bracos";
  for (float s : {-1.0f, 1.0f}) {
    glm::vec3 sh(x + 0.20f * s, 1.46f, 0.08f), el(x + 0.24f * s, 1.30f, 0.40f);
    glm::vec3 hand(x + 0.14f * s, 1.50f, 0.785f); // pegas do manche
    limb(w, "fato", sh, el, 0.05f, 0.045f);
    limb(w, "fato", el, hand - glm::vec3(0, 0, 0.05f), 0.045f, 0.035f);
    w.sphere("luvas", hand, glm::vec3(0.045f, 0.04f, 0.055f), 12, 8);
  }
  if (w.save("models/pilot.obj", "pilot.mtl", "models/pilot.mtl",
             "Piloto gerado por main.cpp"))
    std::cout << "Piloto gerado em models/pilot.obj" << std::endl;
}

// Disco da helice a rodar (como nas fotografias): um disco semitransparente
void generatePropDiscOBJ() {
  ObjWriter w;
  w.material("disco", glm::vec3(0.25f), glm::vec3(0.1f), 8, 0.22f);
  const glm::vec3 c(0.0f, 1.534f, 2.66f);
  const float r = 1.10f;
  for (int i = 0; i < 40; i++) {
    float a0 = 2 * glm::pi<float>() * i / 40,
          a1 = 2 * glm::pi<float>() * (i + 1) / 40;
    w.tri("disco", c, c + r * glm::vec3(std::cos(a0), std::sin(a0), 0),
          c + r * glm::vec3(std::cos(a1), std::sin(a1), 0));
  }
  w.save("models/prop_disc.obj", "prop_disc.mtl", "models/prop_disc.mtl",
         "Disco da helice gerado por main.cpp");
}

// =============================================================================
//  AEROPORTO -> models/airport.obj (gerado pelo programa)
// =============================================================================
void generateAirportOBJ() {
  ObjWriter w;
  addLightFixtures(w);
  addBuildings(w);
  if (w.save("models/airport.obj", "airport.mtl", "models/airport.mtl",
             "Aeroporto gerado por main.cpp: bases das luzes, torre, hangar, "
             "terminal, placa, manga de vento"))
    std::cout << "Aeroporto gerado em models/airport.obj" << std::endl;
  else
    std::cerr << "Aviso: nao consegui gravar models/airport.obj" << std::endl;
}

// =============================================================================
//  TERRENO MONTANHOSO (extra) -> models/terrain.obj
// =============================================================================
//  A altura do terreno e uma FUNCAO h(x, z) (um "heightmap" calculado):
//    1. ruido fractal (fbm): soma de 6 camadas de "value noise", cada uma com o
//       dobro da frequencia e metade da amplitude da anterior -> colinas
//       grandes com detalhe pequeno por cima;
//    2. a mascara poe h = -0.2 (o nivel da relva) no aerodromo e nas cidades, e
//       deixa as montanhas crescer a partir de ~200 m (smoothstep ate 1100 m);
//    3. uma serra (gaussiana) atravessa a aproximacao, e por baixo da rampa de
//       3 graus o terreno nunca sobe acima da trajetoria menos 30 m (um
//       "vale").
//  Com h(x, z) faz-se uma grelha de N x N vertices; a normal de cada vertice
//  vem das diferencas centrais de h (inclinacao em x e em z), por isso a luz e
//  suave. Cada triangulo recebe um material conforme a altura e a inclinacao:
//    relva (baixo) | relva escura | rocha (ingreme ou alto) | neve (picos).
//  A mesma funcao h serve para a colisao do avião com as montanhas.
// =============================================================================
const float TERRAIN_SIZE = 16000.0f; // lado do terreno (m): de -8 km a +8 km
const int TERRAIN_N = 260;           // celulas por lado (~62 m cada)
const float GROUND_Y = -0.2f;        // nivel da relva (a pista esta em y = 0)

struct Town {
  glm::vec2 c;
  float r;
  unsigned seed;
};
// aglomerados espalhados a volta do aerodromo (x, z), raio, semente aleatoria
const std::vector<Town> TOWNS = {
    {{-700, 600}, 260, 11},  {{650, 250}, 220, 22},   {{850, 1750}, 280, 33},
    {{-900, 2050}, 230, 44}, {{-700, -350}, 210, 55}, {{750, -2050}, 200, 66}};

// serra que atravessa o corredor de aproximacao (o avião tem de passar por cima
// dela)
const float RIDGE_Z = -1250.0f, RIDGE_WIDTH = 230.0f;

// valor pseudo-aleatorio em [0,1) para cada ponto inteiro (x, z) da grelha de
// ruido
float hash2(int x, int z) {
  unsigned h = (unsigned)x * 374761393u + (unsigned)z * 668265263u + 53702u;
  h = (h ^ (h >> 13)) * 1274126177u;
  return ((h ^ (h >> 16)) & 0xffffff) / 16777216.0f;
}

// value noise: interpola suavemente os valores aleatorios dos 4 cantos da
// celula
float valueNoise(float x, float z) {
  int xi = (int)std::floor(x), zi = (int)std::floor(z);
  float fx = x - xi, fz = z - zi;
  fx = fx * fx * (3 - 2 * fx); // smoothstep: sem "bicos" nas juntas
  fz = fz * fz * (3 - 2 * fz);
  float a = hash2(xi, zi), b = hash2(xi + 1, zi), c = hash2(xi, zi + 1),
        d = hash2(xi + 1, zi + 1);
  return glm::mix(glm::mix(a, b, fx), glm::mix(c, d, fx), fz);
}

float fbm(float x, float z) {
  float sum = 0, amp = 1, freq = 1, norm = 0;
  for (int i = 0; i < 6; i++) {
    sum += amp * valueNoise(x * freq, z * freq);
    norm += amp;
    amp *= 0.5f;
    freq *= 2.0f;
  }
  return sum / norm; // 0..1
}

// 0 nas zonas planas (aerodromo, cidades) -> 1 longe delas
float flatMask(float x, float z) {
  // retangulo plano: pista, luzes de aproximacao e edificios
  float dx = std::max({-160.0f - x, 0.0f, x - 120.0f});
  float dz = std::max({-350.0f - z, 0.0f, z - (RWY_LEN + 250.0f)});
  float m = glm::smoothstep(200.0f, 1100.0f, std::sqrt(dx * dx + dz * dz));
  for (const Town &t : TOWNS) {
    float d = glm::length(glm::vec2(x, z) - t.c);
    m *= glm::smoothstep(t.r + 80.0f, t.r + 450.0f, d);
  }
  return m;
}

// minimo "suave": igual a min(a, b) longe do ponto onde se cruzam, mas sem
// esquina (k = largura da zona de transicao). Evita degraus no terreno.
float smoothMin(float a, float b, float k) {
  float t = std::max(k - std::fabs(a - b), 0.0f) / k;
  return std::min(a, b) - t * t * k * 0.25f;
}

// altura da rampa de 3 graus (a trajetoria ideal) a distancia z da pista
float glidePathHeight(float z) {
  return (AIM_Z - z) * std::tan(glm::radians(GLIDE_DEG));
}

float terrainHeight(float x, float z) {
  // 1) montanhas: ruido fractal
  float n = fbm(x / 1700.0f + 13.1f, z / 1700.0f + 7.7f);
  float mountains =
      std::pow(glm::clamp((n - 0.28f) / 0.72f, 0.0f, 1.0f), 1.4f) * 1300.0f;
  // 2) serra a atravessar a aproximacao: uma "lomba" gaussiana em z, mais alta
  // para os lados
  float ridgeH = std::min(260.0f + 0.35f * std::fabs(x), 750.0f) *
                 (0.75f + 0.5f * fbm(x / 400.0f, 3.3f));
  float ridge = ridgeH * std::exp(-std::pow((z - RIDGE_Z) / RIDGE_WIDTH, 2.0f));
  float h = flatMask(x, z) * std::max(mountains, ridge);
  // 3) por baixo da rampa de aproximacao o terreno fica pelo menos 30 m abaixo
  // da trajetoria
  //    (30 m de folga; o limite sobe para os lados: e um "vale" por onde se
  //    passa a serra)
  if (z < 0)
    h = smoothMin(
        h,
        std::max(0.0f, glidePathHeight(z) - 30.0f +
                           0.8f * std::max(0.0f, std::fabs(x) - 200.0f)),
        60.0f);
  return GROUND_Y + std::max(h, 0.0f);
}

void generateTerrainOBJ() {
  const int N = TERRAIN_N;
  const float cell = TERRAIN_SIZE / N, x0 = -TERRAIN_SIZE / 2,
              z0 = -TERRAIN_SIZE / 2 + 1000;
  std::vector<float> H((N + 1) * (N + 1));
  auto idx = [N](int i, int j) { return j * (N + 1) + i; };
  for (int j = 0; j <= N; j++)
    for (int i = 0; i <= N; i++)
      H[idx(i, j)] = terrainHeight(x0 + i * cell, z0 + j * cell);

  std::ofstream obj("models/terrain.obj"), mtl("models/terrain.mtl");
  if (!obj || !mtl) {
    std::cerr << "Aviso: nao consegui gravar models/terrain.obj" << std::endl;
    return;
  }
  mtl << "# Materiais do terreno (gerados por main.cpp)\n"
      << "newmtl relva\nKd 0.30 0.48 0.20\nKs 0.02 0.02 0.02\nNs 4\n\n"
      << "newmtl relva_escura\nKd 0.20 0.36 0.15\nKs 0.02 0.02 0.02\nNs 4\n\n"
      << "newmtl rocha\nKd 0.45 0.42 0.38\nKs 0.05 0.05 0.05\nNs 8\n\n"
      << "newmtl neve\nKd 0.95 0.96 0.98\nKs 0.30 0.30 0.30\nNs 24\n";
  obj << "# Terreno gerado por main.cpp: grelha " << N << " x " << N << ", "
      << cell << " m por celula\n"
      << "mtllib terrain.mtl\n";
  for (int j = 0; j <= N; j++)
    for (int i = 0; i <= N; i++)
      obj << "v " << x0 + i * cell << " " << H[idx(i, j)] << " "
          << z0 + j * cell << "\n";
  // normal = (-dh/dx, 1, -dh/dz) normalizada (diferencas centrais)
  for (int j = 0; j <= N; j++)
    for (int i = 0; i <= N; i++) {
      float hl = H[idx(std::max(i - 1, 0), j)],
            hr = H[idx(std::min(i + 1, N), j)];
      float hd = H[idx(i, std::max(j - 1, 0))],
            hu = H[idx(i, std::min(j + 1, N))];
      glm::vec3 n = glm::normalize(
          glm::vec3(-(hr - hl) / (2 * cell), 1.0f, -(hu - hd) / (2 * cell)));
      obj << "vn " << n.x << " " << n.y << " " << n.z << "\n";
    }
  // 2 triangulos por celula, agrupados pelo material
  std::map<std::string, std::vector<int>> faces;
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      int a = idx(i, j) + 1, b = idx(i, j + 1) + 1, c = idx(i + 1, j + 1) + 1,
          d = idx(i + 1, j) + 1;
      for (int t = 0; t < 2; t++) {
        int p = a, q = t ? c : b, r = t ? d : c;
        glm::vec3 P(0, H[p - 1], 0), Q(0, H[q - 1], 0), R(0, H[r - 1], 0);
        float hAvg = (P.y + Q.y + R.y) / 3;
        // inclinacao: diferenca maxima de altura no triangulo / tamanho da
        // celula
        float slope =
            (std::max({P.y, Q.y, R.y}) - std::min({P.y, Q.y, R.y})) / cell;
        std::string m = (hAvg > 520 && slope < 0.9f)    ? "neve"
                        : (slope > 0.55f || hAvg > 330) ? "rocha"
                        : (hAvg > 60)                   ? "relva_escura"
                                                        : "relva";
        faces[m].insert(faces[m].end(), {p, q, r});
      }
    }
  for (auto &[m, f] : faces) {
    obj << "usemtl " << m << "\n";
    for (size_t k = 0; k < f.size(); k += 3)
      obj << "f " << f[k] << "//" << f[k] << " " << f[k + 1] << "//" << f[k + 1]
          << " " << f[k + 2] << "//" << f[k + 2] << "\n";
  }
  std::cout << "Terreno gerado em models/terrain.obj" << std::endl;
}

// =============================================================================
//  CIDADES (extra) -> models/city.obj
// =============================================================================
//  Cada aglomerado: ruas em cruz, predios (varios andares) no centro e casas
//  com telhado de duas aguas a volta. As posicoes vem de um gerador aleatorio
//  com semente fixa (std::mt19937), por isso a cidade e sempre igual. Janelas:
//  pequenos quadrados nas fachadas. ~35% usam o material "janela_acesa", que de
//  noite ganha luz propria (emissive) no fragment shader; as outras ficam
//  escuras. Tambem se guardam candeeiros (pontos de luz) para o brilho noturno.
// =============================================================================

// janelas numa fachada: centro da parede 'c', direcao ao longo da parede
// 'along', normal para fora 'out', largura 'w', andares 'floors'
void addWindows(ObjWriter &w, std::mt19937 &rng, glm::vec3 c, glm::vec3 along,
                glm::vec3 out, float width, int floors, float floorH) {
  std::uniform_real_distribution<float> U(0, 1);
  int cols = std::max(1, (int)(width / 3.2f));
  for (int f = 0; f < floors; f++)
    for (int k = 0; k < cols; k++) {
      float s = (k + 0.5f) / cols - 0.5f;
      glm::vec3 m = c + along * (s * width) +
                    glm::vec3(0, f * floorH + floorH * 0.55f, 0) + out * 0.06f;
      glm::vec3 hx = along * 0.65f, hy(0, 0.75f, 0);
      const char *mat = (U(rng) < 0.35f) ? "janela_acesa" : "janela_apagada";
      quadOut(w, mat, m - hx - hy, m + hx - hy, m + hx + hy, m - hx + hy, out);
    }
}

// caixa com janelas nas 4 faces (casa ou predio), base em (x, y, z)
void building(ObjWriter &w, std::mt19937 &rng, glm::vec3 base, float sx,
              float sz, int floors, const std::string &wall, bool gable) {
  const float FH = 3.0f;
  float h = floors * FH;
  w.box(wall, base, glm::vec3(sx, h, sz));
  addWindows(w, rng, base + glm::vec3(0, 0, sz / 2), {1, 0, 0}, {0, 0, 1},
             sx * 0.8f, floors, FH);
  addWindows(w, rng, base + glm::vec3(0, 0, -sz / 2), {-1, 0, 0}, {0, 0, -1},
             sx * 0.8f, floors, FH);
  addWindows(w, rng, base + glm::vec3(sx / 2, 0, 0), {0, 0, -1}, {1, 0, 0},
             sz * 0.8f, floors, FH);
  addWindows(w, rng, base + glm::vec3(-sx / 2, 0, 0), {0, 0, 1}, {-1, 0, 0},
             sz * 0.8f, floors, FH);
  glm::vec3 t = base + glm::vec3(0, h, 0);
  if (gable) {
    // telhado de duas aguas (cumeeira ao longo de X)
    float rh = 2.2f, ex = sx / 2 + 0.3f, ez = sz / 2 + 0.4f;
    glm::vec3 r0 = t + glm::vec3(-ex, rh, 0), r1 = t + glm::vec3(ex, rh, 0);
    quadOut(w, "telha", t + glm::vec3(-ex, 0, ez), t + glm::vec3(ex, 0, ez), r1,
            r0, {0, 1, 1});
    quadOut(w, "telha", t + glm::vec3(ex, 0, -ez), t + glm::vec3(-ex, 0, -ez),
            r0, r1, {0, 1, -1});
    w.tri("telha", t + glm::vec3(sx / 2, 0, sz / 2),
          t + glm::vec3(sx / 2, 0, -sz / 2), t + glm::vec3(sx / 2, rh, 0));
    w.tri("telha", t + glm::vec3(-sx / 2, 0, -sz / 2),
          t + glm::vec3(-sx / 2, 0, sz / 2), t + glm::vec3(-sx / 2, rh, 0));
  } else {
    w.box("telhado_plano", t, glm::vec3(sx + 0.4f, 0.4f, sz + 0.4f));
  }
}

void addTown(ObjWriter &w, const Town &T) {
  std::mt19937 rng(T.seed);
  std::uniform_real_distribution<float> U(0, 1);
  const char *walls[3] = {"parede1", "parede2", "parede3"};
  float y = 0.0f;
  // ruas em cruz (ligeiramente acima da relva)
  float R = T.r;
  w.quad("estrada", {T.c.x - R, y - 0.1f, T.c.y - 4},
         {T.c.x - R, y - 0.1f, T.c.y + 4}, {T.c.x + R, y - 0.1f, T.c.y + 4},
         {T.c.x + R, y - 0.1f, T.c.y - 4});
  w.quad("estrada", {T.c.x - 4, y - 0.08f, T.c.y - R},
         {T.c.x - 4, y - 0.08f, T.c.y + R}, {T.c.x + 4, y - 0.08f, T.c.y + R},
         {T.c.x + 4, y - 0.08f, T.c.y - R});
  // lotes numa grelha de 24 m; quanto mais longe do centro, menos e mais baixos
  // os edificios
  const float G = 24.0f;
  for (float gx = -R; gx <= R; gx += G)
    for (float gz = -R; gz <= R; gz += G) {
      float d = std::sqrt(gx * gx + gz * gz) / R; // 0 no centro, 1 na borda
      if (d > 1.0f || std::fabs(gx) < 10 || std::fabs(gz) < 10)
        continue; // nao em cima das ruas
      if (U(rng) > 1.05f - 0.6f * d)
        continue; // mais disperso na periferia
      glm::vec3 p(T.c.x + gx + (U(rng) - 0.5f) * 5, y,
                  T.c.y + gz + (U(rng) - 0.5f) * 5);
      const std::string wall = walls[(int)(U(rng) * 3) % 3];
      if (U(rng) < 0.5f)
        cityLamps.push_back(p + glm::vec3(6, 4, 6)); // luz de quintal / rua
      if (d < 0.35f && U(rng) < 0.7f) {
        int floors = 3 + (int)(U(rng) * 6); // predio de 3 a 8 andares
        building(w, rng, p, 14 + U(rng) * 4, 12 + U(rng) * 4, floors, wall,
                 false);
      } else {
        building(w, rng, p, 8 + U(rng) * 4, 7 + U(rng) * 3,
                 (U(rng) < 0.3f) ? 2 : 1, wall, true);
      }
    }
  // candeeiros ao longo das ruas (pontos de luz a 6 m de altura)
  for (float s = -R; s <= R; s += 30) {
    cityLamps.push_back({T.c.x + s, 6, T.c.y + 6});
    cityLamps.push_back({T.c.x + 6, 6, T.c.y + s});
  }
}

void generateCityOBJ() {
  ObjWriter w;
  w.material("parede1", glm::vec3(0.86f, 0.83f, 0.76f), glm::vec3(0.1f), 8);
  w.material("parede2", glm::vec3(0.80f, 0.70f, 0.58f), glm::vec3(0.1f), 8);
  w.material("parede3", glm::vec3(0.92f, 0.92f, 0.90f), glm::vec3(0.1f), 8);
  w.material("telha", glm::vec3(0.62f, 0.26f, 0.15f), glm::vec3(0.1f), 8);
  w.material("telhado_plano", glm::vec3(0.40f, 0.40f, 0.42f), glm::vec3(0.1f),
             8);
  w.material("janela_apagada", glm::vec3(0.12f, 0.14f, 0.18f), glm::vec3(0.6f),
             64);
  w.material("janela_acesa", glm::vec3(0.12f, 0.14f, 0.18f), glm::vec3(0.6f),
             64);
  w.material("estrada", glm::vec3(0.28f, 0.28f, 0.29f), glm::vec3(0.05f), 8);
  cityLamps.clear();
  for (const Town &t : TOWNS)
    addTown(w, t);
  if (w.save("models/city.obj", "city.mtl", "models/city.mtl",
             "Cidades geradas por main.cpp"))
    std::cout << "Cidades geradas em models/city.obj (" << cityLamps.size()
              << " candeeiros)" << std::endl;
}

// =============================================================================
//  MODELO DE VOO (o avião como um ponto com massa + atitude)
// =============================================================================
//  Estado:
//    pos    posicao das rodas principais (m)
//    V      velocidade em relacao ao ar (m/s)
//    gamma  angulo da trajetoria (para onde o avião se desloca; < 0 = a descer)
//    psi    rumo (para onde aponta, no plano horizontal)
//    theta  pitch: para onde aponta o nariz, na vertical
//    phi    roll: inclinacao das asas (> 0 = asa direita em baixo)
//
//  O angulo de ataque e  alfa = theta - gamma  (diferenca entre para onde o
//  nariz aponta e para onde o avião vai). E dele que depende a sustentacao.
//
//  Forcas:   L = 1/2 rho V^2 S CL      (sustentacao, perpendicular a
//  trajetoria)
//            D = 1/2 rho V^2 S CD      (resistencia, contra a trajetoria)
//            T = potencia / V          (tracao da helice, limitada a T_MAX)
//            W = m g                   (peso)
//
//  Equacoes (no ar):
//    dV/dt     = (T cos(alfa) - D) / m - g sin(gamma)
//    dgamma/dt = (L cos(phi) + T sin(alfa) - W cos(gamma)) / (m V)
//    dpsi/dt   = -g tan(phi) / V          (curva provocada pela inclinacao)
//
//  Os comandos (teclado / rato) mudam diretamente theta, phi e psi a uma
//  velocidade angular fixa; o resto (velocidade, subir/descer) vem da fisica.
// =============================================================================
enum Status { FLYING, ON_GROUND, CRASHED };

struct Aircraft {
  glm::vec3 pos;
  float V, gamma, psi, theta, phi;
  float throttle; // 0..1
  int flaps;      // 0, 10, 20 ou 30 graus
  Status status;
  std::string result; // avaliacao da aterragem
} ac;

// comandos atuais (-1..1), juntam teclado e rato
float cmdElev = 0, cmdAil = 0, cmdRud = 0;
bool cmdBrake = false;

float flapFrac() { return ac.flaps / 30.0f; }

// coeficiente de sustentacao: cresce em linha reta com o angulo de ataque,
// limitado a CL max (sem perda de sustentacao: nao cai depois do maximo)
float liftCoef(float alpha) {
  float base = CL0 + FLAP_CL * flapFrac();
  float clMax = CL_MAX + FLAP_CL * flapFrac();
  return glm::clamp(base + CL_A * alpha, -1.0f, clMax);
}

float dragCoef(float cl) {
  return CD0 + FLAP_CD * flapFrac() + K_IND * cl * cl;
}

float thrustAvailable(float V) {
  return std::min(T_MAX, POWER / std::max(V, 1.0f));
}

bool onRunway(const glm::vec3 &p) {
  return std::fabs(p.x) <= RWY_WIDTH / 2 && p.z >= -10 && p.z <= RWY_LEN + 10;
}

// Comeca na rampa de 3 graus, a 60 kt, flaps 30, ja "trimado"
// (pitch e potencia calculados para o avião seguir a rampa sozinho).
void resetAircraft() {
  float g = -glm::radians(GLIDE_DEG);
  ac.pos = glm::vec3(0.0f, (AIM_Z - START_Z) * std::tan(-g), START_Z);
  ac.V = V_APP;
  ac.gamma = g;
  ac.psi = 0;
  ac.phi = 0;
  ac.flaps = 30;
  ac.status = FLYING;
  ac.result = "";

  float q = 0.5f * RHO * ac.V * ac.V, W = MASS * G;
  float alpha = 0, T = 0;
  for (int i = 0; i < 5; i++) { // pequena iteracao de ponto fixo
    float clNeeded =
        (W * std::cos(g) - T * std::sin(alpha)) / (q * WING_S); // dgamma/dt = 0
    alpha = (clNeeded - CL0 - FLAP_CL * flapFrac()) / CL_A;
    float D = q * WING_S * dragCoef(clNeeded);
    T = (D + W * std::sin(g)) / std::cos(alpha); // dV/dt = 0
  }
  ac.theta = alpha + g;
  ac.throttle = glm::clamp(T / thrustAvailable(ac.V), 0.0f, 1.0f);
  printf("Inicio: alt %.1f m, %.0f kt, pitch %.1f graus, potencia %.0f%%, "
         "flaps %d\n",
         ac.pos.y, ac.V / KT, glm::degrees(ac.theta), ac.throttle * 100,
         ac.flaps);
}

// Avaliacao no momento em que as rodas tocam no chao
void touchdown() {
  float vsFpm = ac.V * std::sin(ac.gamma) * FPM; // negativo
  float bank = std::fabs(glm::degrees(ac.phi));
  float pitch = glm::degrees(ac.theta);
  char buf[200];
  if (vsFpm < -1000 || bank > 15 || pitch < -5) {
    ac.status = CRASHED;
    snprintf(buf, sizeof(buf),
             "ACIDENTE (%.0f ft/min, inclinacao %.0f, pitch %.0f)", vsFpm, bank,
             pitch);
  } else {
    ac.status = ON_GROUND;
    std::string q = (vsFpm < -600)   ? "Aterragem dura"
                    : (vsFpm < -300) ? "Aterragem aceitavel"
                                     : "Boa aterragem!";
    if (pitch < 0)
      q += " (roda do nariz primeiro!)";
    if (!onRunway(ac.pos))
      q += " FORA DA PISTA";
    snprintf(buf, sizeof(buf), "%s %.0f ft/min, %.0f kt, a %.0f m da soleira",
             q.c_str(), vsFpm, ac.V / KT, ac.pos.z);
  }
  ac.result = buf;
  std::cout << "TOQUE: " << ac.result << std::endl;
}

void updateAircraft(float dt) {
  if (ac.status == CRASHED)
    return;
  const float W = MASS * G;
  bool ground = (ac.status == ON_GROUND);

  // ---- comandos -> atitude ----
  ac.theta +=
      cmdElev * glm::radians(12.0f) * dt; // 12 graus/s com o manche todo
  if (ground) {
    ac.phi = 0; // asas niveladas no chao
    if (cmdElev <= 0)
      ac.theta -= glm::radians(4.0f) * dt; // o nariz desce sozinho
    ac.theta = glm::clamp(ac.theta, 0.0f,
                          glm::radians(15.0f)); // roda do nariz / cauda no chao
    ac.psi += cmdRud * glm::radians(15.0f) * std::min(1.0f, ac.V / 5.0f) *
              dt; // roda do nariz
  } else {
    ac.phi = glm::clamp(ac.phi + cmdAil * glm::radians(35.0f) * dt,
                        -glm::radians(60.0f), glm::radians(60.0f));
    ac.theta = glm::clamp(ac.theta, -glm::radians(30.0f), glm::radians(30.0f));
    ac.psi += (-G * std::tan(ac.phi) / std::max(ac.V, 5.0f) +
               cmdRud * glm::radians(8.0f)) *
              dt;
  }

  // ---- forcas aerodinamicas ----
  float alpha = ac.theta - ac.gamma;
  float q = 0.5f * RHO * ac.V * ac.V;
  float cl = liftCoef(alpha);
  float L = q * WING_S * cl;
  float D = q * WING_S * dragCoef(cl);
  float T = ac.throttle * thrustAvailable(ac.V);

  if (ground) {
    // no chao: o peso que a asa nao segura vai para as rodas (N) e da atrito
    float N = std::max(0.0f, W - L);
    float mu = cmdBrake ? MU_BRAKE : (onRunway(ac.pos) ? MU_ROLL : MU_GRASS);
    ac.V = std::max(0.0f, ac.V + (T - D - mu * N) / MASS * dt);
    ac.gamma = 0;
    if (L > W)
      ac.status = FLYING; // volta a descolar
  } else {
    float dV = (T * std::cos(alpha) - D) / MASS - G * std::sin(ac.gamma);
    float dGamma =
        (L * std::cos(ac.phi) + T * std::sin(alpha) - W * std::cos(ac.gamma)) /
        (MASS * std::max(ac.V, 5.0f));
    ac.V = std::max(1.0f, ac.V + dV * dt);
    ac.gamma = glm::clamp(ac.gamma + dGamma * dt, -glm::radians(80.0f),
                          glm::radians(80.0f));
  }

  // ---- posicao ----
  glm::vec3 vel = ac.V * glm::vec3(std::cos(ac.gamma) * std::sin(ac.psi),
                                   std::sin(ac.gamma),
                                   std::cos(ac.gamma) * std::cos(ac.psi));
  ac.pos += vel * dt;

  // embater numa montanha (o terreno so sobe longe da pista)
  float hT = terrainHeight(ac.pos.x, ac.pos.z);
  if (ac.status != CRASHED && hT > 0.5f && ac.pos.y <= hT) {
    ac.pos.y = hT;
    ac.status = CRASHED;
    ac.result = "ACIDENTE: embateu no terreno";
    std::cout << ac.result << std::endl;
    return;
  }
  if (ac.status == FLYING && ac.pos.y <= 0.0f) {
    ac.pos.y = 0;
    touchdown();
    ac.gamma = 0;
  }
  if (ac.status == ON_GROUND)
    ac.pos.y = 0;
}

// Matriz model do avião: T(pos) * Ry(rumo) * Rx(-pitch) * Rz(roll)
// O nariz aponta para +Z: levantar o nariz e rodar em X com angulo negativo.
// A origem do modelo esta nas rodas principais, por isso o avião roda a volta
// delas.
glm::mat4 aircraftMatrix() {
  glm::mat4 M = glm::translate(glm::mat4(1.0f), ac.pos);
  M = glm::rotate(M, ac.psi, glm::vec3(0, 1, 0));
  M = glm::rotate(M, -ac.theta, glm::vec3(1, 0, 0));
  M = glm::rotate(M, ac.phi, glm::vec3(0, 0, 1));
  return M;
}

// =============================================================================
//  PECAS QUE SE MEXEM (extra): helice, flaps, ailerons, profundor, leme, manche
// =============================================================================
//  Todas usam a mesma ideia: rodar a volta de um eixo que nao passa pela
//  origem.
//      local = T(p) * R(angulo, eixo) * T(-p)
//  T(-p) leva o ponto p do eixo (centro da helice, dobradica do flap...) para a
//  origem, R roda, T(p) volta a pôr a peca no sitio. Na drawModel faz-se
//  M_aviao * local: a peca mexe-se dentro do avião e depois vai com ele.
//  Pontos e eixos medidos no cessna182.obj (os flaps do modelo estao a ~25
//  graus).
// =============================================================================
glm::mat4 aroundAxis(glm::vec3 p, glm::vec3 axis, float deg) {
  return glm::translate(glm::mat4(1.0f), p) *
         glm::rotate(glm::mat4(1.0f), glm::radians(deg), glm::normalize(axis)) *
         glm::translate(glm::mat4(1.0f), -p);
}

const float FLAP_MODEL_DEG = 25.0f;             // angulo dos flaps no ficheiro
float propAngle = 0, propRPM = 0, flapVis = 30; // estado da animacao
float elevVis = 0, ailVis = 0,
      rudVis = 0; // comandos suavizados (a peca nao salta)

void updateAnimation(float dt) {
  propRPM = (ac.status == CRASHED)
                ? 0.0f
                : 700.0f + 1900.0f * ac.throttle; // ralenti 700 rpm
  propAngle = std::fmod(propAngle + propRPM / 60.0f * 360.0f * dt, 360.0f);
  auto follow = [dt](float &v, float target,
                     float rate) { // aproxima-se a 'rate' por segundo
    v += glm::clamp(target - v, -rate * dt, rate * dt);
  };
  follow(flapVis, (float)ac.flaps,
         10.0f); // os flaps demoram ~3 s a descer 30 graus
  follow(elevVis, cmdElev, 4.0f);
  follow(ailVis, cmdAil, 4.0f);
  follow(rudVis, cmdRud, 4.0f);
}

PartMatrices planeParts() {
  PartMatrices m;
  // helice: roda a volta do eixo do motor (z), que passa no centro do cubo
  // acima de ~1000 rpm as pas nao se veem (fica so o disco)
  m["prop_1_slow"] = (propRPM < 1000)
                         ? aroundAxis({0, 1.534f, 0}, {0, 0, 1}, propAngle)
                         : glm::mat4(0.0f);
  // flaps: dobradica ao longo de x; angulo positivo levanta o bordo de fuga
  glm::mat4 flap =
      aroundAxis({0, 1.849f, -0.465f}, {1, 0, 0}, FLAP_MODEL_DEG - flapVis);
  m["flap_esq"] = m["flap_dir"] = flap;
  // ailerons: a inclinar para a direita (ailVis > 0) o da esquerda desce e o da
  // direita sobe
  glm::mat4 ailL = aroundAxis({2.66f, 1.875f, -0.626f}, {2.70f, 0.151f, 0.321f},
                              -15.0f * ailVis);
  glm::mat4 ailR = aroundAxis({-2.66f, 1.879f, -0.628f},
                              {2.70f, -0.144f, -0.323f}, 15.0f * ailVis);
  m["aileron_esq"] = m["aileron_l"] = ailL;
  m["aileron_dir"] = m["aileron_r"] = ailR;
  // profundor: puxar o manche (elevVis > 0) levanta o bordo de fuga
  m["elevator_r"] = aroundAxis({0, 1.03f, -4.577f}, {1, 0, 0}, 20.0f * elevVis);
  // leme de direcao: eixo inclinado (o bordo de ataque do leme); Q (rudVis > 0)
  // vira para a esquerda
  m["rudder"] =
      aroundAxis({0, 1.10f, -4.62f}, {0, 1.4f, -0.8f}, -20.0f * rudVis);
  // manche: anda para tras/frente com o profundor e roda com os ailerons
  glm::mat4 yoke =
      glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -0.08f * elevVis)) *
      aroundAxis({0.267f, 1.514f, 0}, {0, 0, 1}, 40.0f * ailVis);
  m["steeringwheel"] = m["steeringwheel.001"] = m["steeringwheel.002"] = yoke;
  return m;
}

// piloto: a cabeca acompanha as curvas, as maos seguem o manche
float headYaw = 0, headRoll = 0;
PartMatrices pilotParts(float dt, bool hideHead) {
  float rollDeg = glm::degrees(ac.phi);
  headYaw += glm::clamp(-0.6f * rollDeg - headYaw, -60 * dt,
                        60 * dt); // olha para dentro da curva
  headRoll += glm::clamp(-0.3f * rollDeg - headRoll, -60 * dt,
                         60 * dt); // tenta manter o horizonte
  PartMatrices m;
  m["cabeca"] = hideHead
                    ? glm::mat4(0.0f)
                    : glm::translate(glm::mat4(1.0f), PILOT_NECK) *
                          glm::rotate(glm::mat4(1.0f), glm::radians(headYaw),
                                      glm::vec3(0, 1, 0)) *
                          glm::rotate(glm::mat4(1.0f), glm::radians(headRoll),
                                      glm::vec3(0, 0, 1)) *
                          glm::translate(glm::mat4(1.0f), -PILOT_NECK);
  m["bracos"] =
      glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -0.08f * elevVis));
  return m;
}

// =============================================================================
//  LUZES DE NAVEGACAO DO AVIAO (extra)
// =============================================================================
//  Os mesmos "point sprites" das luzes da pista, mas presos ao avião: a posicao
//  no mundo e M_aviao * posicao no avião. Cada luz so se ve no seu setor:
//    vermelha (asa esquerda): da frente ate 110 graus para a esquerda
//    verde (asa direita): da frente ate 110 graus para a direita
//    branca (cauda): para tras
//  O angulo vem de atan2(x, z) da direcao avião -> camara, em coordenadas do
//  avião. Farol vermelho no leme (pulsa) e strobes brancos nas pontas das asas
//  (flash duplo).
// =============================================================================
void addNavLights(std::vector<float> &data, const glm::mat4 &Mplane,
                  const glm::vec3 &eye, double t) {
  glm::mat3 Rinv = glm::transpose(glm::mat3(Mplane)); // rotacao inversa
  glm::vec3 d = Rinv * (eye - glm::vec3(Mplane[3]));  // camara vista do avião
  float ang = glm::degrees(std::atan2(d.x, d.z)); // 0 = a frente, + = esquerda
  auto add = [&](glm::vec3 p, glm::vec3 c, float size, int sector) {
    glm::vec3 w = glm::vec3(Mplane * glm::vec4(p, 1.0f));
    if (sector == 1 && (ang < -20 || ang > 110))
      return; // vermelha (com 20 graus de sobreposicao a frente)
    if (sector == 2 && (ang > 20 || ang < -110))
      return; // verde
    if (sector == 3 && std::fabs(ang) < 110)
      return; // branca de cauda
    data.insert(data.end(), {w.x, w.y, w.z, c.r, c.g, c.b, size});
  };
  // (um pouco para fora das lampadas do modelo, senao o proprio avião tapava o
  // brilho)
  add({5.80f, 2.13f, 0.42f}, L_RED, 0.30f, 1);
  add({-5.80f, 2.13f, 0.42f}, L_GREEN, 0.30f, 2);
  add({0.0f, 1.06f, -5.42f}, L_WHITE, 0.25f, 3);
  float beacon = std::pow(
      std::max(0.0f, (float)std::sin(t * 2 * glm::pi<float>() * 0.9)), 6.0f);
  if (beacon > 0.05f)
    add({0.0f, 2.80f, -5.36f}, L_RED * beacon * 1.5f, 0.45f, 0);
  float ph = (float)std::fmod(t, 1.3);
  if (ph < 0.05f || (ph > 0.16f && ph < 0.21f)) // flash duplo
    for (float s : {-1.0f, 1.0f})
      add({5.85f * s, 2.08f, 0.16f}, glm::vec3(1.4f), 0.9f, 0);
  // farois de aterragem (asa esquerda), acesos na aproximacao
  if (ac.status != CRASHED && ac.pos.y < 400)
    for (glm::vec3 p :
         {glm::vec3(4.07f, 2.12f, 0.70f), glm::vec3(3.96f, 2.115f, 0.70f)})
      add(p, L_WHITE, 0.35f, 0);
}

// =============================================================================
//  INTERACAO: teclado e rato
// =============================================================================
enum CameraMode { CAM_CHASE, CAM_TOWER, CAM_COCKPIT, NUM_CAMS };
const char *CAM_NAME[] = {"Perseguicao", "Torre", "Cockpit"};
// cockpit: para onde o piloto esta a olhar (arrastar com o botao esquerdo)
float lookYaw = 0.0f, lookPitch = 0.0f;
int cameraMode = CAM_CHASE;

// camara de perseguicao: atras do avião (o angulo e relativo ao rumo)
float orbitYaw = 0.0f, orbitPitch = 10.0f, orbitDist = 22.0f;
// camara da torre: fixa ao lado da pista, com zoom automatico
glm::vec3 towerPos(-70.6f, 16.0f,
                   240.0f); // a janela da torre de controlo (TOWER_POS)
float towerZoom = 1.0f;

bool nightMode = false; // N: dia / noite
bool landCamOn = false; // camara de aterragem visivel neste frame?
int landX = 0, landY = 0, landW = 0,
    landH = 0;        // o seu retangulo (pixeis, y a partir de cima)
bool menuOpen = true; // o programa comeca no menu (jogo em pausa)
bool started = false; // ja se carregou em ENTER pelo menos uma vez?
bool leftDown = false, rightDown = false;
double lastX = 0, lastY = 0;
double yokeX = 0,
       yokeY = 0; // onde o botao direito foi carregado (centro do "manche")
float mouseElev = 0, mouseAil = 0;

void printControls() {
  std::cout
      << "\n===================== CONTROLOS =====================\n"
         " W / S  ou  Setas cima/baixo   nariz para baixo / para cima\n"
         " A / D  ou  Setas esq./dir.    inclinar para a esquerda / direita\n"
         " Q / E                         leme (no chao: roda do nariz)\n"
         " + / -                         mais / menos potencia\n"
         " F / G                         descer / subir flaps (0-10-20-30)\n"
         " B (manter)                    travoes\n"
         " Rato: botao direito + arrastar  manche (cima/baixo = pitch, lados = "
         "roll)\n"
         " Rato: botao esquerdo + arrastar rodar a camara\n"
         " Roda do rato                  zoom\n"
         " C                             camara (Perseguicao / Torre / "
         "Cockpit)\n"
         " N                             dia / noite\n"
         " Esc                           menu (pausa)    R  recomecar\n"
         "=====================================================\n"
         " Objetivo: seguir a rampa ate a pista, fazer o flare (levantar\n"
         " o nariz e tirar a potencia perto do chao) e tocar suavemente.\n"
      << std::endl;
}

void keyCallback(GLFWwindow *w, int key, int, int action, int) {
  if (action != GLFW_PRESS)
    return;
  if (key == GLFW_KEY_ESCAPE) { // abre / fecha o menu
    menuOpen = !menuOpen;
    started = true;
    return;
  }
  if (menuOpen) { // teclas do menu
    if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER ||
        key == GLFW_KEY_SPACE) {
      menuOpen = false;
      started = true;
    }
    if (key == GLFW_KEY_R) {
      resetAircraft();
      menuOpen = false;
      started = true;
    }
    if (key == GLFW_KEY_X)
      glfwSetWindowShouldClose(w, GLFW_TRUE);
    return;
  }
  switch (key) { // teclas durante o jogo
  case GLFW_KEY_R:
    resetAircraft();
    break;
  case GLFW_KEY_C:
    cameraMode = (cameraMode + 1) % NUM_CAMS;
    lookYaw = lookPitch = 0;
    break;
  case GLFW_KEY_N:
    nightMode = !nightMode;
    break;
  case GLFW_KEY_F:
    ac.flaps = std::min(30, ac.flaps + 10);
    std::cout << "Flaps " << ac.flaps << std::endl;
    break;
  case GLFW_KEY_G:
    ac.flaps = std::max(0, ac.flaps - 10);
    std::cout << "Flaps " << ac.flaps << std::endl;
    break;
  }
}

// '+' e '-' como caracteres: funcionam no teclado portugues (e repetem se
// mantidos)
void charCallback(GLFWwindow *, unsigned int c) {
  if (menuOpen)
    return;
  if (c == '+')
    ac.throttle = std::min(1.0f, ac.throttle + 0.05f);
  if (c == '-')
    ac.throttle = std::max(0.0f, ac.throttle - 0.05f);
}

void mouseButtonCallback(GLFWwindow *w, int button, int action, int) {
  glfwGetCursorPos(w, &lastX, &lastY);
  if (button == GLFW_MOUSE_BUTTON_LEFT)
    leftDown = (action == GLFW_PRESS);
  if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    rightDown = (action == GLFW_PRESS);
    yokeX = lastX;
    yokeY = lastY;
    mouseElev = mouseAil = 0; // largar o botao = manche ao centro
  }
}

void cursorCallback(GLFWwindow *, double x, double y) {
  float dx = (float)(x - lastX), dy = (float)(y - lastY);
  lastX = x;
  lastY = y;
  if (menuOpen)
    return;
  if (leftDown && cameraMode == CAM_COCKPIT) {
    lookYaw = glm::clamp(lookYaw - dx * 0.25f, -140.0f, 140.0f);
    lookPitch = glm::clamp(lookPitch - dy * 0.25f, -50.0f, 60.0f);
  } else if (leftDown) {
    orbitYaw -= dx * 0.3f;
    orbitPitch = glm::clamp(orbitPitch + dy * 0.3f, -10.0f, 85.0f);
  }
  if (rightDown) {
    // quanto mais longe do ponto inicial, mais forte o comando (150 px =
    // maximo)
    mouseAil = glm::clamp((float)(x - yokeX) / 150.0f, -1.0f, 1.0f);
    mouseElev = glm::clamp((float)(y - yokeY) / 150.0f, -1.0f,
                           1.0f); // puxar para baixo = nariz para cima
  }
}

void scrollCallback(GLFWwindow *, double, double yoff) {
  float f = std::pow(0.9f, (float)yoff);
  if (cameraMode == CAM_CHASE)
    orbitDist = glm::clamp(orbitDist * f, 8.0f, 400.0f);
  if (cameraMode == CAM_TOWER)
    towerZoom = glm::clamp(towerZoom / f, 0.2f, 5.0f);
}

void framebufferSizeCallback(GLFWwindow *, int w, int h) {
  fbWidth = w;
  fbHeight = h;
  glViewport(0, 0, w, h);
}

// le as teclas mantidas e junta-as com o rato
void readControls() {
  auto down = [](int k) { return glfwGetKey(window, k) == GLFW_PRESS; };
  float kElev = 0, kAil = 0, kRud = 0;
  if (down(GLFW_KEY_S) || down(GLFW_KEY_DOWN))
    kElev += 1; // puxar o manche
  if (down(GLFW_KEY_W) || down(GLFW_KEY_UP))
    kElev -= 1; // empurrar o manche
  if (down(GLFW_KEY_D) || down(GLFW_KEY_RIGHT))
    kAil += 1;
  if (down(GLFW_KEY_A) || down(GLFW_KEY_LEFT))
    kAil -= 1;
  if (down(GLFW_KEY_Q))
    kRud += 1; // rumo para a esquerda
  if (down(GLFW_KEY_E))
    kRud -= 1;
  cmdElev = glm::clamp(kElev + mouseElev, -1.0f, 1.0f);
  cmdAil = glm::clamp(kAil + mouseAil, -1.0f, 1.0f);
  cmdRud = kRud;
  cmdBrake = down(GLFW_KEY_B);
  if (menuOpen)
    cmdElev = cmdAil = cmdRud = 0, cmdBrake = false;
}

std::string phaseName() {
  if (ac.status == CRASHED)
    return "ACIDENTE";
  if (ac.status == ON_GROUND)
    return ac.V < 0.5f ? "Parado" : "No chao";
  return ac.pos.y < 6.0f ? "Flare" : "Aproximacao";
}

// -----------------------------------------------------------------------------
//  Menu (ao iniciar e com Esc) e painel de instrumentos (durante o jogo)
// -----------------------------------------------------------------------------
const glm::vec4 WHITE(1, 1, 1, 1), YELLOW(1.0f, 0.82f, 0.25f, 1),
    GREY(0.70f, 0.74f, 0.80f, 1), GREEN(0.40f, 0.95f, 0.45f, 1),
    RED(1.0f, 0.35f, 0.30f, 1), PANEL(0.06f, 0.08f, 0.12f, 0.88f);

// Desenha uma tecla como uma "tampa" de teclado (retangulo claro com a letra
// escura). Devolve a largura ocupada, para se poder pôr outra a seguir.
float drawKeyCap(float x, float y, const std::string &label) {
  float s = uiScale, h = 22 * s, w = textWidth(label, 1.8f) + 14 * s;
  drawRect(x, y + 2 * s, w, h,
           glm::vec4(0.45f, 0.48f, 0.55f, 1));             // sombra (da relevo)
  drawRect(x, y, w, h, glm::vec4(0.90f, 0.92f, 0.95f, 1)); // tampa
  drawText(x + 7 * s, y + 5 * s, label, 1.8f,
           glm::vec4(0.08f, 0.10f, 0.16f, 1));
  return w;
}

// Coluna "TECLA" de uma linha da tabela: as palavras separadas por '|' sao
// teclas (desenhadas como tampas); "/", "ou", "+ arrastar" e o que esta entre
// parenteses e texto normal.
void drawKeys(float x, float y, const std::string &spec) {
  float s = uiScale;
  std::stringstream ss(spec);
  std::string tok;
  while (std::getline(ss, tok, '|')) {
    bool plain =
        (tok == "/" || tok == "ou" || tok[0] == '(' || tok == "+ arrastar");
    if (plain) {
      drawText(x, y + 5 * s, tok, 1.8f, GREY);
      x += textWidth(tok, 1.8f) + 8 * s;
    } else {
      x += drawKeyCap(x, y, tok) + 8 * s;
    }
  }
}

void drawMenu() {
  float W = (float)fbWidth, H = (float)fbHeight, s = uiScale;
  drawRect(0, 0, W, H, glm::vec4(0, 0, 0, 0.55f)); // escurece o jogo
  float pw = 800 * s, ph = 672 * s, px = (W - pw) / 2, py = (H - ph) / 2;
  drawRect(px, py, pw, ph, PANEL);
  drawRect(px, py, pw, 6 * s, YELLOW); // risca no topo

  float cx = W / 2, y = py + 26 * s;
  drawTextCentered(cx, y, "ATERRAGEM - CESSNA 182", 4.0f, WHITE);
  y += 42 * s;
  drawTextCentered(cx, y,
                   "Trabalho 2 de Computacao Grafica  -  Tome Pedro 53702 PL2",
                   1.7f, GREY);
  y += 34 * s;

  // ---- tabela: uma secao por tipo de comando, cada linha = tecla(s) | o que
  // faz
  float tx = px + 40 * s, tw = pw - 80 * s; // tabela
  float colAcao = tx + 300 * s;             // inicio da coluna "O QUE FAZ"
  float rowH = 28 * s;
  struct Row {
    const char *keys;
    const char *acao;
  };
  struct Section {
    const char *titulo;
    std::vector<Row> rows;
  };
  const Section sections[] = {
      {"PILOTAR",
       {{"W|/|S|ou|CIMA|/|BAIXO", "nariz para baixo / para cima"},
        {"A|/|D|ou|ESQ|/|DIR", "inclinar para a esquerda / direita"},
        {"Q|/|E", "leme (no chao: roda do nariz)"},
        {"+|/|-", "mais / menos potencia"},
        {"F|/|G", "descer / subir flaps (0-10-20-30)"},
        {"B|(manter)", "travoes"},
        {"RATO DIREITO|+ arrastar", "manche com o rato"}}},
      {"CAMARA",
       {{"C", "perseguicao / torre / cockpit"},
        {"RATO ESQUERDO|+ arrastar", "rodar a camara (no cockpit: olhar)"},
        {"RODA DO RATO", "zoom"}}},
      {"JOGO",
       {{"N", "dia / noite (luzes da pista)"},
        {"R", "recomecar a aterragem"},
        {"ESC", "abrir / fechar este menu"}}},
  };
  for (const Section &sec : sections) {
    // cabecalho da secao
    drawRect(tx, y, tw, 24 * s, glm::vec4(1.0f, 0.82f, 0.25f, 0.18f));
    drawRect(tx, y, 4 * s, 24 * s, YELLOW);
    drawText(tx + 12 * s, y + 6 * s, sec.titulo, 1.9f, YELLOW);
    y += 28 * s;
    int i = 0;
    for (const Row &r : sec.rows) {
      if (i++ % 2 == 0)
        drawRect(tx, y - 3 * s, tw, rowH,
                 glm::vec4(1, 1, 1, 0.05f)); // linhas alternadas
      drawKeys(tx + 12 * s, y, r.keys);
      drawText(colAcao, y + 5 * s, r.acao, 1.9f, WHITE);
      y += rowH;
    }
    y += 8 * s;
  }

  y += 4 * s;
  drawTextCentered(
      cx, y, "Objetivo: seguir a rampa de 3 graus ate a pista. Perto do chao,",
      1.7f, GREY);
  y += 18 * s;
  drawTextCentered(cx, y,
                   "tira a potencia e levanta o nariz devagar (flare) para "
                   "tocar suavemente.",
                   1.7f, GREY);
  y += 32 * s;
  drawTextCentered(cx, y,
                   started
                       ? "ENTER - continuar      R - recomecar      X - sair"
                       : "ENTER - comecar      X - sair",
                   2.4f, GREEN);
}

std::string fmt(const char *f, double v) {
  char b[64];
  snprintf(b, sizeof(b), f, v);
  return b;
}

// -----------------------------------------------------------------------------
//  Painel de instrumentos do cockpit (extra): 6 instrumentos classicos em 2D
// -----------------------------------------------------------------------------
//  Angulos dos mostradores em graus, no sentido dos ponteiros do relogio a
//  partir das 12 horas. Em coordenadas de ecra (y para baixo):
//      x = cx + r sin(a),  y = cy - r cos(a)
// -----------------------------------------------------------------------------
const float COCKPIT_PANEL =
    0.32f; // fracao da altura do ecra ocupada pelo painel

glm::vec2 dialPoint(float cx, float cy, float r, float aDeg) {
  float a = glm::radians(aDeg);
  return {cx + r * std::sin(a), cy - r * std::cos(a)};
}

void drawPolygon2D(const std::vector<glm::vec2> &p,
                   const glm::vec4 &col) { // poligono convexo (leque)
  std::vector<float> xy;
  for (size_t i = 1; i + 1 < p.size(); i++)
    xy.insert(xy.end(),
              {p[0].x, p[0].y, p[i].x, p[i].y, p[i + 1].x, p[i + 1].y});
  drawTriangles2D(xy, col);
}

void drawDisc(float cx, float cy, float r, const glm::vec4 &col, int seg = 48) {
  std::vector<glm::vec2> p;
  for (int i = 0; i < seg; i++)
    p.push_back(dialPoint(cx, cy, r, 360.0f * i / seg));
  drawPolygon2D(p, col);
}

void drawSegment(glm::vec2 a, glm::vec2 b, float w, const glm::vec4 &col) {
  glm::vec2 d = b - a;
  float len = glm::length(d);
  if (len < 1e-4f)
    return;
  glm::vec2 n = glm::vec2(-d.y, d.x) / len * (w / 2);
  drawTriangles2D({a.x + n.x, a.y + n.y, b.x + n.x, b.y + n.y, b.x - n.x,
                   b.y - n.y, a.x + n.x, a.y + n.y, b.x - n.x, b.y - n.y,
                   a.x - n.x, a.y - n.y},
                  col);
}

// arco colorido entre os raios r0 e r1, dos angulos a0 a a1
void drawArc(float cx, float cy, float r0, float r1, float a0, float a1,
             const glm::vec4 &col) {
  std::vector<float> xy;
  int seg = std::max(2, (int)std::fabs(a1 - a0) / 4);
  for (int i = 0; i < seg; i++) {
    float b0 = a0 + (a1 - a0) * i / seg, b1 = a0 + (a1 - a0) * (i + 1) / seg;
    glm::vec2 p0 = dialPoint(cx, cy, r0, b0), p1 = dialPoint(cx, cy, r1, b0);
    glm::vec2 p2 = dialPoint(cx, cy, r1, b1), p3 = dialPoint(cx, cy, r0, b1);
    xy.insert(xy.end(), {p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, p0.x, p0.y, p2.x,
                         p2.y, p3.x, p3.y});
  }
  drawTriangles2D(xy, col);
}

void drawNeedle(float cx, float cy, float len, float w, float aDeg,
                const glm::vec4 &col) {
  drawSegment(dialPoint(cx, cy, -len * 0.18f, aDeg),
              dialPoint(cx, cy, len, aDeg), w, col);
  drawDisc(cx, cy, w * 1.2f, glm::vec4(0.25f, 0.25f, 0.27f, 1), 16);
}

// fundo comum: aro metalico + mostrador preto
void dialBase(float cx, float cy, float r) {
  drawDisc(cx, cy, r * 1.08f, glm::vec4(0.32f, 0.33f, 0.35f, 1));
  drawDisc(cx, cy, r, glm::vec4(0.04f, 0.04f, 0.05f, 1));
}

void dialTicks(float cx, float cy, float r, float a0, float a1, int n,
               float lenFrac, float w) {
  for (int i = 0; i <= n; i++) {
    float a = a0 + (a1 - a0) * i / n;
    drawSegment(dialPoint(cx, cy, r * (1 - lenFrac), a),
                dialPoint(cx, cy, r * 0.97f, a), w, WHITE);
  }
}

void dialLabel(float cx, float cy, float r,
               const std::string &s) { // legenda por baixo do mostrador
  drawTextCentered(cx, cy + r * 1.13f, s, 1.3f, GREY);
}

// 1) velocimetro: 0..200 kt em 300 graus (arcos do Cessna 182)
void gaugeAirspeed(float cx, float cy, float r, float kt) {
  dialBase(cx, cy, r);
  auto A = [](float v) {
    return -150.0f + 300.0f * glm::clamp(v, 0.0f, 200.0f) / 200.0f;
  };
  drawArc(cx, cy, r * 0.80f, r * 0.88f, A(49), A(100),
          glm::vec4(0.95f, 0.95f, 0.95f, 1)); // arco branco (flaps)
  drawArc(cx, cy, r * 0.88f, r * 0.96f, A(54), A(140),
          glm::vec4(0.2f, 0.8f, 0.3f, 1)); // verde (normal)
  drawArc(cx, cy, r * 0.88f, r * 0.96f, A(140), A(175),
          glm::vec4(0.95f, 0.8f, 0.1f, 1)); // amarelo (cuidado)
  drawSegment(dialPoint(cx, cy, r * 0.78f, A(175)),
              dialPoint(cx, cy, r * 0.97f, A(175)), 3 * uiScale, RED);
  dialTicks(cx, cy, r, A(0), A(200), 20, 0.12f, 1.5f * uiScale);
  for (int v = 40; v <= 180; v += 20) {
    glm::vec2 p = dialPoint(cx, cy, r * 0.62f, A((float)v));
    drawTextCentered(p.x, p.y - 4 * uiScale, std::to_string(v), 1.2f, WHITE);
  }
  dialLabel(cx, cy, r, "VEL (kt)");
  drawNeedle(cx, cy, r * 0.85f, 3 * uiScale, A(kt), WHITE);
}

// 2) horizonte artificial: ceu/terra cortados por uma linha que roda com a
// inclinacao
//    e desce quando o nariz sobe
void gaugeAttitude(float cx, float cy, float r, float pitchDeg, float rollDeg) {
  drawDisc(cx, cy, r * 1.08f, glm::vec4(0.32f, 0.33f, 0.35f, 1));
  drawDisc(cx, cy, r, glm::vec4(0.20f, 0.50f, 0.90f, 1)); // ceu
  float ph = glm::radians(rollDeg);
  glm::vec2 c(cx, cy), n(-std::sin(ph), -std::cos(ph)),
      d(std::cos(ph), -std::sin(ph));   // n aponta para o ceu
  float k = r * 0.035f;                 // pixeis por grau de pitch
  glm::vec2 H = c - n * (pitchDeg * k); // ponto da linha do horizonte
  // terra = parte do circulo do lado oposto a n (recorte de Sutherland-Hodgman
  // com 1 semiplano)
  std::vector<glm::vec2> circ, ground;
  for (int i = 0; i < 64; i++)
    circ.push_back(dialPoint(cx, cy, r, 360.0f * i / 64));
  std::vector<glm::vec2> cut;
  for (size_t i = 0; i < circ.size(); i++) {
    glm::vec2 a = circ[i], b = circ[(i + 1) % circ.size()];
    float da = glm::dot(a - H, n), db = glm::dot(b - H, n);
    if (da <= 0)
      ground.push_back(a);
    if ((da <= 0) != (db <= 0)) {
      glm::vec2 p = a + (b - a) * (da / (da - db));
      ground.push_back(p);
      cut.push_back(p);
    }
  }
  if (ground.size() >= 3)
    drawPolygon2D(ground, glm::vec4(0.55f, 0.35f, 0.15f, 1));
  if (cut.size() == 2)
    drawSegment(cut[0], cut[1], 2 * uiScale, WHITE);
  // escala de pitch: tracos a +-10 e +-20 graus
  for (int p : {-20, -10, 10, 20}) {
    glm::vec2 m = c - n * ((pitchDeg - p) * k);
    float half = r * (std::abs(p) == 10 ? 0.25f : 0.35f);
    if (glm::length(m - c) < r * 0.8f)
      drawSegment(m - d * half, m + d * half, 1.5f * uiScale, WHITE);
  }
  // avião fixo (amarelo) no centro
  glm::vec4 Y(1.0f, 0.75f, 0.1f, 1);
  drawSegment({cx - r * 0.55f, cy}, {cx - r * 0.18f, cy}, 3.5f * uiScale, Y);
  drawSegment({cx + r * 0.18f, cy}, {cx + r * 0.55f, cy}, 3.5f * uiScale, Y);
  drawDisc(cx, cy, 3.5f * uiScale, Y, 12);
  drawArc(cx, cy, r * 0.97f, r * 1.0f, -60, 60, WHITE);
  dialLabel(cx, cy, r, "HORIZONTE");
}

// 3) altimetro: ponteiro grande = 1000 ft por volta, pequeno = 10000 ft por
// volta
void gaugeAltimeter(float cx, float cy, float r, float ft) {
  dialBase(cx, cy, r);
  dialTicks(cx, cy, r, 0, 360, 50, 0.07f, 1.0f * uiScale);
  dialTicks(cx, cy, r, 0, 360, 10, 0.15f, 2.0f * uiScale);
  for (int i = 0; i < 10; i++) {
    glm::vec2 p = dialPoint(cx, cy, r * 0.68f, 36.0f * i);
    drawTextCentered(p.x, p.y - 4 * uiScale, std::to_string(i), 1.4f, WHITE);
  }
  char b[16];
  snprintf(b, sizeof(b), "%05d", (int)std::max(0.0f, ft));
  drawRect(cx - r * 0.42f, cy + r * 0.16f, r * 0.84f, r * 0.24f,
           glm::vec4(0.15f, 0.15f, 0.16f, 1));
  drawTextCentered(cx, cy + r * 0.2f, b, 1.4f, WHITE);
  dialLabel(cx, cy, r, "ALT (ft)");
  drawNeedle(cx, cy, r * 0.5f, 4 * uiScale, 360.0f * ft / 10000.0f, WHITE);
  drawNeedle(cx, cy, r * 0.88f, 2.5f * uiScale,
             360.0f * std::fmod(std::max(0.0f, ft), 1000.0f) / 1000.0f, WHITE);
}

// 4) coordenador de curva: avião visto de tras que inclina com o roll
void gaugeTurn(float cx, float cy, float r, float rollDeg) {
  dialBase(cx, cy, r);
  float b = glm::clamp(rollDeg, -45.0f, 45.0f);
  for (float m : {-110.0f, -70.0f, 70.0f,
                  110.0f}) // marcas de curva padrao (asas a ~20 graus)
    drawSegment(dialPoint(cx, cy, r * 0.78f, m),
                dialPoint(cx, cy, r * 0.95f, m), 2.5f * uiScale, WHITE);
  drawTextCentered(cx - r * 0.6f, cy + r * 0.35f, "L", 1.4f, WHITE);
  drawTextCentered(cx + r * 0.6f, cy + r * 0.35f, "R", 1.4f, WHITE);
  drawSegment(dialPoint(cx, cy, r * 0.72f, 90 + b),
              dialPoint(cx, cy, r * 0.72f, 270 + b), 4 * uiScale,
              WHITE); // asas
  drawSegment({cx, cy}, dialPoint(cx, cy, r * 0.22f, b), 3 * uiScale,
              WHITE); // cauda
  drawDisc(cx, cy, 5 * uiScale, WHITE, 16);
  dialLabel(cx, cy, r, "CURVA");
}

// 5) indicador de rumo: rosa dos ventos que roda; o rumo atual fica em cima
void gaugeHeading(float cx, float cy, float r, float hdg) {
  dialBase(cx, cy, r);
  for (int h = 0; h < 360; h += 10) {
    float a = h - hdg;
    float len = (h % 30 == 0) ? 0.16f : 0.09f;
    drawSegment(dialPoint(cx, cy, r * (0.95f - len), a),
                dialPoint(cx, cy, r * 0.95f, a), 1.5f * uiScale, WHITE);
  }
  const char *L[4] = {"N", "E", "S", "W"};
  for (int i = 0; i < 4; i++) {
    glm::vec2 p = dialPoint(cx, cy, r * 0.62f, 90.0f * i - hdg);
    drawTextCentered(p.x, p.y - 5 * uiScale, L[i], 1.8f,
                     i == 0 ? YELLOW : WHITE);
  }
  glm::vec4 Y(1.0f, 0.75f, 0.1f, 1);
  drawSegment({cx, cy - r * 0.35f}, {cx, cy + r * 0.3f}, 3 * uiScale,
              Y); // aviaozinho fixo
  drawSegment({cx - r * 0.25f, cy - r * 0.05f},
              {cx + r * 0.25f, cy - r * 0.05f}, 3 * uiScale, Y);
  drawPolygon2D({{cx, cy - r * 1.0f},
                 {cx + 6 * uiScale, cy - r * 1.12f},
                 {cx - 6 * uiScale, cy - r * 1.12f}},
                Y);
  char b[8];
  snprintf(b, sizeof(b), "%03d", ((int)std::round(hdg) % 360 + 360) % 360);
  dialLabel(cx, cy, r, std::string("RUMO ") + b);
}

// 6) variometro (velocidade vertical): 0 as 9 horas, +-2000 ft/min
void gaugeVSI(float cx, float cy, float r, float fpm) {
  dialBase(cx, cy, r);
  auto A = [](float v) {
    return -90.0f + 170.0f * glm::clamp(v, -2000.0f, 2000.0f) / 2000.0f;
  };
  dialTicks(cx, cy, r, A(-2000), A(2000), 8, 0.15f, 2.0f * uiScale);
  for (int v : {5, 10, 15, 20}) {
    for (int s : {1, -1}) {
      glm::vec2 p = dialPoint(cx, cy, r * 0.64f, A((float)(s * v * 100)));
      drawTextCentered(p.x, p.y - 4 * uiScale, std::to_string(v), 1.2f, WHITE);
    }
  }
  drawTextCentered(cx + r * 0.25f, cy - r * 0.2f, "UP", 1.1f, GREY);
  drawTextCentered(cx + r * 0.25f, cy + r * 0.12f, "DN", 1.1f, GREY);
  dialLabel(cx, cy, r, "VS (x100 ft/min)");
  drawNeedle(cx, cy, r * 0.85f, 3 * uiScale, A(fpm), WHITE);
}

// barra vertical (potencia / flaps)
void drawBar(float x, float y, float w, float h, float frac,
             const glm::vec4 &col, const std::string &label) {
  drawRect(x, y, w, h, glm::vec4(0.04f, 0.04f, 0.05f, 1));
  float f = glm::clamp(frac, 0.0f, 1.0f);
  drawRect(x + 3 * uiScale, y + h - (h - 6 * uiScale) * f - 3 * uiScale,
           w - 6 * uiScale, (h - 6 * uiScale) * f, col);
  drawTextCentered(x + w / 2, y + h + 6 * uiScale, label, 1.3f, GREY);
}

void drawCockpitPanel() {
  float W = (float)fbWidth, H = (float)fbHeight, s = uiScale;
  float top = H * (1 - COCKPIT_PANEL);
  // painel com uma "pala" por cima (como o tablier de um avião)
  drawPolygon2D({{0, top + 18 * s},
                 {W * 0.08f, top},
                 {W * 0.92f, top},
                 {W, top + 18 * s},
                 {W, H},
                 {0, H}},
                glm::vec4(0.11f, 0.11f, 0.12f, 1));
  drawRect(0, top + 18 * s, W, 3 * s, glm::vec4(0.2f, 0.2f, 0.22f, 1));
  float r = std::min(H * COCKPIT_PANEL * 0.36f, W / 17.0f);
  float cy = top + 18 * s + (H - top - 18 * s) / 2;
  float gap = r * 2.45f, x0 = W / 2 - gap * 2.5f - r * 0.9f;
  float pitch = glm::degrees(ac.theta), roll = glm::degrees(ac.phi);
  float hdg = std::fmod(-glm::degrees(ac.psi), 360.0f);
  if (hdg < 0)
    hdg += 360;
  gaugeAirspeed(x0, cy, r, ac.V / KT);
  gaugeAttitude(x0 + gap, cy, r, pitch, roll);
  gaugeAltimeter(x0 + 2 * gap, cy, r, ac.pos.y * 3.28084f);
  gaugeTurn(x0 + 3 * gap, cy, r, roll);
  gaugeHeading(x0 + 4 * gap, cy, r, hdg);
  gaugeVSI(x0 + 5 * gap, cy, r, ac.V * std::sin(ac.gamma) * FPM);
  float bx = x0 + 5 * gap + r * 1.6f, bh = r * 1.7f, by = cy - bh / 2 - 6 * s;
  drawBar(bx, by, r * 0.35f, bh, ac.throttle, GREEN, "POT");
  drawBar(bx + r * 0.6f, by, r * 0.35f, bh, ac.flaps / 30.0f,
          glm::vec4(0.9f, 0.9f, 0.9f, 1), "FLAPS");
}

void drawHUD() {
  float W = (float)fbWidth, H = (float)fbHeight, s = uiScale;

  // canto superior esquerdo: atalho para o menu
  drawRect(12 * s, 12 * s, 150 * s, 34 * s, PANEL);
  drawText(24 * s, 22 * s, "ESC", 2.2f, YELLOW);
  drawText(74 * s, 22 * s, "Menu", 2.2f, WHITE);

  // canto superior direito: camara
  std::string cam = std::string("Camara: ") + CAM_NAME[cameraMode] + "  (C)";
  float cw = textWidth(cam, 2.0f) + 24 * s;
  drawRect(W - cw - 12 * s, 12 * s, cw, 34 * s, PANEL);
  drawText(W - cw, 22 * s, cam, 2.0f, WHITE);

  if (cameraMode == CAM_COCKPIT) {
    drawCockpitPanel();
  } else {
    // canto inferior esquerdo: instrumentos
    float rampa =
        ac.pos.y - (AIM_Z - ac.pos.z) *
                       std::tan(glm::radians(GLIDE_DEG)); // + acima, - abaixo
    std::vector<std::pair<std::string, std::string>> rows = {
        {"FASE", phaseName()},
        {"VEL", fmt("%.0f kt", ac.V / KT)},
        {"ALT", fmt("%.0f ft", ac.pos.y * 3.28084f)},
        {"VS", fmt("%.0f ft/min", ac.V * std::sin(ac.gamma) * FPM)},
        {"PITCH", fmt("%.1f graus", glm::degrees(ac.theta))},
        {"INCL", fmt("%.0f graus", glm::degrees(ac.phi))},
        {"POT", fmt("%.0f %%", ac.throttle * 100)},
        {"FLAPS", fmt("%.0f graus", ac.flaps)},
    };
    if (ac.status == FLYING && ac.pos.z < AIM_Z)
      rows.push_back({"RAMPA", fmt("%+.0f m", rampa)});
    float bh = (rows.size() * 24 + 20) * s, by = H - bh - 12 * s;
    drawRect(12 * s, by, 270 * s, bh, PANEL);
    float y = by + 12 * s;
    for (auto &r : rows) {
      drawText(26 * s, y, r.first, 2.0f, YELLOW);
      drawText(120 * s, y, r.second, 2.0f, WHITE);
      y += 24 * s;
    }
  }

  // moldura da camara de aterragem (a imagem ja foi desenhada no retangulo)
  if (landCamOn) {
    float x = (float)landX, y = (float)landY, w = (float)landW,
          h = (float)landH, b = 3 * s;
    drawRect(x - b, y - b, w + 2 * b, b, PANEL);
    drawRect(x - b, y + h, w + 2 * b, b, PANEL);
    drawRect(x - b, y, b, h, PANEL);
    drawRect(x + w, y, b, h, PANEL);
    drawRect(x, y, w, 26 * s, glm::vec4(0, 0, 0, 0.45f));
    if (std::fmod(glfwGetTime(), 1.0) < 0.6)
      drawDisc(x + 14 * s, y + 13 * s, 5 * s, RED); // "REC" a piscar
    drawText(x + 26 * s, y + 7 * s, "CAMARA DE ATERRAGEM", 1.8f, WHITE);
  }

  // avisos ao centro
  float cy = 70 * s;
  if (cmdBrake && ac.status == ON_GROUND) {
    drawTextCentered(W / 2, cy, "TRAVOES", 2.6f, YELLOW);
    cy += 34 * s;
  }
  if (!ac.result.empty()) {
    glm::vec4 c = (ac.status == CRASHED)
                      ? RED
                      : (ac.result.rfind("Boa", 0) == 0 ? GREEN : YELLOW);
    float tw = textWidth(ac.result, 2.4f) + 40 * s;
    drawRect(W / 2 - tw / 2, cy - 10 * s, tw, 64 * s, PANEL);
    drawTextCentered(W / 2, cy, ac.result, 2.4f, c);
    drawTextCentered(W / 2, cy + 28 * s, "R - recomecar", 2.0f, GREY);
  }
}

// =============================================================================
//  MAIN
// =============================================================================
int main() {
  if (!glfwInit()) {
    std::cerr << "Erro ao iniciar o GLFW" << std::endl;
    return -1;
  }
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // obrigatorio no macOS
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window =
      glfwCreateWindow(1280, 720, "Aterragem - Cessna 182", nullptr, nullptr);
  if (!window) {
    std::cerr << "Erro ao criar a janela" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "Erro ao iniciar o GLEW" << std::endl;
    return -1;
  }

  glfwSetKeyCallback(window, keyCallback);
  glfwSetCharCallback(window, charCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetCursorPosCallback(window, cursorCallback);
  glfwSetScrollCallback(window, scrollCallback);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  glfwGetFramebufferSize(window, &fbWidth,
                         &fbHeight); // em ecras Retina e o dobro da janela
  glViewport(0, 0, fbWidth, fbHeight);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);
  // Sem face culling: as asas e a helice do modelo sao finas e devem ver-se dos
  // dois lados.

  const glm::vec3 SKY(0.62f, 0.78f, 0.92f);
  glClearColor(SKY.r, SKY.g, SKY.b, 1.0f);

  GLuint prog = createProgram();
  initOverlay();
  initSkybox(); // se falhar, fica o glClearColor (cor do ceu)
  initGlow();

  // ---- carregar os .obj ----
  buildRunwayLights();
  generateAirportOBJ();
  generateTerrainOBJ();
  generateCityOBJ();
  generatePilotOBJ();
  generatePropDiscOBJ();
  // plane = Cessna 182 (o que se pilota); plane152 = Cessna 152 (os
  // estacionados na placa)
  Model runway, plane, plane152, airport, terrain, city, pilot, propDisc;
  if (!loadOBJ("models/runway.obj", runway) ||
      !loadOBJ("models/c182/cessna182.obj", plane) ||
      !loadOBJ("models/cessna152.obj", plane152) ||
      !loadOBJ("models/airport.obj", airport) ||
      !loadOBJ("models/terrain.obj", terrain) ||
      !loadOBJ("models/city.obj", city) ||
      !loadOBJ("models/pilot.obj", pilot) ||
      !loadOBJ("models/prop_disc.obj", propDisc)) {
    std::cerr << "Corre o programa a partir da pasta do trabalho (make run)."
              << std::endl;
    return -1;
  }

  // matriz model da pista: T * S * R (ver as constantes RWY_*)
  const glm::mat4 runwayMatrix =
      glm::translate(glm::mat4(1.0f),
                     glm::vec3(0, 0, RWY_THRESH_OBJ * RWY_SCALE)) *
      glm::scale(glm::mat4(1.0f), glm::vec3(RWY_SCALE)) *
      glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0, 1, 0));

  // ---- luz pontual ----
  const glm::vec3 lightPos(-250.0f, 400.0f, 150.0f);
  const glm::vec3 lightColor(1.0f, 0.97f, 0.90f);

  // unidades de textura dos samplers: 0 = textura do material, 1 = cubemap
  // (reflexo do ceu)
  glUseProgram(prog);
  glUniform1i(glGetUniformLocation(prog, "tex"), 0);
  glUniform1i(glGetUniformLocation(prog, "skyCube"), 1);

  printControls();
  resetAircraft();

  double lastTime = glfwGetTime();
  while (!glfwWindowShouldClose(window)) {
    double now = glfwGetTime();
    float dt = (float)std::min(now - lastTime, 0.1);
    lastTime = now;

    readControls();

    // ---- fisica em passos pequenos (estavel mesmo com poucos FPS) ----
    if (!menuOpen) {
      float t = dt;
      while (t > 0) {
        float step = std::min(t, 0.005f);
        updateAircraft(step);
        t -= step;
      }
      updateAnimation(dt); // helice, flaps, superficies de comando
    }

    // ---- camara ----
    glm::mat4 Mplane = aircraftMatrix();
    glm::vec3 target = ac.pos + glm::vec3(0, 1.0f, 0);
    glm::vec3 eye;
    float fov = 45.0f;
    if (cameraMode == CAM_CHASE) {
      // atras do avião: o angulo da orbita soma-se ao rumo
      float y = ac.psi + glm::radians(orbitYaw), p = glm::radians(orbitPitch);
      eye = target + orbitDist * glm::vec3(-std::sin(y) * std::cos(p),
                                           std::sin(p),
                                           -std::cos(y) * std::cos(p));
      eye.y = std::max(eye.y, terrainHeight(eye.x, eye.z) +
                                  2.0f); // nunca abaixo do chao / montanha
    } else if (cameraMode == CAM_TOWER) {
      eye = towerPos;
      float d = glm::length(target - eye);
      fov = glm::clamp(glm::degrees(2.0f * std::atan(12.0f / (d * towerZoom))),
                       2.0f, 60.0f); // zoom automatico
    }
    glm::vec3 up(0, 1, 0);
    float nearP = 1.0f;
    int vpY = 0, vpH = fbHeight; // zona do ecra onde se desenha a cena 3D
    if (cameraMode == CAM_COCKPIT) {
      // olhos do piloto, em coordenadas do avião -> mundo (a camara roda com o
      // avião)
      eye = glm::vec3(Mplane * glm::vec4(PILOT_EYE, 1.0f));
      glm::mat4 R = Mplane;
      R[3] = glm::vec4(0, 0, 0, 1); // so a rotacao
      glm::mat4 look = glm::rotate(glm::mat4(1.0f), glm::radians(lookYaw),
                                   glm::vec3(0, 1, 0));
      look = glm::rotate(look, glm::radians(-(lookPitch - 3.0f)),
                         glm::vec3(1, 0, 0)); // 3 graus para baixo
      glm::vec3 fwd = glm::vec3(R * look * glm::vec4(0, 0, 1, 0));
      up = glm::vec3(R * glm::vec4(0, 1, 0, 0));
      target = eye + fwd;
      fov = 60.0f;
      nearP = 0.3f; // o vidro e o tablier estao muito perto
      vpY = (int)(fbHeight * COCKPIT_PANEL); // a cena fica por cima do painel
      vpH = fbHeight - vpY;
    }
    glm::mat4 V = glm::lookAt(eye, target, up);
    glm::mat4 P = glm::perspective(
        glm::radians(fov), (float)fbWidth / std::max(vpH, 1), nearP, 10000.0f);

    // ---- dia / noite ----
    glm::vec3 tint =
        nightMode ? glm::vec3(0.16f, 0.18f, 0.30f) : glm::vec3(1.0f);
    glm::vec3 fogCol = SKY * tint;
    glm::vec3 lightC =
        lightColor *
        (nightMode ? 0.35f : 1.0f); // de noite a luz pontual faz de lua
    glClearColor(fogCol.r, fogCol.g, fogCol.b, 1.0f);

    // pecas que se mexem (iguais para as duas vistas deste frame)
    PartMatrices planeLocal = planeParts();
    PartMatrices pilotLocal = pilotParts(
        menuOpen ? 0.0f : dt,
        cameraMode == CAM_COCKPIT); // no cockpit a camara esta na cabeca
    // Desenha o mundo visto de uma camara (V, P, eye). Chama-se uma vez para o
    // ecra e outra para a "camara de aterragem" no canto.
    auto renderScene = [&](const glm::mat4 &V, const glm::mat4 &P,
                           const glm::vec3 &eye, float fov, int viewH,
                           bool cockpit) {
      glUseProgram(prog);
      glUniformMatrix4fv(glGetUniformLocation(prog, "V"), 1, GL_FALSE,
                         glm::value_ptr(V));
      glUniformMatrix4fv(glGetUniformLocation(prog, "P"), 1, GL_FALSE,
                         glm::value_ptr(P));
      glUniform3fv(glGetUniformLocation(prog, "lightPos"), 1,
                   glm::value_ptr(lightPos));
      glUniform3fv(glGetUniformLocation(prog, "lightColor"), 1,
                   glm::value_ptr(lightC));
      glUniform3f(glGetUniformLocation(prog, "attenuation"), 1.0f, 0.0002f,
                  0.0000002f);
      glUniform1f(glGetUniformLocation(prog, "ambient"),
                  nightMode ? 0.18f : 0.30f);
      glUniform3fv(glGetUniformLocation(prog, "viewPos"), 1,
                   glm::value_ptr(eye));
      glUniform3fv(glGetUniformLocation(prog, "fogColor"), 1,
                   glm::value_ptr(fogCol));
      glUniform1f(glGetUniformLocation(prog, "fogDensity"),
                  0.00022f); // menos nevoeiro: ver as montanhas
      glUniform1f(glGetUniformLocation(prog, "clipZ"), 1e9f);
      glUniform1f(glGetUniformLocation(prog, "reflectivity"), 0.0f);
      glUniform3fv(glGetUniformLocation(prog, "skyTint"), 1,
                   glm::value_ptr(tint));
      glActiveTexture(GL_TEXTURE1); // cubemap do ceu na unidade 1
      glBindTexture(GL_TEXTURE_CUBE_MAP, skyTex);
      glActiveTexture(GL_TEXTURE0);
      float planeRefl = skyTex ? 0.35f : 0.0f;
      // no cockpit esconde-se a helice (tapava a vista): tudo o que esta a
      // frente de PROP_CLIP_Z
      float clip = cockpit ? PROP_CLIP_Z : 1e9f;

      // 1) objetos opacos
      drawModel(runway, prog, runwayMatrix);
      drawModel(airport, prog, glm::mat4(1.0f), nightMode);
      drawModel(terrain, prog, glm::mat4(1.0f));
      drawModel(city, prog, glm::mat4(1.0f), nightMode);
      glUniform1f(glGetUniformLocation(prog, "clipZ"), clip);
      glUniform1f(glGetUniformLocation(prog, "reflectivity"), planeRefl);
      drawModel(plane, prog, Mplane, false, false, &planeLocal);
      glUniform1f(glGetUniformLocation(prog, "reflectivity"), 0.0f);
      drawModel(pilot, prog, Mplane, false, false,
                &pilotLocal); // mesma matriz do avião + as das pecas
      glUniform1f(glGetUniformLocation(prog, "clipZ"), 1e9f);
      std::vector<glm::mat4> parked;
      for (const glm::vec3 &p : PARKED) // o Cessna 152, estacionado na placa
        parked.push_back(glm::rotate(glm::translate(glm::mat4(1.0f), p),
                                     glm::radians(90.0f), glm::vec3(0, 1, 0)));
      for (const glm::mat4 &M : parked)
        drawModel(plane152, prog, M);
      drawSkybox(V, P, tint); // depois dos opacos: so preenche o fundo
      glUseProgram(prog);

      // 2) objetos transparentes (vidros, disco da helice), por cima e sem
      // escrever profundidade
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glDepthMask(GL_FALSE);
      for (const glm::mat4 &M : parked)
        drawModel(plane152, prog, M, false, true);
      glUniform1f(glGetUniformLocation(prog, "clipZ"), clip);
      if (propRPM >= 1000)
        drawModel(propDisc, prog, Mplane, false, true);
      glUniform1f(glGetUniformLocation(prog, "reflectivity"), planeRefl);
      drawModel(plane, prog, Mplane, false, true, &planeLocal);
      glUniform1f(glGetUniformLocation(prog, "reflectivity"), 0.0f);
      glUniform1f(glGetUniformLocation(prog, "clipZ"), 1e9f);
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);

      // 3) pontos de luz: pista, cidades e luzes de navegacao do avião
      std::vector<float> glow;
      addRunwayLights(glow, ac.pos + glm::vec3(0, 1.6f, 0), now, nightMode);
      addNavLights(glow, Mplane, eye, now);
      drawGlowPoints(glow, P * V, fov, viewH, nightMode);
    };

    // ---- desenhar: vista principal ----
    glViewport(0, 0, fbWidth, fbHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, vpY, fbWidth, vpH);
    renderScene(V, P, eye, fov, vpH, cameraMode == CAM_COCKPIT);

    // ---- camara de aterragem (extra): a mesma cena, vista de outra camara,
    // num retangulo ---- glViewport diz onde se desenha; glScissor limita
    // tambem o glClear a esse retangulo.
    landCamOn =
        !menuOpen &&
        ((ac.status == FLYING && ac.pos.y < 35.0f && ac.pos.z > -700.0f) ||
         (ac.status == ON_GROUND && ac.V > 1.0f));
    if (landCamOn) {
      landW = (int)(fbWidth * 0.30f);
      landH = landW * 9 / 16;
      landX = fbWidth - landW - (int)(12 * uiScale);
      landY = (int)(140 * uiScale);       // y de cima (por baixo das mensagens)
      int glY = fbHeight - landY - landH; // o OpenGL conta a partir de baixo
      glm::vec3 camPos(32.0f, 1.8f,
                       AIM_Z + 60.0f); // ao lado da pista, junto ao toque
      glm::vec3 look = ac.pos + glm::vec3(0, 1.0f, 0);
      float d = glm::length(look - camPos);
      float lfov = glm::clamp(glm::degrees(2.0f * std::atan(7.0f / d)), 2.0f,
                              50.0f); // zoom automatico
      glm::mat4 V2 = glm::lookAt(camPos, look, glm::vec3(0, 1, 0));
      glm::mat4 P2 = glm::perspective(glm::radians(lfov), (float)landW / landH,
                                      1.0f, 10000.0f);
      glEnable(GL_SCISSOR_TEST);
      glViewport(landX, glY, landW, landH);
      glScissor(landX, glY, landW, landH);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      renderScene(V2, P2, camPos, lfov, landH, false);
      glDisable(GL_SCISSOR_TEST);
    }

    // ---- 2D por cima: painel ou menu ----
    glViewport(0, 0, fbWidth, fbHeight);
    uiScale = std::max(
        0.5f, fbHeight / 720.0f); // texto do mesmo tamanho em ecras Retina
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(overlayProg);
    glUniform2f(glGetUniformLocation(overlayProg, "screen"), (float)fbWidth,
                (float)fbHeight);
    if (menuOpen)
      drawMenu();
    else
      drawHUD();
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteProgram(prog);
  glfwTerminate();
  return 0;
}

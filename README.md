# Trabalho 2 de CG: simulação da aterragem (Exercício 2)

Tomé Pedro, 53702, PL2

## Compilar e correr
```
make run
```
Tem de ser corrido nesta pasta, porque o programa lê os modelos de `models/`.
No macOS precisa de `brew install glfw glew glm`.

## Ficheiros
- `FUNCIONALIDADES.txt`: lista de todas as funcionalidades (obrigatórias e extras): para que servem, como se usam e o que mostram
- `src/stb_easy_font.h`: fonte de texto de Sean Barrett (domínio público), usada no menu e no painel
- `src/main.cpp`: o programa todo (leitor de OBJ, geração da pista, shaders Phong, simulação, controlos)
- `models/c182/cessna182.obj` + `.mtl` + texturas: o avião que se pilota, Cessna 182 Skylane (modelo de VTX no Sketchfab, `.glb`), convertido para OBJ com `tools/converter_c182.py` (Blender): ficaram de fora as capas "Remove Before Flight", esferas auxiliares e malhas de colisão; origem nas rodas principais, nariz para +Z, em metros. As texturas foram reduzidas (no máximo 2048 px) e renomeadas, e o `.mtl` reescrito, com `tools/texturas_c182.py`
- `models/cessna152.obj` + `.mtl`: Cessna 152 (modelo low poly de dejunes, itch.io, convertido de FBX para OBJ), usado nos dois aviões estacionados na placa
- `models/runway.obj` + `.mtl`: a pista do professor (link do enunciado, `RUNWAY 8.blend`), exportada do Blender com `tools/exportar_pista.py` (números de texto convertidos em malha, modificadores aplicados, triangulada, Y para cima). A geometria não foi alterada; no `.mtl` só se corrigiram as cores. Na cena é colocada pela matriz `T·S(0,25)·Ry(180°)`: fica com 30 m × 1830 m e a soleira da "09" em z = 0
- `models/airport.obj` + `.mtl`: aeroporto gerado pelo programa (relva, bases das luzes, torre de controlo, hangar, terminal, placa de estacionamento, caminho de rolagem, manga de vento)
- `models/terrain.obj` + `.mtl`: terreno montanhoso gerado pelo programa (grelha de 200 × 200 células de 80 m)
- `models/city.obj` + `.mtl`: 6 povoações geradas pelo programa (casas, prédios, ruas, janelas)
- `models/skybox/*.jpg`: as 6 faces da skybox, geradas por `tools/gerar_skybox.py`
- `src/stb_image.h`: leitor de imagens de Sean Barrett (domínio público), usado para a skybox e para as texturas do avião

## Como funciona
O programa abre no menu, com os comandos e o jogo em pausa. Durante o jogo, o canto superior esquerdo mostra "ESC Menu" para voltares lá; o canto inferior esquerdo tem o painel de instrumentos.

Pilotas tu o Cessna 182. O avião começa estabilizado na rampa de 3°, a 1,55 km da pista 09 (a ~87 m de altura, antes de uma serra que tens de passar por cima), a 65 kt, com flaps a 30° e 35% de potência. Se não tocares em nada, segue a rampa sozinho até à pista. Para tocar suavemente tens de fazer o flare: perto do chão, tira a potência e levanta o nariz.

O movimento vem de um modelo de voo simples com dados do Cessna 182: massa de 1406 kg, asa de 16,2 m², 230 hp, sustentação máxima a ~54 kt sem flaps e ~48 kt com flaps 30 (não há perda: o CL fica limitado), flaps, atrito das rodas e travões.

No toque, a aterragem é avaliada:
| Resultado | Condição |
|---|---|
| Boa aterragem | descida mais suave do que 300 ft/min |
| Aceitável | entre 300 e 600 ft/min |
| Dura | entre 600 e 1000 ft/min |
| Acidente | mais de 1000 ft/min, asas inclinadas mais de 15° ou nariz mais de 5° em baixo |

Também avisa se tocares primeiro com a roda do nariz ou fora da pista.

O painel mostra: fase, velocidade (kt), altitude (ft), velocidade vertical (ft/min), pitch, inclinação, potência, flaps, desvio em relação à rampa e resultado.

## Extras
- **Skybox**: cubemap com céu, nuvens (ruído fractal) e sol na direção da luz pontual. O horizonte tem a cor do nevoeiro, para não se ver a junção com o chão.
- **Luzes da pista**: bordas (brancas, a cada 60 m), soleira (verde), fim da pista (vermelho), luzes de aproximação com flash sequencial e PAPI (4 luzes ao lado do ponto de mira: 2 brancas + 2 vermelhas = na rampa; 4 brancas = alto; 4 vermelhas = baixo). Vêem-se melhor de noite (tecla N).
- **Texturas** (avião): o leitor de OBJ lê também as coordenadas de textura (`vt`) e o `map_Kd` de cada material. Cada vértice no VBO passa a ter 8 floats (posição, normal, uv). A imagem é lida com stb_image (virada na vertical, porque no OpenGL v = 0 é em baixo), enviada com `glTexImage2D` e com mipmaps. No fragment shader, a cor difusa é `Kd × texture(tex, uv)`, usada no Phong como antes; o alfa da textura recorta letreiros (alpha test) e torna os vidros transparentes.
- **Vidros transparentes**: os materiais com `d < 1` (vidros dos aviões) são desenhados depois dos opacos, com blending e sem escrever no depth buffer, para se ver o interior da cabine.
- **Vista do cockpit** (tecla C até "Cockpit"): câmara nos olhos do piloto, que roda com o avião. Arrastar com o botão esquerdo permite olhar à volta. Em baixo há um painel com os 6 instrumentos clássicos, todos a mexer: velocímetro (0–200 kt, com os arcos branco, verde e amarelo do C182), horizonte artificial, altímetro, coordenador de curva, indicador de rumo e variómetro, além das barras de potência e flaps. A hélice é escondida nesta vista.
- **Edifícios do aeroporto**: torre de controlo (é dela que a câmara "Torre" olha), hangar, terminal com janelas que se acendem à noite, placa de estacionamento com dois Cessnas parados e a manga de vento.

- **Piloto** (gerado pelo programa, `models/pilot.obj`): boneco de elipsoides, troncos de cone e caixas, sentado no lugar da esquerda, com fato azul UBI, capacete branco com risca azul e viseira escura. Tem 3 objetos (linhas `o` no .obj), cada um com a sua matriz: o corpo vai com o avião, a cabeça roda à volta do pescoço (olha para dentro das curvas e tenta manter o horizonte) e os braços seguem o manche.
- **Peças que se mexem**: hélice (roda conforme a potência; acima de ~1000 rpm vê-se um disco semitransparente), flaps (descem e sobem devagar com F/G), ailerons, profundor, leme de direção e manche (seguem os comandos). Todas rodam à volta de um eixo que não passa pela origem: `local = T(p) · R(ângulo, eixo) · T(−p)`, e a matriz final é `M_avião · local`. O leitor de OBJ passou a agrupar por objeto + material, para cada peça ter um nome; os flaps e ailerons, que no modelo eram um só objeto, foram separados com `tools/separar_superficies.py`.
- **Luzes de navegação**: vermelha na asa esquerda, verde na direita e branca na cauda (cada uma só se vê no seu setor, calculado com `atan2` da direção para a câmara, em coordenadas do avião), farol vermelho a pulsar no leme, strobes brancos com flash duplo e faróis de aterragem. São os mesmos point sprites das luzes da pista, com a posição `M_avião · p`.
- **Câmara de aterragem**: perto do chão (abaixo de 35 m) e enquanto o avião rola, aparece no canto superior direito a mesma cena vista de uma câmara ao lado da pista, com zoom automático. Desenha-se a cena duas vezes por frame: a segunda com `glViewport` e `glScissor` limitados ao retângulo.
- **Reflexo do céu**: na pintura e nos vidros do avião, o fragment shader calcula `reflect(−V, N)` e lê a cubemap da skybox nessa direção, misturando-a com a cor (mais quando se vê de lado: Fresnel). Só reflete o céu, não a pista nem as montanhas.
- **Terreno montanhoso**: a altura h(x, z) vem de ruído fractal (fbm com 6 camadas de value noise). É plano no aeródromo e nas povoações; as montanhas começam a ~200 m da zona da pista e sobem até ~1300 m. A meio da aproximação há uma serra que a atravessa: por baixo da rampa de 3° o terreno fica sempre pelo menos 30 m abaixo da trajetória (um 'vale' com mínimo suave), mas para os lados sobe muito mais. A cor depende da altura e da inclinação: relva, relva escura, rocha e neve. As normais vêm das diferenças centrais de h. Embater no terreno conta como acidente.
- **Cidades**: 6 aglomerados a 0,6–1,5 km da pista , com ruas em cruz, prédios de 3 a 8 andares no centro e casas com telhado de duas águas na periferia, colocados com um gerador aleatório de semente fixa. À noite, ~35% das janelas acendem-se (luz própria no shader) e os candeeiros brilham.

## Controlos
| Tecla / rato | Ação |
|---|---|
| W / S ou ↑ / ↓ | nariz para baixo / para cima |
| A / D ou ← / → | inclinar para a esquerda / direita |
| Q / E | leme (no chão: roda do nariz) |
| + / - | mais / menos potência |
| F / G | descer / subir flaps (0-10-20-30) |
| B (manter) | travões |
| Arrastar com o botão direito | manche: cima/baixo = pitch, lados = inclinação |
| Arrastar com o botão esquerdo | rodar a câmara (no cockpit: olhar à volta) |
| Roda do rato | zoom |
| C | câmara: Perseguição / Torre (zoom automático) / Cockpit |
| N | dia / noite |
| R | recomeçar |
| Esc | menu (pausa): Enter continua, R recomeça, X sai |

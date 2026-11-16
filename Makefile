# Trabalho 2 de CG - Aterragem (Tome Pedro, 53702, PL2)
#   make       compila
#   make run   compila e corre (tem de correr nesta pasta por causa de models/)
#   make clean apaga o executavel

UNAME := $(shell uname -s)

SRC := src/main.cpp
EXE := bin/aterragem

ifeq ($(UNAME), Darwin)
    # macOS (bibliotecas instaladas com o Homebrew)
    CXX      := clang++
    CXXFLAGS := -std=c++17 -O2 -Wall -DGL_SILENCE_DEPRECATION \
                -I/opt/homebrew/include -I/usr/local/include
    LDFLAGS  := -L/opt/homebrew/lib -L/usr/local/lib -lglfw -lGLEW -framework OpenGL
else
    # Linux
    CXX      := g++
    CXXFLAGS := -std=c++17 -O2 -Wall
    LDFLAGS  := -lglfw -lGLEW -lGL
endif

all: $(EXE)

$(EXE): $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE) $(LDFLAGS)

run: $(EXE)
	./$(EXE)

clean:
	rm -rf bin

.PHONY: all run clean

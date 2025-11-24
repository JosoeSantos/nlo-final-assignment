# Documentação Completa - Biblioteca de Otimização Numérica (`otimo.py`)

Esta documentação apresenta uma descrição completa de todas as classes e métodos implementados na biblioteca de otimização numérica `otimo.py`.

## Índice

1. [Classes Base](#classes-base)
2. [Métodos de Otimização Unidimensional](#métodos-de-otimização-unidimensional)
3. [Métodos de Otimização Multidimensional Irrestrita](#métodos-de-otimização-multidimensional-irrestrita)
4. [Métodos de Otimização Restrita](#métodos-de-otimização-restrita)
5. [Funções Utilitárias](#funções-utilitárias)
6. [Exemplos de Uso](#exemplos-de-uso)

---

## Classes Base

### `Solution`

**Descrição**: Classe para armazenar e gerenciar os resultados de um problema de otimização.

**Atributos**:
- `x`: Ponto ótimo encontrado (ndarray)
- `fx`: Valor da função objetivo no ponto ótimo (float)
- `iter`: Número de iterações realizadas (int)
- `aval`: Número de avaliações da função objetivo (int)
- `xhist`: Histórico dos pontos visitados (ndarray)
- `fxhist`: Histórico dos valores da função objetivo (ndarray)

**Métodos**:

#### `__init__(x, fx=None, iter=None, aval=None, xhist=None, fxhist=None)`
Inicializa uma solução de otimização.

**Parâmetros**:
- `x`: Ponto ótimo encontrado
- `fx`: Valor da função objetivo (opcional)
- `iter`: Número de iterações (opcional)
- `aval`: Número de avaliações (opcional)
- `xhist`: Histórico de pontos (opcional)
- `fxhist`: Histórico de valores (opcional)

#### `resultados(func, xlim, ylim, levels=None)`
Plota os resultados da otimização com gráficos de contorno e convergência.

**Parâmetros**:
- `func`: Função objetivo para plotagem
- `xlim`: Limites do eixo x [min, max]
- `ylim`: Limites do eixo y [min, max]
- `levels`: Número de níveis de contorno (padrão: 30)

#### `__str__()`
Retorna uma representação textual da solução.

### `Unidimensional`

**Descrição**: Classe base abstrata para métodos de otimização unidimensional.

**Métodos**:
- `__init__()`: Inicialização básica
- `solve()`: Método abstrato para implementação por subclasses

### `Irrestrita`

**Descrição**: Classe base para métodos de otimização multidimensional irrestrita.

**Métodos**:
- `__init__(maxit, maxaval)`: Inicialização com limites de iterações e avaliações
- `resolva(func, x0)`: Método base que retorna configuração inicial

### `Restrita`

**Descrição**: Classe base para métodos de otimização restrita.

**Métodos**:
- `__init__(maxit, maxaval, precisao)`: Inicialização com parâmetros de convergência
- `resolva(func, x0, restricoes, tipo_restricoes)`: Configuração inicial para problemas restritos

---

## Métodos de Otimização Unidimensional

### `Eliminacao(Unidimensional)`

**Descrição**: Método de eliminação para encontrar um intervalo que contém o mínimo.

**Parâmetros**:
- `passo`: Passo inicial para busca (padrão: 1e-2)
- `aceleracao`: Fator de aceleração do passo (padrão: 1.5)
- `maxaval`: Número máximo de avaliações (padrão: 100)

**Método**:
- `resolva(func)`: Retorna (limite_inferior, limite_superior, num_avaliacoes)

### `Exaustiva(Eliminacao)`

**Descrição**: Busca exaustiva em intervalos divididos em pontos equidistantes.

**Parâmetros Adicionais**:
- `npontos`: Número de pontos para divisão do intervalo (padrão: 8)
- `precisao`: Precisão de convergência (padrão: 1e-3)

**Método**:
- `resolva(func)`: Retorna (ponto_otimo, num_avaliacoes)

### `Dicotomica(Eliminacao)`

**Descrição**: Método dicotômico para otimização unidimensional.

**Parâmetros Adicionais**:
- `precisao`: Precisão de convergência (padrão: 1e-3)

**Algoritmo**: Utiliza dois pontos intermediários para reduzir o intervalo de busca pela metade a cada iteração.

### `Bissecao(Eliminacao)`

**Descrição**: Método de bissecção com três pontos de avaliação.

**Parâmetros**: Herda de `Eliminacao` com adição de `precisao`

**Características**: Usa pontos u, v e c para dividir o intervalo em três partes e eliminar regiões.

### `Fibonacci(Eliminacao)`

**Descrição**: Método de Fibonacci para busca ótima.

**Parâmetros Adicionais**:
- `maxiter`: Número máximo de iterações (padrão: 100)
- `precisao`: Precisão de convergência (padrão: 1e-3)

**Características**: Utiliza a sequência de Fibonacci para determinar pontos de avaliação otimamente espaçados.

### `SecaoAurea(Eliminacao)`

**Descrição**: Método da seção áurea (golden section).

**Parâmetros**: Herda de `Eliminacao` com `precisao`

**Características**: Usa a razão áurea (0.618) para determinar pontos de busca, mantendo a propriedade de redução ótima do intervalo.

### `Interpolacao(Unidimensional)`

**Descrição**: Classe base para métodos baseados em interpolação.

**Parâmetros**:
- `precisao`: Precisão de convergência (padrão: 1e-3)
- `maxaval`: Número máximo de avaliações (padrão: 100)

### `Quadratica(Interpolacao)`

**Descrição**: Interpolação quadrática para encontrar o mínimo.

**Parâmetros Adicionais**:
- `passo`: Passo para pontos iniciais (padrão: 1e-5)

**Algoritmo**: Constrói um polinômio quadrático usando três pontos e encontra seu mínimo analiticamente.

### `QuasiNewton(Interpolacao)`

**Descrição**: Método quasi-Newton unidimensional.

**Parâmetros Adicionais**:
- `pertubacao`: Perturbação para cálculo de derivadas (padrão: 1e-8)
- `maxiter`: Número máximo de iterações (padrão: 200)

**Características**: Usa aproximações de primeira e segunda derivadas para método de Newton.

### `Secante(Interpolacao)`

**Descrição**: Método da secante para encontrar raízes da derivada.

**Parâmetros**: Herda de `Interpolacao` com `pertubacao`

**Algoritmo**: Aproxima a derivada e usa o método da secante para encontrar onde ela é zero.

---

## Métodos de Otimização Multidimensional Irrestrita

### `DirecoesAleatorias(Irrestrita)`

**Descrição**: Método de direções aleatórias para otimização.

**Parâmetros**:
- `unidimensional`: Método de busca unidimensional
- `maxit`: Número máximo de iterações (padrão: 10000)
- `maxaval`: Número máximo de avaliações (padrão: 10000)
- `precisao`: Precisão de convergência (padrão: 1e-3)

**Algoritmo**: 
1. Gera direções aleatórias
2. Verifica se a direção aponta para minimização
3. Executa busca unidimensional na direção escolhida

### `Gradiente(Irrestrita)`

**Descrição**: Método do gradiente (steepest descent).

**Parâmetros**:
- `unidimensional`: Método de busca unidimensional
- `diferenca`: Tipo de diferenciação ('progressiva', 'regressiva', 'central')
- `maxit`, `maxaval`, `precisao`: Critérios de parada

**Algoritmo**:
1. Calcula o gradiente da função
2. Define direção como -gradiente
3. Busca unidimensional para determinar passo ótimo
4. Atualiza posição

### `Newton(Irrestrita)`

**Descrição**: Método de Newton para otimização.

**Parâmetros**:
- `unidimensional`: Método de busca unidimensional
- `maxit`, `maxaval`, `precisao`: Critérios de parada

**Algoritmo**:
1. Calcula gradiente e matriz hessiana
2. Direção: d = -H⁻¹∇f
3. Busca unidimensional se necessário
4. Verifica singularidade da hessiana

### `DFP(Irrestrita)`

**Descrição**: Método quasi-Newton Davidon-Fletcher-Powell.

**Parâmetros**: Similar ao Newton

**Características**:
- Aproxima a inversa da hessiana usando fórmula DFP
- Atualização: Hₖ₊₁ = Hₖ + vvᵀ/(vᵀr) - HᵣᵣᵀH/(rᵀHr)
- Mantém propriedade de definida positiva

### `BFGS(Irrestrita)`

**Descrição**: Método quasi-Newton Broyden-Fletcher-Goldfarb-Shanno.

**Parâmetros**: Similar ao Newton

**Características**:
- Atualização BFGS da aproximação da hessiana inversa
- Geralmente mais robusto que DFP
- Fórmula: Hₖ₊₁ = Hₖ + (1 + rᵀHr/rᵀv)vvᵀ/(vᵀr) - (vᵀH + Hrᵀv)/(rᵀv)

### `QuasiNewton(Irrestrita)`

**Descrição**: Implementação genérica de quasi-Newton.

**Parâmetros Adicionais**:
- `qsi`: Parâmetro de interpolação entre DFP e BFGS (padrão: 0.5)

**Características**:
- Combina atualizações DFP e BFGS: H = (1-ξ)H_DFP + ξH_BFGS
- Permite ajuste entre os métodos

### `GradienteConjugado(Irrestrita)`

**Descrição**: Método do gradiente conjugado.

**Parâmetros**:
- `unidimensional`: Método de busca unidimensional
- `formula`: Fórmula para β ('fletcher-reeves' ou 'polak-ribiere')
- `maxit`, `maxaval`, `precisao`: Critérios de parada

**Algoritmo**:
1. Direção inicial: d₀ = -∇f₀
2. Busca unidimensional: αₖ
3. Atualização: xₖ₊₁ = xₖ + αₖdₖ
4. Novo gradiente: ∇fₖ₊₁
5. Parâmetro β: Fletcher-Reeves ou Polak-Ribière
6. Nova direção: dₖ₊₁ = -∇fₖ₊₁ + βₖdₖ

### `HookeJeeves(Irrestrita)`

**Descrição**: Método de busca padrão Hooke-Jeeves.

**Parâmetros**:
- `lamb`: Passo em cada coordenada (padrão: 0.1)
- `alpha`: Aceleração na direção de busca (padrão: 1.0)
- `maxit`, `maxaval`, `precisao`: Critérios de parada

**Algoritmo**:
1. Busca exploratória em cada coordenada
2. Se melhoria encontrada, busca na direção do movimento
3. Reduz passo se não houver melhoria

### `NelderMeadSimplex(Irrestrita)`

**Descrição**: Método do simplex de Nelder-Mead.

**Parâmetros**:
- `reflexao`: Coeficiente de reflexão (padrão: 1.0)
- `contracao`: Coeficiente de contração (padrão: 0.5)
- `expansao`: Coeficiente de expansão (padrão: 2.0)
- `encolhimento`: Coeficiente de encolhimento (padrão: 0.5)

**Algoritmo**:
1. Constrói simplex inicial com n+1 pontos
2. Operações: reflexão, expansão, contração, encolhimento
3. Não requer cálculo de derivadas
4. Adequado para funções não-diferenciáveis

---

## Métodos de Otimização Restrita

### `PenalidadeInterior(Restrita)`

**Descrição**: Método de penalidade interior (barreira).

**Características**:
- Para restrições de desigualdade g(x) ≥ 0 ou g(x) ≤ 0
- Função penalizada: F(x,μ) = f(x) - μ∑1/(gᵢ(x))
- Reduz μ a cada iteração
- Ponto inicial deve ser factível

### `PenalidadeExterior(Restrita)`

**Descrição**: Método de penalidade exterior.

**Parâmetros**:
- `maxit`: Iterações máximas
- `maxaval`: Avaliações máximas  
- `precisao`: Precisão de convergência

**Método**:
```python
resolva(func, x0, restricoes, tipo_restricoes, irrestrito, 
        penalidade=1.0, aceleracao=1.5, print=True)
```

**Características**:
- Para restrições g(x) ≥ 0 ou g(x) ≤ 0
- Para restrições h(x) = 0
- Função penalizada: F(x,ρ) = f(x) + ρ∑max(0, gᵢ(x))² + ρ∑(hᵢ(x))²
- Aumenta ρ a cada iteração
- Ponto inicial pode ser infactível

### `LagrangeanoAumentado(Restrita)`

**Descrição**: Método do Lagrangeano Aumentado.

**Parâmetros**: Similar à penalidade exterior

**Método**:
```python
resolva(func, x0, restricoes, tipo_restricoes, irrestrito,
        penalidade=1.0, aceleracao=1.2, print=True)
```

**Características**:
- Combina multiplicadores de Lagrange com penalidade
- Melhor convergência que penalidade pura

---

## Funções Utilitárias

### `gradiente(x, func, fx=None, metodo='progressiva', delta=1e-10)`

**Descrição**: Calcula o gradiente numérico de uma função.

**Parâmetros**:
- `x`: Ponto de avaliação
- `func`: Função objetivo
- `fx`: Valor da função em x (opcional, evita recálculo)
- `metodo`: Tipo de diferenciação
  - `'progressiva'`: f'(x) ≈ (f(x+h) - f(x))/h
  - `'regressiva'`: f'(x) ≈ (f(x) - f(x-h))/h  
  - `'central'`: f'(x) ≈ (f(x+h) - f(x-h))/(2h)
- `delta`: Passo para diferenciação

**Retorna**: (gradiente, num_avaliacoes)

### `hessiana(x, func, fx=None, grad=None, delta=1e-10)`

**Descrição**: Calcula a matriz hessiana numérica.

**Parâmetros**:
- `x`: Ponto de avaliação
- `func`: Função objetivo
- `fx`: Valor da função (opcional)
- `grad`: Gradiente (opcional) 
- `delta`: Passo para diferenciação

**Características**:
- Usa diferenças centrais para segundas derivadas
- Hᵢⱼ ≈ (f(x+hᵢ+hⱼ) - f(x+hᵢ-hⱼ) - f(x-hᵢ+hⱼ) + f(x-hᵢ-hⱼ))/(4hᵢhⱼ)
- Simétrica por construção

**Retorna**: (hessiana, num_avaliacoes)

### `avalia_funcao(func, x)`

**Descrição**: Wrapper para avaliação consistente de funções.

**Parâmetros**:
- `func`: Função a ser avaliada
- `x`: Ponto de avaliação

**Características**:
- Garante formato consistente dos dados
- Trata casos especiais de entrada
- Facilita debugging e instrumentação

---

## Exemplos de Uso

### 1. Otimização Unidimensional

```python
import numpy as np
from otimo import SecaoAurea

# Função objetivo
def f(x):
    return x**2 - 4*x + 3

# Método de otimização
metodo = SecaoAurea(precisao=1e-6)

# Resolução
x_otimo, avaliacoes = metodo.resolva(f)
print(f"Ótimo: {x_otimo}, Avaliações: {avaliacoes}")
```

### 2. Otimização Multidimensional Irrestrita

```python
from otimo import Gradiente, SecaoAurea, Solution

# Função de Rosenbrock
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# Configuração
busca_1d = SecaoAurea(precisao=1e-6)
otimizador = Gradiente(busca_1d, precisao=1e-6)

# Resolução
solucao = otimizador.resolva(rosenbrock, np.array([0.0, 0.0]))

print(f"Ponto ótimo: {solucao.x}")
print(f"Valor ótimo: {solucao.fx}")
print(f"Iterações: {solucao.iter}")

# Plotagem
solucao.resultados(rosenbrock, [-2, 2], [-1, 3])
```

### 3. Otimização com Restrições

```python
from otimo import PenalidadeExterior, GradienteConjugado, SecaoAurea

# Problema: min f(x) = x₁² + x₂²
#          s.a. x₁ + x₂ ≥ 1
#               x₁, x₂ ≥ 0

def objetivo(x):
    return x[0]**2 + x[1]**2

def restricao1(x):
    return x[0] + x[1] - 1  # x₁ + x₂ ≥ 1

def restricao2(x):
    return x[0]  # x₁ ≥ 0

def restricao3(x):
    return x[1]  # x₂ ≥ 0

# Configuração
restricoes = [restricao1, restricao2, restricao3]
tipos = np.array(['>', '>', '>'])
x0 = np.array([1.0, 1.0])

busca_1d = SecaoAurea(precisao=1e-6)
irrestrito = GradienteConjugado(busca_1d, precisao=1e-6)
restrito = PenalidadeExterior(precisao=1e-6)

# Resolução
solucao = restrito.resolva(objetivo, x0, restricoes, tipos, 
                          irrestrito, penalidade=1.0, aceleracao=2.0)

print(f"Solução: {solucao.x}")
print(f"Valor objetivo: {solucao.fx}")
```

### 4. Comparação de Métodos

```python
import time
from otimo import *

def sphere(x):
    return np.sum(x**2)

x0 = np.array([5.0, 5.0])
busca_1d = SecaoAurea(precisao=1e-6)

# Métodos a comparar
metodos = {
    'Gradiente': Gradiente(busca_1d),
    'Newton': Newton(busca_1d),
    'BFGS': BFGS(busca_1d),
    'Nelder-Mead': NelderMeadSimplex()
}

for nome, metodo in metodos.items():
    inicio = time.time()
    solucao = metodo.resolva(sphere, x0.copy())
    tempo = time.time() - inicio
    
    print(f"\n{nome}:")
    print(f"  Solução: {solucao.x}")
    print(f"  Valor: {solucao.fx:.2e}")
    print(f"  Iterações: {solucao.iter}")
    print(f"  Avaliações: {solucao.aval}")
    print(f"  Tempo: {tempo:.3f}s")
```

### 5. Análise de Convergência

```python
import matplotlib.pyplot as plt

# Função de teste
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

x0 = np.array([0.0, 0.0])
busca_1d = SecaoAurea(precisao=1e-8)
metodo = BFGS(busca_1d, precisao=1e-8)

solucao = metodo.resolva(himmelblau, x0)

# Análise de convergência
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.semilogy(solucao.fxhist)
plt.xlabel('Iteração')
plt.ylabel('Valor da Função')
plt.title('Convergência do Valor da Função')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(solucao.xhist[:, 0], solucao.xhist[:, 1], 'o-')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Trajetória de Convergência')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Observações sobre Implementação

### Critérios de Convergência

A biblioteca implementa múltiplos critérios de parada:

1. **Número máximo de iterações** (`maxit`)
2. **Número máximo de avaliações** (`maxaval`)
3. **Precisão do gradiente**: ‖∇f‖ < ε
4. **Precisão do ponto**: ‖xₖ₊₁ - xₖ‖ < ε
5. **Precisão da função**: |fₖ₊₁ - fₖ| < ε

### Busca Unidimensional

Todos os métodos de gradiente requerem um método de busca unidimensional. Os disponíveis são:

- **SecaoAurea**: Mais robusto, recomendado para uso geral
- **Quadratica**: Rápido quando a função é bem comportada
- **Fibonacci**: Ótimo teoricamente, mas pode ser lento
- **Dicotomica**: Simples e confiável

### Métodos Quasi-Newton

- **DFP**: Histórico, pode perder definida positiva
- **BFGS**: Mais robusto, recomendado
- **QuasiNewton**: Combina DFP e BFGS, experimental

### Métodos Livres de Derivada

Para funções não-diferenciáveis:
- **NelderMeadSimplex**: Mais robusto
- **HookeJeeves**: Mais simples
- **DirecoesAleatorias**: Para exploração global

### Escolha do Método de Restrições

- **PenalidadeInterior**: Quando x₀ é factível e restrições são g(x) ≥ 0
- **PenalidadeExterior**: Quando x₀ pode ser infactível
- **LagrangeanoAumentado**: Melhor convergência, mais complexo

### Dicas de Performance

1. **Use gradiente central** para funções suaves
2. **SecaoAurea com precisão adequada** para busca 1D
3. **BFGS** para problemas de médio porte
4. **GradienteConjugado** para problemas grandes
5. **Ajuste parâmetros de penalidade** experimentalmente

Esta documentação fornece uma base completa para uso efetivo da biblioteca de otimização numérica, cobrindo desde conceitos básicos até implementações avançadas.

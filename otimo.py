"""
Biblioteca de Otimização Numérica
==================================

Esta biblioteca implementa diversos algoritmos de otimização para problemas
com e sem restrições, incluindo métodos de busca unidimensional, métodos
de gradiente, métodos quasi-Newton e métodos de penalidade.

Módulos principais:
- Otimização Unidimensional: Eliminação, Exaustiva, Dicotômica, Bissecção, 
  Fibonacci, Seção Áurea, Interpolação Quadrática, Quasi-Newton, Secante
- Otimização Irrestrita: Direções Aleatórias, Gradiente, Newton, BFGS, DFP, 
  Quasi-Newton, Gradiente Conjugado, Hooke-Jeeves, Nelder-Mead Simplex
- Otimização Restrita: Penalidade Interior, Penalidade Exterior, 
  Lagrangeano Aumentado

Funcionalidades:
- Plotagem de resultados com trajetórias de convergência
- Análise numérica de gradientes e hessianas
- Controle de critérios de parada e tolerâncias
- Histórico completo de iterações

Exemplo de uso:
    >>> import numpy as np
    >>> from otimo import Gradiente, SecaoAurea, Solution
    >>> 
    >>> # Definir função objetivo
    >>> def rosenbrock(x):
    ...     return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    >>> 
    >>> # Configurar algoritmo
    >>> otimizador = Gradiente(SecaoAurea(precisao=1e-6))
    >>> 
    >>> # Resolver
    >>> solucao = otimizador.resolva(rosenbrock, [0, 0])
    >>> print(solucao)

Autor: André
Data: 2025
Versão: 1.0
"""

from typing import Callable, List, Union, Optional
import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from numpy.linalg import inv, norm

class Solution:
    """
    Classe para armazenar a solução de um problema de otimização.
    
    Esta classe encapsula todos os resultados de um processo de otimização,
    incluindo a solução ótima, valor da função objetivo, histórico de
    convergência e métricas de performance.
    
    Attributes:
        x (ndarray): Ponto ótimo encontrado
        fx (float): Valor da função objetivo no ponto ótimo
        iter (int): Número de iterações realizadas
        aval (int): Número de avaliações da função objetivo
        xhist (ndarray): Histórico dos pontos visitados
        fxhist (ndarray): Histórico dos valores da função objetivo
        criterio_parada (str): Critério de parada ativado na otimização
    """
    
    def __init__(self, x: ndarray | None, fx: float| None = None,
                 iter: int | None = None, aval: int | None = None,
                 xhist: ndarray | None = None, fxhist: ndarray | None = None,
                 criterio_parada: str | None = None):
        """
        Inicializa uma solução de otimização.
        
        Args:
            x: Ponto ótimo encontrado
            fx: Valor da função objetivo no ponto ótimo
            iter: Número de iterações realizadas
            aval: Número de avaliações da função objetivo
            xhist: Histórico dos pontos visitados durante a otimização
            fxhist: Histórico dos valores da função objetivo
        """
        if x is not None:
            self.x = x.reshape(x.size)
        else:
            self.x = None
        self.fx = fx
        self.iter = iter
        self.aval = aval
        self.xhist = xhist
        self.fxhist = fxhist
        self.criterio_parada = None

    def resultados(self, func: Callable, xlim: List[float],
                   ylim: List[float], levels: int | None = None):
        """
        Plota os resultados da otimização.
        
        Gera gráficos de contorno da função objetivo com a trajetória
        de convergência e um gráfico da evolução do valor da função.
        
        Args:
            func: Função objetivo a ser plotada
            xlim: Limites do eixo x [min, max]
            ylim: Limites do eixo y [min, max]
            levels: Número de níveis de contorno (padrão: 30)
        """
    
        # Malha para plotar contorno
        x1, x2 = np.meshgrid(np.linspace(xlim[0], xlim[1]),
                             np.linspace(ylim[0], ylim[1]))

        # Avalia função para plotar contorno
        f = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                f[i, j] = avalia_funcao(func, np.array([x1[i, j], x2[i, j]]))

        # Plota trajetória
        _, axis = plt.subplots(ncols=2, figsize=[2*6.4, 4.8])
        if levels is not None:
            axis[0].contour(x1, x2, f, levels=levels)
        else:
            axis[0].contour(x1, x2, f, levels=30)
            axis[0].plot(self.xhist[:, 0], self.xhist[:, 1], '--*r')
            axis[0].set_xlabel(r'$x_1$')
            axis[0].set_ylabel(r'$x_2$')
            axis[0].set_title('Problema')
            axis[0].grid()

        # Plota convergencia
        axis[1].plot(self.fxhist, '--*')
        axis[1].set_xlabel('Iterações')
        axis[1].set_ylabel(r'$f(\mathbf{x})$')
        axis[1].set_title('Convergência')
        axis[1].grid()

        plt.tight_layout()
        plt.show()
    
    def __str__(self):
        """
        Representação em string da solução.
        
        Returns:
            str: Resumo da solução com ponto ótimo, iterações e avaliações
        """
        mensagem = ''
        mensagem += 'Solução ótima: ' + str(self.x) + '\n'
        mensagem += 'Número de iterações: %d\n' % self.iter
        mensagem += 'Número de avaliações: %d' % self.aval
        if self.fx is not None:
            mensagem += '\nValor da função objetivo: %.4f' % self.fx
        if self.criterio_parada is not None:
            mensagem += '\nCritério de parada: ' + self.criterio_parada
        return mensagem

class Unidimensional:
    """
    Classe base para métodos de otimização unidimensional.
    
    Esta classe serve como base para todos os algoritmos de otimização
    unidimensional, definindo a interface comum que deve ser implementada
    pelas subclasses.
    """
    
    def __init__(self):
        """Inicializa o otimizador unidimensional."""
        pass
        
    def solve(self):
        """
        Método placeholder para resolução do problema.
        
        Deve ser sobrescrito pelas subclasses.
        """
        pass
    
class Eliminacao(Unidimensional):
    """
    Método de eliminação para otimização unidimensional.
    
    Este método encontra um intervalo que contém o mínimo da função
    através de busca com passo crescente.
    
    Attributes:
        passo (float): Passo inicial para a busca
        aceleracao (float): Fator de aceleração do passo
        maxaval (int): Número máximo de avaliações da função
    """

    def __init__(self, passo: float = 1e-2, aceleracao: float = 1.5,
                 maxaval: int = 100):
        """
        Inicializa o método de eliminação.
        
        Args:
            passo: Passo inicial para a busca (padrão: 1e-2)
            aceleracao: Fator de aceleração do passo (padrão: 1.5)
            maxaval: Número máximo de avaliações da função (padrão: 100)
        """
        self.passo = passo
        self.aceleracao = aceleracao
        self.maxaval = maxaval

    def resolva(self, func: Callable) -> tuple[float, float, int]:
        """
        Encontra um intervalo que contém o mínimo da função.
        
        Args:
            func: Função objetivo a ser minimizada
            
        Returns:
            tuple: (limite_inferior, limite_superior, num_avaliacoes)
        """
        # Intervalo inicial
        passo = self.passo
        a = 0
        b = passo
        
        # Avaliação do intervalo inicial
        fa = avalia_funcao(func, a)
        fb = avalia_funcao(func, b)
        avaliacoes = 2
        
        # Marca o limite inferior anterior
        anterior = 0
        
        # Enquanto eu estiver descendo
        while fb < fa:
            
            # Salvo o valor do limite inferior anterior
            anterior = a
            
            # Novo limite inferior é o atual limite máximo
            a = b
            fa = fb
            
            # Acelero o passo
            passo = passo*self.aceleracao
            
            # Dou um passo a frente no intervalo
            b += passo
            fb = avalia_funcao(func, b)
            avaliacoes += 1
                    
        # Retorno o intervalo final
        return anterior, b, avaliacoes

class Exaustiva(Eliminacao):
    """
    Classe que implementa o método de busca exaustiva para otimização.
    Esta classe herda de Eliminacao e utiliza uma estratégia de busca exaustiva
    que divide iterativamente o intervalo em pontos equidistantes para encontrar
    o mínimo de uma função.
    Attributes:
        npontos (int): Número de pontos equidistantes utilizados na divisão do intervalo
        precisao (float): Critério de parada baseado no tamanho do intervalo
        passo (float): Passo inicial para o método de eliminação (herdado)
        aceleracao (float): Fator de aceleração (herdado)
        maxaval (int): Número máximo de avaliações da função permitidas
    Methods:
        resolva(func): Encontra o mínimo da função utilizando busca exaustiva
            Args:
                func (Callable): Função a ser minimizada
            Returns:
                tuple[float, int]: Tupla contendo a aproximação do ponto de mínimo
                                 e o número total de avaliações da função realizadas
            O método primeiro utiliza o algoritmo de eliminação da classe pai para
            estabelecer um intervalo inicial [a,b], depois refina este intervalo
            iterativamente dividindo-o em npontos equidistantes, avaliando a função
            em cada ponto e selecionando o subintervalo que contém o menor valor.
            O processo continua até que o tamanho do intervalo seja menor que a
            precisão especificada ou o número máximo de avaliações seja atingido.
    """

    def __init__(self, npontos: int = 8, precisao: float = 1e-3, 
                 passo: float = 0.01, aceleracao: float = 1.5,
                 maxaval: int = 100):
        """
        Inicializa o método de busca exaustiva.

        Args:
            npontos (int): Número de pontos equidistantes para dividir o intervalo (padrão: 8)
            precisao (float): Critério de parada baseado no tamanho do intervalo (padrão: 1e-3)
            passo (float): Passo inicial para o método de eliminação (padrão: 0.01)
            aceleracao (float): Fator de aceleração do passo (padrão: 1.5)
            maxaval (int): Número máximo de avaliações da função permitidas (padrão: 100)
        """
        super().__init__(passo, aceleracao, maxaval=maxaval)
        self.precisao = precisao
        self.npontos = npontos

    def resolva(self, func: Callable) -> tuple[float, int]:
        """
        Resolve o problema de otimização usando busca uniforme por intervalos.
        Este método implementa um algoritmo de otimização que divide iterativamente
        o intervalo de busca em pontos equidistantes, avalia a função objetivo em
        cada ponto e reduz o intervalo mantendo apenas a região ao redor do ponto
        de menor valor encontrado.
    
        Args:
            func (Callable): Função objetivo a ser minimizada. Deve aceitar um
                            argumento numérico e retornar um valor numérico.
        Returns:
            tuple[float, int]: Tupla contendo:
                - float: Aproximação do ponto ótimo (mínimo da função)
                - int: Número total de avaliações da função realizadas
        Raises:
            Pode propagar exceções da função objetivo ou dos métodos auxiliares.
        Notes:
            - O algoritmo continua até que o intervalo seja menor que self.precisao
              ou o número de avaliações exceda self.maxaval
            - Utiliza self.npontos para definir quantos pontos equidistantes são
              avaliados em cada iteração
            - A cada iteração, o intervalo é reduzido mantendo três pontos: o ponto
              de mínimo e seus vizinhos imediatos
            - O resultado final é o ponto médio do intervalo final
        """

        a, b, avaliacoes = super().resolva(func)  
        fa = avalia_funcao(func, a)
        fb = avalia_funcao(func, b)
        avaliacoes += 2
        
        # Enquanto o meu intervalo não for reduzido
        # a um tamanho suficientemente pequeno
        while (b-a) > self.precisao and avaliacoes < self.maxaval:
            
            # Divide o intervalo em npontos equidistantes
            x = np.linspace(a, b, self.npontos)
            
            # Vetor que vai carregar a avaliação de cada ponto
            fx = np.zeros(x.size)
            
            # O primeiro e o último pontos são a e b
            fx[0] = fa
            fx[-1] = fb
            
            # Avalia todos os outros pontos
            for n in range(1, self.npontos-1):
                fx[n] = avalia_funcao(func, x[n])
                avaliacoes += 1
                
            # Acha quem é o menor
            n = np.argmin(fx)
            
            # O novo limite inferior é o ponto anterior do menor
            a = x[n-1]
            fa = fx[n-1]

            # O novo limite superior é o ponto anterior do menor
            b = x[n+1]
            fb = fx[n+1]
                
        # A aproximação do meu ótimo é o meio do meu intervalo
        return (a+b)/2, avaliacoes

class Dicotomica(Eliminacao):
    """
    Implementa o método de busca dicotômica para otimização unidimensional.
    A busca dicotômica é um algoritmo de otimização que reduz iterativamente o intervalo
    de busca através da divisão do intervalo em segmentos e comparação dos valores da
    função em pontos estratégicos. Este método é eficiente para encontrar o mínimo de
    funções unimodais.
    O algoritmo funciona dividindo o intervalo atual em duas partes através de dois
    pontos intermediários calculados com base em um delta de 25% do comprimento do
    intervalo. Com base na comparação dos valores da função nesses pontos, uma metade
    do intervalo é eliminada a cada iteração.
    Attributes:
        precisao (float): Critério de parada baseado no tamanho do intervalo
        passo (float): Passo inicial herdado da classe pai Eliminacao
        aceleracao (float): Fator de aceleração do passo herdado da classe pai
        maxaval (int): Número máximo de avaliações da função permitidas
    Methods:
        resolva(func): Executa o algoritmo de busca dicotômica para encontrar o ótimo
    Example:
        >>> dicotomica = Dicotomica(precisao=1e-4, passo=0.1)
        >>> otimo, avaliacoes = dicotomica.resolva(lambda x: x**2 - 4*x + 3)
        >>> print(f"Ótimo encontrado: {otimo}, Avaliações: {avaliacoes}")
        Esta classe herda de Eliminacao e utiliza o método de eliminação para
        estabelecer o intervalo inicial de busca antes de aplicar a busca dicotômica.
    """

    def __init__(self, precisao: float = 1e-3, passo: float = 0.01, 
                 aceleracao: float = 1.5, maxaval: int = 100):
        """
        Inicializa o método de busca dicotômica.

        Args:
            precisao (float): Critério de parada baseado no tamanho do intervalo (padrão: 1e-3)
            passo (float): Passo inicial para o método de eliminação (padrão: 0.01)
            aceleracao (float): Fator de aceleração do passo (padrão: 1.5)
            maxaval (int): Número máximo de avaliações da função permitidas (padrão: 100)
        """
        super().__init__(passo=passo, aceleracao=aceleracao, maxaval=maxaval)
        self.precisao = precisao

    def resolva(self, func: Callable) -> tuple[float, int]:
        """
        Resolve um problema de otimização unidimensional usando o método de busca uniforme.
        Este método implementa um algoritmo de busca uniforme que reduz iterativamente
        o intervalo de busca dividindo-o em segmentos e comparando os valores da função
        em pontos estratégicos.
        Args:
            func (Callable): Função objetivo a ser otimizada. Deve aceitar um valor
                            numérico como entrada e retornar um valor numérico.
        Returns:
            tuple[float, int]: Uma tupla contendo:
                - float: Aproximação do ponto ótimo (meio do intervalo final)
                - int: Número total de avaliações da função realizadas
        Note:
            O algoritmo continua até que o tamanho do intervalo seja menor que
            self.precisao ou até atingir o número máximo de avaliações (self.maxaval).
            A cada iteração, dois pontos intermediários são calculados usando um
            delta de 25% do comprimento do intervalo atual.
        """
        a, b, avaliacoes = super().resolva(func)        

        # Enquanto o meu intervalo não for reduzido
        # a um tamanho suficientemente pequeno
        while (b-a) > self.precisao and avaliacoes < self.maxaval:
            
            # Calcula o comprimento do intervalo
            L = b-a
            
            # Calcula o valor de delta
            delta = .25*L
            
            # Determina os dois pontos intermediarios
            u = a + L/2-delta/2
            v = a + L/2+delta/2
            fu = avalia_funcao(func, u)
            fv = avalia_funcao(func, v)
            avaliacoes += 2
            
            # Se fv for maior que fu, então v é o novo b
            if fu < fv:
                b = v
                fb = fv
                
            # Se fu for maior que fv, então u é o novo a
            else:
                a = u
                fa = fu
        
        # A aproximação do meu ótimo é o meio do meu intervalo
        return (a+b)/2, avaliacoes

class Bissecao(Eliminacao):
    """
    Implementa o método de busca por bisseção (busca ternária) para otimização unidimensional.
    Esta classe herda da classe Eliminacao e implementa o algoritmo de busca ternária,
    que é um método eficiente para encontrar o mínimo de funções unimodais. O algoritmo
    funciona dividindo iterativamente o intervalo de busca em três partes iguais e
    eliminando um terço do intervalo a cada iteração com base na comparação dos valores
    da função em pontos estratégicos.
    Attributes:
        precisao (float): Critério de parada baseado no tamanho do intervalo
        passo (float): Passo inicial para o método de eliminação (herdado)
        aceleracao (float): Fator de aceleração do passo (herdado)
        maxaval (int): Número máximo de avaliações da função permitidas (herdado)
    Methods:
        resolva(func): Executa o algoritmo de busca ternária para encontrar o mínimo
    Example:
        >>> bissecao = Bissecao(precisao=1e-4, maxaval=50)
        >>> def funcao_teste(x):
        ...     return (x - 2)**2
        >>> minimo, num_avaliacoes = bissecao.resolva(funcao_teste)
        >>> print(f"Mínimo encontrado: {minimo}, Avaliações: {num_avaliacoes}")
        - A função objetivo deve ser unimodal no intervalo de busca
        - O método garante convergência para funções contínuas e unimodais
        - A precisão final depende do critério de parada especificado
        - O número de avaliações é limitado pelo parâmetro maxaval
    """

    def __init__(self, precisao: float = 1e-3, passo: float = 0.01, 
                 aceleracao: float = 1.5, maxaval: int = 100):
        """
        Inicializa o método de busca por bisseção.

        Args:
            precisao (float): Critério de parada baseado no tamanho do intervalo (padrão: 1e-3)
            passo (float): Passo inicial para o método de eliminação (padrão: 0.01)
            aceleracao (float): Fator de aceleração do passo (padrão: 1.5)
            maxaval (int): Número máximo de avaliações da função permitidas (padrão: 100)
        """
        super().__init__(passo=passo, aceleracao=aceleracao, maxaval=maxaval)
        self.precisao = precisao

    def resolva(self, func: Callable) -> tuple[float, int]:
        """
        Resolve o problema de otimização usando o método de busca ternária.
        Este método implementa o algoritmo de busca ternária para encontrar o mínimo
        de uma função unimodal em um intervalo. O algoritmo divide iterativamente
        o intervalo em três partes e elimina um terço do intervalo a cada iteração
        com base na comparação dos valores da função em pontos estratégicos.

        Args:
            func (Callable): Função objetivo a ser minimizada. Deve ser uma função
                            unimodal (ter apenas um mínimo) no intervalo de busca.
        Returns:
            tuple[float, int]: Uma tupla contendo:
                - float: Aproximação da posição do mínimo da função
                - int: Número total de avaliações da função realizadas
        Notes:
            - O método continua iterando enquanto o tamanho do intervalo for maior
              que a precisão especificada e o número de avaliações não exceder
              o máximo permitido
            - A cada iteração, dois pontos internos são avaliados (u e v) além
              do ponto central (c)
            - Dependendo dos valores da função nestes pontos, um terço do intervalo
              é eliminado
            - A solução final é retornada como o ponto médio do intervalo final
        """
        a, b, avaliacoes = super().resolva(func)        

        # Calcula e avalia o ponto no meio do intervalo
        c = (a+b)/2
        fc = avalia_funcao(func, c)
        avaliacoes += 1

        # Enquanto o meu intervalo não for reduzido
        # a um tamanho suficientemente pequeno
        while (b-a) > self.precisao and avaliacoes < self.maxaval:        
            
            # Calcula o comprimento do intervalo
            L = b-a
            
            # Ponto médio entre a e c
            u = a + L/4
            
            # Ponto médio entre c e b
            v = a + L*3/4
            
            # Avaliações
            fu = avalia_funcao(func, u)
            fv = avalia_funcao(func, v)
            avaliacoes += 2
            
            # Se fu é o menor, então excluímos o intervalo (c, b)
            if fu < fc and fc < fv:
                b, fb = c, fc
                c, fc = u, fu

            # Se fv é o menor, então excluímos o intervalo (a, c)
            elif fu > fc and fc > fv:
                a, fa = c, fc
                c, fc = v, fv
            
            # Se fc é o menor, então excluímos os intervalos (a, u) e (v, c)
            elif fu > fc and fv > fc:
                a, fa = u, fu
                b, fb = v, fv
            
        # A aproximação do meu ótimo é o meio do meu intervalo
        return (a+b)/2, avaliacoes

class Fibonacci(Eliminacao):
    """
    Implementa o método de busca de Fibonacci para otimização unidimensional.
    O método de Fibonacci é uma técnica de eliminação por intervalos que utiliza
    a sequência de Fibonacci para determinar os pontos de avaliação, garantindo
    uma redução ótima do intervalo de busca a cada iteração.
    Attributes:
        precisao (float): Tolerância para o critério de parada baseado no 
            tamanho do intervalo.
        maxiter (int): Número máximo de iterações permitidas.
    Methods:
        resolva(func): Encontra o mínimo da função no intervalo determinado
            pelo método de eliminação da classe pai.
    Notes:
        - Herda de Eliminacao para determinar o intervalo inicial de busca
        - Utiliza a propriedade da sequência de Fibonacci para posicionamento
          ótimo dos pontos de avaliação
        - Garante convergência com número pré-determinado de avaliações da função
        - Ideal para funções unimodais onde se deseja minimizar o número de
          avaliações da função objetivo
    Example:
        >>> fibonacci = Fibonacci(maxiter=50, precisao=1e-4)
        >>> resultado, num_avaliacoes = fibonacci.resolva(lambda x: x**2)

    """

    def __init__(self, maxiter: int = 100, precisao: float = 1e-3, 
                 passo: float = 0.01, aceleracao: float = 1.5,
                 maxaval: int = 100):
        """
        Inicializa o método de busca de Fibonacci.

        Args:
            maxiter (int): Número máximo de iterações permitidas (padrão: 100)
            precisao (float): Tolerância para o critério de parada (padrão: 1e-3)
            passo (float): Passo inicial para o método de eliminação (padrão: 0.01)
            aceleracao (float): Fator de aceleração do passo (padrão: 1.5)
            maxaval (int): Número máximo de avaliações da função permitidas (padrão: 100)
        """
        super().__init__(passo=passo, aceleracao=aceleracao, maxaval=maxaval)
        self.precisao = precisao
        self.maxiter = maxiter

    def resolva(self, func: Callable) -> tuple[float, int]:
        """
        Resolve um problema de otimização unidimensional usando o método de busca de Fibonacci.
        O método de Fibonacci é um algoritmo de busca por seção áurea que utiliza os números
        da sequência de Fibonacci para determinar os pontos de avaliação no intervalo de busca,
        reduzindo iterativamente o intervalo até encontrar o ótimo com a precisão desejada.
        Args:
            func (Callable): Função objetivo a ser otimizada. Deve aceitar um único argumento
                            numérico e retornar um valor numérico.
        Returns:
            tuple[float, int]: Uma tupla contendo:
                - float: Aproximação do ponto ótimo (centro do intervalo final)
                - int: Número total de avaliações da função realizadas
        Notes:
            - O método utiliza os atributos da classe: maxiter (máximo de iterações),
              precisao (tolerância para convergência) e maxaval (máximo de avaliações)
            - A busca continua até que o comprimento do intervalo seja menor que a precisão
              especificada, o número máximo de iterações seja atingido, ou o limite de
              avaliações seja excedido
            - O algoritmo assume que a função é unimodal no intervalo inicial
        """
        a, b, avaliacoes = super().resolva(func)        

        # Contador de termos de Fibonacci
        k = 1
        
        # Calcula o comprimento do intervalo
        L = b-a
        
        # Calcula a sequência de Fibonacci
        F = np.zeros(self.maxiter+1)
        F[:2] = 1.
        for i in range(2, F.size):
            F[i] = F[i-1] + F[i-2]
        F = F[::-1]
        
        # Determina dois pontos médios
        u = b - F[k]/F[k-1]*L
        v = a + F[k]/F[k-1]*L
        
        # Avalia
        fu = avalia_funcao(func, u)
        fv = avalia_funcao(func, v)
        avaliacoes += 2
        
        # Avança para o próximo termo
        k += 1

        # Enquanto o meu intervalo não for reduzido
        # a um tamanho suficientemente pequeno
        while (k <= self.maxiter and (b-a) > self.precisao
               and avaliacoes < self.maxaval):        
            
            # Exclui intervalo (v, c)
            if fu < fv:
                b, fb = v, fv
                v, fv = u, fu
                L = b-a
                u = b - F[k]/F[k-1]*L
                fu = func(u)
            
            # Exclui intervalo (a, u)
            else:
                a, fa = u, fu
                u, fu = v, fv
                L = b-a
                v = a + F[k]/F[k-1]*L
                fv = func(v)
            
            avaliacoes += 1
            
            # Avança para o próximo termo
            k += 1
        
        # A aproximação do meu ótimo é o meio do meu intervalo
        return (a+b)/2, avaliacoes

class SecaoAurea(Eliminacao):
    """
    Implementa o método da Seção Áurea para otimização unidimensional.
    A Seção Áurea é um método de busca que utiliza a proporção áurea (0.618) para
    dividir o intervalo de busca de forma eficiente, mantendo a redução do intervalo
    a cada iteração sem perder a propriedade de convergência.
    Herda de Eliminacao e utiliza o método de eliminação inicial para estabelecer
    o intervalo de busca, seguido pela aplicação do algoritmo da seção áurea.
    Attributes:
        precisao (float): Tolerância para o critério de parada baseado no tamanho
                         do intervalo. Default: 1e-3
        passo (float): Passo inicial para o método de eliminação herdado. Default: 0.01
        aceleracao (float): Fator de aceleração para o método de eliminação herdado. 
                           Default: 1.5
        maxaval (int): Número máximo de avaliações da função permitidas. Default: 100
    Methods:
        resolva(func): Encontra o mínimo da função utilizando o método da seção áurea.
                      Retorna uma tupla com (ponto_ótimo, número_de_avaliações).
    Example:
        >>> otimizador = SecaoAurea(precisao=1e-4)
        >>> resultado, avaliacoes = otimizador.resolva(lambda x: x**2 - 4*x + 3)
        >>> print(f"Mínimo encontrado em x = {resultado} com {avaliacoes} avaliações")
    """

    def __init__(self, precisao: float = 1e-3, passo: float = 0.01, 
                 aceleracao: float = 1.5, maxaval: int = 100):
        """
        Inicializa o otimizador com os parâmetros especificados.

        Args:
            precisao (float, optional): Tolerância para critério de convergência. 
                Defaults to 1e-3.
            passo (float, optional): Tamanho do passo inicial para o algoritmo. 
                Defaults to 0.01.
            aceleracao (float, optional): Fator de aceleração aplicado durante a otimização. 
                Defaults to 1.5.
            maxaval (int, optional): Número máximo de avaliações da função objetivo. 
                Defaults to 100.
        """   
        super().__init__(passo=passo, aceleracao=aceleracao, maxaval=maxaval)
        self.precisao = precisao

    def resolva(self, func:Callable) -> tuple[float, int]:
        """
        Resolve um problema de otimização unidimensional usando o método da Seção Áurea.
        Este método implementa o algoritmo de busca pela seção áurea (golden section search)
        para encontrar o mínimo de uma função unimodal em um intervalo. O algoritmo utiliza
        a proporção áurea (0.618) para dividir iterativamente o intervalo de busca,
        mantendo sempre dois pontos internos para comparação.
        Args:
            func (Callable): Função objetivo a ser minimizada. Deve aceitar um valor
                            numérico como entrada e retornar um valor numérico.
        Returns:
            tuple[float, int]: Uma tupla contendo:
                - float: Aproximação do ponto ótimo (mínimo) da função
                - int: Número total de avaliações da função realizadas durante a busca
        Note:
            O algoritmo para quando o comprimento do intervalo se torna menor que
            self.precisao ou quando o número de avaliações excede self.maxaval.
            A aproximação final é calculada como o ponto médio do intervalo final.
        """
        a, b, avaliacoes = super().resolva(func)        

        # Calcula o comprimento do intervalo
        L = b-a
        
        # Determina dois pontos médios
        u = b - .618*L
        v = a + .618*L
        
        # Avalia
        fu = avalia_funcao(func, u)
        fv = avalia_funcao(func, v)
        avaliacoes += 2

        # Enquanto o meu intervalo não for reduzido
        # a um tamanho suficientemente pequeno
        while (b-a) > self.precisao and avaliacoes < self.maxaval:        
            
            if fu < fv:
                
                # Exclui o intervalo (v, b)
                b = v
                
                # Atualiza o novo comprimento do intervalo
                L = b-a
                
                # Faz de u o novo v
                v, fv = u, fu
                
                # Calcula o novo u
                u = b -.618*L
                fu = avalia_funcao(func, u)
            
            # Se fu > fv
            else:
                
                # Exclui o intervalo (a, u)
                a = u
                
                # Atualiza o novo comprimento do intervalo
                L = b-a
                
                # Faz de v o novo u
                u, fu = v, fv
                
                # Calcula o novo u
                v = a + .618*L
                fv = avalia_funcao(func, v)
            
            avaliacoes += 1
        
        # A aproximação do meu ótimo é o meio do meu intervalo
        return (a+b)/2, avaliacoes

class Interpolacao(Unidimensional):
    """
    Classe para otimização unidimensional usando método de interpolação.
    Esta classe herda de Unidimensional e implementa um algoritmo de otimização
    baseado em interpolação para encontrar o mínimo de uma função unidimensional.
    Attributes:
        precisao (float): Precisão desejada para a convergência do algoritmo.
            Valor padrão é 1e-3.
        maxaval (int): Número máximo de avaliações da função permitidas.
            Valor padrão é 100.
    Methods:
        resolva(func): Resolve o problema de otimização para a função fornecida.
    Example:
        >>> interpolacao = Interpolacao(precisao=1e-4, maxaval=200)
        >>> resultado = interpolacao.resolva(minha_funcao)
    """

    def __init__(self, precisao: float = 1e-3, maxaval: int = 100):
        """
        Inicializa o otimizador de interpolação com os parâmetros especificados.
        
        Args:
            precisao (float): Precisão desejada para a convergência do algoritmo.
                Default é 1e-3.
            maxaval (int): Número máximo de avaliações da função permitidas.
                Default é 100.
        """
        self.precisao = precisao
        self.maxaval = maxaval
    
    def resolva(self, func: Callable) -> float:
        return avalia_funcao(func, 0.)

class Quadratica(Interpolacao):
    """
    Implementa o método de interpolação quadrática para otimização de funções.
    A classe Quadratica utiliza interpolação polinomial de segundo grau para encontrar
    o mínimo de uma função. O método inicia com três pontos equidistantes e ajusta
    iterativamente um polinômio quadrático através destes pontos, refinando a busca
    pelo ponto ótimo.

    Atributos:
        passo (float): Tamanho do passo inicial usado para estabelecer os três pontos
                       iniciais de interpolação.
    Herança:
        Herda de Interpolacao, obtendo os atributos precisao e maxaval.
    Exemplo:
        >>> otimizador = Quadratica(precisao=1e-4, passo=0.1)
        >>> def funcao(x):
        ...     return x**2 + 2*x + 1
        >>> x_min, avaliacoes = otimizador.resolva(funcao)
    """

    def __init__(self, precisao: float = 1e-3, passo: float = 1e-5,
                 maxaval: int = 100):
        """
        Inicializa o otimizador de interpolação quadrática.

        Args:
        
            precisao (float): Precisão desejada para a convergência do algoritmo.
                Default é 1e-3.
            passo (float): Tamanho do passo inicial usado para estabelecer os três
                pontos iniciais de interpolação. Default é 1e-5.
            maxaval (int): Número máximo de avaliações da função permitidas.
                Default é 100.
        """
        super().__init__(precisao=precisao, maxaval=maxaval)
        self.passo = passo
    
    def resolva(self, func: Callable) -> tuple[float, int]:
        f0 = super().resolva(func)
        
        # O ponto inicial é sempre zero
        A, fA = 0., f0
        
        # O ponto do meio é um passo
        B, fB = self.passo, avalia_funcao(func, self.passo)
        
        # O terceiro são dois passos
        C, fC = 2*self.passo, avalia_funcao(func, 2*self.passo)
        avaliacoes = 2
        
        # Termos do polinômio equivalente: a + b*x + c*x^2
        a = (fA*B*C*(C-B) + fB*C*A*(A-C) + fC*A*B*(B-A))/((A-B)*(B-C)*(C-A))
        b = (fA*(B*B-C*C) + fB*(C*C-A*A) + fC*(A*A-B*B))/((A-B)*(B-C)*(C-A))
        c = - (fA*(B-C) + fB*(C-A) + fC*(A-B))/((A-B)*(B-C)*(C-A))
        
        # Ponto de ótimo do polinômio aproximado
        xopt = -b/2/c
        fopt = avalia_funcao(func, xopt)
        avaliacoes += 1
        
        # Enquanto a avaliação do mínimo do polinônimo aproximado
        # não for suficientemente igual ao f(x)
        while (np.abs((a+b*xopt+c*xopt**2 - fopt)/fopt) > self.precisao
               and avaliacoes < self.maxaval):
            
            # Atualização dos três pontos de interpolação
            if xopt > B and fopt < fB:
                A, fA = B, fB
                B, fB = xopt, fopt
            elif xopt > B and fopt > fB:
                C, fC = xopt, fopt
            elif xopt < B and fopt < fB:
                C, fC = B, fB
                B, fB = xopt, fopt
            elif xopt < B and fopt > fB:
                A, fA = xopt, fopt
        
            # Novos termos do polinômio equivalente: a + b*x + c*x^2
            a = (fA*B*C*(C-B) + fB*C*A*(A-C) + fC*A*B*(B-A))/((A-B)*(B-C)
                                                              * (C-A))
            b = (fA*(B*B-C*C) + fB*(C*C-A*A) + fC*(A*A-B*B))/((A-B)*(B-C)
                                                              * (C-A))
            c = - (fA*(B-C) + fB*(C-A) + fC*(A-B))/((A-B)*(B-C)*(C-A))
            
            # Novo ponto de ótimo do polinômio aproximado
            xopt = -b/2/c
            fopt = avalia_funcao(func, xopt)
            avaliacoes += 1
        
        return xopt, avaliacoes

class QuasiNewtonUnidimensional(Interpolacao):
    """
    Classe para otimização unidimensional usando o método Quasi-Newton.
    Esta classe implementa uma variação do método de Newton para encontrar o mínimo
    de uma função unidimensional, utilizando aproximações numéricas das derivadas
    primeira e segunda através de diferenças finitas.
    Attributes:
        pertubacao (float): Valor da perturbação utilizada para calcular as 
                           derivadas numéricas por diferenças finitas.
        maxiter (int): Número máximo de iterações permitidas.
    Methods:
        resolva(func): Encontra o ponto de mínimo da função utilizando o método
                       Quasi-Newton com aproximações numéricas das derivadas.
    Example:
        >>> otimizador = QuasiNewtonUnidimensional(precisao=1e-6, pertubacao=1e-10)
        >>> resultado, avaliacoes = otimizador.resolva(lambda x: x**2 + 2*x + 1)
    Notes:
        - O método utiliza diferenças finitas centradas para estimar a primeira derivada
        - A segunda derivada é estimada usando diferenças finitas de segunda ordem
        - O algoritmo para quando |f'(x)| < precisao ou quando atinge o número máximo
          de iterações ou avaliações de função
        - Herda funcionalidades da classe Interpolacao para controle de precisão
          e número máximo de avaliações
    """

    def __init__(self, precisao: float = 1e-3, pertubacao: float = 1e-8, 
                 maxiter: int = 200, maxaval: int = 100):
        """
        Inicializa o otimizador Quasi-Newton unidimensional.
        
        Args:
            precisao (float): Precisão desejada para a convergência do algoritmo.
                Default é 1e-3.
            pertubacao (float): Valor da perturbação utilizada para calcular as
                derivadas numéricas. Default é 1e-8.
            maxiter (int): Número máximo de iterações permitidas. Default é 200.
            maxaval (int): Número máximo de avaliações da função permitidas.
                Default é 100.
        """

        super().__init__(precisao=precisao, maxaval=maxaval)
        self.pertubacao = pertubacao
        self.maxiter = maxiter
    
    def resolva(self, func: Callable) -> tuple[float, int]:
        """
        Resolve um problema de otimização usando o método de Newton com estimativa numérica das derivadas.
        Este método implementa o algoritmo de Newton para encontrar pontos críticos de uma função,
        utilizando diferenças finitas para estimar a primeira e segunda derivadas. O algoritmo
        itera até que a primeira derivada seja suficientemente próxima de zero ou até que os
        critérios de parada sejam atingidos.
        Args:
            func (Callable): Função objetivo a ser otimizada. Deve aceitar um valor numérico
                            como entrada e retornar um valor numérico.
        Returns:
            tuple[float, int]: Uma tupla contendo:
                - float: O ponto ótimo encontrado (valor de x)
                - int: Número total de avaliações da função realizadas durante o processo
        Notes:
            - O método utiliza diferenças finitas centradas para estimar as derivadas
            - A primeira derivada é estimada como: f'(x) ≈ (f(x+δ) - f(x-δ)) / (2δ)
            - A segunda derivada é estimada como: f''(x) ≈ (f(x+δ) - 2f(x) + f(x-δ)) / δ²
            - O algoritmo para quando |f'(x)| < precisao, k >= maxiter ou avaliacoes >= maxaval
            - Utiliza a fórmula de Newton: x_{k+1} = x_k - δ*f'(x_k)/f''(x_k)
        Attributes utilizados:
            - self.pertubacao: Valor de δ para cálculo das diferenças finitas
            - self.precisao: Tolerância para convergência da primeira derivada
            - self.maxiter: Número máximo de iterações permitidas
            - self.maxaval: Número máximo de avaliações da função permitidas
        """

        f0 = super().resolva(func)
        
        # Pertubação pequena para estimativa da derivada
        delta = self.pertubacao
        
        # Ponto inicial
        x, fx = 0., f0
        
        # Calcula f(x+delta) e f(x-delta)
        fxm = avalia_funcao(func, x-delta)
        fxp = avalia_funcao(func, x+delta)
        avaliacoes = 2

        # Estima a primeira derivada
        fp = (fxp-fxm)/(2*delta)
        
        # Estima a segunda derivada
        fpp = (fxp-2*fx+fxm)/(delta**2)
        
        # Enquanto a primeira derivada não for
        # tão próxima de zero
        k = 0
        while (np.abs(fp) > self.precisao and k < self.maxiter
               and avaliacoes < self.maxaval):
            
            # Fórmula de Newton
            x = x - delta*fp/fpp
        
            # Recalcula f(x+delta) e f(x-delta) para novo ponto
            fxm = avalia_funcao(func, x-delta)
            fxp = avalia_funcao(func, x+delta)
            avaliacoes += 2

            # Estima a primeira derivada
            fp = (fxp-fxm)/(2*delta)
            
            # Estima a segunda derivada
            fpp = (fxp-2*fx+fxm)/(delta**2)
            
            k += 1
        
        return x, avaliacoes

class Secante(Interpolacao):
    """
    Classe que implementa o método da secante para otimização unidimensional.
    O método da secante é uma técnica de otimização que encontra o ponto onde a derivada
    de uma função é zero, utilizando aproximações lineares (secantes) da derivada.
    Este método é uma variação do método de Newton que não requer o cálculo analítico
    da segunda derivada.
    Herda da classe Interpolacao e implementa um algoritmo iterativo que:
    1. Inicia com uma pequena perturbação para estimar a derivada numericamente
    2. Encontra dois pontos com derivadas de sinais opostos
    3. Usa o método da secante para convergir ao ponto onde a derivada é zero
    Attributes:
        pertubacao (float): Valor da perturbação usada para calcular a derivada numérica.
                           Valor padrão: 1e-8
    Methods:
        resolva(func): Encontra o ponto ótimo da função usando o método da secante.
    Args:
        precisao (float, optional): Tolerância para convergência. Padrão: 1e-3
        pertubacao (float, optional): Perturbação para derivada numérica. Padrão: 1e-8  
        maxaval (int, optional): Número máximo de avaliações da função. Padrão: 100
    Returns:
        tuple[float, int]: Tupla contendo o ponto ótimo encontrado e o número de
                          avaliações da função realizadas.
    Example:
        >>> secante = Secante(precisao=1e-4, pertubacao=1e-6)
        >>> ponto_otimo, num_avaliacoes = secante.resolva(minha_funcao)
    """

    def __init__(self, precisao: float = 1e-3, pertubacao: float = 1e-8, 
                 maxaval: int = 100):
        """
        Inicializa o otimizador com parâmetros de configuração.
        Args:
            precisao (float, optional): Tolerância para critério de convergência. 
                Valor padrão é 1e-3.
            pertubacao (float, optional): Valor da perturbação utilizada para 
                cálculo de derivadas numéricas. Valor padrão é 1e-8.
            maxaval (int, optional): Número máximo de avaliações da função objetivo 
                permitidas. Valor padrão é 100.
        Attributes:
            pertubacao (float): Valor da perturbação armazenado na instância.
            maxaval: int = 100):
        """
        super().__init__(precisao=precisao, maxaval=maxaval)
        self.pertubacao = pertubacao
    
    def resolva(self, func: Callable) -> tuple[float, int]:
        """
        Resolve um problema de otimização usando o método da secante para encontrar
        o ponto onde a derivada da função é zero.
        O método utiliza uma abordagem em duas fases:
        1. Fase de busca: Encontra um intervalo [A, B] onde a derivada muda de sinal
        2. Fase de refinamento: Usa o método da secante para encontrar o zero da derivada
        Args:
            func (Callable): Função objetivo a ser otimizada
        Returns:
            tuple[float, int]: Tupla contendo:
                - x (float): Ponto ótimo encontrado onde a derivada é aproximadamente zero
                - avaliacoes (int): Número total de avaliações da função realizadas
        Notes:
            - A derivada é estimada numericamente usando diferenças finitas
            - O método para quando |f'(x)| <= precisao ou quando o número máximo
              de avaliações é atingido
            - Utiliza perturbação pequena (delta) para calcular a derivada numérica
            - O algoritmo assume que existe um ponto ótimo na região de busca
        """

        f0 = super().resolva(func)
        
        # Pertubação pequena para estimativa da derivada
        delta = self.pertubacao
        
        # Ponto inicial
        A, fA = 0, f0
        
        # Derivada no ponto inicial
        fD = avalia_funcao(func, delta)
        fpA = (fD-fA)/delta
        
        # Segundo ponto e sua derivada
        t0, fpt0 = delta, (avalia_funcao(func, 2*delta)-fD)/delta
        avaliacoes = 2
        
        # Enquanto a derivada do segundo ponto não estiver acima de zero
        while fpt0 < 0 and avaliacoes < self.maxaval:
            
            # O primeiro ponto é atualizado
            A, fpA = t0, fpt0
            
            # É dado um novo passo e calculado a sua derivada
            t0 = 2*t0
            fpt0 = ((avalia_funcao(func, t0+delta)-avalia_funcao(func, t0))
                    / delta)
            avaliacoes += 2

        # O ponto A é um ponto com derivada negativa
        # enquanto o ponto B é um ponto com derivada positiva
        B, fpB = t0, fpt0
        
        while True:
            
            # Atualização de x:
            # O ponto onde a reta que une os dois pontos (secante)
            # toca o zero no eixo-y da derivada
            x = A - (fpA*(B-A))/(fpB-fpA)
            
            # Reavalia a derivada
            fpx = (avalia_funcao(func, x+delta)-avalia_funcao(func, x))/delta
            avaliacoes += 2
            
            # Se a derivada for suficientemente
            # próxima de zero
            if np.abs(fpx) <= self.precisao:
                break
            elif avaliacoes >= self.maxaval:
                break
            
            # A derivada está acima de zero,
            # então atualiza B
            if fpx >= 0:
                B, fpB = x, fpx
            
            # A derivada está abaixo de zero,
            # então atualiza B
            else:
                A, fpA = x, fpx
            
        return x, avaliacoes

class Irrestrita:
    """
    Classe base para métodos de otimização irrestrita multidimensional.

    Esta classe fornece a estrutura comum para todos os algoritmos de
    otimização sem restrições, incluindo inicialização de variáveis
    e critérios de parada.

    Attributes:
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações da função objetivo
    """

    def __init__(self, maxit: int = 10000, maxaval: int = 10000):
        """
        Inicializa o otimizador irrestrito.

        Args:
            maxit: Número máximo de iterações (padrão: 10000)
            maxaval: Número máximo de avaliações da função (padrão: 10000)
        """
        self.maxit = maxit
        self.maxaval = maxaval

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> (
            Solution
    ):
        """
        Método base para inicialização das variáveis do processo iterativo.

        Este método inicializa as variáveis comuns a todos os algoritmos de
        otimização irrestrita.

        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)

        Returns:
            tuple: (x, fx, xhist, fxhist, iter, aval) - Variáveis inicializadas
        """
        # Definição inicial das variáveis do processo iterativo
        if type(x0) is list:
            x = np.array(x0, dtype=float).reshape((-1, 1))
        else:
            x = np.array(x0, dtype=float).reshape((-1, 1))
        fx = avalia_funcao(func, x)
        xhist = [np.copy(x)]
        fxhist = [fx]
        iter = 0
        aval = 1
        return x, fx, xhist, fxhist, iter, aval

class DirecoesAleatorias(Irrestrita):
    """
    Método de direções aleatórias para otimização multidimensional.

    Este método gera direções aleatórias e usa otimização unidimensional
    para encontrar o passo ótimo em cada direção.

    Attributes:
        unidimensional: Instância de um método de otimização unidimensional
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações da função
    """

    def __init__(self, unidimensional: Unidimensional, maxit: int = 10000, 
                 maxaval: int = 10000):
        """
        Inicializa o método de direções aleatórias.

        Args:
            unidimensional: Instância de um método de otimização unidimensional
            maxit: Número máximo de iterações (padrão: 10000)
            maxaval: Número máximo de avaliações (padrão: 10000)
        """
        super().__init__(maxit=maxit, maxaval=maxaval)
        self.unidimensional = unidimensional

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> Solution:
        """
        Resolve um problema de otimização usando direções aleatórias.
        Este método gera direções aleatórias a partir do ponto inicial e
        utiliza um método de otimização unidimensional para encontrar o
        passo ótimo em cada direção. O processo continua até que o número
        máximo de iterações ou avaliações seja atingido.

        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Inicializa as variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0)

        # Critério de parada
        while iter < self.maxit and aval < self.maxaval:
        
            # Define a direção de busca aleatoriamente
            d = np.random.normal(size=x.shape)
            
            # Se a direção obtida não aponta para minimização
            if avalia_funcao(func, x + 1e-15*d) > fx:
                d = -d
            aval += 1
                
            # A função que representará nossa otimização unidimensional
            def theta(alpha):
                fx = avalia_funcao(func, x + alpha*d)
                return fx

            # Otimização unidimensional para determinar o passo na direção d
            alpha, na = self.unidimensional.resolva(theta)
            aval += na
                
            # Atualiza
            x = x + alpha*d
            fx = func(x)
            aval += 1
            iter += 1
                
            xhist.append(np.copy(x))
            fxhist.append(fx)

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        else:
            criterio_parada = "Critério de convergência atingido"

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)

        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)

def gradiente(x: ndarray, func: Callable, fx: Optional[float] = None, 
              metodo: str = 'progressiva', delta: float = 1e-10) -> tuple[ndarray, int]:
    """
    Calcula o gradiente numérico de uma função.

    Este função computa o gradiente usando diferenças finitas progressivas,
    regressivas ou centrais.

    Args:
        x: Ponto onde calcular o gradiente
        func: Função objetivo
        fx: Valor da função em x (opcional, para economia de avaliações)
        metodo: Método de diferenciação ('progressiva', 'regressiva', 'central')
        delta: Perturbação para diferenciação numérica (padrão: 1e-10)

    Returns:
        tuple: (gradiente, numero_avaliacoes)
        
    Example:
        >>> import numpy as np
        >>> def f(x): return x[0]**2 + x[1]**2
        >>> grad, naval = gradiente(np.array([1, 2]), f)
    """
    avaliacoes = 0

    if fx is None:
        fx = avalia_funcao(func, x)
        avaliacoes +=1

    # Inicializa o vetor gradiente
    grad = np.zeros((x.size, 1))
    
    # Para cada variável
    for n in range(x.size):
        
        # Vetor com 1 na posição da variável onde a derivada será calculada
        e = np.zeros(x.size)
        e[n] = 1
        
        # Estima a derivada no ponto
        if metodo == 'regressiva': # O(delta)
            grad[n] = (fx - avalia_funcao(func, x.flatten() - delta*e))/delta 
            avaliacoes += 1
        elif metodo == 'central': # O(delta**2)
            grad[n] = (avalia_funcao(func, x.flatten() + delta*e)
                       - avalia_funcao(func, x.flatten() - delta*e))/(2*delta)
            avaliacoes += 2
        else: # O(delta)
            grad[n] = (avalia_funcao(func, x.flatten() + delta*e)-fx)/delta
            avaliacoes += 1
    
    return grad, avaliacoes

class Gradiente(Irrestrita):
    """
    Método do gradiente descendente para otimização multidimensional.

    Este método usa o gradiente da função para determinar a direção de
    descida mais íngreme, combinado com busca unidimensional para o passo.

    Attributes:
        unidimensional: Método de otimização unidimensional para busca linear
        diferenca (str): Tipo de diferença finita para cálculo do gradiente
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações
        precisao (float): Tolerância para convergência
    """

    def __init__(self, unidimensional: Unidimensional, 
                 diferenca: str = 'progressiva', maxit: int = 10000, 
                 maxaval: int = 10000, precisao: float = 1e-3):
        """
        Inicializa o método do gradiente descendente.

        Args:
            unidimensional: Instância de um método de otimização unidimensional
            diferenca: Método de diferenciação ('progressiva', 'regressiva', 'central')
            maxit: Número máximo de iterações (padrão: 10000)
            maxaval: Número máximo de avaliações (padrão: 10000)
            precisao: Tolerância para convergência (padrão: 1e-3)
        """
        super().__init__(maxit=maxit, maxaval=maxaval)
        self.unidimensional = unidimensional
        self.diferenca = diferenca
        self.precisao = precisao

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> Solution:
        """
        Resolve um problema de otimização usando o método do gradiente descendente.
        Este método itera até que o gradiente seja suficientemente pequeno ou
        até que o número máximo de iterações ou avaliações seja atingido.
        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Inicializa as variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0)
        grad = 2*self.precisao*np.ones(x.size)
        
        # Critério de parada
        while (iter < self.maxit and aval < self.maxaval
               and norm(grad) > self.precisao):
        
            # Define a direção de busca baseado no gradiente
            grad, na = gradiente(x, func, fx=fx, metodo=self.diferenca)
            aval += na
            d = -grad

            # A função que representará nossa otimização unidimensional
            def theta(alpha):
                fx = avalia_funcao(func, x.flatten() + alpha*d.flatten())
                return fx

            # Otimização unidimensional para determinar o passo na direção d
            alpha, na = self.unidimensional.resolva(theta)
            aval += na
                
            # Atualiza
            x = x + alpha*d
            fx = avalia_funcao(func, x)
            aval += 1
            iter += 1
                
            xhist.append(np.copy(x))
            fxhist.append(fx)

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        elif norm(grad) <= self.precisao:
            criterio_parada = "Gradiente suficientemente pequeno"
        else:
            criterio_parada = "Critério de convergência atingido"

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)

        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)

def hessiana(x: ndarray, func: Callable, fx: Optional[float] = None, 
             grad: Optional[ndarray] = None, delta: float = 1e-10) -> tuple[ndarray, int]:
    """
    Calcula a matriz Hessiana numérica de uma função.

    Esta função computa a matriz de segundas derivadas usando diferenças
    finitas do gradiente.

    Args:
        x: Ponto onde calcular a Hessiana
        func: Função objetivo
        fx: Valor da função em x (opcional)
        grad: Gradiente em x (opcional, se não fornecido será calculado)
        delta: Perturbação para diferenciação numérica (padrão: 1e-10)

    Returns:
        tuple: (hessiana, numero_avaliacoes)
        
    Example:
        >>> import numpy as np
        >>> def f(x): return x[0]**2 + x[1]**2
        >>> H, naval = hessiana(np.array([1, 2]), f)
    """
    if grad is None:
        grad, navaliacoes = gradiente(x, func, fx=fx, delta=delta)
    else:
        navaliacoes = 0
    
    # A Hessiana é uma matriz quadrada do tamanho do número de variáveis
    H = np.zeros((x.size, x.size))    
    
    # Para cada variável...
    for n in range(x.size):
        
        # Perturbação na n-ésima variável
        e = np.zeros(x.shape)
        e[n] = 1

        # Calcula o gradiente nesse ponto perturbado
        gpert, nava = gradiente(x + delta*e, func)
        
        navaliacoes += nava

        # Calcula uma coluna da Hessiana
        H[:, n] = (gpert.flatten()-grad.flatten())/delta

    return H, navaliacoes

class Newton(Irrestrita):
    """
    Método de Newton multidimensional para otimização.

    Este método usa tanto o gradiente quanto a matriz Hessiana para
    encontrar a direção de Newton, oferecendo convergência quadrática
    próximo ao ótimo.

    Attributes:
        unidimensional: Método de otimização unidimensional para busca linear
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações
        precisao (float): Tolerância para convergência
    """
    def __init__(self, unidimensional: Unidimensional, maxit: int = 10000,
                 maxaval: int = 10000, precisao: float = 1e-3):
        """
        Inicializa o método de Newton.

        Args:
            unidimensional: Instância de um método de otimização unidimensional
            maxit: Número máximo de iterações (padrão: 10000)
            maxaval: Número máximo de avaliações (padrão: 10000)
            precisao: Tolerância para convergência (padrão: 1e-3)
        """
        super().__init__(maxit=maxit, maxaval=maxaval)
        self.unidimensional = unidimensional
        self.precisao = precisao

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> Solution:
        """
        Resolve um problema de otimização usando o método de Newton.
        Este método itera até que o gradiente seja suficientemente pequeno ou
        até que o número máximo de iterações ou avaliações seja atingido.

        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0)
        grad = 2*self.precisao*np.ones(x.shape)
        # Critério de parada
        while (iter < self.maxit and aval < self.maxaval
               and norm(grad) > self.precisao):
        
            # Define a direção de busca
            grad, avg = gradiente(x, func, fx=fx)
            H, avh = hessiana(x, func, grad=grad)
            aval += avg + avh

            try:
                d = -inv(H) @ grad
                if d.T @ grad > 0:
                    d = -d
            
            # Matriz singular
            except:
                print("Matriz singular encontrada. Parando o algortimo...")
                xhist = np.array(xhist)
                fxhist = np.array(fxhist)
                return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                                fxhist=fxhist)

            # A função que representará nossa otimização unidimensional
            def theta(alpha):
                fx = func(x.flatten() + alpha*d.flatten())
                return fx

            # Otimização unidimensional para determinar o passo na direção d
            alpha, na = self.unidimensional.resolva(theta)
            aval += na
                
            # Atualiza
            x = x + alpha*d
            fx = avalia_funcao(func, x)
            
            aval += 1
            iter += 1
                
            xhist.append(np.copy(x))
            fxhist.append(fx)

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        elif norm(grad) <= self.precisao:
            criterio_parada = "Gradiente suficientemente pequeno"
        else:
            criterio_parada = "Critério de convergência atingido"

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)

        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)

class DFP(Irrestrita):
    """
    Método Davidon-Fletcher-Powell (DFP).
    Este é um método quasi-Newton que aproxima a matriz Hessiana usando
    informações do gradiente. É um dos métodos mais antigos e amplamente
    utilizados para otimização irrestrita.
    Attributes:
        unidimensional: Método de otimização unidimensional para busca linear
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações
        precisao (float): Tolerância para convergência
    """

    def __init__(self, unidimensional: Unidimensional, maxit: int = 10000, 
                 maxaval: int = 10000, precisao: float = 1e-3):
        """
        Inicializa o método DFP.

        Args:
            unidimensional: Instância de um método de otimização unidimensional
            maxit: Número máximo de iterações (padrão: 10000)
            maxaval: Número máximo de avaliações (padrão: 10000)
            precisao: Tolerância para convergência (padrão: 1e-3)
        """
        super().__init__(maxit=maxit, maxaval=maxaval)
        self.unidimensional = unidimensional
        self.precisao = precisao

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> Solution:
        """
        Resolve um problema de otimização usando o método DFP.
        Este método itera até que o gradiente seja suficientemente pequeno ou
        até que o número máximo de iterações ou avaliações seja atingido.
        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Inicializa as variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0)
        
        # Aproximação inicial da inversa da matriz hessiana
        Hh = np.eye(x.size)

        # Primeira estimativa do gradiente
        g, na = gradiente(x, func=func, fx=fx)
        aval += na
        
        # Critério de parada
        while (iter < self.maxit and aval < self.maxaval
               and norm(g) > self.precisao):
        
            # Determina a direção de busca
            d = - Hh @ g
            
            # Função de otimização unidimensional
            # A função que representará nossa otimização unidimensional
            def theta(alpha):
                fx = avalia_funcao(func, x.flatten() + alpha*d.flatten())
                return fx

            # Determina o passo ótimo    
            alpha, na = self.unidimensional.resolva(theta)
            aval += na

            # Grava informações antes do passo
            xanterior = x.copy()
            ganterior = g.copy()
            
            # Atualiza
            x = x + alpha*d
            fx = avalia_funcao(func, x.flatten())
            aval += 1

            # Estima novo gradiente
            g, na = gradiente(x, func=func, fx=fx)
            aval += na
            
            xhist.append(np.copy(x))
            fxhist.append(fx)

            # Atualiza vetores v e r
            v = xanterior-x
            r = ganterior-g
            
            # Coloca na forma vetor-coluna
            v = v.reshape((-1, 1))
            r = r.reshape((-1, 1))

            # Atualização de Hh
            Hh = Hh + v@v.T/(v.T@r) - Hh@r@r.T@Hh/(r.T@Hh@r)
            
            iter += 1

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        elif norm(g) <= self.precisao:
            criterio_parada = "Gradiente suficientemente pequeno"
        else:
            criterio_parada = "Critério de convergência atingido"

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)

        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)

class BFGS(Irrestrita):
    """
    Método Broyden-Fletcher-Goldfarb-Shanno (BFGS).

    Este é um método quasi-Newton que aproxima a inversa da matriz Hessiana
    usando informações do gradiente. É considerado um dos melhores métodos
    quasi-Newton para otimização irrestrita.

    Attributes:
        unidimensional: Método de otimização unidimensional para busca linear
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações
        precisao (float): Tolerância para convergência
    """

    def __init__(self, unidimensional: Unidimensional, maxit: int = 10000, 
                 maxaval: int = 10000, precisao: float = 1e-3):
        """
        Inicializa o método BFGS.

        Args:
            unidimensional: Instância de um método de otimização unidimensional
            maxit: Número máximo de iterações (padrão: 10000)
            maxaval: Número máximo de avaliações (padrão: 10000)
            precisao: Tolerância para convergência (padrão: 1e-3)
        """
        super().__init__(maxit=maxit, maxaval=maxaval)
        self.unidimensional = unidimensional
        self.precisao = precisao

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> Solution:
        """
        Resolve um problema de otimização usando o método BFGS.
        Este método itera até que o gradiente seja suficientemente pequeno ou
        até que o número máximo de iterações ou avaliações seja atingido.
        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Inicializa as variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0)
        
        # Aproximação inicial da inversa da matriz hessiana
        Hh = np.eye(x.size)

        # Primeira estimativa do gradiente
        g, na = gradiente(x, func=func, fx=fx)
        aval += na
        
        # Critério de parada
        while (iter < self.maxit and aval < self.maxaval
               and norm(g) > self.precisao):
        
            # Determina a direção de busca
            d = - Hh @ g
            
            # Função de otimização unidimensional
            # A função que representará nossa otimização unidimensional
            def theta(alpha):
                fx = avalia_funcao(func, x + alpha*d)
                return fx

            # Determina o passo ótimo    
            alpha, na = self.unidimensional.resolva(theta)
            aval += na

            # Grava informações antes do passo
            xanterior = x.copy()
            ganterior = g.copy()
            
            # Atualiza
            x = x + alpha*d
            fx = avalia_funcao(func, x)
            aval += 1

            # Estima novo gradiente
            g, na = gradiente(x, func=func, fx=fx) 
            aval += na
            
            xhist.append(np.copy(x))
            fxhist.append(fx)

            # Atualiza vetores v e r
            v = xanterior-x
            r = ganterior-g
            
            # Coloca na forma vetor-coluna
            v = v.reshape((-1, 1))
            r = r.reshape((-1, 1))

            # Atualização de Hh
            Hh = (Hh + (1 + r.T@Hh@r/(r.T@v))*v@v.T/(v.T@r) 
                  - (v@r.T@Hh + Hh@r@v.T)/(r.T@v))

            iter += 1

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        elif norm(g) <= self.precisao:
            criterio_parada = "Gradiente suficientemente pequeno"
        else:
            criterio_parada = "Critério de convergência atingido"

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)

        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)        

class QuasiNewton(Irrestrita):
    """
    Método Quasi-Newton para otimização multidimensional.
    Este método combina o conceito de direções de Newton com uma
    atualização da matriz Hessiana aproximada, utilizando uma combinação
    de métodos DFP e BFGS. É um método eficiente para otimização sem
    restrições, especialmente quando a matriz Hessiana é difícil de
    calcular diretamente.
    Attributes:
        unidimensional: Método de otimização unidimensional para busca linear
        qsi (float): Parâmetro de combinação entre DFP e BFGS
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações
        precisao (float): Tolerância para convergência
    """

    def __init__(self, unidimensional: Unidimensional, qsi: float = 0.5, 
                 maxit: int = 10000, maxaval: int = 10000,
                 precisao: float = 1e-3):
        """
        Inicializa o método Quasi-Newton.

        Args:
            unidimensional: Instância de um método de otimização unidimensional
            qsi: Parâmetro de combinação entre DFP e BFGS (padrão: 0.5)
            maxit: Número máximo de iterações (padrão: 10000)
            maxaval: Número máximo de avaliações (padrão: 10000)
            precisao: Tolerância para convergência (padrão: 1e-3)
        """
        super().__init__(maxit=maxit, maxaval=maxaval)
        self.unidimensional = unidimensional
        self.qsi = qsi
        self.precisao = precisao

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> Solution:
        """
        Resolve um problema de otimização usando o método Quasi-Newton.
        Este método itera até que o gradiente seja suficientemente pequeno ou
        até que o número máximo de iterações ou avaliações seja atingido.
        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Inicializa as variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0)
        
        # Aproximação inicial da inversa da matriz hessiana
        Hh = np.eye(x.size)

        # Primeira estimativa do gradiente
        g, na = gradiente(x, func=func, fx=fx)
        aval += na
        
        # Critério de parada
        while (iter < self.maxit and aval < self.maxaval
               and norm(g) > self.precisao):
        
            # Determina a direção de busca
            d = - Hh @ g
            
            # Função de otimização unidimensional
            # A função que representará nossa otimização unidimensional
            def theta(alpha):
                fx = avalia_funcao(func, x + alpha*d)
                return fx

            # Determina o passo ótimo    
            alpha, na = self.unidimensional.resolva(theta)
            aval += na

            # Grava informações antes do passo
            xanterior = x.copy()
            ganterior = g.copy()
            
            # Atualiza
            x = x + alpha*d
            fx = func(x)
            aval += 1

            # Estima novo gradiente
            g, na = gradiente(x, func=func, fx=fx) 
            aval += na
            
            xhist.append(np.copy(x))
            fxhist.append(fx)

            # Atualiza vetores v e r
            v = xanterior-x
            r = ganterior-g
            
            # Coloca na forma vetor-coluna
            v = v.reshape((-1, 1))
            r = r.reshape((-1, 1))

            # Atualização de Hh
            C_DFP = v@v.T/(v.T@r) - Hh@r@r.T@Hh/(r.T@Hh@r)
            C_BFGS = ((1 + r.T@Hh@r/(r.T@v))*v@v.T/(v.T@r) 
                      - (v@r.T@Hh + Hh@r@v.T)/(r.T@v))
            
            Hh += (1-self.qsi)*C_DFP + self.qsi*C_BFGS
            iter += 1

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        elif norm(g) <= self.precisao:
            criterio_parada = "Gradiente suficientemente pequeno"
        else:
            criterio_parada = "Critério de convergência atingido"

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)

        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)        

class GradienteConjugado(Irrestrita):
    """
    Método do gradiente conjugado para otimização.

    Este método usa direções conjugadas para acelerar a convergência,
    sendo especialmente eficaz para funções quadráticas. Implementa
    as fórmulas de Fletcher-Reeves e Polak-Ribière.

    Attributes:
        unidimensional: Método de otimização unidimensional para busca linear
        formula (str): Fórmula para cálculo do parâmetro beta
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações
        precisao (float): Tolerância para convergência
    """

    def __init__(self, unidimensional: Unidimensional,
                 formula: str = 'polak-ribiere', maxit: int = 10000, 
                 maxaval: int = 10000, precisao: float = 1e-3):
        """
        Inicializa o algoritmo de otimização.
        Args:
            unidimensional (Unidimensional): Objeto responsável pela busca unidimensional.
            formula (str, optional): Fórmula a ser utilizada no algoritmo. 
                Padrão é 'polak-ribiere'.
            maxit (int, optional): Número máximo de iterações permitidas. 
                Padrão é 10000.
            maxaval (int, optional): Número máximo de avaliações da função objetivo. 
                Padrão é 10000.
            precisao (float, optional): Precisão desejada para o critério de parada. 
                Padrão é 1e-3.
        """
        super().__init__(maxit=maxit, maxaval=maxaval)
        self.unidimensional = unidimensional
        self.formula = formula
        self.precisao = precisao

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> Solution:
        """
        Resolve um problema de otimização usando o método do Gradiente Conjugado.
        Este método utiliza direções conjugadas para encontrar o mínimo de uma função,
        combinando informações do gradiente atual com direções de busca anteriores
        para acelerar a convergência comparado ao método do gradiente simples.
        Parameters
        ----------
        func : Callable
            Função objetivo a ser minimizada. Deve aceitar um array numpy como
            entrada e retornar um valor escalar.
        x0 : List[float] | ndarray
            Ponto inicial para a otimização. Pode ser uma lista de floats ou
            um array numpy.
        Returns
        -------
        Solution
            Objeto contendo os resultados da otimização:
            - x: ponto ótimo encontrado
            - fx: valor da função no ponto ótimo
            - iter: número de iterações realizadas
            - aval: número de avaliações da função
            - xhist: histórico dos pontos visitados
            - fxhist: histórico dos valores da função
        Notes
        -----
        O método implementa duas fórmulas para o cálculo do parâmetro beta:
        - Fletcher-Reeves: beta = (r.T@r)/(ranterior.T@ranterior)
        - Polak-Ribière: beta = (r.T@(r-ranterior))/(ranterior.T@ranterior)
        A direção de busca é reinicializada a cada n iterações (onde n é a dimensão
        do problema) para garantir convergência.
        O algoritmo para quando:
        - O número máximo de iterações é atingido, ou
        - O número máximo de avaliações da função é atingido, ou
        - A norma do gradiente é menor que a precisão especificada
        """
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0)
        
        # Aproximação inicial da inversa da matriz hessiana
        Hh = np.eye(x.size)

        # Primeira estimativa do gradiente
        g, na = gradiente(x, func=func, fx=fx)
        aval += na
        
        # Define resíduos e direção de busca
        r = -g
        d = r.copy()
        
        # Critério de parada
        while (iter < self.maxit and aval < self.maxaval
               and norm(g) > self.precisao):
            
            # Função de otimização unidimensional
            # A função que representará nossa otimização unidimensional
            def theta(alpha):
                fx = avalia_funcao(func, x + alpha*d)
                return fx

            # Determina o passo ótimo    
            alpha, na = self.unidimensional.resolva(theta)
            aval += na
            
            # Atualiza
            x = x + alpha*d
            fx = avalia_funcao(func, x)
            aval += 1
            
            # Registra o histórico
            xhist.append(np.copy(x))
            fxhist.append(fx)

            # Salvo a informação do resíduo antes de calcular o próximo
            ranterior = r.copy() # Guarda valor anterior
            ranterior = ranterior.reshape((-1, 1))

            # Estima novo gradiente
            g, na = gradiente(x, func=func, fx=fx) 
            aval += na
            
            # Cálculo do novo resíduo
            r = -g
            
            # Cálculo do conjugado do gradiente
            if self.formula == "fletcher-reeves":
                beta = (r.T@r/(ranterior.T@ranterior)).item()
            else: # Polak-Ribière
                beta = (r.T@(r-ranterior)/(ranterior.T@ranterior)).item()

            # Atualiza a direção de busca
            d = r + beta*d
        
            # Verifica iterações
            if np.mod(iter+1, x.size) == 0:
                d = r.copy()

            iter += 1

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        elif norm(g) <= self.precisao:
            criterio_parada = "Gradiente suficientemente pequeno"
        else:
            criterio_parada = "Critério de convergência atingido"

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)

        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)  

class HookeJeeves(Irrestrita):
    """
    Método de Hooke-Jeeves (busca por padrão).

    Este método não requer derivadas, usando buscas exploratórias ao longo
    das coordenadas seguidas de movimentos na direção do padrão identificado.

    Attributes:
        lamb (float): Passo para busca por coordenadas (passo_coordenada)
        alpha (float): Passo para busca por padrão (passo_direcao)
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações
        precisao (float): Tolerância para convergência
    """

    def __init__(self, passo_coordenada: float = 0.5,
                 passo_direcao: float = 0.1, maxit: int = 10000,
                 maxaval: int = 10000, precisao: float = 1e-3):
        """
        Inicializa o método de Hooke-Jeeves.
        Args:
            passo_coordenada (float): Passo para busca por coordenadas (padrão: 0.5)
            passo_direcao (float): Passo para busca por padrão (padrão: 0.1)
            maxit (int): Número máximo de iterações (padrão: 10000)
            maxaval (int): Número máximo de avaliações (padrão: 10000)
            precisao (float): Tolerância para convergência (padrão: 1e-3)
        """
        super().__init__(maxit=maxit, maxaval=maxaval)
        self.lamb = passo_coordenada
        self.alpha = passo_direcao
        self.precisao = precisao

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> Solution:
        """
        Resolve um problema de otimização usando o método de Hooke-Jeeves.
        Este método itera até que o número máximo de iterações ou avaliações
        seja atingido, ou até que a solução converja dentro da precisão especificada.
        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Inicializa as variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0)
        
        # Número de variáveis
        N = x.size
        
        lamb = self.lamb
        alpha = self.alpha
        
        while iter < self.maxit and aval < self.maxaval:
                
            """ Busca em cada coordenada """
            y = np.copy(x)
            xt = np.copy(x)
            for n in range(N):
                
                # Dou um passo em um eixo
                xt[n] = x[n] + lamb
                fxt = avalia_funcao(func, xt)
                aval += 1
                
                # Se eu melhorei dando esse passo
                if fxt < fx:
                    y[n] = xt[n]
                    continue # Salvo esse passo e vou para próxima variável
                
                # Se eu tiver piorado, dou um passo no sentido contrário
                xt[n] = x[n] - lamb
                fxt = avalia_funcao(func, xt)
                aval += 1
                
                # Se eu melhorei dando esse passo
                if fxt < fx:
                    y[n] = xt[n]
                    continue # Salvo esse passo e vou para próxima variável
                
                # Caso contrário, eu fico com o mesmo valor
                y[n] = x[n]

            """ Busca na direção """
            fy = func(y)
            aval += 1
            
            # Se minha busca por coordenada retornou uma solução melhor
            if fy < fx:
                
                # Tento dar um passo na direção final que eu andei
                z = y + alpha*(y-x)
                fz = avalia_funcao(func, z)
                aval += 1
                
                # Caso esse passo seja melhor, atualiza 
                if fz < fy:
                    x = z
                    fx = fz
                
                # Fica com o anterior e diminui o passo nessa direção
                else:
                    x = y
                    fx = fy
                    alpha = alpha/2
            
            # Reduz o passo em cada coordenada
            else:
                lamb = lamb/2
            
            xhist.append(np.copy(x))
            fxhist.append(fx)
            iter += 1
            
            if iter > 5 and norm(x-xhist[-5]) < self.precisao:
                break

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        else:
            criterio_parada = ("Diferença entre as últimas soluções menor que "
                               + "a precisão")

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)

        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)  

class NelderMeadSimplex(Irrestrita):
    """
    Método Nelder-Mead (simplex) para otimização.

    Este método usa um simplex (poliedro de n+1 vértices em n dimensões)
    que se adapta à forma da função através de operações de reflexão,
    expansão, contração e encolhimento.

    Attributes:
        alpha (float): Coeficiente de reflexão
        beta (float): Coeficiente de contração
        gamma (float): Coeficiente de expansão
        delta (float): Coeficiente de encolhimento
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações
        precisao (float): Tolerância para convergência
    """

    def __init__(self, reflexao: float = 1.0, contracao: float = 0.5, 
                 expansao: float = 2.0, encolhimento: float = 0.5, 
                 maxit: int = 10000, maxaval: int = 10000, 
                 precisao: float = 1e-3):
        """
        Inicializa o método Nelder-Mead Simplex.

        Args:
            reflexao (float): Coeficiente de reflexão (padrão: 1.0)
            contracao (float): Coeficiente de contração (padrão: 0.5)
            expansao (float): Coeficiente de expansão (padrão: 2.0)
            encolhimento (float): Coeficiente de encolhimento (padrão: 0.5)
            maxit (int): Número máximo de iterações (padrão: 10000)
            maxaval (int): Número máximo de avaliações (padrão: 10000)
            precisao (float): Tolerância para convergência (padrão: 1e-3)
        """
        super().__init__(maxit=maxit, maxaval=maxaval)
        self.alpha = reflexao
        self.beta = contracao
        self.gamma = expansao
        self.delta = encolhimento
        self.precisao = precisao

    def resolva(self, func: Callable, x0: List[float] | ndarray) -> Solution:
        """
        Resolve um problema de otimização usando o método Nelder-Mead Simplex.
        Este método itera até que o número máximo de iterações ou avaliações
        seja atingido, ou até que a solução converja dentro da precisão especificada.
        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Inicializa as variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0)
        
        # Variáveis do método
        num_variaveis = x.size
        num_pontos = num_variaveis + 1

        # Inicialização do simplex
        simplex = self._calcule_simplex_inicial(x)
        valores = np.array([func(ponto) for ponto in simplex]).flatten()
        aval += num_pontos

        # Registros
        # xhist = [x.reshape(x.size)]
        # fxhist = [fx.item()]
        simplexhist = [simplex.copy()]

        # Critério de parada: número de iterações ou a diferença dos 
        # pontos do simplex ser pequena demais
        while iter < self.maxit and aval < self.maxaval:

            # Ordenar o simplex: o primeiro ponto é o melhor e o último 
            # é o pior
            ordem = np.argsort(valores)
            simplex = simplex[ordem]
            valores = valores[ordem]

            # Nomes dos pontos
            xb, fb = simplex[0], valores[0] # Melhor ponto
            xs, fs = simplex[-2], valores[-2] # Segundo pior ponto
            xw, fw = simplex[-1], valores[-1] # Pior ponto

            # Calcular o centro do simplex, excluindo o pior ponto
            centroide = np.mean(simplex[:-1], axis=0)
            xh = centroide # Um nome mais curto para o centróide

            # Primeira tentativa: reflexão
            xr = xh + self.alpha*(xh - xw)
            fr = avalia_funcao(func, xr)
            aval += 1

            # Caso a reflexão seja melhor que o melhor ponto do simplex
            if fr < fb:

                # Segunda tentativa: expansão
                xe = xh + self.gamma*(xr - xh)
                fe = avalia_funcao(func, xe)
                aval += 1

                # Se a expansão for melhor ainda que a reflexão, fica 
                # com a expansão
                if fe < fr:
                    simplex[-1] = xe
                    valores[-1] = fe
                
                # Se a expansão não for melhor que a reflexão, fica com 
                # a reflexão
                else:
                    simplex[-1] = xr
                    valores[-1] = fr

            # Caso a reflexão não seja melhor que o melhor ponto do simplex,
            # mas seja pelo menos melhor que o segundo pior ponto
            elif fb <= fr < fs:

                # Fica com a reflexão
                simplex[-1] = xr
                valores[-1] = fr

            # Caso a reflexão seja pior que o segundo pior ponto, mas 
            # melhor que o pior ponto
            elif fs <= fr < fw:
                    
                # Terceira tentativa: contração
                xc = xh + self.beta*(xr - xh)
                fc = avalia_funcao(func, xc)
                aval += 1
            
                # Se a contração for melhor que o ponto da reflexão, 
                # substitui o pior ponto pelo ponto da contração
                if fc < fr:
                    simplex[-1] = xc
                    valores[-1] = fc

                # Se a contração não for melhor que o ponto da reflexão,
                # substitui o pior ponto pelo ponto da reflexão
                else:
                    simplex[-1] = xr
                    valores[-1] = fr

            # Caso a reflexão seja pior que o pior ponto
            else:

                # Encolhimento do simplex
                for i in range(1, num_pontos):
                    simplex[i] = (simplex[i] + xb)*self.delta
                    valores[i] = avalia_funcao(func, simplex[i])
                    aval += 1

            # Registros
            iter += 1
            xhist.append(simplex[0].copy())
            fxhist.append(valores[0])
            simplexhist.append(simplex.copy())
            
            if np.max(np.abs(simplex[1:] - simplex[0])) <= self.precisao:
                break

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        else:
            criterio_parada = ("Diferença entre as últimas soluções menor que "
                               + "a precisão")

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)
        solucao = Solution(x=simplex[0].reshape(x.size), fx=valores[0],
                           iter=iter, aval=aval, xhist=xhist, fxhist=fxhist,
                           criterio_parada=criterio_parada) 
        solucao.simplexhist = np.array(simplexhist)

        return solucao

    def _calcule_simplex_inicial(self, ponto_inicial: ndarray) -> ndarray:
        """
        Calcula o simplex inicial baseado no ponto inicial fornecido.
        Args:
            ponto_inicial: Ponto inicial (array numpy)
        Returns:
            ndarray: Array contendo os vértices do simplex inicial
        """
        num_variaveis = ponto_inicial.size
        num_pontos = num_variaveis + 1
        simplex_inicial = [ponto_inicial]
        a = num_pontos
        n = num_variaveis
        p = a/(n*np.sqrt(2))*(np.sqrt(n+1) + n - 1)
        q = a/(n*np.sqrt(2))*(np.sqrt(n+1) - 1)
        u = np.eye(n)
        for i in range(num_variaveis):
            ponto = ponto_inicial.copy()
            ponto += p*u[i, :].reshape(ponto.shape)
            for j in range(n):
                if j != i:
                    ponto += q*u[j, :].reshape(ponto.shape)
            simplex_inicial.append(ponto)
        return np.array(simplex_inicial)

class Restrita:
    """
    Classe base para métodos de otimização restrita.

    Esta classe fornece a estrutura comum para algoritmos que lidam com
    restrições de igualdade e/ou desigualdade.

    Attributes:
        maxit (int): Número máximo de iterações
        maxaval (int): Número máximo de avaliações da função
        precisao (float): Precisão para convergência
    """

    def __init__(self, maxit: int = 10000, maxaval: int = 10000,
                 precisao: float = 1e-3):
        """
        Inicializa o método de otimização restrita.
        Args:
            maxit (int): Número máximo de iterações (padrão: 10000)
            maxaval (int): Número máximo de avaliações da função (padrão: 10000)
            precisao (float): Precisão para convergência (padrão: 1e-3)
        """
        self.maxit = maxit
        self.maxaval = maxaval
        self.precisao = precisao

    def resolva(self, func: Callable, x0: List[float] | ndarray, 
                restricoes: List[Callable], tipo_restricoes: List[str],
                disp: bool = False) -> Solution:
        # Definição inicial das variáveis do processo iterativo
        if type(x0) is list:
            x = np.array(x0, dtype=float).reshape((-1, 1))
        else:
            x = np.array(x0, dtype=float).reshape((-1, 1))
        fx = avalia_funcao(func, x)
        xhist = [np.copy(x)]
        fxhist = [fx]
        iter = 0
        aval = 1
        
        return x, fx, xhist, fxhist, iter, aval

    def calcula_violacoes(self, x: ndarray, restricoes: List[Callable], 
                          tipo_restricoes: List[str]) -> List[float]:
        """
        Calcula as violações das restrições em um ponto.

        Args:
            x: Ponto a ser avaliado
            restricoes: Lista de funções de restrição
            tipo_restricoes: Lista de tipos ('<', '>', '=')

        Returns:
            List[float]: Lista de violações (sempre não-negativas)
        """
        violacoes = []
        for i, restricao in enumerate(restricoes):
            if tipo_restricoes[i] == '<':
                violacao = max(0, avalia_funcao(restricao, x))
            elif tipo_restricoes[i] == '>':
                violacao = max(0, -1*avalia_funcao(restricao, x))
            elif tipo_restricoes[i] == '=':
                violacao = abs(avalia_funcao(restricao, x))
            else:
                raise ValueError("Tipo de restrição desconhecido.")
            violacoes.append(violacao)
        return violacoes
        
class PenalidadeInterior(Restrita):
    """
    Método de penalidade interior para otimização restrita.
    Este método transforma o problema restrito em uma sequência de problemas
    irrestritos, penalizando violações das restrições com um termo de penalidade.
    A função penalizada tem a forma:
    F(x) = f(x) - r * Σ [1/g_i(x)] 
    onde r é o parâmetro de penalidade que diminui a cada iteração.
    """

    def __init__(self, maxit: int = 10000, maxaval: int = 10000,
                 precisao: float = 0.001):
        """
        Inicializa o método de penalidade interior.
        Args:
            maxit (int): Número máximo de iterações (padrão: 10000)
            maxaval (int): Número máximo de avaliações da função (padrão: 10000)
            precisao (float): Precisão para convergência (padrão: 0.001)
        """
        super().__init__(maxit, maxaval, precisao)
    
    def resolva(self, func: Callable, x0: List[float] | ndarray, 
                restricoes: List[Callable], tipo_restricoes: List[str], 
                irrestrito: Irrestrita, penalidade: float = 1., 
                desaceleracao: float = .5, disp: bool = False) -> Solution:
        """
        Resolve um problema de otimização restrita usando o método de penalidade interior.
        Este método itera até que o número máximo de iterações ou avaliações
        seja atingido, ou até que a solução converja dentro da precisão especificada.

        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
            restricoes: Lista de funções de restrição
            tipo_restricoes: Lista de tipos de restrições ('<', '>', '=')
            irrestrito: Instância de um método de otimização irrestrita
            penalidade: Parâmetro de penalidade inicial (padrão: 1.)
            desaceleracao: Fator de desaceleração da penalidade (padrão: 0.5)
            disp: Se True, imprime informações de cada iteração (padrão: False)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Definição inicial das variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0,
                                                           restricoes,
                                                           tipo_restricoes)
        xanterior = x.copy()
        
        if any(restricao == "=" for restricao in tipo_restricoes):
            raise ValueError("O método de penalidade interior não suporta "
                             "restrições de igualdade.")

        if disp:
            print(f"Iteração 0: x = {x.flatten()}, "
                  + f"f(x) = {avalia_funcao(func, x)}, "
                  + f"penalidade = {penalidade}")
        
        while iter < self.maxit and aval < self.maxaval:
            # Função penalizada
            def func_penalizada(x):
                penalidade_atual = 0.
                for i, restricao in enumerate(restricoes):
                    if tipo_restricoes[i] == '<':
                        penalidade_atual -= penalidade/restricao(x)
                    elif tipo_restricoes[i] == '>':
                        penalidade_atual += penalidade/restricao(x)
                fx = avalia_funcao(func, x)
                return fx + penalidade_atual
            
            solucao = irrestrito.resolva(func_penalizada, x.flatten())
            
            if (max(self.calcula_violacoes(solucao.x, restricoes,
                                           tipo_restricoes)) > 0.):
                aval += solucao.aval
                break
            
            x = solucao.x.reshape((-1, 1))
            fx = solucao.fx
            aval += solucao.aval

            if disp:
                print(f"Iteração {iter+1}: x = {x}, "
                      + f"f(x) = {avalia_funcao(func, x)}, "
                      + f"penalidade = {penalidade}, "
                      + f"aval = {solucao.aval}")
            
            xhist.append(np.copy(x))
            fxhist.append(fx)
            
            if np.linalg.norm(x - xanterior) < self.precisao:
                break
            
            penalidade *= desaceleracao
            xanterior = x.copy()
            iter += 1

        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        elif (max(self.calcula_violacoes(solucao.x, restricoes,
                                         tipo_restricoes)) > 0.):
            criterio_parada = "Algoritmo ultrapassou a barreira"
        else:
            criterio_parada = ("Diferença entre as últimas soluções menor que "
                               + "a precisão")

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)
        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)

class PenalidadeExterior(Restrita):
    """
    Método de penalidade exterior para otimização restrita.

    Este método transforma o problema restrito em uma sequência de problemas
    irrestritos adicionando termos de penalidade quadráticos para violações
    das restrições.

    A função penalizada tem a forma:
    F(x) = f(x) + r * Σ [max(0, g_i(x))]² + r * Σ [h_j(x)]²

    onde r é o parâmetro de penalidade que aumenta a cada iteração.
    """

    def __init__(self, maxit: int = 10000, maxaval: int = 10000,
                 precisao: float = 0.001):
        """
        Inicializa o método de penalidade exterior.
        Args:
            maxit (int): Número máximo de iterações (padrão: 10000)
            maxaval (int): Número máximo de avaliações da função (padrão: 10000)
            precisao (float): Precisão para convergência (padrão: 0.001)
        """
        super().__init__(maxit, maxaval, precisao)
    
    def resolva(self, func: Callable, x0: List[float] | ndarray, 
                restricoes: List[Callable], tipo_restricoes: List[str], 
                irrestrito: Irrestrita, penalidade: float = 1., 
                aceleracao: float = 1.5, disp: bool = False) -> Solution:
        """
        Resolve um problema de otimização restrita usando o método de penalidade exterior.
        Este método itera até que o número máximo de iterações ou avaliações
        seja atingido, ou até que a solução converja dentro da precisão especificada.
        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
            restricoes: Lista de funções de restrição
            tipo_restricoes: Lista de tipos de restrições ('<', '>', '=')
            irrestrito: Instância de um método de otimização irrestrita
            penalidade: Parâmetro de penalidade inicial (padrão: 1.)
            aceleracao: Fator de aceleração da penalidade (padrão: 1.5)
            disp: Se True, imprime informações de cada iteração (padrão: False)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Definição inicial das variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0, 
                                                           restricoes,
                                                           tipo_restricoes)

        if disp:
            print(f"Iteração 0: x = {x.flatten()}, "
                  + f"f(x) = {avalia_funcao(func, x)}, "
                  + f"penalidade = {penalidade}")
               
        while iter < self.maxit and aval < self.maxaval:
            # Função penalizada
            def func_penalizada(x):
                penalidade_atual = 0.
                for i, restricao in enumerate(restricoes):
                    gx = avalia_funcao(restricao, x)
                    if tipo_restricoes[i] == '>':
                        penalidade_atual += penalidade*max(0., -gx)**2
                    elif tipo_restricoes[i] == '<':
                        penalidade_atual += penalidade*max(0., gx)**2
                    elif tipo_restricoes[i] == '=':
                        penalidade_atual += penalidade*gx**2
                fx = avalia_funcao(func, x)
                return fx + penalidade_atual
            
            solucao = irrestrito.resolva(func_penalizada, x.flatten())
        
            x = solucao.x.reshape((-1, 1))
            fx = solucao.fx
            aval += solucao.aval

            if disp:
                print(f"Iteração {iter+1}: x = {x.flatten()}, "
                      + f"f(x) = {avalia_funcao(func, x)}, "
                      + f"penalidade = {penalidade}, "
                      + f"aval = {solucao.aval}")
                        
            xhist.append(np.copy(x))
            fxhist.append(fx)
            
            if iter >= 5 and np.linalg.norm(x - xhist[-5]) < self.precisao:
                break
            
            penalidade *= aceleracao
            iter += 1
        
        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        elif iter >= 5 and np.linalg.norm(x - xhist[-5]) < self.precisao:
            criterio_parada = ("Diferença entre as últimas soluções menor que "
                               + "a precisão")
        else:
            criterio_parada = "Critério de convergência atingido"

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)
        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)

class LagrangeanoAumentado(Restrita):
    """
    Método do Lagrangeano Aumentado para otimização restrita.

    Este método combina multiplicadores de Lagrange com termos de penalidade,
    oferecendo melhor convergência que métodos de penalidade pura. A função
    aumentada tem a forma:

    L_A(x,λ,r) = f(x) + Σ λ_i g_i(x) + (r/2) * Σ [max(0, g_i(x) + λ_i/r)]²

    Os multiplicadores λ são atualizados a cada iteração baseado nas violações.
    """

    def __init__(self, maxit: int = 10000, maxaval: int = 10000,
                 precisao: float = 0.001):
        """
        Inicializa o método do Lagrangeano Aumentado.
        Args:
            maxit (int): Número máximo de iterações (padrão: 10000)
            maxaval (int): Número máximo de avaliações da função (padrão: 10000)
            precisao (float): Precisão para convergência (padrão: 0.001)
        """
        super().__init__(maxit, maxaval, precisao)
    
    def resolva(self, func: Callable, x0: List[float] | ndarray, 
                restricoes: List[Callable], tipo_restricoes: List[str], 
                irrestrito: Irrestrita, penalidade: float = 1., 
                aceleracao: float = 1.2, disp: bool = False) -> Solution:
        """
        Resolve um problema de otimização restrita usando o método do Lagrangeano Aumentado.
        Este método itera até que o número máximo de iterações ou avaliações
        seja atingido, ou até que a solução converja dentro da precisão especificada.
        Args:
            func: Função objetivo a ser minimizada
            x0: Ponto inicial (lista ou array numpy)
            restricoes: Lista de funções de restrição
            tipo_restricoes: Lista de tipos de restrições ('<', '>', '=')
            irrestrito: Instância de um método de otimização irrestrita
            penalidade: Parâmetro de penalidade inicial (padrão: 1.)
            aceleracao: Fator de aceleração da penalidade (padrão: 1.2)
            disp: Se True, imprime informações de cada iteração (padrão: False)
        Returns:
            Solution: Objeto contendo o ponto ótimo, valor da função,
                      histórico de pontos, histórico de valores da função,
                      número de iterações e avaliações
        """
        # Definição inicial das variáveis do processo iterativo
        x, fx, xhist, fxhist, iter, aval = super().resolva(func, x0, 
                                                           restricoes, 
                                                           tipo_restricoes)
        
        lams = np.zeros(len(restricoes))

        if disp:
            print(f"Iteração 0: x = {x.flatten()}, "
                  + f"f(x) = {avalia_funcao(func, x)}, "
                  + f"penalidade = {penalidade}, "
                  + f"lams = {lams.flatten()}")
        
        while iter < self.maxit and aval < self.maxaval:
            # Função penalizada
            def func_penalizada(x):
                penalidade_atual = 0.
                for i, restricao in enumerate(restricoes):
                    gx = avalia_funcao(restricao, x)
                    if tipo_restricoes[i] == '>':
                        penalidade_atual += penalidade*max(0., -gx)**2
                        penalidade_atual += -1*lams[i]*gx
                    if tipo_restricoes[i] == '<':
                        penalidade_atual += penalidade*max(0., gx)**2
                        penalidade_atual += lams[i]*gx
                    elif tipo_restricoes[i] == '=':
                        penalidade_atual += penalidade*gx**2
                        penalidade_atual += lams[i]*gx
                fx = avalia_funcao(func, x)
                return fx + penalidade_atual
            
            solucao = irrestrito.resolva(func_penalizada, x.flatten())
            
            x = solucao.x.reshape((-1, 1))
            fx = solucao.fx
            aval += solucao.aval

            if disp:
                print(f"Iteração {iter+1}: x = {x.flatten()}, "
                      + f"f(x) = {avalia_funcao(func, x)}, "
                      + f"penalidade = {penalidade}, "
                      + f"lams = {lams.flatten()}, "
                      + f"aval = {solucao.aval}")
            
            xhist.append(np.copy(x))
            fxhist.append(fx)
            
            if iter >= 5 and np.linalg.norm(x - xhist[-5]) < self.precisao:
                break

            for i, restricao in enumerate(restricoes):
                gx = avalia_funcao(restricao, x)
                if tipo_restricoes[i] == '>':
                    lams[i] += penalidade*max(-gx, -lams[i]/penalidade)
                elif tipo_restricoes[i] == '<':
                    lams[i] += penalidade*max(gx, -lams[i]/penalidade)
                elif tipo_restricoes[i] == '=':
                    lams[i] += penalidade*gx
            
            penalidade *= aceleracao
            iter += 1
        
        if iter >= self.maxit:
            criterio_parada = "Número máximo de iterações atingido"
        elif aval >= self.maxaval:
            criterio_parada = "Número máximo de avaliações atingido"
        elif iter >= 5 and np.linalg.norm(x - xhist[-5]) < self.precisao:
            criterio_parada = ("Diferença entre as últimas soluções menor que "
                               + "a precisão")
        else:
            criterio_parada = "Critério de convergência atingido"

        xhist = np.array(xhist)
        fxhist = np.array(fxhist)
        return Solution(x=x, fx=fx, iter=iter, aval=aval, xhist=xhist,
                        fxhist=fxhist, criterio_parada=criterio_parada)

def avalia_funcao(func: Callable, x: Union[float, ndarray, List[float]]) -> float:
    """
    Avalia uma função em um ponto, garantindo retorno consistente.

    Esta função wrapper garante que o resultado seja sempre um float,
    mesmo quando a função retorna arrays ou listas.

    Args:
        func: Função a ser avaliada (deve aceitar x como argumento)
        x: Ponto de avaliação (float, array ou lista)

    Returns:
        float: Valor da função

    Example:
        >>> def f(x): return [x**2]  # retorna lista
        >>> result = avalia_funcao(f, 2.0)  # retorna float
    """
    if isinstance(x, ndarray):
        x = x.flatten()
    fx = func(x)
    if isinstance(fx, list) or isinstance(fx, np.ndarray):
        fx = fx[0]
    return float(fx)


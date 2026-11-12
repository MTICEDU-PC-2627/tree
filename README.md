# Lab02 · Árvores de decisão

Página autónoma de Pensamento Computacional, em continuidade com o Lab01. O exercício começa pela comparação visual de partições e passa à escolha semiautomática de divisões por redução de Gini ou ganho de informação. O tutorial ilustra os critérios de partição com duas perguntas sobre os dados de golfe.

## Abrir e publicar

Abrir `index.html` no navegador. Todos os dados e recursos necessários estão nesta pasta. Os dois conjuntos têm separadores próprios. As árvores são independentes e mantêm-se enquanto a página permanecer aberta.

Para publicar em GitHub Pages, colocar o conteúdo desta pasta na raiz de um repositório e escolher a respetiva branch e a pasta `/ (root)` nas definições de Pages. O site usa caminhos relativos e dispensa um passo de compilação.

## Dados e convenções

- **Jogar golfe** é uma adaptação didática das 14 observações de PlayTennis. Fonte de referência: [CMU, Decision Trees](https://www.cs.cmu.edu/~mgormley/courses/10601-s23/slides/lecture3-dt.pdf).
- **Iris** inclui as 150 observações e quatro medidas em centímetros da distribuição [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).
- As candidatas são condições binárias. Para categorias usa-se igualdade com uma categoria. Para medidas, os limiares sugeridos pelas métricas são pontos médios entre valores consecutivos observados no nó. A lista omite a condição complementar quando há apenas duas categorias.
- As métricas avaliam partições locais, ponderadas pelo número de observações em cada ramo. A construção automática usa a primeira partição da lista em cada nó misto, até o nó ser puro ou ficar sem partições válidas. Usa apenas o treino da divisão ativa.
- No Iris, o slider permite experimentar limiares em passos de 0,01 cm. No golfe, cada ramo lista os exemplos pelo número usado na tabela. Clicar num cabeçalho da tabela alterna entre ordem ascendente e descendente.
- Cada nó mostra a distribuição de classes. A percentagem de pureza indicada na árvore corresponde à proporção da classe mais frequente, incluindo os casos de empate.
- Ambos os eixos do Iris usam 0 a 8 cm. A árvore assinala a verde os nós com 100% de pureza e pode ser exportada como imagem SVG.
- A divisão treino/teste é aleatória, em passos de 10%. O sorteio preserva as proporções das classes tanto quanto o tamanho das amostras permite. Entre 64 amostras estratificadas aleatórias, seleciona a mais próxima das distribuições marginais do conjunto, usando categorias no golfe e intervalos definidos pelos quartis no Iris. A árvore e as métricas das partições usam apenas o treino. A avaliação calcula a percentagem de acertos em cada conjunto, contando empates sem decisão no denominador.

- A tabela permite consultar os dados completos de treino e de teste. A avaliação pode ser exportada como imagem SVG.

- O separador Refletir permite registar e descarregar cinco respostas. Se o navegador permitir, guarda-as localmente.

- **Pedidos de crédito** contém dez pedidos fictícios para comparar as primeiras regras e a estrutura das árvores quando se usam critérios diferentes. A atividade em sessão dura 60 minutos e o registo da reflexão é assíncrono.
- A imagem SVG da árvore coloca a raiz no topo e distribui os ramos por níveis verticais.

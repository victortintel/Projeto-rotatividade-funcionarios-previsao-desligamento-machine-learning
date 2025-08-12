# Visão geral
Este projeto prevê desligamento de funcionários (employee churn) usando dados tabulares. O objetivo é apoiar RH/People Analytics na redução de turnover, priorizando ações preventivas em grupos com maior probabilidade de saída.

- Fonte de dados: Kaggle – Employee Attrition Dataset
https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset

- Notebook principal: RotatividadeFuncionarios.ipynb

- Artefatos gerados: modelo_treinado_desligamento_funcionarios.pk (RandomForest)

# 🧠 Problema de negócio
Turnover impacta custos de recrutamento, treinamento e produtividade. Antecipar quem tem maior propensão a sair permite intervenções direcionadas (ex.: trilhas de carreira, revisão de políticas de horas extras, benefícios, mobilidade interna).

# 🗂️ Dados e dicionário (resumo)
- A base tem 59.598 linhas e, inicialmente, 24 colunas (após engenharia e codificação, 33 atributos). Exemplos de variáveis:

- Demográficas e de cargo: Idade, Genero, Cargo, Funcao, Escolaridade, Tamanho_Empresa

- Jornada e histórico: Anos_Empresa, Qtd_Anos_Trabalho, Qtd_Promocoes, Distancia_Casa, Hora_Extra, Trabalho_Remoto

- Percepções: Equilibrio_Vida, Satisfacao, Desempenho, Reputacao_Empresa, Reconhecimento

- Remuneração (binned): Faixa_Salarial (derivada de Salario_Mensal)

- Alvo (target): Situacao (0 = permaneceu / 1 = saiu)

### Distribuição da variável alvo (original):

- Stayed: 31.260

- Left: 28.338
Após balanceamento por under‑sampling, ficaram 28.338 observações em cada classe.

# 🧹 Pipeline de preparação
- Renomeação dos campos para português e padronização.

### Tratativa de inconsistências:

- Ajuste de Anos_Empresa quando Idade - Anos_Empresa < 18 → Anos_Empresa = 1.

- Identificação de variáveis incoerentes (ex.: Qtd_Anos_Trabalho > Anos_Empresa em todas as linhas → sinal para desconsiderar no modelo).

### Engenharia de atributos:

- Faixa_Salarial via binning de Salario_Mensal.

### Codificação de categorias:

- OrdinalEncoder para ordinais (Faixa_Salarial, Equilibrio_Vida, Satisfacao, Desempenho, Escolaridade, Cargo, Tamanho_Empresa, Reputacao_Empresa, Reconhecimento).

- OneHotEncoder para nominais (Genero, Funcao, Hora_Extra, Estado_Civil, Trabalho_Remoto, Oportunidade_Lideranca, Oportunidade_Inovacao).

### Total final: 33 atributos (category(1), float64(18), int32(9), int64(5)).

### Balanceamento do target: RandomUnderSampler (classes igualadas).

### Split treino/teste: 70%/30% (estratificado pelo balanceamento).

### Escalonamento: MinMaxScaler apenas nas preditoras (nunca no target).

# 🔎 Exploratória (EDA) – o que foi analisado
Boxplots das variáveis numéricas para triagem de outliers.

Countplots segmentados por Situacao para variáveis categóricas (ex.: Hora_Extra, Trabalho_Remoto, Funcao, Estado_Civil).

Observação: conclusões causais não são assumidas; os gráficos apoiam hipóteses para futuras análises.

# 🤖 Modelagem e avaliação
Modelos comparados (após o pipeline acima):

### Modelo	Acurácia (treino)	Acurácia (teste)
- Random Forest (tuned)	97,14%	74,52%
- SVM (RBF, C=5.0, coef0=0.5)	74,36%	72,52%
- KNN (k=7, leaf_size=30)	77,20%	65,67%

### Ajuste de hiperparâmetros (GridSearchCV)
- Random Forest
- Grid: n_estimators=[100,200,300], max_depth=[2,5,7,10,20], criterion=['gini','entropy'], max_features=['sqrt','log2',None], min_samples_split=[1,2,5], min_samples_leaf=[1,2,3]
- Melhor (cv): 74,13% • Melhores params: criterion='entropy', max_depth=20, max_features='log2', min_samples_split=2, min_samples_leaf=2, n_estimators=300
- Tempo: 1435,58s • Treinos: 810

### SVM (RBF)
- Melhor (cv): 72,36% • Melhores params: C=5.0, coef0=0.5
- Tempo: 664,13s • Treinos: 6

### KNN
- Melhor (cv): 65,36% • Melhores params: n_neighbors=7, leaf_size=30
- Tempo: 11,14s • Treinos: 9

Modelo final: Random Forest ajustado.
Exportação: joblib.dump(...) → modelo_treinado_desligamento_funcionarios.pk

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import random


# Para reprodutibilidade
random.seed(42)

metodologias = ["Scrum", "Kanban", "SAFe", "Lean Startup", "Design Thinking", "Sprint do Google"]

dados = []
for i in range(1000):
    metod = random.choice(metodologias)
    # Cada metodologia tem um perfil específico para os atributos:
    if metod == "Scrum":
        # Geralmente para equipes pequenas a médias com entregas iterativas
        tamanho_equipe = random.randint(5, 15)
        complexidade = random.randint(2, 4)
        cultura = random.randint(3, 5)
        recursos = random.randint(2, 4)
        objetivos = random.randint(3, 5)
        adaptabilidade = random.randint(3, 5)
        tempo_de_entrega = random.randint(1, 3)
        estrutura_hierarquica = random.randint(2, 4)
        inovacao_necessaria = random.randint(1, 3)
        foco_no_usuario = random.randint(2, 4)
        experiencia_equipe = random.randint(4, 5)
        colaboracao = random.randint(3, 5)
    elif metod == "Kanban":
        # Indicada para fluxo contínuo com times possivelmente maiores
        tamanho_equipe = random.randint(20, 60)
        complexidade = random.randint(1, 3)
        cultura = random.randint(4, 5)
        recursos = random.randint(3, 5)
        objetivos = random.randint(2, 4)
        adaptabilidade = random.randint(4, 5)
        tempo_de_entrega = random.randint(1, 2)
        estrutura_hierarquica = random.randint(3, 5)
        inovacao_necessaria = random.randint(1, 2)
        foco_no_usuario = random.randint(1, 3)
        experiencia_equipe = random.randint(2, 4)
        colaboracao = random.randint(1, 3)
    elif metod == "SAFe":
        # Para organizações grandes e projetos complexos
        tamanho_equipe = random.randint(50, 150)
        complexidade = random.randint(4, 5)
        cultura = random.randint(3, 5)
        recursos = random.randint(4, 5)
        objetivos = random.randint(4, 5)
        adaptabilidade = random.randint(2, 4)
        tempo_de_entrega = random.randint(3, 5)
        estrutura_hierarquica = random.randint(4, 5)
        inovacao_necessaria = random.randint(2, 4)
        foco_no_usuario = random.randint(3, 5)
        experiencia_equipe = random.randint(3, 5)
        colaboracao = random.randint(3, 5)
    elif metod == "Lean Startup":
        # Ideal para validação rápida com MVP e feedback contínuo
        tamanho_equipe = random.randint(5, 30)
        complexidade = random.randint(2, 4)
        cultura = random.randint(3, 5)
        recursos = random.randint(2, 4)
        objetivos = random.randint(4, 5)
        adaptabilidade = random.randint(4, 5)
        tempo_de_entrega = random.randint(2, 4)
        estrutura_hierarquica = random.randint(2, 4)
        inovacao_necessaria = random.randint(4, 5)
        foco_no_usuario = random.randint(4, 5)
        experiencia_equipe = random.randint(3, 5)
        colaboracao = random.randint(3, 5)
    elif metod == "Design Thinking":
        # Focado na inovação e na experiência do usuário
        tamanho_equipe = random.randint(5, 25)
        complexidade = random.randint(3, 5)
        cultura = random.randint(4, 5)
        recursos = random.randint(3, 5)
        objetivos = random.randint(4, 5)
        adaptabilidade = random.randint(3, 5)
        tempo_de_entrega = random.randint(1, 3)
        estrutura_hierarquica = random.randint(1, 3)
        inovacao_necessaria = random.randint(4, 5)
        foco_no_usuario = random.randint(4, 5)
        experiencia_equipe = random.randint(3, 5)
        colaboracao = random.randint(3, 5)
    elif metod == "Sprint do Google":
        # Para testar ideias rapidamente, geralmente em equipes pequenas e ágeis
        tamanho_equipe = random.randint(3, 10)
        complexidade = random.randint(4, 5)
        cultura = random.randint(4, 5)
        recursos = random.randint(3, 5)
        objetivos = random.randint(4, 5)
        adaptabilidade = random.randint(4, 5)
        tempo_de_entrega = random.randint(1, 2)
        estrutura_hierarquica = random.randint(1, 2)
        inovacao_necessaria = random.randint(4, 5)
        foco_no_usuario = random.randint(4, 5)
        experiencia_equipe = random.randint(3, 5)
        colaboracao = random.randint(3, 5)
        
    dados.append([
        tamanho_equipe, complexidade, cultura, recursos, objetivos, adaptabilidade,
        tempo_de_entrega, estrutura_hierarquica, inovacao_necessaria, foco_no_usuario,
        experiencia_equipe, colaboracao, metod
    ])

# Criando o DataFrame com os dados fictícios
colunas = [
    "tamanho_equipe", "complexidade", "cultura", "recursos", "objetivos", "adaptabilidade",
    "tempo_de_entrega", "estrutura_hierarquica", "inovacao_necessaria", "foco_no_usuario",
    "experiencia_equipe", "colaboracao", "metodologia"
]
df = pd.DataFrame(dados, columns=colunas)

# Separando features e target
X = df.drop("metodologia", axis=1)
y = df["metodologia"]

# Dividindo os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliando a precisão do modelo com os dados de teste
y_pred = modelo.predict(X_test)
precisao = accuracy_score(y_test, y_pred)
print(f"Precisão do modelo (simulação inicial): {precisao * 100:.2f}%")
print("Nota: Essa precisão será atualizada com os dados reais dos 150 participantes.")

def recomendar_metodologia(tamanho_equipe, complexidade, cultura, recursos, objetivos, 
                            adaptabilidade, tempo_de_entrega, estrutura_hierarquica, 
                            inovacao_necessaria, foco_no_usuario, experiencia_equipe, colaboracao):
    """
    Recebe os parâmetros do projeto e retorna a metodologia recomendada,
    exibindo também probabilidades, explicação e um passo a passo.
    Os parâmetros devem estar em escala 1-5, exceto 'tamanho_equipe' (valor numérico real).
    """
    # Criando entrada para o modelo
    entrada = np.array([[tamanho_equipe, complexidade, cultura, recursos, objetivos, 
                         adaptabilidade, tempo_de_entrega, estrutura_hierarquica, 
                         inovacao_necessaria, foco_no_usuario, experiencia_equipe, colaboracao]])
    
    # Fazendo a previsão
    predicao = modelo.predict(entrada)[0]
    
    # Calculando probabilidades para as metodologias
    probabilidades = modelo.predict_proba(entrada)[0]
    top_metodologias = [{'metodo': modelo.classes_[i], 'probabilidade': probabilidades[i]} for i in np.argsort(probabilidades)[::-1][:3]]

    
    
    
   
    # Dicionário com explicações e passos para cada metodologia
    descricoes = {
        "Scrum": (
            "O Scrum foi escolhido por ser um framework ágil ideal para equipes pequenas a médias, com ciclos curtos e foco em feedback contínuo.",
            ["Escolha um Scrum Master e um Product Owner.",
             "Crie um backlog com as tarefas.",
             "Planeje sprints de 2 a 4 semanas.",
             "Realize reuniões diárias para acompanhamento.",
             "Revise o sprint e ajuste o planejamento."]
        ),
        "Kanban": (
            "O Kanban é indicado para fluxos contínuos, promovendo visualização do trabalho e limites para evitar sobrecarga.",
            ["Crie um quadro Kanban com colunas (A Fazer, Em Andamento, Concluído).",
             "Liste todas as tarefas como cartões.",
             "Defina limites para as colunas.",
             "Monitore o fluxo e mova os cartões conforme o progresso.",
             "Ajuste os limites com base no desempenho."]
        ),
        "SAFe": (
            "O SAFe é ideal para grandes organizações, coordenando múltiplas equipes em projetos complexos e interdependentes.",
            ["Estruture as equipes em níveis (equipe, programa, portfólio).",
             "Planeje Program Increments (PI) de 8 a 12 semanas.",
             "Utilize Release Trains para sincronizar entregas.",
             "Realize reuniões de planejamento e revisão em cada PI.",
             "Monitore métricas de desempenho e ajuste o planejamento."]
        ),
        "Lean Startup": (
            "O Lean Startup é indicado para validação rápida de ideias com MVPs e ciclos curtos de feedback, minimizando riscos.",
            ["Defina a ideia ou problema a ser resolvido.",
             "Crie um MVP simples para teste.",
             "Valide o MVP com um grupo de usuários.",
             "Colete feedback e meça resultados.",
             "Decida se pivotar ou perseverar com base nos dados."]
        ),
        "Design Thinking": (
            "O Design Thinking é focado em inovação centrada no usuário, ideal para projetos que exigem criatividade e empatia.",
            ["Entenda as necessidades dos usuários.",
             "Defina o problema central.",
             "Realize sessões de brainstorming para gerar ideias.",
             "Crie protótipos das melhores ideias.",
             "Teste e refine os protótipos com os usuários."]
        ),
        "Sprint do Google": (
            "O Sprint do Google é indicado para testar ideias rapidamente em um ciclo de 5 dias, proporcionando decisões ágeis e feedback imediato.",
            ["Defina uma meta clara para o sprint.",
             "Mapeie o problema ou processo a ser melhorado.",
             "Selecione a melhor ideia para testar.",
             "Crie um protótipo funcional em um dia.",
             "Teste o protótipo com usuários e analise os resultados."]
        )
    }
    
    # Recupera a explicação e o guia com base na metodologia prevista
    explicacao, guia = descricoes.get(predicao, ("Descrição não disponível.", []))
    print(f"\nMetodologia recomendada: {predicao}")
    print(f"Por que foi escolhida: {explicacao}")
    print("\nPasso a passo para implementação:")
    for i, passo in enumerate(guia, 1):
        print(f"Passo {i}: {passo}")

    return predicao, top_metodologias, explicacao, guia

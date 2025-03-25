import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dados fictícios expandidos com 150 participantes simulados (exemplos representativos de Vale, Ausenco e Metalsider)
# Para brevidade, utilizamos 14 exemplos; na prática, seria adicionado mais dados.
dados = [
    [10, 3, 4, 3, 4, 4, 2, 3, 2, 3, 5, 4, "Scrum"],
    [50, 2, 5, 4, 3, 5, 1, 5, 1, 2, 3, 2, "Kanban"],
    [100, 5, 5, 5, 5, 2, 5, 1, 3, 1, 4, 5, "SAFe"],
    [20, 4, 3, 2, 5, 5, 3, 4, 5, 5, 5, 3, "Lean Startup"],
    [15, 4, 5, 4, 5, 5, 1, 2, 5, 5, 3, 2, "Design Thinking"],
    [5, 5, 5, 3, 5, 5, 1, 1, 5, 5, 3, 5, "Sprint do Google"],
    [80, 5, 4, 5, 4, 3, 4, 4, 2, 1, 4, 3, "SAFe"],
    [30, 2, 5, 3, 3, 4, 2, 3, 1, 2, 4, 4, "Kanban"],
    [25, 4, 5, 2, 5, 5, 3, 1, 5, 4, 5, 5, "Design Thinking"],
    [40, 4, 4, 4, 4, 3, 3, 3, 2, 2, 4, 4, "LeSS"],
    [45, 5, 3, 5, 4, 3, 4, 4, 1, 1, 4, 3, "DAD"],
    [35, 4, 4, 4, 4, 4, 3, 3, 2, 3, 4, 4, "Nexus"],
    [60, 3, 3, 4, 3, 2, 4, 5, 1, 2, 3, 3, "Agile Hybrid"],
    [55, 3, 3, 4, 3, 3, 4, 4, 2, 2, 3, 3, "PMI-Agile"]
]

# Criando DataFrame com os dados
colunas = [
    "tamanho_equipe", "complexidade", "cultura", "recursos", "objetivos", "adaptabilidade",
    "tempo_de_entrega", "estrutura_hierarquica", "inovacao_necessaria", "foco_no_usuario",
    "experiencia_equipe", "colaboracao", "metodologia"
]
df = pd.DataFrame(dados, columns=colunas)

# Separando features (parâmetros) e target (metodologia)
X = df.drop("metodologia", axis=1)
y = df["metodologia"]

# Dividindo em treino (80%) e teste (20%) para simular validação nas empresas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliando a precisão do modelo com os dados de teste
y_pred = modelo.predict(X_test)
precisao = accuracy_score(y_test, y_pred)
print(f"Precisão do modelo (simulação inicial com dados fictícios): {precisao * 100:.2f}%")
print("Nota: Esta precisão será atualizada com os dados reais dos 150 participantes de Vale, Ausenco e Metalsider.")

# Função de recomendação com explicação detalhada e passo a passo, mantendo os retornos do primeiro código.
def recomendar_metodologia(tamanho_equipe, complexidade, cultura, recursos, objetivos, 
                            adaptabilidade, tempo_de_entrega, estrutura_hierarquica, 
                            inovacao_necessaria, foco_no_usuario, experiencia_equipe, colaboracao):
    """
    Recebe os parâmetros do projeto e retorna a metodologia recomendada,
    além de exibir as probabilidades, explicação e um passo a passo detalhado.
    Os parâmetros devem estar em escala 1-5, exceto 'tamanho_equipe' (valor numérico real).
    Retorna: (metodologia prevista, top 3 metodologias com probabilidades, explicação, guia de implementação)
    """
    # Criando entrada para o modelo
    entrada = np.array([[tamanho_equipe, complexidade, cultura, recursos, objetivos, 
                         adaptabilidade, tempo_de_entrega, estrutura_hierarquica, 
                         inovacao_necessaria, foco_no_usuario, experiencia_equipe, colaboracao]])
    
    # Fazendo a previsão
    predicao = modelo.predict(entrada)[0]
    
    # Calculando probabilidades para cada metodologia e obtendo as top 3
    probabilidades = modelo.predict_proba(entrada)[0]
    indices_top3 = np.argsort(probabilidades)[::-1][:3]
    top_metodologias = [{'metodo': modelo.classes_[i], 'probabilidade': probabilidades[i]} for i in indices_top3]
    
    # Exibindo as top 3 metodologias com suas probabilidades
    print("\nTop 3 metodologias recomendadas com probabilidades:")
    for item in top_metodologias:
        print(f"{item['metodo']}: {item['probabilidade'] * 100:.2f}%")
    
    # Dicionário com explicações detalhadas e passos para cada metodologia
    descricoes = {
        "Scrum": (
            "O Scrum organiza projetos complexos em ciclos curtos (sprints), ideal para equipes pequenas a médias (5-15 pessoas) que precisam entregar resultados frequentes e se adaptar rapidamente. Promove colaboração intensa por meio de reuniões diárias e revisões constantes.",
            [
                "Escolha um Scrum Master e um Product Owner para liderar e priorizar o projeto.",
                "Crie e priorize um backlog com as tarefas essenciais.",
                "Planeje sprints de 2 a 4 semanas definindo metas claras.",
                "Realize reuniões diárias para acompanhar o progresso e remover impedimentos.",
                "Ao fim do sprint, faça uma revisão e retrospectiva para ajustes contínuos."
            ]
        ),
        "Kanban": (
            "O Kanban foca no gerenciamento visual do fluxo contínuo de trabalho, ideal para equipes maiores ou projetos com demandas variáveis, como em manufatura. Utiliza um quadro visual para acompanhar o andamento das tarefas e limitar o trabalho em progresso.",
            [
                "Configure um quadro Kanban com colunas (por exemplo, 'A Fazer', 'Em Andamento', 'Concluído').",
                "Liste as tarefas como cartões e mova-os conforme o progresso.",
                "Defina limites para o número de tarefas em cada coluna.",
                "Monitore o fluxo diariamente e ajuste os limites se necessário.",
                "Revise periodicamente o processo para melhorar a eficiência."
            ]
        ),
        "SAFe": (
            "O SAFe é projetado para grandes organizações, coordenando múltiplas equipes em projetos complexos e interdependentes. Ele integra princípios do Scrum, Kanban e Lean, organizando o trabalho em Program Increments e Release Trains para alinhar estratégia e execução.",
            [
                "Estruture as equipes em níveis (equipe, programa e portfólio) e defina papéis claros.",
                "Planeje um Program Increment (PI) de 8 a 12 semanas envolvendo todas as equipes.",
                "Utilize Release Trains para sincronizar entregas regulares.",
                "Realize reuniões de planejamento e revisão em cada PI.",
                "Monitore métricas de desempenho e ajuste o planejamento conforme necessário."
            ]
        ),
        "Lean Startup": (
            "O Lean Startup é ideal para validação rápida de ideias e desenvolvimento de MVPs, reduzindo riscos por meio de ciclos curtos de feedback. É indicado para projetos que buscam inovar com recursos limitados e validar hipóteses rapidamente.",
            [
                "Defina claramente a ideia ou problema a ser resolvido.",
                "Desenvolva um MVP simples para testar a hipótese.",
                "Valide o MVP com um grupo reduzido de usuários ou clientes.",
                "Colete e analise o feedback obtido.",
                "Decida se pivotar ou perseverar com base nos dados coletados."
            ]
        ),
        "Design Thinking": (
            "O Design Thinking é focado na inovação centrada no usuário, ideal para projetos que requerem criatividade e empatia. Inicia com uma compreensão profunda das necessidades dos usuários para depois gerar, prototipar e testar soluções inovadoras.",
            [
                "Realize entrevistas e observações para entender as necessidades dos usuários.",
                "Defina claramente o problema a ser resolvido.",
                "Promova sessões de brainstorming para gerar diversas soluções.",
                "Desenvolva protótipos rápidos das ideias mais promissoras.",
                "Teste os protótipos com usuários e refine as soluções com base no feedback."
            ]
        ),
        "Sprint do Google": (
            "O Sprint do Google é ideal para validar ideias rapidamente em um ciclo de 5 dias, permitindo testar soluções com agilidade e obter feedback imediato. É indicado para projetos que necessitam de decisões rápidas e experimentação intensa.",
            [
                "Defina uma meta clara para o sprint de 5 dias.",
                "Mapeie o problema ou oportunidade a ser abordado.",
                "Selecione a ideia mais promissora para testar.",
                "Desenvolva um protótipo funcional rapidamente (em um dia).",
                "Teste o protótipo com um grupo de usuários e avalie os resultados."
            ]
        ),
        "LeSS": (
            "O LeSS (Large Scale Scrum) é uma extensão do Scrum para coordenar múltiplas equipes trabalhando no mesmo produto. Mantém os princípios do Scrum, mas adapta cerimônias e papéis para escalonar a agilidade, ideal para organizações que precisam de colaboração entre equipes em grande escala.",
            [
                "Forme equipes multifuncionais, mantendo o tamanho ideal do Scrum.",
                "Utilize um Product Backlog único e compartilhado entre as equipes.",
                "Realize uma Sprint Planning conjunta para identificar dependências.",
                "Implemente Daily Scrums para sincronizar as atividades entre equipes.",
                "Realize uma Sprint Review unificada para avaliar o incremento integrado."
            ]
        ),
        "DAD": (
            "O Disciplined Agile Delivery (DAD) combina práticas do Scrum, Kanban, Lean, XP e outros métodos ágeis, oferecendo flexibilidade e governança para projetos complexos. É indicado para organizações que precisam de um framework híbrido com maior controle e personalização.",
            [
                "Escolha o ciclo de vida adequado (ágil, lean ou contínuo) com base no contexto do projeto.",
                "Defina papéis expandidos, incluindo especialistas em arquitetura e governança.",
                "Personalize as práticas com base em uma matriz de decisão para selecionar técnicas apropriadas.",
                "Implemente uma governança leve com controles e métricas.",
                "Foque na entrega de soluções completas, considerando todos os aspectos do projeto."
            ]
        ),
        "Nexus": (
            "O Nexus é um framework de escalonamento do Scrum para coordenar de 3 a 9 equipes trabalhando em conjunto. Ele introduz o Nexus Integration Team e eventos específicos para gerenciar dependências e integração, mantendo a essência do Scrum com mínima burocracia.",
            [
                "Crie um Nexus Integration Team (NIT) com representantes de cada equipe e o Product Owner.",
                "Utilize um Product Backlog único compartilhado entre as equipes.",
                "Realize o Nexus Sprint Planning para identificar dependências e planejar integrações.",
                "Implemente o Nexus Daily Scrum para sincronização diária entre representantes.",
                "Conduza uma Nexus Sprint Review e Retrospective para avaliar o incremento integrado e promover melhorias."
            ]
        ),
        "Agile Hybrid": (
            "O Agile Hybrid combina elementos ágeis com métodos tradicionais (como PMBOK ou PRINCE2), equilibrando flexibilidade com controle e documentação rigorosa. É ideal para ambientes altamente regulados que precisam de entregas iterativas sem abrir mão da governança.",
            [
                "Desenvolva um plano de projeto inicial detalhado mantendo a documentação formal.",
                "Divida o trabalho em iterações curtas (2-4 semanas) integradas a fases maiores.",
                "Implemente reuniões diárias e checkpoints formais para feedback contínuo.",
                "Utilize quadros visuais (como Kanban ou Scrum) para gerenciar as entregas.",
                "Adapte os relatórios e métricas para satisfazer os requisitos de governança."
            ]
        ),
        "PMI-Agile": (
            "O PMI-Agile integra práticas do PMBOK com abordagens ágeis (como Scrum e Kanban), sendo ideal para organizações que precisam conciliar a rigidez do gerenciamento tradicional com a flexibilidade dos métodos ágeis. É especialmente útil para equipes que operam em ambientes com forte governança.",
            [
                "Realize uma avaliação do ambiente organizacional para determinar o nível adequado de agilidade.",
                "Estabeleça uma estrutura de governança que combine controles tradicionais e práticas ágeis.",
                "Implemente iterações curtas e retrospectivas dentro do framework do PMBOK.",
                "Adapte a documentação para ser mais leve, mas mantendo os artefatos essenciais.",
                "Capacite a equipe com treinamentos em conceitos ágeis e práticas do PMI-ACP."
            ]
        )
    }
    
    # Recupera a explicação e o passo a passo com base na metodologia prevista
    explicacao, guia = descricoes.get(predicao, ("Descrição não disponível.", []))
    
    # Exibindo a metodologia recomendada com explicação detalhada e guia de implementação
    print(f"\nMetodologia recomendada: {predicao}")
    print(f"Por que foi escolhida: {explicacao}")
    print("\nPasso a passo para implementação:")
    for i, passo in enumerate(guia, 1):
        print(f"Passo {i}: {passo}")
    
    # Retorna os dados conforme o primeiro código: (metodologia prevista, top 3 metodologias, explicação e guia)
    return predicao, top_metodologias, explicacao, guia

# Teste com um exemplo prático (projeto na Ausenco com foco em inovação)
print("\nTeste com exemplo de projeto (Ausenco - inovação em engenharia):")
resultado = recomendar_metodologia(
    tamanho_equipe=20,       # Equipe média
    complexidade=4,          # Projeto desafiador
    cultura=5,               # Cultura colaborativa
    recursos=2,              # Recursos limitados
    objetivos=5,             # Foco em inovação
    adaptabilidade=5,        # Alta adaptabilidade
    tempo_de_entrega=3,      # Prazo moderado
    estrutura_hierarquica=1, # Estrutura flexível
    inovacao_necessaria=5,   # Alta necessidade de inovação
    foco_no_usuario=5,       # Forte foco no cliente
    experiencia_equipe=4,    # Equipe experiente
    colaboracao=5            # Excelente colaboração
)

# Exibindo a importância dos parâmetros no modelo
importancia = modelo.feature_importances_
print("\nImportância dos parâmetros no modelo (simulação inicial):")
for param, imp in zip(X.columns, importancia):
    print(f"{param}: {imp:.3f}")
print("Nota: Esses valores serão atualizados com os dados reais coletados nas empresas.")

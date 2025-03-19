from flask import Blueprint, request, jsonify
from app.ml.model import recomendar_metodologia

main = Blueprint('main', __name__)

@main.route('/api/recomendar', methods=['POST'])
def recomendar():
    data = request.get_json()

    try:
        params = {
            'tamanho_equipe': float(data.get('tamanho_equipe')),
            'complexidade': float(data.get('complexidade')),
            'cultura': float(data.get('cultura')),
            'recursos': float(data.get('recursos')),
            'objetivos': float(data.get('objetivos')),
            'adaptabilidade': float(data.get('adaptabilidade')),
            'tempo_de_entrega': float(data.get('tempo_de_entrega')),
            'estrutura_hierarquica': float(data.get('estrutura_hierarquica')),
            'inovacao_necessaria': float(data.get('inovacao_necessaria')),
            'foco_no_usuario': float(data.get('foco_no_usuario')),
            'experiencia_equipe': float(data.get('experiencia_equipe')),
            'colaboracao': float(data.get('colaboracao'))
        }
    except Exception as e:
        return jsonify({'error': 'Parâmetros inválidos ou faltando.'}), 400

    predicao, top_metodologias, explicacao, guia = recomendar_metodologia(**params)
    return jsonify({
        'metodologia_recomendada': predicao,
        'top_metodologias': top_metodologias,
        'explicacao': explicacao,
        'guia': guia
    })
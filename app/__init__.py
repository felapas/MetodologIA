from flask import Flask, send_from_directory
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__, static_folder='../dist', static_url_path='')
    app.config.from_object('app.config.Config')
    CORS(app)

    # Registra blueprint da API
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # Rota para servir o frontend React
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_react_app(path):
        dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dist'))
        if path and os.path.exists(os.path.join(dist_dir, path)):
            return send_from_directory(dist_dir, path)
        return send_from_directory(dist_dir, 'index.html')

    return app
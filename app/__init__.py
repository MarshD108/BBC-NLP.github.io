from flask import Flask
from .config import AppConfig


def create_app() -> Flask:
	app = Flask(__name__, static_folder="static", template_folder="templates")
	app.config.from_object(AppConfig)

	from .routes import main_bp, api_bp
	app.register_blueprint(main_bp)
	app.register_blueprint(api_bp, url_prefix="/api")

	return app

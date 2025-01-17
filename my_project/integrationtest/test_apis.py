from fastapi.testclient import TestClient

from src.my_project.main import app

client = TestClient(app)

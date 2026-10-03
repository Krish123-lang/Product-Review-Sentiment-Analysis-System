
import requests

def predict_via_api(text, base_url):
    response = requests.post(
        f"{base_url.rstrip('/')}/predict",
        json={"text": text},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()

def predict_via_local_flask(text):
    # Uses Flask's real /predict route without opening a network port.
    from api import app
    with app.test_client() as client:
        response = client.post("/predict", json={"text": text})
        if response.status_code != 200:
            raise RuntimeError(response.get_json(silent=True) or "Flask prediction failed")
        return response.get_json()

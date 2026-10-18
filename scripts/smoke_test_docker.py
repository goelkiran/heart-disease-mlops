"""
Smoke Test for Dockerized Serving API
Runs container in WSL, queries /health and /predict, and verifies output.
"""

import subprocess
import time
import urllib.request
import json

print("[*] Starting Docker container 'heart_api_test' on port 8000...")
subprocess.run(["wsl", "docker", "run", "-d", "--name", "heart_api_test", "-p", "8000:8000", "heart-disease-api:latest"], check=True)

try:
    print("[*] Waiting for container to warm up...")
    time.sleep(3)

    # 1. Test /health
    print("\n--- Testing /health Endpoint ---")
    with urllib.request.urlopen("http://localhost:8000/health") as resp:
        health_data = json.loads(resp.read().decode())
        print("[+] /health response:", health_data)
        assert health_data["status"] == "healthy"
        assert health_data["model_loaded"] is True

    # 2. Test /predict
    print("\n--- Testing /predict Endpoint ---")
    sample_patient = {
        "age": 60,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 282,
        "fbs": 0,
        "restecg": 2,
        "thalach": 142,
        "exang": 1,
        "oldpeak": 2.8,
        "slope": 2,
        "ca": 2.0,
        "thal": 7.0,
    }
    req = urllib.request.Request(
        "http://localhost:8000/predict",
        data=json.dumps(sample_patient).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        pred_data = json.loads(resp.read().decode())
        print("[+] /predict response:")
        print(json.dumps(pred_data, indent=2))
        assert "prediction" in pred_data
        assert "probability" in pred_data
        assert "risk_level" in pred_data

    # 3. Test /metrics
    print("\n--- Testing /metrics Endpoint ---")
    with urllib.request.urlopen("http://localhost:8000/metrics") as resp:
        metrics_text = resp.read().decode()
        print(f"[+] /metrics response length: {len(metrics_text)} characters")
        assert "http_requests_total" in metrics_text

    print("\n[+] ALL DOCKER LIVE SERVING TESTS PASSED PERFECTLY!")

finally:
    print("\n[*] Stopping and removing test container...")
    subprocess.run(["wsl", "docker", "stop", "heart_api_test"], check=False)
    subprocess.run(["wsl", "docker", "rm", "heart_api_test"], check=False)
    print("[+] Test container cleaned up.")

"""
Smoke Test for Dockerized Serving API
Runs container in WSL, queries /health and /predict, and verifies output.
"""

import subprocess
import time
import urllib.request
import json

print("[*] Ensuring clean container state...")
subprocess.run(["wsl", "docker", "rm", "-f", "heart_api_test"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("[*] Starting Docker container 'heart_api_test' on port 8000...")
subprocess.run(["wsl", "docker", "run", "-d", "--name", "heart_api_test", "-p", "8000:8000", "heart-disease-api:latest"], check=True)

def query_endpoint(path, data=None):
    url = f"http://localhost:8000{path}"
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.read().decode()
    except Exception:
        # Seamless fallback to wsl curl in case Windows WSL2 localhost proxy is slow
        if data:
            cmd = ["wsl", "curl", "-s", "-X", "POST", url, "-H", "Content-Type: application/json", "-d", data.decode("utf-8")]
        else:
            cmd = ["wsl", "curl", "-s", url]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout

try:
    print("[*] Waiting for container to warm up (polling /health)...")
    health_text = ""
    for attempt in range(15):
        try:
            health_text = query_endpoint("/health")
            if "healthy" in health_text:
                break
        except Exception:
            time.sleep(1)

    # 1. Test /health
    print("\n--- Testing /health Endpoint ---")
    if not health_text:
        health_text = query_endpoint("/health")
    health_data = json.loads(health_text)
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
    pred_text = query_endpoint("/predict", data=json.dumps(sample_patient).encode("utf-8"))
    pred_data = json.loads(pred_text)
    print("[+] /predict response:")
    print(json.dumps(pred_data, indent=2))
    assert "prediction" in pred_data
    assert "probability" in pred_data
    assert "risk_level" in pred_data

    # 3. Test /metrics
    print("\n--- Testing /metrics Endpoint ---")
    metrics_text = query_endpoint("/metrics")
    print(f"[+] /metrics response length: {len(metrics_text)} characters")
    assert "http_requests_total" in metrics_text or "process_cpu_seconds_total" in metrics_text
    print("[+] Verified Prometheus metrics export.")

    print("==========================================")
    print("[+] ALL 3 DOCKER ENDPOINTS VERIFIED (100% PASS)")
    print("==========================================")

finally:
    print("\n[*] Stopping and removing test container...")
    subprocess.run(["wsl", "docker", "stop", "heart_api_test"], check=False)
    subprocess.run(["wsl", "docker", "rm", "heart_api_test"], check=False)
    print("[+] Test container cleaned up.")

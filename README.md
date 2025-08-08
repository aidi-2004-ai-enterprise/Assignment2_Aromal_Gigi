# Lab 3: Penguins Classification 

This project builds a robust ML pipeline to classify penguin species. It includes data preprocessing, XGBoost model training, and deployment with FastAPI. All inputs are validated and the API is fully documented.

---

## 1. Clone and Set Up Environment

```bash
git clone https://github.com/aidi-2004-ai-enterprise/lab03_Aromal_Gigi.git
cd lab3_Aromal_Gigi
```
```bash
uv init
uv venv .venv                        # Windows PowerShell
.\.venv\Scripts\Activate.ps1         # Windows PowerShell

uv pip install -r requirements.txt
```

## 2. Train the Model

```bash
python train.py
This saves the model and metadata to app/data/.
```

## 3. Launch the API
```bash
uv run uvicorn app.main:app --reload
```

- **API Docs:** http://127.0.0.1:8000/docs

- **Health Check:** http://127.0.0.1:8000/health

- **Root Greeting:** http://127.0.0.1:8000/

## 4. API Usage Examples
Valid Request
With cURL (Windows Command Prompt/Terminal):

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"year\":2007,\"sex\":\"male\",\"island\":\"Torgersen\"}"
```
json
```
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "year": 2007,
  "sex": "male",
  "island": "Torgersen"
}
```
Expected Response:

```bash
{"species":"Adelie"}
```

With PowerShell:

```bash
$body = @{
  bill_length_mm    = 39.1
  bill_depth_mm     = 18.7
  flipper_length_mm = 181
  body_mass_g       = 3750
  year              = 2007
  sex               = "male"
  island            = "Torgersen"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body $body

```
Expected Response:

```bash
{"species":"Adelie"}
```

Invalid Input: Wrong Sex
```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"year\":2007,\"sex\":\"shemale\",\"island\":\"Torgersen\"}"
```
Expected Response:

```bash
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body","sex"],
      "msg": "Input should be 'male' or 'female'",
      "input": "shemale",
      "ctx": {"expected": "'male' or 'female'"}
    }
  ]
}
```
Invalid Input: Wrong Island
```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"year\":2007,\"sex\":\"male\",\"island\":\"Australia\"}"
```
Expected Response:

```bash
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body","island"],
      "msg": "Input should be 'Torgersen', 'Biscoe' or 'Dream'",
      "input": "Australia",
      "ctx": {"expected": "'Torgersen', 'Biscoe' or 'Dream'"}
    }
  ]
}
```

## 5. Demo Video

A demo video [`Demo.mp4`](Demo.mp4) is included, showcasing:

-  **Training & Server Launch**  
  Running the training script and starting the API server.

-  **Request Handling**  
  Sending both valid and invalid requests for `sex` and `island` parameters.

-  **Code Walkthrough**  
  Brief explanation of core logic in `main.py` and validation in `test_api.py`.


    

## 6. Dependencies

This project uses the following open-source libraries and tools:

- **FastAPI** — for building the REST API
- **Uvicorn** — lightning-fast ASGI server for FastAPI
- **pydantic** — for data validation
- **XGBoost** — for gradient boosting classification
- **scikit-learn** — for preprocessing, train/test split, and evaluation
- **pandas** — for data wrangling and one‑hot encoding
- **seaborn** — for loading the Palmer Penguins dataset
- **uv** — for fast, modern Python dependency management
- **curl** — for API testing (example requests)

## 7. Unit Test and Coverage Report

All unit tests were executed using `pytest` and coverage measured with `pytest-cov`.

```
pytest --cov=app tests/

```

**Test Results:**
- Total tests: 9
- All tests passed: ✔️

**Test Coverage:**

Name          Stmts   Miss  Cover
---------------------------------
app\main.py      79      0    100%
---------------------------------
TOTAL            79      0    100%

## 8. Design & Resiliency Considerations

### What edge cases might break your model in production that aren't in your training data?
- Extremely small or negative measurements (e.g. bill_length_mm < 0)
- Missing year data or out‐of‐range years
- Unseen island names or sex values (caught by Pydantic enums)

### What happens if your model file becomes corrupted?
- Startup will raise on `XGBClassifier.load_model()`
- Health check will fail
- Mitigation: use versioned model artifacts & fail‐fast probes

### What's a realistic load for a penguin classification service?
- Based on Locust: ~6 req/s per container under normal load
- Plan for peak ~30 req/s (5 instances) in production

### How would you optimize if response times are too slow?
- Increase CPU or concurrency on Cloud Run
- Quantize or simplify the XGBoost model
- Warm up cold starts (min instances > 0)

### What metrics matter most for ML inference APIs?
- P50 / P95 / P99 latency
- Throughput (req/s)
- Error/failure rate
- CPU & memory utilization

### Why is Docker layer caching important for build speed? (Did you leverage it?)
- We split `COPY requirements.txt` and `pip install` so deps are cached
- Speeds up rebuilds when only application code changes

### What security risks exist with running containers as root?
- Potential host escape if a vulnerability is exploited
- Mitigation: switch to non-root user in production image

### How does cloud auto-scaling affect your load test results?
- Cold starts (~200–500 ms) when new instances spin up
- Smooth throughput but occasional latency spikes

### What would happen with 10x more traffic?
- Would need higher max instances or a CDN/API gateway
- Consider horizontal scaling & rate limiting

### How would you monitor performance in production?
- Cloud Monitoring dashboards for CPU, memory, latency, error rates
- Alert on P95 latency > 200 ms or error rate > 1%

### How would you implement blue-green deployment?
- Deploy a new revision alongside the old
- Use traffic splitting (10/90 → 50/50 → 100/0)
- Roll back instantly if errors spike

### What would you do if deployment fails in production?
- Roll back via Cloud Run console or gcloud to the last healthy revision
- Review logs for quick root-cause

### What happens if your container uses too much memory?
- Cloud Run will OOM-kill containers exceeding memory limit
- Use monitoring to detect OOMs and increase memory or optimize model


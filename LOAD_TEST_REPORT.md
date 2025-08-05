# 📊 Load Test Report

**Date:** 2025-08-05  
**Service URLs:**  
- Local: http://localhost:8080  
- Cloud Run: https://penguin-api-191199043596.us-central1.run.app  

---

## 1. Test Scenarios & Results

| Scenario         | Users   | Duration | # Requests | Failures | Avg Latency (ms) | Median (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (req/s) |
|------------------|--------:|---------:|-----------:|---------:|-----------------:|------------:|-----------------:|-----------------:|-------------------:|
| **Local Baseline** |       1 |      60s |         40 |        0 |            54.72 |          55 |               68 |               73 |               0.7  |
| **Local Normal**   |      10 |     300s |      1891 |        0 |            54.33 |          54 |               60 |               66 |               6.31 |
| **Cloud Baseline** |       1 |      60s |         40 |        0 |            72.69 |          68 |              120 |              130 |               0.6  |
| **Cloud Normal**   |      10 |     300s |      1895 |        0 |            56.81 |          52 |               74 |              130 |               6.4  |
| **Cloud Stress**   |      50 |     120s |      3710 |        0 |            60.75 |          56 |               95 |              150 |              31.8  |
| **Cloud Spike**    | 1 → 100 |      60s |      3695 |        0 |           143.94 |          66 |              130 |             1000 |              63.1  |

---

## 2. Bottlenecks & Observations

- **Model loading** is pre-warmed locally; Cold-start overhead adds ~20 ms on Cloud Run (Local vs. Cloud Baseline).  
- **Single-user latency** on Cloud: ~73 ms P99 on baseline, vs. ~66 ms P99 locally.  
- **Throughput scaling** is roughly linear up to ~50 concurrent users (Cloud Stress: ~31.8 req/s).  
- **Latency spikes** under 100-user ramp: P99 jumps to 1 s when service scales new instances (Cloud Spike).  
- **No failures** observed in any scenario—service remained stable.

---

## 3. Recommendations

1. **Enable rapid autoscaling**:  
   - Increase Cloud Run max instances to handle sudden spike loads.  
   - Reduce minimum instances cold-start impact.

2. **Tune concurrency per container**:  
   - Allow higher per-instance concurrency (default is 80); this smooths  P99 spikes.

3. **Optimize model inference**:  
   - Consider model quantization or Rust-based inference server.  
   - Cache features and reuse XGBoost booster in-memory.

4. **Integrate CI load tests**:  
   - Automate baseline and normal scenario in GitHub Actions.  
   - Alert on latency regressions.

5. **Monitor metrics**:  
   - Use Cloud Monitoring dashboards for CPU, memory, latency, and error rates.  
   - Set alerts on elevated P95 latency or error count.

---

## 4. Conclusion

All load tests completed without errors. The API handles nominal traffic well (up to ~50 users) with sub-100 ms P95 latencies. Spikes to 100 users reveal cold-start and scaling delays (P99 up to 1 s). Implementing autoscaling tweaks and inference optimizations will achieve smoother performance under production loads.

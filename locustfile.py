from locust import HttpUser, task, between

class PenguinPredictUser(HttpUser):
    wait_time = between(1, 2)  # seconds between tasks

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={
                "bill_length_mm": 40.0,
                "bill_depth_mm": 18.0,
                "flipper_length_mm": 195,
                "body_mass_g": 4000,
                "year": 2008,
                "sex": "male",
                "island": "Biscoe"
            }
        )

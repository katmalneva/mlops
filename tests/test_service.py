from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from clothing_mlops.data_pipeline import prediction_example
from clothing_mlops.service import app, refresh_model


class ServiceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        refresh_model()
        cls.client = TestClient(app)

    def test_root_endpoint(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["message"], "Clothing value prediction service")
        self.assertEqual(body["example_request"], prediction_example())

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "model_loaded": True},
        )

    def test_predict_endpoint(self) -> None:
        response = self.client.post("/predict", json=prediction_example())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_status"], "ok")
        self.assertIsInstance(body["predicted_sale_price"], float)

    def test_predict_validation(self) -> None:
        response = self.client.post("/predict", json={"brand": "Patagonia"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

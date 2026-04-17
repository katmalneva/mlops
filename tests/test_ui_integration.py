from __future__ import annotations

import json
import re
import unittest

from fastapi.testclient import TestClient

import clothing_mlops.service as service


class UiIntegrationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        service.refresh_model()
        cls.client = TestClient(service.app)
        cls._original_model = service._serving_model
        cls._original_loaded = service._model_loaded

    @classmethod
    def tearDownClass(cls) -> None:
        service._serving_model = cls._original_model
        service._model_loaded = cls._original_loaded

    def test_homepage_catalog_matches_api_curve_items(self) -> None:
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        html = home.text

        match = re.search(r"const ITEMS = (\[.*?\]);", html, flags=re.DOTALL)
        self.assertIsNotNone(match, "Could not find embedded ITEMS catalog in homepage script.")
        embedded_items = json.loads(match.group(1))
        embedded_names = [item["name"] for item in embedded_items]

        api_response = self.client.get("/api")
        self.assertEqual(api_response.status_code, 200)
        api_items = api_response.json()["curve_items"]

        self.assertEqual(embedded_names, api_items)

    def test_predict_response_reflects_loaded_model_behavior(self) -> None:
        # Force deterministic fallback model so UI payload and prediction behavior align.
        service._serving_model = service.PlaceholderModel()
        service._model_loaded = True

        payload = {
            "brand": "Patagonia",
            "category": "Jacket",
            "size": "M",
            "condition": "used_very_good",
            "color": "Blue",
            "material": "Polyester",
            "listing_price": 100.0,
            "shipping_price": 20.0,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        body = response.json()
        expected = round((payload["listing_price"] + payload["shipping_price"]) * 0.87, 2)
        self.assertEqual(body["predicted_sale_price"], expected)
        self.assertEqual(body["model_status"], "ok")

    def test_lifetime_curve_ui_values_are_consistent(self) -> None:
        payload = {
            "item_name": "Fear of God Essentials Hoodie",
            "purchase_price": 200.0,
        }
        response = self.client.post("/api/lifetime-curve", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()

        points = body["points"]
        self.assertEqual(points[0]["label"], "Year 0")
        self.assertEqual(points[0]["value"], 200.0)
        self.assertEqual(points[1]["value"], 176.0)
        self.assertEqual(points[-1]["label"], "Year 6")
        self.assertEqual(points[-1]["value"], 106.0)


if __name__ == "__main__":
    unittest.main()

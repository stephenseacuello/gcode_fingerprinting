"""
Python client for G-Code Fingerprinting API.

Example usage:
    client = GCodeAPIClient("http://localhost:8000")
    result = client.predict(sensor_data)
    print(result['gcode_sequence'])
"""

import requests
import numpy as np
from typing import Dict, List, Optional


class GCodeAPIClient:
    """Python client for the G-Code Fingerprinting API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """
        Check API health status.

        Returns:
            Health status dictionary
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_info(self) -> Dict:
        """
        Get model information.

        Returns:
            Model metadata dictionary
        """
        response = self.session.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        continuous: np.ndarray,
        categorical: np.ndarray,
        return_fingerprint: bool = False,
        temperature: float = 1.0,
        method: str = "greedy",
    ) -> Dict:
        """
        Predict G-code sequence from sensor data.

        Args:
            continuous: Continuous sensor data [T, 135]
            categorical: Categorical features [T, 4]
            return_fingerprint: Whether to include fingerprint in response
            temperature: Sampling temperature
            method: Generation method ('greedy', 'beam_search', etc.)

        Returns:
            Prediction result dictionary
        """
        payload = {
            "sensor_data": {
                "continuous": continuous.tolist(),
                "categorical": categorical.tolist(),
            },
            "return_fingerprint": return_fingerprint,
            "inference_config": {
                "method": method,
                "temperature": temperature,
            }
        }

        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()

    def batch_predict(
        self,
        continuous_batch: List[np.ndarray],
        categorical_batch: List[np.ndarray],
        return_fingerprint: bool = False,
    ) -> Dict:
        """
        Predict G-code sequences for multiple samples.

        Args:
            continuous_batch: List of continuous sensor arrays
            categorical_batch: List of categorical feature arrays
            return_fingerprint: Whether to include fingerprints

        Returns:
            Batch prediction results
        """
        sensor_data_batch = [
            {
                "continuous": cont.tolist(),
                "categorical": cat.tolist(),
            }
            for cont, cat in zip(continuous_batch, categorical_batch)
        ]

        payload = {
            "sensor_data_batch": sensor_data_batch,
            "return_fingerprint": return_fingerprint,
        }

        response = self.session.post(f"{self.base_url}/batch_predict", json=payload)
        response.raise_for_status()
        return response.json()

    def get_fingerprint(
        self,
        continuous: np.ndarray,
        categorical: np.ndarray,
    ) -> Dict:
        """
        Extract machine fingerprint from sensor data.

        Args:
            continuous: Continuous sensor data [T, 135]
            categorical: Categorical features [T, 4]

        Returns:
            Fingerprint dictionary with embedding and metadata
        """
        payload = {
            "sensor_data": {
                "continuous": continuous.tolist(),
                "categorical": categorical.tolist(),
            }
        }

        response = self.session.post(f"{self.base_url}/fingerprint", json=payload)
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = GCodeAPIClient("http://localhost:8000")

    # Check health
    health = client.health_check()
    print("API Status:", health)

    # Get model info
    info = client.get_info()
    print("\nModel Info:", info)

    # Create sample data
    continuous = np.random.randn(64, 135).astype(np.float32)
    categorical = np.random.randint(0, 5, size=(64, 4)).astype(np.int64)

    # Single prediction
    print("\n=== Single Prediction ===")
    result = client.predict(continuous, categorical, return_fingerprint=True)
    print("G-code:", result['gcode_sequence'])
    print("Inference time:", f"{result['inference_time_ms']:.2f}ms")
    if result.get('fingerprint'):
        print("Fingerprint dim:", len(result['fingerprint']))

    # Batch prediction
    print("\n=== Batch Prediction ===")
    batch_results = client.batch_predict(
        [continuous, continuous],
        [categorical, categorical],
    )
    print(f"Processed {len(batch_results['predictions'])} samples")
    print(f"Total time: {batch_results['total_inference_time_ms']:.2f}ms")

    # Fingerprint extraction
    print("\n=== Fingerprint Extraction ===")
    fp_result = client.get_fingerprint(continuous, categorical)
    print(f"Fingerprint dimension: {fp_result['embedding_dim']}")
    print(f"Norm: {fp_result['norm']:.4f}")

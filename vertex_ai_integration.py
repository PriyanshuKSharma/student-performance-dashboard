import os
from google.cloud import aiplatform
from google.oauth2 import service_account

# Set your Google Cloud project ID and region
PROJECT_ID = "your-project-id"
REGION = "us-central1"

# Path to your service account key file
SERVICE_ACCOUNT_FILE = "path/to/your/service-account-file.json"

def initialize_vertex_ai():
    """
    Initialize the Vertex AI client with authentication.
    """
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE
    )
    aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)

def predict_vertex_ai(endpoint_id: str, instances: list):
    """
    Make a prediction request to a deployed Vertex AI endpoint.

    Args:
        endpoint_id (str): The ID of the deployed endpoint.
        instances (list): A list of instances to predict on.

    Returns:
        list: Prediction results.
    """
    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )
    endpoint = client.endpoint_path(PROJECT_ID, REGION, endpoint_id)

    response = client.predict(
        endpoint=endpoint,
        instances=instances,
        parameters={},
    )
    return response.predictions

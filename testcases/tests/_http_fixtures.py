"""
HTTP-only fixtures that work without model imports.

These fixtures allow tests to run against any backend that implements the API spec,
without requiring direct access to SQLAlchemy models. This is crucial for:
1. Testing student implementations before all models are complete
2. Testing against external/deployed services
3. Running a subset of tests without full database setup
"""
import pytest
import os
from typing import Generator, Optional, Dict, Any
import uuid


# Unique suffix to avoid email conflicts across test runs
_test_run_id = uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """
    Get the base URL for API testing.

    Override with TEST_API_URL environment variable for external testing.
    Defaults to empty string for TestClient usage.
    """
    return os.getenv("TEST_API_URL", "")


@pytest.fixture
def http_client():
    """
    Create an HTTP test client without database dependency.

    This fixture tries to import the FastAPI app and create a TestClient.
    If the app is not available, the test is skipped with a helpful message.
    """
    try:
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            yield client

    except ImportError as e:
        pytest.skip(
            f"\n\nFastAPI app not available for HTTP testing.\n"
            f"To fix: Implement app/main.py with a FastAPI application.\n"
            f"Import error: {e}"
        )


@pytest.fixture
def create_user_via_api(http_client) -> callable:
    """
    Factory fixture to create users via the API (no model imports needed).

    Returns:
        A function that creates users and returns the response data.

    Usage:
        user = create_user_via_api("test@example.com", "password123", "Test User")
    """
    created_users = []

    def _create(
        email: Optional[str] = None,
        password: str = "testpassword123",
        full_name: str = "Test User",
        role: str = "student"
    ) -> Optional[Dict[str, Any]]:
        if email is None:
            email = f"test_{uuid.uuid4().hex[:8]}_{_test_run_id}@test.com"

        response = http_client.post("/api/v1/auth/register", json={
            "email": email,
            "password": password,
            "full_name": full_name,
            "role": role
        })

        if response.status_code == 201:
            data = response.json()
            created_users.append({"email": email, "password": password, **data})
            return {"email": email, "password": password, **data}

        return None

    yield _create

    # Note: Cleanup would require DELETE endpoint or DB access
    # For test isolation, use transaction rollback in db_session fixture


@pytest.fixture
def get_auth_token(http_client) -> callable:
    """
    Get an authentication token via API login.

    Returns:
        A function that takes email/password and returns the access token.

    Usage:
        token = get_auth_token("test@example.com", "password123")
        headers = {"Authorization": f"Bearer {token}"}
    """
    def _get_token(email: str, password: str) -> Optional[str]:
        response = http_client.post("/api/v1/auth/login", json={
            "email": email,
            "password": password
        })

        if response.status_code == 200:
            return response.json().get("access_token")

        return None

    return _get_token


@pytest.fixture
def auth_headers_via_api(http_client, create_user_via_api, get_auth_token) -> Dict[str, str]:
    """
    Create a test user and return auth headers, all via API.

    This fixture creates a fresh student user and returns authentication headers.
    """
    user = create_user_via_api(role="student")
    if user is None:
        pytest.skip(
            "\n\nCould not create test user via API.\n"
            "To fix: Implement POST /api/v1/auth/register endpoint."
        )

    token = get_auth_token(user["email"], user["password"])
    if token is None:
        pytest.skip(
            "\n\nCould not authenticate test user via API.\n"
            "To fix: Implement POST /api/v1/auth/login endpoint."
        )

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def instructor_auth_headers_via_api(http_client, create_user_via_api, get_auth_token) -> Dict[str, str]:
    """
    Create a test instructor and return auth headers, all via API.
    """
    user = create_user_via_api(role="instructor", full_name="Test Instructor")
    if user is None:
        pytest.skip(
            "\n\nCould not create test instructor via API.\n"
            "To fix: Implement POST /api/v1/auth/register with instructor role."
        )

    token = get_auth_token(user["email"], user["password"])
    if token is None:
        pytest.skip(
            "\n\nCould not authenticate test instructor via API.\n"
            "To fix: Implement POST /api/v1/auth/login endpoint."
        )

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers_via_api(http_client, create_user_via_api, get_auth_token) -> Dict[str, str]:
    """
    Create a test admin and return auth headers, all via API.
    """
    user = create_user_via_api(role="admin", full_name="Test Admin")
    if user is None:
        pytest.skip(
            "\n\nCould not create test admin via API.\n"
            "To fix: Implement POST /api/v1/auth/register with admin role."
        )

    token = get_auth_token(user["email"], user["password"])
    if token is None:
        pytest.skip(
            "\n\nCould not authenticate test admin via API.\n"
            "To fix: Implement POST /api/v1/auth/login endpoint."
        )

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_user_credentials() -> Dict[str, str]:
    """
    Return test user credentials for manual authentication.
    """
    return {
        "email": f"testuser_{uuid.uuid4().hex[:8]}@test.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }


@pytest.fixture
def sample_checkin_data() -> Dict[str, Any]:
    """
    Return sample check-in request data.
    """
    return {
        "session_id": str(uuid.uuid4()),
        "latitude": 1.3483,  # NTU coordinates
        "longitude": 103.6831,
        "location_accuracy_meters": 10.0,
        "device_fingerprint": f"device_{uuid.uuid4().hex[:8]}"
    }


@pytest.fixture
def sample_course_data() -> Dict[str, Any]:
    """
    Return sample course creation data.
    """
    return {
        "code": f"CS{uuid.uuid4().hex[:4].upper()}",
        "name": "Test Course",
        "semester": "AY2024-25 Sem 1",
        "venue_latitude": 1.3483,
        "venue_longitude": 103.6831,
        "venue_name": "Test Venue",
        "geofence_radius_meters": 100.0
    }


@pytest.fixture
def sample_session_data() -> Dict[str, Any]:
    """
    Return sample session creation data.
    """
    from datetime import datetime, timedelta

    now = datetime.utcnow()

    return {
        "name": "Test Session",
        "session_type": "lecture",
        "scheduled_start": (now + timedelta(minutes=5)).isoformat() + "Z",
        "scheduled_end": (now + timedelta(hours=2)).isoformat() + "Z",
        "checkin_opens_at": (now - timedelta(minutes=10)).isoformat() + "Z",
        "checkin_closes_at": (now + timedelta(minutes=30)).isoformat() + "Z",
        "require_liveness_check": True,
        "require_face_match": False
    }

"""
HTTP-only pytest fixtures for SAIV testing.

These fixtures work against any running implementation without model imports.
All test data is created via HTTP API calls.

Environment Variables:
    TEST_BACKEND_URL: Backend API URL (default: http://localhost:8000)
    TEST_FACE_URL: Face recognition service URL (default: http://localhost:8001)

Usage:
    # Start services first
    docker-compose up -d

    # Run tests
    export TEST_BACKEND_URL=http://localhost:8000
    export TEST_FACE_URL=http://localhost:8001
    pytest tests/public/ -v
"""
import pytest
import httpx
import os
import uuid
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta

# Import scoring plugin for per-test point tracking
pytest_plugins = ['scoring.plugin']

# =============================================================================
# Configuration
# =============================================================================

BACKEND_URL = os.getenv("TEST_BACKEND_URL", "http://localhost:8000")
FACE_URL = os.getenv("TEST_FACE_URL", "http://localhost:8001")
TEST_TIMEOUT = 30.0

# Unique test run identifier to prevent email collisions across test runs
_test_run_id = uuid.uuid4().hex[:8]


# =============================================================================
# HTTP Client Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def backend_url() -> str:
    """Get the backend API URL."""
    return BACKEND_URL


@pytest.fixture(scope="session")
def face_url() -> str:
    """Get the face recognition service URL."""
    return FACE_URL


@pytest.fixture(scope="session")
def http_client():
    """
    Session-scoped HTTP client for backend API.

    This client persists across all tests in a session for efficiency.
    """
    with httpx.Client(base_url=BACKEND_URL, timeout=TEST_TIMEOUT) as client:
        # Verify backend is running
        try:
            response = client.get("/health")
            if response.status_code != 200:
                pytest.skip(
                    f"\n\nBackend service not healthy.\n"
                    f"URL: {BACKEND_URL}\n"
                    f"Status: {response.status_code}\n"
                    f"To fix: Start the backend service with 'docker-compose up -d'"
                )
        except httpx.ConnectError:
            pytest.skip(
                f"\n\nBackend service not running.\n"
                f"URL: {BACKEND_URL}\n"
                f"To fix: Start the backend service with 'docker-compose up -d'"
            )
        yield client


@pytest.fixture(scope="session")
def face_http_client():
    """
    Session-scoped HTTP client for face recognition service.
    """
    with httpx.Client(base_url=FACE_URL, timeout=TEST_TIMEOUT) as client:
        # Verify face service is running
        try:
            response = client.get("/health")
            if response.status_code != 200:
                pytest.skip(
                    f"\n\nFace recognition service not healthy.\n"
                    f"URL: {FACE_URL}\n"
                    f"Status: {response.status_code}\n"
                    f"To fix: Start the face service with 'docker-compose up -d'"
                )
        except httpx.ConnectError:
            pytest.skip(
                f"\n\nFace recognition service not running.\n"
                f"URL: {FACE_URL}\n"
                f"To fix: Start the face service with 'docker-compose up -d'"
            )
        yield client


# Backward compatibility aliases
@pytest.fixture
def client(http_client):
    """Alias for http_client for backward compatibility with existing tests."""
    return http_client


@pytest.fixture
def face_client(face_http_client):
    """Alias for face_http_client for backward compatibility."""
    return face_http_client


# =============================================================================
# User Creation Fixtures
# =============================================================================

@pytest.fixture
def test_student(http_client) -> Dict[str, Any]:
    """
    Create a test student via API.

    Returns:
        Dictionary with id, email, password, full_name, role
    """
    email = f"student_{_test_run_id}_{uuid.uuid4().hex[:6]}@test.com"
    password = "testpassword123"

    response = http_client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Test Student",
        "role": "student"
    })

    if response.status_code != 201:
        pytest.fail(
            f"\n\nCannot create test student.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/auth/register endpoint."
        )

    user_data = response.json()
    return {
        "id": user_data["id"],
        "email": email,
        "password": password,
        "full_name": user_data.get("full_name", "Test Student"),
        "role": "student"
    }


@pytest.fixture
def test_instructor(http_client) -> Dict[str, Any]:
    """
    Create a test instructor via API.

    Returns:
        Dictionary with id, email, password, full_name, role
    """
    email = f"instructor_{_test_run_id}_{uuid.uuid4().hex[:6]}@test.com"
    password = "testpassword123"

    response = http_client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Test Instructor",
        "role": "instructor"
    })

    if response.status_code != 201:
        pytest.fail(
            f"\n\nCannot create test instructor.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/auth/register endpoint."
        )

    user_data = response.json()
    return {
        "id": user_data["id"],
        "email": email,
        "password": password,
        "full_name": "Test Instructor",
        "role": "instructor"
    }


@pytest.fixture
def test_admin(http_client) -> Dict[str, Any]:
    """
    Create a test admin via API.

    Returns:
        Dictionary with id, email, password, full_name, role
    """
    email = f"admin_{_test_run_id}_{uuid.uuid4().hex[:6]}@test.com"
    password = "testpassword123"

    response = http_client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Test Admin",
        "role": "admin"
    })

    if response.status_code != 201:
        pytest.fail(
            f"\n\nCannot create test admin.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/auth/register endpoint."
        )

    user_data = response.json()
    return {
        "id": user_data["id"],
        "email": email,
        "password": password,
        "full_name": "Test Admin",
        "role": "admin"
    }


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture
def auth_headers_student(http_client, test_student) -> Dict[str, str]:
    """Get authentication headers for test student."""
    response = http_client.post("/api/v1/auth/login", json={
        "email": test_student["email"],
        "password": test_student["password"]
    })
    if response.status_code != 200:
        pytest.fail(
            f"\n\nLogin failed for test student.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/auth/login endpoint."
        )

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_instructor(http_client, test_instructor) -> Dict[str, str]:
    """Get authentication headers for test instructor."""
    response = http_client.post("/api/v1/auth/login", json={
        "email": test_instructor["email"],
        "password": test_instructor["password"]
    })
    if response.status_code != 200:
        pytest.fail(
            f"\n\nLogin failed for test instructor.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/auth/login endpoint."
        )

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_admin(http_client, test_admin) -> Dict[str, str]:
    """Get authentication headers for test admin."""
    response = http_client.post("/api/v1/auth/login", json={
        "email": test_admin["email"],
        "password": test_admin["password"]
    })
    if response.status_code != 200:
        pytest.fail(
            f"\n\nLogin failed for test admin.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/auth/login endpoint."
        )

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# =============================================================================
# Course and Session Fixtures
# =============================================================================

@pytest.fixture
def test_course(http_client, auth_headers_admin) -> Dict[str, Any]:
    """
    Create a test course via API.

    Note: Course creation requires admin role.

    Returns:
        Dictionary with id, code, name, and other course properties
    """
    course_data = {
        "code": f"CS{uuid.uuid4().hex[:4].upper()}",
        "name": "Test Course",
        "semester": "AY2024-25 Sem 1",
        "venue_latitude": 1.3483,  # NTU coordinates
        "venue_longitude": 103.6831,
        "venue_name": "NTU LT1",
        "geofence_radius_meters": 100.0,
        "require_device_binding": True,
        "risk_threshold": 0.5
    }

    response = http_client.post(
        "/api/v1/courses/",
        headers=auth_headers_admin,
        json=course_data
    )

    if response.status_code not in [200, 201]:
        pytest.fail(
            f"\n\nCannot create test course.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/courses/ endpoint."
        )

    data = response.json()
    return {
        "id": data["id"],
        "code": data.get("code", course_data["code"]),
        "name": data.get("name", course_data["name"]),
        **data
    }


@pytest.fixture
def test_enrollment(http_client, test_student, test_course, auth_headers_admin) -> Dict[str, Any]:
    """
    Enroll test student in test course via admin API.

    Returns:
        Dictionary with enrollment id, student_id, course_id
    """
    response = http_client.post(
        "/api/v1/admin/enrollments/",
        headers=auth_headers_admin,
        json={
            "student_id": test_student["id"],
            "course_id": test_course["id"]
        }
    )

    if response.status_code not in [200, 201]:
        # Try fallback to regular enrollments endpoint
        response = http_client.post(
            "/api/v1/enrollments/",
            headers=auth_headers_admin,
            json={
                "student_id": test_student["id"],
                "course_id": test_course["id"]
            }
        )

    if response.status_code not in [200, 201]:
        pytest.fail(
            f"\n\nCannot create enrollment.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/admin/enrollments/ or POST /api/v1/enrollments/ endpoint."
        )

    data = response.json()
    return {
        "id": data.get("id"),
        "student_id": test_student["id"],
        "course_id": test_course["id"],
        **data
    }


@pytest.fixture
def test_session(http_client, test_course, auth_headers_instructor, auth_headers_admin) -> Dict[str, Any]:
    """
    Create a test attendance session via API and activate it.

    Returns:
        Dictionary with id, name, status, and other session properties
    """
    from datetime import timezone
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    session_data = {
        "course_id": test_course["id"],
        "name": "Test Lecture",
        "session_type": "lecture",
        "scheduled_start": (now + timedelta(minutes=5)).isoformat() + "Z",
        "scheduled_end": (now + timedelta(hours=2)).isoformat() + "Z",
        "checkin_opens_at": (now - timedelta(minutes=10)).isoformat() + "Z",
        "checkin_closes_at": (now + timedelta(minutes=30)).isoformat() + "Z",
        "require_liveness_check": True,
        "require_face_match": False,
        "risk_threshold": 0.5
    }

    response = http_client.post(
        "/api/v1/sessions/",
        headers=auth_headers_instructor,
        json=session_data
    )

    if response.status_code not in [200, 201]:
        pytest.fail(
            f"\n\nCannot create test session.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/sessions/ endpoint."
        )

    data = response.json()
    session_id = data["id"]

    # Activate the session for check-ins using admin API
    activate_resp = http_client.patch(
        f"/api/v1/admin/sessions/{session_id}/status",
        headers=auth_headers_admin,
        json={"status": "active"}
    )

    if activate_resp.status_code == 200:
        data["status"] = "active"

    return {
        "id": session_id,
        "name": data.get("name", session_data["name"]),
        "status": data.get("status", "active"),
        **data
    }


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def test_device(http_client, test_student, auth_headers_student) -> Dict[str, Any]:
    """
    Register a test device via API.

    Returns:
        Dictionary with id, device_fingerprint, and other device properties
    """
    device_fingerprint = f"test_device_{uuid.uuid4().hex[:8]}"

    device_data = {
        "device_fingerprint": device_fingerprint,
        "device_name": "Test Device",
        "platform": "web",
        "browser": "Chrome"
    }

    response = http_client.post(
        "/api/v1/devices/",
        headers=auth_headers_student,
        json=device_data
    )

    # Try alternative endpoint if first fails
    if response.status_code not in [200, 201]:
        response = http_client.post(
            "/api/v1/devices/register",
            headers=auth_headers_student,
            json=device_data
        )

    if response.status_code not in [200, 201]:
        pytest.fail(
            f"\n\nCannot register test device.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/devices/ or /api/v1/devices/register endpoint."
        )

    data = response.json()
    return {
        "id": data.get("id"),
        "device_fingerprint": device_fingerprint,
        **data
    }


# =============================================================================
# Test State Manipulation (Admin API)
# =============================================================================

@pytest.fixture
def deactivated_student(http_client, auth_headers_admin) -> Dict[str, Any]:
    """
    Create a student and deactivate them via admin API.

    This fixture is used for testing that inactive users cannot log in.

    Returns:
        Dictionary with id, email, password, is_active=False
    """
    # Create student
    email = f"inactive_{_test_run_id}_{uuid.uuid4().hex[:6]}@test.com"
    password = "testpassword123"

    reg_response = http_client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Inactive Student",
        "role": "student"
    })

    if reg_response.status_code != 201:
        pytest.fail(
            f"\n\nCannot create student for deactivation test.\n"
            f"Status: {reg_response.status_code}\n"
            f"Response: {reg_response.text[:300]}\n"
            f"To fix: Implement POST /api/v1/auth/register endpoint."
        )

    user_id = reg_response.json()["id"]

    # Deactivate via admin API
    deactivate_response = http_client.patch(
        f"/api/v1/admin/users/{user_id}/deactivate",
        headers=auth_headers_admin
    )

    if deactivate_response.status_code not in [200, 204]:
        pytest.fail(
            f"\n\nCannot deactivate user via admin API.\n"
            f"Status: {deactivate_response.status_code}\n"
            f"Response: {deactivate_response.text[:300]}\n"
            f"To fix: Implement PATCH /api/v1/admin/users/{{id}}/deactivate endpoint."
        )

    return {
        "id": user_id,
        "email": email,
        "password": password,
        "is_active": False
    }


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def unique_email() -> Callable[[str], str]:
    """
    Generate a unique email for testing.

    Usage:
        email = unique_email("prefix")  # Returns "prefix_abc123@test.com"
    """
    def _generate(prefix: str = "test") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}@test.com"
    return _generate


@pytest.fixture
def ntu_coordinates() -> Dict[str, Any]:
    """Return NTU campus coordinates for geolocation testing."""
    return {
        "latitude": 1.3483,
        "longitude": 103.6831,
        "venue_name": "NTU Campus"
    }


# =============================================================================
# Face Test Data Fixtures
# =============================================================================

def create_test_image_base64(width: int = 128, height: int = 128, color: tuple = (100, 150, 200)) -> str:
    """
    Create a simple test image encoded as base64.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: RGB tuple for solid color fill

    Returns:
        Base64-encoded PNG image string
    """
    try:
        from PIL import Image

        # Create a solid color image
        img = Image.new('RGB', (width, height), color)

        # Save to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except ImportError:
        # Fallback: return minimal valid PNG
        # This is a 1x1 red pixel PNG
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def create_synthetic_face_base64(width: int = 256, height: int = 256) -> str:
    """
    Create a synthetic uniform skin-tone image for deepfake testing.
    Real faces have texture variation; synthetic/uniform images should fail liveness.

    Returns:
        Base64-encoded PNG with uniform skin color
    """
    # Approximate skin tone RGB
    skin_color = (255, 220, 185)
    return create_test_image_base64(width, height, skin_color)


@pytest.fixture
def test_image_base64() -> str:
    """Provide a simple test image as base64 string."""
    return create_test_image_base64()


@pytest.fixture
def synthetic_face_base64() -> str:
    """Provide a synthetic uniform face image for anti-spoofing tests."""
    return create_synthetic_face_base64()


@pytest.fixture
def face_test_data() -> Dict[str, str]:
    """
    Fixture providing test images for face recognition tests.

    Returns:
        dict with keys: real_face_1, real_face_2, non_face, synthetic_face
    """
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'faces')

    data = {
        # Default to synthetic images
        "real_face_1": create_test_image_base64(256, 256, (180, 140, 120)),
        "real_face_1_alt": create_test_image_base64(256, 256, (175, 135, 115)),
        "real_face_2": create_test_image_base64(256, 256, (160, 120, 100)),
        "non_face": create_test_image_base64(256, 256, (50, 100, 50)),  # Green landscape-like
        "synthetic_uniform": create_synthetic_face_base64(),
        "blurry_face": create_test_image_base64(64, 64, (180, 140, 120)),  # Low res
        "empty": "",  # Empty string for edge case
    }

    # Try to load real face images if available
    if os.path.exists(test_data_dir):
        real_faces_dir = os.path.join(test_data_dir, 'real_faces')

        if os.path.exists(real_faces_dir):
            for filename in os.listdir(real_faces_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(real_faces_dir, filename)
                    try:
                        with open(filepath, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')

                        # Map filename to key
                        key = filename.rsplit('.', 1)[0]  # Remove extension
                        data[key] = img_data
                    except Exception:
                        pass  # Skip files that can't be read

        # Load non-faces
        non_faces_dir = os.path.join(test_data_dir, 'non_faces')
        if os.path.exists(non_faces_dir):
            for filename in os.listdir(non_faces_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(non_faces_dir, filename)
                    try:
                        with open(filepath, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                        key = f"non_face_{filename.rsplit('.', 1)[0]}"
                        data[key] = img_data
                    except Exception:
                        pass

    return data


@pytest.fixture
def attack_test_data() -> Dict[str, str]:
    """
    Fixture providing attack images for anti-spoofing tests.

    Returns:
        dict with keys: print_attack, screen_attack, synthetic_face
    """
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'attacks')

    data = {
        # Default to synthetic attack simulations
        "print_attack_1": create_test_image_base64(256, 256, (200, 180, 170)),
        "print_attack_2": create_test_image_base64(256, 256, (195, 175, 165)),
        "screen_phone": create_test_image_base64(256, 256, (190, 170, 160)),
        "screen_tablet": create_test_image_base64(256, 256, (185, 165, 155)),
        "synthetic_uniform": create_synthetic_face_base64(),
    }

    # Try to load real attack images if available
    if os.path.exists(test_data_dir):
        for subdir in ['print_attacks', 'screen_attacks', 'synthetic']:
            subdir_path = os.path.join(test_data_dir, subdir)
            if os.path.exists(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        filepath = os.path.join(subdir_path, filename)
                        try:
                            with open(filepath, 'rb') as f:
                                img_data = base64.b64encode(f.read()).decode('utf-8')
                            key = f"{subdir}_{filename.rsplit('.', 1)[0]}"
                            data[key] = img_data
                        except Exception:
                            pass

    return data


# =============================================================================
# Bulk User Creation Fixture (for stress tests)
# =============================================================================

@pytest.fixture
def create_bulk_users(http_client, auth_headers_admin) -> Callable:
    """
    Factory fixture to create multiple users via admin API.

    Usage:
        users = create_bulk_users(count=10, role="student")

    Returns:
        List of dictionaries with user data
    """
    def _create(count: int = 10, role: str = "student") -> list:
        users_data = [
            {
                "email": f"bulk_{_test_run_id}_{i}_{uuid.uuid4().hex[:4]}@test.com",
                "password": "testpassword123",
                "full_name": f"Bulk User {i}",
                "role": role
            }
            for i in range(count)
        ]

        response = http_client.post(
            "/api/v1/admin/users/bulk",
            headers=auth_headers_admin,
            json={"users": users_data}
        )

        if response.status_code not in [200, 201]:
            # Fallback: create users one by one
            created_users = []
            for user_data in users_data:
                reg_response = http_client.post("/api/v1/auth/register", json=user_data)
                if reg_response.status_code == 201:
                    data = reg_response.json()
                    created_users.append({
                        "id": data["id"],
                        "email": user_data["email"],
                        "password": user_data["password"],
                        "full_name": user_data["full_name"],
                        "role": role
                    })
            return created_users

        result = response.json()
        users = result.get("users", [])

        # Add passwords to the result (since they're not returned by the API)
        for i, user in enumerate(users):
            if i < len(users_data):
                user["password"] = users_data[i]["password"]

        return users

    return _create

"""
PUBLIC TESTS: Frontend & Dashboard API Contract Tests (8 points)

These tests verify that the backend API endpoints work correctly with
the expected request/response formats used by the frontend and dashboard.

This is NOT browser-based testing - it validates API contract compliance.

Students can run these tests locally to validate their implementation.
"""
import pytest
from datetime import datetime, timedelta
import uuid


class TestFrontendAuthContract:
    """Test frontend authentication contract (2 points)"""

    @pytest.mark.points(1, category="frontend_contract")
    def test_login_response_format(self, client, test_student):
        """Test login returns expected format for frontend."""
        response = client.post("/api/v1/auth/login", json={
            "email": test_student["email"],
            "password": test_student["password"]
        })

        assert response.status_code == 200
        data = response.json()

        # Frontend expects these exact fields
        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

        # Frontend expects user object in response
        assert "user" in data
        user = data["user"]
        assert "id" in user
        assert "email" in user
        assert "role" in user

    def test_registration_response_format(self, client, unique_email):
        """Test registration returns expected format for frontend."""
        email = unique_email("frontend_test")
        response = client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "securepass123",
            "full_name": "Frontend Test User",
            "role": "student"
        })

        assert response.status_code == 201
        data = response.json()

        # Frontend expects these fields
        assert "id" in data
        assert "email" in data
        assert "full_name" in data
        assert "role" in data
        assert "is_active" in data

    @pytest.mark.points(1, category="frontend_contract")
    def test_token_refresh_format(self, client, test_student):
        """Test token refresh returns expected format."""
        # Login first
        login_resp = client.post("/api/v1/auth/login", json={
            "email": test_student["email"],
            "password": test_student["password"]
        })
        refresh_token = login_resp.json()["refresh_token"]

        # Refresh
        response = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token
        })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data


class TestFrontendCheckinContract:
    """Test frontend check-in contract (2 points)"""

    @pytest.mark.points(1, category="frontend_contract")
    def test_checkin_request_format(self, client, test_student, test_session,
                                    test_enrollment, auth_headers_student):
        """Test check-in accepts expected request format from frontend."""
        # This is the format the frontend sends
        response = client.post("/api/v1/checkins/", headers=auth_headers_student, json={
            "session_id": test_session["id"],
            "latitude": 1.3483,
            "longitude": 103.6831,
            "location_accuracy_meters": 10.0,
            "device_fingerprint": f"frontend_device_{uuid.uuid4().hex[:6]}",
            # Optional fields
            "liveness_challenge_response": None,
            "qr_code": None
        })

        assert response.status_code in [201, 400]  # 400 if already checked in

    @pytest.mark.points(1, category="frontend_contract")
    def test_checkin_response_format(self, client, test_student, test_session,
                                     test_enrollment, auth_headers_student):
        """Test check-in returns expected response format for frontend."""
        response = client.post("/api/v1/checkins/", headers=auth_headers_student, json={
            "session_id": test_session["id"],
            "latitude": 1.3483,
            "longitude": 103.6831,
            "device_fingerprint": f"frontend_device_{uuid.uuid4().hex[:6]}"
        })

        if response.status_code == 201:
            data = response.json()
            # Frontend expects these fields for status display
            assert "id" in data
            assert "status" in data
            assert "checked_in_at" in data
            assert "risk_score" in data

            # Status should be one of expected values
            assert data["status"] in ["pending", "approved", "flagged", "rejected"]

    def test_my_checkins_list_format(self, client, auth_headers_student):
        """Test my-checkins returns list format for frontend."""
        response = client.get("/api/v1/checkins/my-checkins",
                              headers=auth_headers_student)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestDashboardSessionContract:
    """Test dashboard session management contract (2 points)"""

    @pytest.mark.points(1, category="dashboard_contract")
    def test_active_sessions_format(self, client, test_session):
        """Test active sessions returns expected format for dashboard."""
        response = client.get("/api/v1/sessions/active")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        if len(data) > 0:
            session = data[0]
            # Dashboard expects these fields for display
            assert "id" in session
            assert "name" in session
            assert "status" in session
            assert "scheduled_start" in session
            assert "scheduled_end" in session

    @pytest.mark.points(1, category="dashboard_contract")
    def test_session_details_format(self, client, test_session, auth_headers_student):
        """Test session details returns expected format for dashboard."""
        response = client.get(f"/api/v1/sessions/{test_session['id']}", headers=auth_headers_student)

        assert response.status_code == 200
        data = response.json()

        # Dashboard needs these for session management
        assert "id" in data
        assert "course_id" in data
        assert "name" in data
        assert "status" in data
        assert "scheduled_start" in data
        assert "scheduled_end" in data
        assert "checkin_opens_at" in data
        assert "checkin_closes_at" in data

    def test_session_checkins_format(self, client, test_session, auth_headers_instructor):
        """Test session check-ins returns expected format for dashboard."""
        response = client.get(
            f"/api/v1/checkins/session/{test_session['id']}",
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestDashboardMetricsContract:
    """Test dashboard metrics access (2 points)"""

    @pytest.mark.points(1, category="dashboard_contract")
    def test_courses_list_format(self, client, test_course, auth_headers_student):
        """Test courses list returns expected format for dashboard."""
        response = client.get("/api/v1/courses/", headers=auth_headers_student)

        assert response.status_code == 200
        data = response.json()

        # Handle both list and paginated response formats
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        else:
            items = data

        assert isinstance(items, list)

        if len(items) > 0:
            course = items[0]
            # Dashboard needs these for course display
            assert "id" in course
            assert "code" in course
            assert "name" in course
            assert "semester" in course

    @pytest.mark.points(1, category="dashboard_contract")
    def test_user_profile_format(self, client, auth_headers_instructor, test_instructor):
        """Test user profile returns expected format for dashboard."""
        response = client.get("/api/v1/users/me", headers=auth_headers_instructor)

        assert response.status_code == 200
        data = response.json()

        # Dashboard needs role for permission checks
        assert "id" in data
        assert "email" in data
        assert "role" in data
        assert data["role"] in ["student", "instructor", "ta", "admin"]

    def test_health_endpoint_format(self, client):
        """Test health endpoint returns expected format for monitoring."""
        response = client.get("/health")

        # Health endpoint should return 200 or 503 (if DB/Redis unavailable)
        assert response.status_code in [200, 503]
        data = response.json()

        # Should have API status at minimum
        assert "api" in data or "status" in data


class TestErrorResponseContract:
    """Test error response format consistency (bonus)"""

    def test_401_error_format(self, client):
        """Test 401 error returns consistent format."""
        response = client.get("/api/v1/users/me")

        # Should be 401 or 403 for unauthorized
        assert response.status_code in [401, 403]
        data = response.json()
        assert "detail" in data

    def test_400_error_format(self, client, test_student, auth_headers_student):
        """Test 400 error returns consistent format."""
        # Try to register with existing email
        response = client.post("/api/v1/auth/register", json={
            "email": test_student["email"],
            "password": "password123",
            "full_name": "Duplicate User"
        })

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_422_validation_error_format(self, client):
        """Test 422 validation error returns expected format."""
        response = client.post("/api/v1/auth/register", json={
            "email": "invalid-email",  # Invalid email format
            "password": "123",  # Too short
            "full_name": ""  # Empty
        })

        # Should be 422 for validation error or 400 for custom validation
        assert response.status_code in [400, 422]

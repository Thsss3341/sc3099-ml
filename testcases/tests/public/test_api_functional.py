"""
PUBLIC TESTS: Backend API Functional Tests (28 points)

These tests verify that the core API functionality works correctly:
- User registration and authentication (5 pts)
- User management (5 pts)
- Course management (4 pts)
- Session management (4 pts)
- Check-in workflow (8 pts)
- Device management (2 pts)

Students can run these tests locally to validate their implementation.
"""
import pytest
import uuid


class TestAuthentication:
    """Test authentication endpoints (5 points)"""

    @pytest.mark.points(1, category="authentication")
    def test_user_registration(self, client, unique_email):
        """Test new user registration."""
        email = unique_email("newstudent")
        response = client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "securepass123",
            "full_name": "New Student",
            "role": "student"
        })
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == email
        assert data["role"] == "student"
        assert "id" in data

    @pytest.mark.points(1, category="authentication")
    def test_duplicate_registration_fails(self, client, test_student):
        """Test that duplicate email registration fails."""
        response = client.post("/api/v1/auth/register", json={
            "email": test_student["email"],
            "password": "password123",
            "full_name": "Duplicate User"
        })
        assert response.status_code == 400

    @pytest.mark.points(1, category="authentication")
    def test_user_login_success(self, client, test_student):
        """Test successful login."""
        response = client.post("/api/v1/auth/login", json={
            "email": test_student["email"],
            "password": test_student["password"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == test_student["email"]

    @pytest.mark.points(1, category="authentication")
    def test_login_wrong_password_fails(self, client, test_student):
        """Test login with wrong password fails."""
        response = client.post("/api/v1/auth/login", json={
            "email": test_student["email"],
            "password": "wrongpassword"
        })
        assert response.status_code == 401

    @pytest.mark.points(1, category="authentication")
    def test_token_refresh(self, client, test_student):
        """Test token refresh functionality."""
        # Login first
        login_response = client.post("/api/v1/auth/login", json={
            "email": test_student["email"],
            "password": test_student["password"]
        })
        refresh_token = login_response.json()["refresh_token"]

        # Refresh token
        response = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data


class TestUserManagement:
    """Test user management endpoints (5 points)"""

    @pytest.mark.points(2, category="user_management")
    def test_get_current_user(self, client, test_student, auth_headers_student):
        """Test retrieving current user information."""
        response = client.get("/api/v1/users/me", headers=auth_headers_student)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_student["email"]
        assert data["full_name"] == test_student["full_name"]

    @pytest.mark.points(2, category="user_management")
    def test_update_user_profile(self, client, auth_headers_student):
        """Test updating user profile."""
        response = client.put("/api/v1/users/me", headers=auth_headers_student, json={
            "full_name": "Updated Name",
            "camera_consent": True,
            "geolocation_consent": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"
        assert data["camera_consent"] is True
        assert data["geolocation_consent"] is False

    @pytest.mark.points(1, category="user_management")
    def test_unauthorized_access_fails(self, client):
        """Test that accessing protected endpoints without auth fails."""
        response = client.get("/api/v1/users/me")
        assert response.status_code in [401, 403]  # Either unauthorized or forbidden


class TestCourseManagement:
    """Test course endpoints (4 points)"""

    @pytest.mark.points(2, category="courses")
    def test_list_courses(self, client, test_course, auth_headers_student):
        """Test listing all active courses."""
        response = client.get("/api/v1/courses/", headers=auth_headers_student)
        assert response.status_code == 200
        data = response.json()
        # Handle both list and paginated response formats
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        else:
            items = data
        assert isinstance(items, list)
        assert len(items) > 0
        # Find the test course
        course_found = any(c.get("code") == test_course["code"] for c in items)
        assert course_found, f"Test course {test_course['code']} not found in courses list"

    @pytest.mark.points(2, category="courses")
    def test_get_course_details(self, client, test_course, auth_headers_student):
        """Test retrieving course details."""
        response = client.get(f"/api/v1/courses/{test_course['id']}", headers=auth_headers_student)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_course["id"]
        assert data["code"] == test_course["code"]
        assert data["name"] == test_course["name"]


class TestSessionManagement:
    """Test session endpoints (4 points)"""

    @pytest.mark.points(2, category="sessions")
    def test_list_active_sessions(self, client, test_session):
        """Test listing active sessions."""
        response = client.get("/api/v1/sessions/active")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Session may or may not be included depending on timing
        # Just verify the endpoint returns a list

    @pytest.mark.points(2, category="sessions")
    def test_get_session_details(self, client, test_session, auth_headers_student):
        """Test retrieving session details."""
        response = client.get(f"/api/v1/sessions/{test_session['id']}", headers=auth_headers_student)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_session["id"]
        assert data["name"] == test_session["name"]
        assert data["status"] in ["scheduled", "active", "closed", "cancelled"]


class TestCheckInWorkflow:
    """Test check-in functionality (8 points)"""

    @pytest.mark.points(3, category="checkin")
    def test_successful_checkin(self, client, test_student, test_session, test_enrollment, auth_headers_student):
        """Test successful check-in to a session."""
        response = client.post("/api/v1/checkins/", headers=auth_headers_student, json={
            "session_id": test_session["id"],
            "latitude": 1.3483,  # NTU coordinates
            "longitude": 103.6831,
            "location_accuracy_meters": 10.0,
            "device_fingerprint": f"test_device_{uuid.uuid4().hex[:8]}"
        })
        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == test_session["id"]
        assert data["student_id"] == test_student["id"]
        assert data["status"] in ["pending", "approved", "flagged"]
        assert "risk_score" in data

    @pytest.mark.points(3, category="checkin")
    def test_duplicate_checkin_fails(self, client, test_student, test_session, test_enrollment, auth_headers_student):
        """Test that checking in twice to the same session fails."""
        device_fp = f"test_device_{uuid.uuid4().hex[:8]}"

        # First check-in
        first_response = client.post("/api/v1/checkins/", headers=auth_headers_student, json={
            "session_id": test_session["id"],
            "latitude": 1.3483,
            "longitude": 103.6831,
            "device_fingerprint": device_fp
        })
        assert first_response.status_code == 201, f"First check-in should succeed: {first_response.text}"

        # Second check-in (should fail)
        response = client.post("/api/v1/checkins/", headers=auth_headers_student, json={
            "session_id": test_session["id"],
            "latitude": 1.3483,
            "longitude": 103.6831,
            "device_fingerprint": device_fp
        })
        assert response.status_code == 400

    @pytest.mark.points(2, category="checkin")
    def test_list_my_checkins(self, client, test_student, test_session, test_enrollment, auth_headers_student):
        """Test retrieving student's check-in history."""
        # Create a check-in first
        device_fp = f"test_device_{uuid.uuid4().hex[:8]}"
        client.post("/api/v1/checkins/", headers=auth_headers_student, json={
            "session_id": test_session["id"],
            "latitude": 1.3483,
            "longitude": 103.6831,
            "device_fingerprint": device_fp
        })

        # Get check-ins
        response = client.get("/api/v1/checkins/my-checkins", headers=auth_headers_student)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestDeviceManagement:
    """Test device management (2 points)"""

    @pytest.mark.points(2, category="devices")
    def test_list_my_devices(self, client, test_student, test_device, auth_headers_student):
        """Test listing user's devices."""
        response = client.get("/api/v1/devices/my-devices", headers=auth_headers_student)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Find the test device
        device_found = any(d.get("id") == test_device["id"] for d in data)
        assert device_found, f"Test device not found in devices list"

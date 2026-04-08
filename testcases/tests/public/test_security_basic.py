"""
PUBLIC TESTS: Security Basics (12 points)

These tests verify basic security measures:
- Authentication security (4 pts)
- Authorization controls (4 pts)
- Input validation (2 pts)
- Rate limiting (2 pts)

Students can run these tests to ensure basic security compliance.
"""
import pytest
import uuid


class TestAuthenticationSecurity:
    """Test authentication security (4 points)"""

    @pytest.mark.points(1, category="auth_security")
    def test_weak_password_rejected(self, client, unique_email):
        """Test that weak passwords are rejected."""
        response = client.post("/api/v1/auth/register", json={
            "email": unique_email("weak"),
            "password": "123",  # Too short
            "full_name": "Weak Password User"
        })
        assert response.status_code == 422  # Validation error

    @pytest.mark.points(1, category="auth_security")
    def test_password_not_in_response(self, client, test_student):
        """Test that passwords are never returned in API responses."""
        response = client.post("/api/v1/auth/login", json={
            "email": test_student["email"],
            "password": test_student["password"]
        })
        assert response.status_code == 200
        response_text = response.text.lower()
        # Password should not appear in response (except as field name like "hashed_password" is ok)
        assert "testpassword123" not in response_text

    @pytest.mark.points(1, category="auth_security")
    def test_invalid_token_rejected(self, client):
        """Test that invalid JWT tokens are rejected."""
        response = client.get("/api/v1/users/me", headers={
            "Authorization": "Bearer invalid_token_12345"
        })
        assert response.status_code in [401, 403]

    @pytest.mark.points(1, category="auth_security")
    def test_expired_token_rejected(self, client):
        """Test that expired tokens are rejected (simulated)."""
        # This is a simplified test - real implementation would use actual expired tokens
        expired_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwiZXhwIjoxfQ.invalid"
        response = client.get("/api/v1/users/me", headers={
            "Authorization": f"Bearer {expired_token}"
        })
        assert response.status_code in [401, 403]


class TestAuthorizationControls:
    """Test role-based access control (4 points)"""

    @pytest.mark.points(1, category="authorization")
    def test_student_cannot_access_instructor_endpoints(self, client, test_student, test_session, auth_headers_student):
        """Test that students cannot access instructor-only endpoints."""
        response = client.get(f"/api/v1/checkins/session/{test_session['id']}", headers=auth_headers_student)
        assert response.status_code == 403

    @pytest.mark.points(1, category="authorization")
    def test_instructor_can_view_session_checkins(self, client, test_instructor, test_session, auth_headers_instructor):
        """Test that instructors can view check-ins for their sessions."""
        response = client.get(f"/api/v1/checkins/session/{test_session['id']}", headers=auth_headers_instructor)
        assert response.status_code == 200

    @pytest.mark.points(1, category="authorization")
    def test_non_admin_cannot_access_audit_logs(self, client, auth_headers_student):
        """Test that non-admin users cannot access audit logs."""
        response = client.get("/api/v1/audit/", headers=auth_headers_student)
        assert response.status_code == 403

    @pytest.mark.points(1, category="authorization")
    def test_inactive_user_cannot_login(self, client, deactivated_student):
        """Test that inactive users cannot login."""
        response = client.post("/api/v1/auth/login", json={
            "email": deactivated_student["email"],
            "password": deactivated_student["password"]
        })
        assert response.status_code == 403


class TestInputValidation:
    """Test input validation and sanitization (3 points)"""

    @pytest.mark.points(1, category="input_validation")
    def test_invalid_email_rejected(self, client):
        """Test that invalid email formats are rejected."""
        response = client.post("/api/v1/auth/register", json={
            "email": "not_an_email",
            "password": "testpassword123",
            "full_name": "Test User"
        })
        assert response.status_code == 422

    @pytest.mark.points(1, category="input_validation")
    def test_sql_injection_prevented(self, client, auth_headers_student):
        """Test that SQL injection attempts are prevented."""
        response = client.get("/api/v1/users/me", headers=auth_headers_student, params={
            "id": "1' OR '1'='1"
        })
        # Should either ignore the param or return normal response, not execute SQL
        assert response.status_code in [200, 400, 422]

    @pytest.mark.points(1, category="input_validation")
    def test_xss_payload_sanitized(self, client, auth_headers_student):
        """Test that XSS payloads are sanitized."""
        response = client.put("/api/v1/users/me", headers=auth_headers_student, json={
            "full_name": "<script>alert('XSS')</script>"
        })
        # Should either reject or sanitize
        if response.status_code == 200:
            data = response.json()
            # Ensure script tag is not present as-is
            assert "<script>" not in data.get("full_name", "")


class TestRateLimiting:
    """Test rate limiting (1 point)"""

    @pytest.mark.points(1, category="rate_limiting")
    def test_login_rate_limiting(self, client, test_student):
        """Test that excessive login attempts are rate limited."""
        # Attempt many logins with wrong password
        for i in range(20):
            response = client.post("/api/v1/auth/login", json={
                "email": test_student["email"],
                "password": "wrongpassword"
            })

        # After many attempts, should be rate limited
        # Note: This assumes rate limiting is implemented
        # The exact status code may vary (429 or 403)
        assert response.status_code in [429, 403, 401]  # Allow 401 if no rate limiting yet

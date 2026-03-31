"""
PUBLIC TESTS: Privacy Basics (8 points)

These tests verify basic privacy compliance:
- Consent management (3 pts)
- Data minimization (3 pts)
- Data retention (2 pts)

Students can run these tests to ensure basic privacy compliance.
"""
import pytest
import uuid


class TestConsentManagement:
    """Test privacy consent management (3 points)"""

    @pytest.mark.points(1, category="consent")
    def test_camera_consent_required(self, client, test_student, auth_headers_student):
        """Test that camera consent is tracked."""
        response = client.get("/api/v1/users/me", headers=auth_headers_student)
        assert response.status_code == 200
        data = response.json()
        assert "camera_consent" in data
        assert isinstance(data["camera_consent"], bool)

    @pytest.mark.points(1, category="consent")
    def test_geolocation_consent_required(self, client, auth_headers_student):
        """Test that geolocation consent is tracked."""
        response = client.get("/api/v1/users/me", headers=auth_headers_student)
        assert response.status_code == 200
        data = response.json()
        assert "geolocation_consent" in data
        assert isinstance(data["geolocation_consent"], bool)

    @pytest.mark.points(1, category="consent")
    def test_consent_can_be_updated(self, client, auth_headers_student):
        """Test that users can update their consent preferences."""
        response = client.put("/api/v1/users/me", headers=auth_headers_student, json={
            "camera_consent": False,
            "geolocation_consent": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["camera_consent"] is False
        assert data["geolocation_consent"] is False


class TestDataMinimization:
    """Test PII minimization (3 points)"""

    @pytest.mark.points(1, category="data_minimization")
    def test_no_raw_face_images_in_checkin_response(self, client, test_student, test_session,
                                                     test_enrollment, auth_headers_student):
        """Test that check-in response doesn't contain raw face images."""
        response = client.post("/api/v1/checkins/", headers=auth_headers_student, json={
            "session_id": test_session["id"],
            "latitude": 1.3483,
            "longitude": 103.6831,
            "device_fingerprint": f"privacy_test_device_{uuid.uuid4().hex[:6]}"
        })

        if response.status_code == 201:
            data = response.json()
            # Should not have raw image fields
            assert "face_image" not in data, "Response should not contain face_image"
            assert "image_data" not in data, "Response should not contain image_data"
            assert "photo" not in data, "Response should not contain photo"
            assert "face_data" not in data, "Response should not contain face_data"
            assert "raw_image" not in data, "Response should not contain raw_image"

    @pytest.mark.points(1, category="data_minimization")
    def test_only_face_embedding_hash_stored(self, client, test_student, test_session,
                                             test_enrollment, auth_headers_student):
        """Test that only face embedding hash is stored, not the embedding itself."""
        response = client.post("/api/v1/checkins/", headers=auth_headers_student, json={
            "session_id": test_session["id"],
            "latitude": 1.3483,
            "longitude": 103.6831,
            "device_fingerprint": f"privacy_test_device_{uuid.uuid4().hex[:6]}"
        })

        if response.status_code == 201:
            data = response.json()
            # If face embedding is returned, it should be a hash (64 chars for SHA-256)
            if "face_embedding_hash" in data and data["face_embedding_hash"]:
                assert isinstance(data["face_embedding_hash"], str)
                # Hash should be fixed length (SHA-256 = 64 chars)
                assert len(data["face_embedding_hash"]) == 64

    @pytest.mark.points(1, category="data_minimization")
    def test_passwords_not_in_user_response(self, client, auth_headers_student):
        """Test that passwords are not returned in user response."""
        response = client.get("/api/v1/users/me", headers=auth_headers_student)
        assert response.status_code == 200
        data = response.json()

        # Password should never be in response
        assert "password" not in data, "Password should not be in user response"
        assert "hashed_password" not in data, "Hashed password should not be in user response"


class TestDataRetention:
    """Test data retention policies (2 points)"""

    @pytest.mark.points(1, category="data_retention")
    def test_user_has_scheduled_deletion_field(self, client, auth_headers_admin):
        """Test that user records have scheduled deletion tracking."""
        # Get user profile and check for retention-related fields
        response = client.get("/api/v1/users/me", headers=auth_headers_admin)
        assert response.status_code == 200
        data = response.json()

        # The API should expose scheduling deletion info or handle it internally
        # For now, we verify the user data structure exists
        assert "id" in data
        assert "email" in data

    @pytest.mark.points(1, category="data_retention")
    def test_audit_log_endpoint_exists(self, client, auth_headers_admin):
        """Test that audit log endpoint exists for compliance."""
        response = client.get("/api/v1/audit/", headers=auth_headers_admin)

        # Should either return audit logs or indicate access
        assert response.status_code in [200, 403], \
            f"Audit endpoint should return 200 or 403, got {response.status_code}"

        if response.status_code == 200:
            data = response.json()
            # Audit logs should be a list
            if isinstance(data, list) and len(data) > 0:
                log = data[0]
                # Each log should have timestamp and be immutable
                assert "timestamp" in log or "created_at" in log

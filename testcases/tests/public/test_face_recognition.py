"""
PUBLIC TESTS: Face Recognition Service Tests (15 points)

These tests verify the face recognition microservice functionality:
- Service health and availability (2 points)
- Face enrollment (4 points)
- Face matching/verification (4 points)
- Privacy compliance (2 points)
- Risk assessment (3 points)

Students can run these tests locally to validate their implementation.

NOTE: Liveness detection is a BONUS feature (see test_liveness_bonus.py)
"""
import pytest
import base64
import os
from io import BytesIO
from pathlib import Path

# Try to import PIL, fallback if not available
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Path to sample images directory (relative to test file or project root)
SAMPLE_IMAGES_DIR = Path(__file__).parent.parent.parent / "sample_images"


def get_sample_image_base64(image_name: str) -> str:
    """Load a sample image from the sample_images directory as base64."""
    image_path = SAMPLE_IMAGES_DIR / image_name
    if image_path.exists():
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    # Fallback to synthetic image if sample not found
    return create_test_image_base64(256, 256)


def get_real_face_image_base64() -> str:
    """Get a real face image for enrollment/matching tests."""
    # Try obama.jpg first (clear frontal face)
    return get_sample_image_base64("obama.jpg")


def get_different_face_image_base64() -> str:
    """Get a different person's face for non-matching tests."""
    # Use biden.jpg (different person)
    return get_sample_image_base64("biden.jpg")


def get_same_person_different_image_base64() -> str:
    """Get a different image of the same person for matching tests."""
    # Use obama2.jpg (same person, different image)
    return get_sample_image_base64("obama2.jpg")


def get_partial_face_image_base64() -> str:
    """Get an image with a partial/occluded face."""
    return get_sample_image_base64("obama_partial_face.jpg")


def create_test_image_base64(width=128, height=128, color=(200, 150, 100)):
    """Create a simple test image as base64."""
    if HAS_PIL:
        img = Image.new('RGB', (width, height), color)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        # Minimal valid 1x1 PNG fallback
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def create_synthetic_face_base64(width=256, height=256):
    """Create uniform skin-tone image for anti-spoofing tests."""
    skin_color = (255, 220, 185)  # Approximate skin tone
    return create_test_image_base64(width, height, skin_color)


# =============================================================================
# Face Service Health Tests (2 points)
# =============================================================================

class TestFaceServiceHealth:
    """Test face recognition service availability (2 points)"""

    @pytest.mark.points(1, category="face_service")
    def test_health_check(self, face_client):
        """
        Test face recognition service health endpoint.

        The /health endpoint should return:
        - HTTP 200 status
        - JSON with status="healthy"
        """
        response = face_client.get("/health")
        assert response.status_code == 200, \
            f"Health check failed with status {response.status_code}"

        data = response.json()
        assert data["status"] == "healthy", \
            f"Service status should be 'healthy', got '{data.get('status')}'"
        assert "service" in data, \
            "Response should include 'service' field"

    @pytest.mark.points(1, category="face_service")
    def test_root_endpoint_lists_endpoints(self, face_client):
        """
        Test that root endpoint lists available API endpoints.

        The / endpoint should return a list of available endpoints
        including the required face recognition endpoints.
        """
        response = face_client.get("/")
        assert response.status_code == 200, \
            f"Root endpoint failed with status {response.status_code}"

        data = response.json()
        assert "endpoints" in data, \
            "Root should list available endpoints"

        # Check for required endpoints
        endpoints = data["endpoints"]
        assert "/face/enroll" in endpoints or "/liveness/check" in endpoints, \
            "Should list face recognition endpoints"


# =============================================================================
# Face Enrollment Tests (4 points)
# =============================================================================

class TestFaceEnrollment:
    """Test face enrollment functionality (4 points)"""

    @pytest.mark.points(1.5, category="face_enrollment")
    def test_enroll_face_success(self, face_client):
        """
        Test successful face enrollment with valid real face image.

        A valid face image should enroll successfully and return:
        - enrollment_successful: true
        - face_template_hash: 64-character hex string
        - quality_score: between 0 and 1
        """
        # Use a real face image for enrollment
        real_face_image = get_real_face_image_base64()

        response = face_client.post("/face/enroll", json={
            "user_id": "test-user-001",
            "image": real_face_image,
            "camera_consent": True
        })

        # Endpoint should exist
        assert response.status_code != 404, \
            "POST /face/enroll endpoint not found. Implement this endpoint."

        # Accept 200 or 201 for success
        assert response.status_code in [200, 201], \
            f"Enrollment with real face should succeed, got status {response.status_code}"

        data = response.json()

        # Check enrollment was successful
        assert data.get("enrollment_successful", False), \
            "Enrollment with real face image should succeed"
        assert "face_template_hash" in data, \
            "Successful enrollment must return face_template_hash"
        assert "quality_score" in data, \
            "Successful enrollment must return quality_score"

        # Quality score should be reasonably high for a clear face image
        assert data["quality_score"] >= 0.5, \
            f"Quality score should be >= 0.5 for clear face, got {data['quality_score']}"

    @pytest.mark.points(1, category="face_enrollment")
    def test_enroll_response_format(self, face_client):
        """
        Test that enrollment response has correct format.

        Response should include:
        - enrollment_successful (boolean)
        - face_template_hash (string, if successful)
        - quality_score (float 0-1)
        - details (object with additional info)
        """
        real_face_image = get_real_face_image_base64()

        response = face_client.post("/face/enroll", json={
            "user_id": "test-user-002",
            "image": real_face_image,
            "camera_consent": True
        })

        assert response.status_code in [200, 201], \
            f"Enrollment endpoint returned {response.status_code}"

        data = response.json()

        # Required fields
        assert "enrollment_successful" in data, \
            "Response must include 'enrollment_successful' field"
        assert isinstance(data["enrollment_successful"], bool), \
            "'enrollment_successful' must be a boolean"

        # Quality score should be present and valid
        assert "quality_score" in data, \
            "Response must include quality_score"
        assert 0 <= data["quality_score"] <= 1, \
            "quality_score must be between 0 and 1"

        # If enrollment successful, hash must be present
        if data["enrollment_successful"]:
            assert "face_template_hash" in data, \
                "Successful enrollment must include face_template_hash"
            assert len(data["face_template_hash"]) == 64, \
                "face_template_hash should be 64 chars (SHA-256 hex)"

    @pytest.mark.points(0.5, category="face_enrollment")
    def test_enroll_no_face_rejected(self, face_client):
        """
        Test that images without faces are rejected.

        Non-face images (solid colors, landscapes) should:
        - Return enrollment_successful: false, OR
        - Return HTTP 400 Bad Request
        """
        # Use a simple solid color image (clearly not a face)
        non_face_image = create_test_image_base64(128, 128, (50, 100, 50))

        response = face_client.post("/face/enroll", json={
            "user_id": "test-user-003",
            "image": non_face_image,
            "camera_consent": True
        })

        # Should either fail gracefully (400) or return enrollment_successful=false
        if response.status_code == 200 or response.status_code == 201:
            data = response.json()
            # If returns 200, enrollment should have failed
            assert data.get("enrollment_successful") is False, \
                "Non-face images should not enroll successfully"
        elif response.status_code == 400:
            # 400 is acceptable for no face detected
            pass
        else:
            # 404 means endpoint not implemented
            assert response.status_code != 404, \
                "POST /face/enroll endpoint not found"

    @pytest.mark.points(1, category="face_enrollment")
    def test_enroll_requires_consent(self, face_client):
        """
        Test that enrollment requires camera consent.

        Enrollment with camera_consent=false should:
        - Return HTTP 400 Bad Request, OR
        - Return enrollment_successful: false
        """
        response = face_client.post("/face/enroll", json={
            "user_id": "test-user-004",
            "image": create_test_image_base64(256, 256),
            "camera_consent": False  # No consent
        })

        # Should reject without consent
        if response.status_code in [200, 201]:
            data = response.json()
            # If returns 200, should not have enrolled
            assert data.get("enrollment_successful") is False, \
                "Should not enroll without camera consent"
        elif response.status_code == 400:
            # 400 is the expected response
            pass
        else:
            assert response.status_code != 404, \
                "POST /face/enroll endpoint not found"


# =============================================================================
# Face Matching/Verification Tests (4 points)
# =============================================================================

class TestFaceMatching:
    """Test face matching/verification functionality (4 points)"""

    @pytest.fixture
    def enrolled_hash(self, face_client):
        """Enroll a face using real face image and return the hash for matching tests."""
        # Use a real face image for enrollment
        real_face_image = get_real_face_image_base64()

        response = face_client.post("/face/enroll", json={
            "user_id": "matching-test-user",
            "image": real_face_image,
            "camera_consent": True
        })

        if response.status_code in [200, 201]:
            data = response.json()
            if data.get("face_template_hash"):
                return data["face_template_hash"]

        # Fall back to liveness check to get a hash
        response = face_client.post("/liveness/check", json={
            "challenge_response": real_face_image,
            "challenge_type": "passive"
        })

        if response.status_code == 200:
            data = response.json()
            if data.get("face_embedding_hash"):
                return data["face_embedding_hash"]

        # Last fallback: return a dummy hash (should not happen with real face)
        pytest.fail("Could not enroll face or get hash from liveness check")

    @pytest.mark.points(1.5, category="face_matching")
    def test_face_verify_same_image(self, face_client, enrolled_hash):
        """
        Test that the same person's face matches with enrolled hash.

        When verifying with the same person's image,
        the match should pass with a high score.
        """
        # Use the same face image
        same_face_image = get_real_face_image_base64()

        response = face_client.post("/face/verify", json={
            "image": same_face_image,
            "reference_template_hash": enrolled_hash
        })

        # Fall back to /face/match if /face/verify not found
        if response.status_code == 404:
            response = face_client.post("/face/match", json={
                "image": same_face_image,
                "reference_hash": enrolled_hash
            })

        assert response.status_code != 404, \
            "Neither /face/verify nor /face/match endpoint found"

        assert response.status_code == 200, \
            f"Face verification failed with status {response.status_code}"

        data = response.json()

        # Same image should match with high score
        assert "match_passed" in data, \
            "Response should include match_passed"
        assert "match_score" in data, \
            "Response should include match_score"

        # Same image should definitely match
        assert data["match_passed"] is True, \
            f"Same face image should match, got match_passed={data['match_passed']}"
        assert data["match_score"] >= 0.7, \
            f"Same face should have high match score, got {data['match_score']}"

    @pytest.mark.points(1.5, category="face_matching")
    def test_face_verify_different_person(self, face_client, enrolled_hash):
        """
        Test that a different person's face doesn't match.

        A different person's face should:
        - Have match_passed: false
        - Have a lower match score than same person
        """
        # Use a different person's face (biden vs enrolled obama)
        different_face_image = get_different_face_image_base64()

        response = face_client.post("/face/verify", json={
            "image": different_face_image,
            "reference_template_hash": enrolled_hash
        })

        # Fall back to /face/match
        if response.status_code == 404:
            response = face_client.post("/face/match", json={
                "image": different_face_image,
                "reference_hash": enrolled_hash
            })

        assert response.status_code != 404, \
            "Face verification endpoint not found"

        assert response.status_code == 200, \
            f"Face verification failed with status {response.status_code}"

        data = response.json()

        # Different person should NOT match
        assert "match_passed" in data, \
            "Response should include match_passed"
        assert data["match_passed"] is False, \
            "Different person's face should NOT match"

        # Different person should have different hash
        if "current_template_hash" in data:
            assert data["current_template_hash"] != enrolled_hash, \
                "Different people should produce different hashes"

    @pytest.mark.points(0.5, category="face_matching")
    def test_face_verify_response_format(self, face_client, enrolled_hash):
        """
        Test that verification response has correct format.

        Response should include:
        - match_passed (boolean)
        - match_score (float 0-1)
        - current_template_hash or face_embedding_hash (string)
        """
        real_face_image = get_real_face_image_base64()

        response = face_client.post("/face/verify", json={
            "image": real_face_image,
            "reference_template_hash": enrolled_hash
        })

        # Fall back to /face/match
        if response.status_code == 404:
            response = face_client.post("/face/match", json={
                "image": real_face_image,
                "reference_hash": enrolled_hash
            })

        assert response.status_code == 200, \
            f"Verification returned {response.status_code}"

        data = response.json()

        # Check required fields
        assert "match_passed" in data, \
            "Response must include match_passed"
        assert isinstance(data["match_passed"], bool), \
            "match_passed must be a boolean"

        assert "match_score" in data, \
            "Response must include match_score"
        assert 0 <= data["match_score"] <= 1, \
            "match_score must be between 0 and 1"

        # Should have a current hash
        assert "current_template_hash" in data or "face_embedding_hash" in data, \
            "Response should include current face hash"

    @pytest.mark.points(0.5, category="face_matching")
    def test_face_verify_no_face(self, face_client, enrolled_hash):
        """
        Test verification handles no-face images gracefully.

        When verifying an image with no detectable face:
        - Should return match_passed: false, OR
        - Should return HTTP 400
        """
        # Solid color - no face
        no_face_image = create_test_image_base64(128, 128, (100, 100, 100))

        response = face_client.post("/face/verify", json={
            "image": no_face_image,
            "reference_template_hash": enrolled_hash
        })

        # Fall back
        if response.status_code == 404:
            response = face_client.post("/face/match", json={
                "image": no_face_image,
                "reference_hash": enrolled_hash
            })

        # Should handle gracefully
        if response.status_code == 200:
            data = response.json()
            # No face should not match
            assert data.get("match_passed") is False or \
                   data.get("face_detected") is False, \
                   "No-face images should not match"
        elif response.status_code == 400:
            # 400 is acceptable
            pass
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")


# =============================================================================
# Privacy Compliance Tests (2 points)
# =============================================================================

class TestPrivacyCompliance:
    """Test privacy compliance - hash-only storage (2 points)"""

    @pytest.mark.points(1, category="privacy")
    def test_face_embedding_hash_format(self, face_client):
        """
        Test that face embedding is returned as SHA-256 hash.

        The face_template_hash or face_embedding_hash should be:
        - Exactly 64 characters long (SHA-256 hex)
        - All lowercase hexadecimal characters
        """
        # Use real face image for enrollment
        real_face_image = get_real_face_image_base64()

        response = face_client.post("/face/enroll", json={
            "user_id": "privacy-test-user",
            "image": real_face_image,
            "camera_consent": True
        })

        hash_value = None

        if response.status_code in [200, 201]:
            data = response.json()
            hash_value = data.get("face_template_hash")

        # Fall back to liveness check
        if not hash_value:
            response = face_client.post("/liveness/check", json={
                "challenge_response": real_face_image,
                "challenge_type": "passive"
            })

            if response.status_code == 200:
                data = response.json()
                hash_value = data.get("face_embedding_hash")

        assert hash_value is not None, \
            "Should get face_template_hash from enrollment or face_embedding_hash from liveness"

        # Validate SHA-256 format
        assert len(hash_value) == 64, \
            f"Hash should be 64 characters (SHA-256), got {len(hash_value)}"
        assert all(c in '0123456789abcdef' for c in hash_value.lower()), \
            "Hash should contain only hexadecimal characters"

    @pytest.mark.points(1, category="privacy")
    def test_no_raw_image_in_response(self, face_client):
        """
        Test that responses do not contain raw image data.

        Privacy requirement: Never return raw image data in responses.
        Only hashes should be returned.
        """
        real_face_image = get_real_face_image_base64()

        # Test enrollment response
        response = face_client.post("/face/enroll", json={
            "user_id": "privacy-test-user-2",
            "image": real_face_image,
            "camera_consent": True
        })

        assert response.status_code in [200, 201], \
            f"Enrollment should succeed, got {response.status_code}"

        response_text = response.text.lower()

        # Should not contain base64 image patterns
        assert "data:image" not in response_text, \
            "Response should not contain data:image URI"

        # Should not contain the input image (first 100 chars of base64)
        assert real_face_image[:100].lower() not in response_text, \
            "Response should not contain input image data"

        # Also test liveness response
        response = face_client.post("/liveness/check", json={
            "challenge_response": real_face_image,
            "challenge_type": "passive"
        })

        if response.status_code == 200:
            response_text = response.text.lower()
            assert "data:image" not in response_text, \
                "Liveness response should not contain image data"
            assert real_face_image[:100].lower() not in response_text, \
                "Liveness response should not contain input image data"


# =============================================================================
# Risk Assessment Tests (3 points)
# =============================================================================

class TestRiskAssessment:
    """Test risk assessment functionality (3 points)"""

    @pytest.mark.points(1, category="risk_assessment")
    def test_risk_assess_response_format(self, face_client):
        """
        Test that risk assessment returns correct response format.

        Response should include:
        - risk_score (float 0-1, where 0=safe, 1=risky)
        - risk_level (LOW, MEDIUM, HIGH, or CRITICAL)
        - signals (object with individual signal scores)
        - pass_threshold (boolean)
        """
        response = face_client.post("/risk/assess", json={
            "liveness_score": 0.8,
            "face_match_score": 0.9,
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

        assert response.status_code != 404, \
            "POST /risk/assess endpoint not found"

        assert response.status_code == 200, \
            f"Risk assessment failed with status {response.status_code}"

        data = response.json()

        # Required fields
        assert "risk_score" in data, \
            "Response must include 'risk_score'"
        assert "risk_level" in data, \
            "Response must include 'risk_level'"
        assert "pass_threshold" in data, \
            "Response must include 'pass_threshold'"

        # Validate risk_score range
        assert 0 <= data["risk_score"] <= 1, \
            "risk_score must be between 0 and 1"

        # Validate risk_level values
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"], \
            f"Invalid risk_level: {data['risk_level']}"

    @pytest.mark.points(1, category="risk_assessment")
    def test_high_scores_produce_low_risk(self, face_client):
        """
        Test that high liveness and face match scores result in low risk.

        Good signals should produce:
        - risk_score < 0.5
        - pass_threshold = true
        """
        response = face_client.post("/risk/assess", json={
            "liveness_score": 0.95,
            "face_match_score": 0.95,
            "ip_address": "8.8.8.8",
            "user_agent": "Mozilla/5.0",
            "geolocation": {
                "latitude": 1.3483,
                "longitude": 103.6831,
                "accuracy": 10
            }
        })

        assert response.status_code == 200

        data = response.json()

        # High scores should result in low risk
        assert data["risk_score"] < 0.5, \
            f"High scores should produce low risk, got {data['risk_score']}"
        assert data["pass_threshold"] is True, \
            "Good signals should pass threshold"

    @pytest.mark.points(1, category="risk_assessment")
    def test_low_scores_produce_high_risk(self, face_client):
        """
        Test that low liveness and face match scores result in high risk.

        Bad signals should produce:
        - risk_score >= 0.5
        - pass_threshold = false
        """
        response = face_client.post("/risk/assess", json={
            "liveness_score": 0.2,
            "face_match_score": 0.3,
            "ip_address": "10.0.0.1",  # Private IP (VPN indicator)
            "user_agent": "vpn-client"  # VPN keyword
        })

        assert response.status_code == 200

        data = response.json()

        # Low scores should result in high risk
        assert data["risk_score"] >= 0.5, \
            f"Low scores should produce high risk, got {data['risk_score']}"
        assert data["pass_threshold"] is False, \
            "Bad signals should fail threshold"

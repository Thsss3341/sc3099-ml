"""
PUBLIC TESTS: Integration & Performance Tests (4 points)

These tests verify end-to-end workflows and basic performance:
- End-to-end check-in flow (2 pts)
- Response latency (1 pt)
- Concurrent users (1 pt)

Students can run these tests to ensure all modules work together.
"""
import pytest
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestEndToEndCheckIn:
    """Test complete check-in workflow (2 points)"""

    @pytest.mark.points(2, category="integration")
    def test_complete_student_checkin_flow(self, client, test_course, auth_headers_instructor, auth_headers_admin):
        """Test the complete student check-in workflow from registration to approval."""
        from datetime import datetime, timedelta

        # 1. Student registers
        student_email = f"integration_student_{uuid.uuid4().hex[:6]}@test.com"
        register_resp = client.post("/api/v1/auth/register", json={
            "email": student_email,
            "password": "testpass123",
            "full_name": "Integration Test Student",
            "role": "student"
        })
        assert register_resp.status_code == 201
        student_data = register_resp.json()
        student_id = student_data["id"]

        # 2. Student logs in
        login_resp = client.post("/api/v1/auth/login", json={
            "email": student_email,
            "password": "testpass123"
        })
        assert login_resp.status_code == 200
        student_token = login_resp.json()["access_token"]
        student_headers = {"Authorization": f"Bearer {student_token}"}

        # 3. Student updates consent
        consent_resp = client.put("/api/v1/users/me", headers=student_headers, json={
            "camera_consent": True,
            "geolocation_consent": True
        })
        assert consent_resp.status_code == 200

        # 4. Enroll student in course via admin API
        enroll_resp = client.post("/api/v1/admin/enrollments/", headers=auth_headers_admin, json={
            "student_id": student_id,
            "course_id": test_course["id"]
        })
        # If admin enrollments endpoint not available, try regular endpoint
        if enroll_resp.status_code == 404:
            enroll_resp = client.post("/api/v1/enrollments/", headers=auth_headers_admin, json={
                "student_id": student_id,
                "course_id": test_course["id"]
            })
        assert enroll_resp.status_code in [200, 201], f"Enrollment failed: {enroll_resp.text}"

        # 5. Create active session (instructor)
        now = datetime.utcnow()
        session_resp = client.post("/api/v1/sessions/", headers=auth_headers_instructor, json={
            "course_id": test_course["id"],
            "name": "Integration Test Lecture",
            "session_type": "lecture",
            "scheduled_start": (now + timedelta(minutes=5)).isoformat() + "Z",
            "scheduled_end": (now + timedelta(hours=2)).isoformat() + "Z",
            "checkin_opens_at": (now - timedelta(minutes=10)).isoformat() + "Z",
            "checkin_closes_at": (now + timedelta(minutes=30)).isoformat() + "Z",
            "require_liveness_check": False,  # Simplified for integration test
            "risk_threshold": 0.5
        })
        assert session_resp.status_code in [200, 201], f"Session creation failed: {session_resp.text}"
        session_id = session_resp.json()["id"]

        # 5b. Activate the session using admin API
        activate_resp = client.patch(
            f"/api/v1/admin/sessions/{session_id}/status",
            headers=auth_headers_admin,
            json={"status": "active"}
        )
        assert activate_resp.status_code == 200, f"Session activation failed: {activate_resp.text}"

        # 6. Student lists active sessions
        sessions_resp = client.get("/api/v1/sessions/active")
        assert sessions_resp.status_code == 200

        # 7. Student checks in
        checkin_resp = client.post("/api/v1/checkins/", headers=student_headers, json={
            "session_id": session_id,
            "latitude": 1.3483,
            "longitude": 103.6831,
            "device_fingerprint": f"integration_test_device_{uuid.uuid4().hex[:6]}"
        })
        assert checkin_resp.status_code == 201
        checkin_data = checkin_resp.json()
        assert checkin_data["status"] in ["pending", "approved", "flagged"]

        # 8. Student views their check-ins
        my_checkins_resp = client.get("/api/v1/checkins/my-checkins", headers=student_headers)
        assert my_checkins_resp.status_code == 200
        assert len(my_checkins_resp.json()) > 0


class TestBasicPerformance:
    """Test basic performance requirements (2 points)"""

    @pytest.mark.points(1, category="performance")
    def test_auth_endpoint_latency(self, client, test_student):
        """Test that authentication completes within 2 seconds."""
        start = time.time()
        response = client.post("/api/v1/auth/login", json={
            "email": test_student["email"],
            "password": test_student["password"]
        })
        duration = time.time() - start

        assert response.status_code == 200
        assert duration < 2.0, f"Login took {duration:.2f}s, should be < 2s"

    @pytest.mark.points(1, category="performance")
    def test_concurrent_logins(self, client):
        """Test handling of concurrent login requests."""
        def perform_login():
            try:
                # Register unique user
                email = f"perf_{uuid.uuid4().hex[:8]}@test.com"
                client.post("/api/v1/auth/register", json={
                    "email": email,
                    "password": "testpass123",
                    "full_name": "Performance Test",
                    "role": "student"
                })

                # Login
                response = client.post("/api/v1/auth/login", json={
                    "email": email,
                    "password": "testpass123"
                })
                return response.status_code == 200
            except Exception:
                return False

        # Run 10 concurrent logins
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(perform_login) for _ in range(10)]
            results = [f.result() for f in as_completed(futures)]

        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Success rate {success_rate:.0%} < 80%"

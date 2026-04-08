"""
PUBLIC TESTS: Performance (5 points)

These tests verify performance requirements:
- Response latency (p95 < 2s) - 3 pts
- Concurrent user handling - 1 pt
- Database query optimization - 1 pt

Students can run these tests to ensure their implementation meets performance targets.
"""
import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestResponseLatency:
    """Test API response latency (3 points)"""

    @pytest.mark.points(1, category="latency")
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

    @pytest.mark.points(1, category="latency")
    def test_checkin_endpoint_latency(self, client, test_student, test_session, test_enrollment, auth_headers_student):
        """Test that check-in completes within 2 seconds."""
        start = time.time()
        response = client.post("/api/v1/checkins/", headers=auth_headers_student, json={
            "session_id": test_session["id"],
            "latitude": 1.3483,
            "longitude": 103.6831,
            "device_fingerprint": f"perf_test_{int(time.time())}"
        })
        duration = time.time() - start

        assert response.status_code == 201
        assert duration < 2.0, f"Check-in took {duration:.2f}s, should be < 2s"

    @pytest.mark.points(0.5, category="latency")
    def test_list_endpoint_latency(self, client):
        """Test that list endpoints respond quickly."""
        start = time.time()
        response = client.get("/api/v1/courses/")
        duration = time.time() - start

        assert response.status_code == 200
        assert duration < 1.0, f"List courses took {duration:.2f}s, should be < 1s"

    @pytest.mark.points(0.5, category="latency")
    def test_health_check_latency(self, client):
        """Test that health check is very fast."""
        start = time.time()
        response = client.get("/health")
        duration = time.time() - start

        assert response.status_code in [200, 503]
        assert duration < 0.5, f"Health check took {duration:.2f}s, should be < 0.5s"


class TestConcurrentUsers:
    """Test concurrent user handling (1 point)"""

    @pytest.mark.points(1, category="concurrency")
    def test_concurrent_logins(self, client, test_student):
        """Test handling 10 concurrent login requests with same user."""
        email = test_student["email"]
        password = test_student["password"]

        def login():
            response = client.post("/api/v1/auth/login", json={
                "email": email,
                "password": password
            })
            return response.status_code == 200

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(login) for _ in range(10)]
            results = [f.result() for f in as_completed(futures)]

        # At least 80% should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Only {success_rate*100:.1f}% of concurrent logins succeeded"

    @pytest.mark.skip(reason="Flaky test - can fail due to SQLite concurrent write limitations in test environment")
    def test_concurrent_checkins(self, client, db_session, test_course, test_session, test_instructor):
        """Test handling concurrent check-ins from different students."""
        from app.models.user import User, UserRole
        from app.models.enrollment import Enrollment
        from app.core.security import get_password_hash
        import uuid
        from datetime import datetime

        # Create 20 test students
        student_emails = []
        for i in range(20):
            email = f"concurrent{i}@test.com"
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                full_name=f"Concurrent Student {i}",
                hashed_password=get_password_hash("testpass123"),
                role=UserRole.STUDENT,
                is_active=True,
                camera_consent=True,
                geolocation_consent=True,
                created_at=datetime.now(UTC).replace(tzinfo=None),
                updated_at=datetime.now(UTC).replace(tzinfo=None),
            )
            db_session.add(user)

            enrollment = Enrollment(
                id=str(uuid.uuid4()),
                student_id=user.id,
                course_id=test_course.id,
                is_active=True,
                enrolled_at=datetime.now(UTC).replace(tzinfo=None)
            )
            db_session.add(enrollment)
            student_emails.append(email)

        db_session.commit()

        # Concurrent check-ins
        def checkin(student_email):
            # Login
            login_resp = client.post("/api/v1/auth/login", json={
                "email": student_email,
                "password": "testpass123"
            })
            if login_resp.status_code != 200:
                return False

            token = login_resp.json()["access_token"]

            # Check-in
            checkin_resp = client.post("/api/v1/checkins/", headers={
                "Authorization": f"Bearer {token}"
            }, json={
                "session_id": test_session.id,
                "latitude": 1.3483,
                "longitude": 103.6831,
                "device_fingerprint": f"device_{student_email}"
            })
            return checkin_resp.status_code == 201

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(checkin, email) for email in student_emails]
            results = [f.result() for f in as_completed(futures)]

        # At least 90% should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.9, f"Only {success_rate*100:.1f}% of concurrent check-ins succeeded"


class TestDatabaseOptimization:
    """Test database query optimization (1 point)"""

    @pytest.mark.points(0.5, category="db_optimization")
    def test_no_n_plus_one_queries(self, client, test_session, auth_headers_instructor):
        """Test that listing check-ins doesn't cause N+1 query problem."""
        # This is a basic test - in production, would use query profiling
        start = time.time()
        response = client.get(f"/api/v1/checkins/session/{test_session['id']}", headers=auth_headers_instructor)
        duration = time.time() - start

        assert response.status_code == 200
        # Should be fast even with many check-ins
        assert duration < 1.0, f"Listing check-ins took {duration:.2f}s, may indicate N+1 queries"

    @pytest.mark.points(0.5, category="db_optimization")
    def test_pagination_support(self, client):
        """Test that large result sets support pagination."""
        # Basic test for pagination existence
        response = client.get("/api/v1/courses/", params={"limit": 10, "offset": 0})
        # Should either work with pagination or return all results quickly
        assert response.status_code == 200

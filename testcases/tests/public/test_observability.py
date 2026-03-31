"""
PUBLIC TESTS: Observability & Admin API Tests (12 bonus points)

These tests validate the admin and observability endpoints for
dashboard functionality, session management, and data export.

Students can run these tests locally to validate their implementation.
These are bonus points that count towards extra credit.
"""
import pytest
from datetime import datetime, timedelta, UTC
import uuid


class TestStatsEndpoints:
    """Test statistics endpoints (4 bonus points)"""

    @pytest.mark.points(1, category="observability_bonus")
    def test_stats_overview(self, client, auth_headers_instructor):
        """Test stats overview returns system-wide statistics."""
        response = client.get("/api/v1/stats/overview", headers=auth_headers_instructor)

        assert response.status_code == 200
        data = response.json()

        # Required fields for dashboard overview
        assert "total_sessions" in data
        assert "active_sessions" in data
        assert "total_courses" in data
        assert "total_students" in data
        assert "today_checkins" in data
        assert "flagged_pending" in data
        assert "approval_rate" in data

        # Values should be non-negative
        assert data["total_sessions"] >= 0
        assert data["approval_rate"] >= 0

    @pytest.mark.points(1, category="observability_bonus")
    def test_stats_session(self, client, test_session, auth_headers_instructor):
        """Test session stats returns attendance details."""
        response = client.get(
            f"/api/v1/stats/sessions/{test_session['id']}",
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields for session analytics
        assert "session_id" in data
        assert "session_name" in data
        assert "total_enrolled" in data
        assert "checked_in_count" in data
        assert "approved_count" in data
        assert "flagged_count" in data
        assert "attendance_rate" in data
        assert "average_risk_score" in data

    @pytest.mark.points(1, category="observability_bonus")
    def test_stats_course(self, client, test_course, auth_headers_instructor):
        """Test course stats returns course-level analytics."""
        response = client.get(
            f"/api/v1/stats/courses/{test_course['id']}",
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields for course analytics
        assert "course_id" in data
        assert "course_code" in data
        assert "total_enrolled" in data
        assert "total_sessions" in data
        assert "average_attendance_rate" in data
        assert "flagged_checkins" in data

    @pytest.mark.points(1, category="observability_bonus")
    def test_stats_student(self, client, test_student, test_enrollment, auth_headers_instructor):
        """Test student stats returns individual attendance record."""
        response = client.get(
            f"/api/v1/stats/students/{test_student['id']}",
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields for student analytics
        assert "student_id" in data
        assert "student_name" in data
        assert "total_enrolled_courses" in data
        assert "total_sessions" in data
        assert "attended_sessions" in data
        assert "attendance_rate" in data
        assert "recent_sessions" in data


class TestSessionManagement:
    """Test session CRUD operations (3 bonus points)"""

    @pytest.mark.points(1, category="observability_bonus")
    def test_list_sessions_with_filters(self, client, test_session, auth_headers_instructor):
        """Test listing sessions with filter parameters."""
        response = client.get(
            "/api/v1/sessions/",
            params={"limit": 10, "offset": 0},
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()

        # Should return paginated response
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert isinstance(data["items"], list)

    @pytest.mark.points(1, category="observability_bonus")
    def test_create_session(self, client, test_course, auth_headers_instructor):
        """Test creating a new session."""
        now = datetime.now(UTC)
        session_data = {
            "course_id": test_course["id"],
            "name": f"Test Session {uuid.uuid4().hex[:8]}",
            "session_type": "lecture",
            "scheduled_start": (now + timedelta(hours=1)).isoformat(),
            "scheduled_end": (now + timedelta(hours=2)).isoformat()
        }

        response = client.post(
            "/api/v1/sessions/",
            json=session_data,
            headers=auth_headers_instructor
        )

        assert response.status_code == 201
        data = response.json()

        assert "id" in data
        assert data["name"] == session_data["name"]
        assert data["status"] == "scheduled"

    @pytest.mark.points(1, category="observability_bonus")
    def test_update_session(self, client, test_session, auth_headers_instructor):
        """Test updating a session."""
        update_data = {
            "name": f"Updated Session {uuid.uuid4().hex[:8]}"
        }

        response = client.patch(
            f"/api/v1/sessions/{test_session['id']}",
            json=update_data,
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]


class TestEnrollmentManagement:
    """Test enrollment endpoints (2 bonus points)"""

    @pytest.mark.points(1, category="observability_bonus")
    def test_list_course_enrollments(self, client, test_course, test_enrollment,
                                      auth_headers_instructor):
        """Test listing enrollments for a course."""
        response = client.get(
            f"/api/v1/enrollments/course/{test_course['id']}",
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()

        assert "course_id" in data
        assert "total_enrolled" in data
        assert "students" in data
        assert isinstance(data["students"], list)

    @pytest.mark.points(1, category="observability_bonus")
    def test_my_enrollments(self, client, test_enrollment, auth_headers_student):
        """Test getting student's own enrollments."""
        response = client.get(
            "/api/v1/enrollments/my-enrollments",
            headers=auth_headers_student
        )

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        if len(data) > 0:
            enrollment = data[0]
            assert "course_id" in enrollment
            assert "course_code" in enrollment
            assert "is_active" in enrollment


class TestCheckinFiltering:
    """Test check-in filtering and flagged queue (2 bonus points)"""

    @pytest.mark.points(1, category="observability_bonus")
    def test_list_all_checkins(self, client, auth_headers_instructor):
        """Test listing all check-ins with filters."""
        response = client.get(
            "/api/v1/checkins/",
            params={"limit": 10},
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()

        assert "items" in data
        assert "total" in data
        assert isinstance(data["items"], list)

    @pytest.mark.points(1, category="observability_bonus")
    def test_flagged_checkins_queue(self, client, auth_headers_instructor):
        """Test getting flagged check-ins queue."""
        response = client.get(
            "/api/v1/checkins/flagged",
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()

        assert "items" in data
        assert "total" in data
        # All items should be flagged or appealed
        for item in data["items"]:
            assert item["status"] in ["flagged", "appealed"]


class TestExportEndpoints:
    """Test data export endpoints (1 bonus point)"""

    @pytest.mark.points(1, category="observability_bonus")
    def test_export_session_attendance_json(self, client, test_session,
                                            auth_headers_instructor):
        """Test exporting session attendance as JSON."""
        response = client.get(
            f"/api/v1/export/session/{test_session['id']}",
            params={"format": "json"},
            headers=auth_headers_instructor
        )

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert "summary" in data
        assert "records" in data
        assert isinstance(data["records"], list)

        # Summary should have attendance stats
        summary = data["summary"]
        assert "total_enrolled" in summary
        assert "attendance_rate" in summary


class TestAuditLogFiltering:
    """Test audit log advanced filtering"""

    def test_audit_logs_with_filters(self, client, auth_headers_admin):
        """Test listing audit logs with filters."""
        response = client.get(
            "/api/v1/audit/",
            params={"limit": 10},
            headers=auth_headers_admin
        )

        assert response.status_code == 200
        data = response.json()

        assert "items" in data
        assert "total" in data

    def test_audit_summary(self, client, auth_headers_admin):
        """Test audit log summary endpoint."""
        response = client.get(
            "/api/v1/audit/summary",
            params={"days": 7},
            headers=auth_headers_admin
        )

        assert response.status_code == 200
        data = response.json()

        assert "period_days" in data
        assert "total_logs" in data
        assert "by_action" in data


class TestAdminUserEndpoints:
    """Test admin user management endpoints"""

    def test_list_users_admin(self, client, auth_headers_admin):
        """Test listing all users (admin only)."""
        response = client.get(
            "/api/v1/users/",
            headers=auth_headers_admin
        )

        assert response.status_code == 200
        data = response.json()

        assert "items" in data
        assert "total" in data
        assert isinstance(data["items"], list)

    def test_list_users_forbidden_for_instructor(self, client, auth_headers_instructor):
        """Test listing all users is forbidden for non-admin."""
        response = client.get(
            "/api/v1/users/",
            headers=auth_headers_instructor
        )

        # Should be 401, 403, or similar forbidden status
        assert response.status_code in [401, 403]


class TestCourseManagement:
    """Test course CRUD operations"""

    def test_create_course_admin(self, client, auth_headers_admin):
        """Test creating a course (admin only)."""
        course_data = {
            "code": f"TEST{uuid.uuid4().hex[:4].upper()}",
            "name": "Test Course",
            "semester": "AY2024-25 Sem 1"
        }

        response = client.post(
            "/api/v1/courses/",
            json=course_data,
            headers=auth_headers_admin
        )

        assert response.status_code == 201
        data = response.json()
        assert data["code"] == course_data["code"]

    def test_create_course_forbidden_for_instructor(self, client, auth_headers_instructor):
        """Test creating a course is forbidden for non-admin."""
        course_data = {
            "code": "FORBIDDEN101",
            "name": "Forbidden Course",
            "semester": "AY2024-25 Sem 1"
        }

        response = client.post(
            "/api/v1/courses/",
            json=course_data,
            headers=auth_headers_instructor
        )

        assert response.status_code in [401, 403]


class TestDeviceManagement:
    """Test device management endpoints"""

    def test_my_devices(self, client, auth_headers_student):
        """Test getting own devices."""
        response = client.get(
            "/api/v1/devices/my-devices",
            headers=auth_headers_student
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_all_devices_admin(self, client, auth_headers_admin):
        """Test listing all devices (admin only)."""
        response = client.get(
            "/api/v1/devices/",
            headers=auth_headers_admin
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data

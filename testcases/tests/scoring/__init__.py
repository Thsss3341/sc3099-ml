"""
SAIV Capstone Scoring Plugin

A pytest plugin that provides per-test scoring with point markers.

Usage:
    @pytest.mark.points(2, category="authentication")
    def test_user_login(self, client):
        ...

Running tests will show a scoring summary at the end.
"""

from .plugin import ScoringPlugin

__all__ = ['ScoringPlugin']

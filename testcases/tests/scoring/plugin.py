"""
Pytest Scoring Plugin for SAIV Capstone Project

This plugin tracks points for each test and provides a scoring summary.
Each test can be marked with @pytest.mark.points(N, category="...") to
assign point values.
"""
import pytest
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


@dataclass
class TestScore:
    """Score data for a single test."""
    node_id: str
    name: str
    max_points: float
    earned_points: float
    passed: bool
    category: Optional[str] = None
    file_path: Optional[str] = None
    duration: float = 0.0
    error_message: Optional[str] = None

    @property
    def status_icon(self) -> str:
        return "✓" if self.passed else "✗"


@dataclass
class ScoringSession:
    """Accumulates scoring data across test session."""
    tests: Dict[str, TestScore] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def total_points(self) -> float:
        return sum(t.max_points for t in self.tests.values())

    @property
    def earned_points(self) -> float:
        return sum(t.earned_points for t in self.tests.values())

    @property
    def score_percentage(self) -> float:
        if self.total_points == 0:
            return 0.0
        return (self.earned_points / self.total_points) * 100

    @property
    def letter_grade(self) -> str:
        pct = self.score_percentage
        if pct >= 90:
            return "A"
        elif pct >= 80:
            return "B"
        elif pct >= 70:
            return "C"
        elif pct >= 60:
            return "D"
        else:
            return "F"

    def by_category(self) -> Dict[str, List[TestScore]]:
        """Group scores by category."""
        categories: Dict[str, List[TestScore]] = {}
        for test in self.tests.values():
            cat = test.category or "uncategorized"
            categories.setdefault(cat, []).append(test)
        return categories

    def by_file(self) -> Dict[str, List[TestScore]]:
        """Group scores by file."""
        files: Dict[str, List[TestScore]] = {}
        for test in self.tests.values():
            filepath = test.file_path or "unknown"
            files.setdefault(filepath, []).append(test)
        return files


class ScoringPlugin:
    """Main pytest plugin for scoring."""

    def __init__(self, config: Any):
        self.config = config
        self.session = ScoringSession()
        self._enabled = True

    @pytest.hookimpl(tryfirst=True)
    def pytest_configure(self, config: Any) -> None:
        """Register custom markers."""
        config.addinivalue_line(
            "markers",
            "points(value, category=None): Assign point value to test. "
            "Example: @pytest.mark.points(2, category='authentication')"
        )

    @pytest.hookimpl
    def pytest_sessionstart(self, session: Any) -> None:
        """Initialize scoring session."""
        self.session.start_time = datetime.now()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
        """Capture test results and calculate points."""
        outcome = yield
        report = outcome.get_result()

        if call.when != "call":
            return

        # Get points marker
        marker = item.get_closest_marker("points")
        if marker is None:
            return  # Test has no points assigned

        max_points = marker.args[0] if marker.args else 0
        category = marker.kwargs.get("category")

        # Extract file path from node_id
        file_path = item.nodeid.split("::")[0] if "::" in item.nodeid else None
        if file_path:
            file_path = Path(file_path).name

        # Determine earned points
        if report.passed:
            earned = max_points
            passed = True
            error_msg = None
        elif report.skipped:
            # Skipped tests earn 0 but don't count as failed
            earned = 0
            passed = False
            error_msg = "SKIPPED"
        else:
            passed = False
            error_msg = str(report.longrepr)[:200] if report.longrepr else "Test failed"
            earned = 0

        self.session.tests[item.nodeid] = TestScore(
            node_id=item.nodeid,
            name=item.name,
            max_points=max_points,
            earned_points=earned,
            passed=passed,
            category=category,
            file_path=file_path,
            duration=report.duration if hasattr(report, 'duration') else 0.0,
            error_message=error_msg
        )

    @pytest.hookimpl
    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        """Finalize scoring session."""
        self.session.end_time = datetime.now()

        # Write JSON report if requested
        report_path = self.config.getoption("score_report", None)
        if report_path:
            self._write_json_report(report_path)

        # Write HTML report if requested
        html_path = self.config.getoption("score_html", None)
        if html_path:
            self._write_html_report(html_path)

    @pytest.hookimpl
    def pytest_terminal_summary(self, terminalreporter: Any, exitstatus: int, config: Any) -> None:
        """Output scoring summary to terminal."""
        if config.getoption("no_score_summary", False):
            return

        if not self.session.tests:
            return  # No scored tests

        tr = terminalreporter

        tr.write_sep("=", "SCORING SUMMARY", bold=True, yellow=True)
        tr.write_line("")

        # Group by file
        for file_path, tests in sorted(self.session.by_file().items()):
            file_max = sum(t.max_points for t in tests)
            file_earned = sum(t.earned_points for t in tests)
            passed = sum(1 for t in tests if t.passed)
            total = len(tests)

            tr.write_line(f"  {file_path}: {file_earned:.1f}/{file_max:.1f} pts ({passed}/{total} tests)")

            # Show individual failing tests
            for test in tests:
                if not test.passed:
                    tr.write_line(f"    {test.status_icon} {test.name}: 0/{test.max_points:.0f} pts")

        tr.write_line("")
        tr.write_sep("-", "")

        # Category breakdown (if categories used)
        categories = self.session.by_category()
        if len(categories) > 1 or "uncategorized" not in categories:
            tr.write_line("  By Category:")
            for category, tests in sorted(categories.items()):
                cat_max = sum(t.max_points for t in tests)
                cat_earned = sum(t.earned_points for t in tests)
                tr.write_line(f"    {category}: {cat_earned:.1f}/{cat_max:.1f} pts")
            tr.write_line("")

        # Final score
        tr.write_line(
            f"  TOTAL SCORE: {self.session.earned_points:.1f}/{self.session.total_points:.1f} "
            f"({self.session.score_percentage:.1f}%)"
        )
        tr.write_line(f"  LETTER GRADE: {self.session.letter_grade}")
        tr.write_sep("=", "")

    def _write_json_report(self, path: str) -> None:
        """Generate JSON score report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_points": self.session.total_points,
                "earned_points": self.session.earned_points,
                "percentage": self.session.score_percentage,
                "letter_grade": self.session.letter_grade,
                "tests_passed": sum(1 for t in self.session.tests.values() if t.passed),
                "tests_failed": sum(1 for t in self.session.tests.values() if not t.passed),
                "tests_total": len(self.session.tests),
            },
            "by_file": {},
            "by_category": {},
            "tests": []
        }

        # File breakdown
        for filepath, tests in self.session.by_file().items():
            report["by_file"][filepath] = {
                "max_points": sum(t.max_points for t in tests),
                "earned_points": sum(t.earned_points for t in tests),
                "passed": sum(1 for t in tests if t.passed),
                "failed": sum(1 for t in tests if not t.passed),
            }

        # Category breakdown
        for category, tests in self.session.by_category().items():
            report["by_category"][category] = {
                "max_points": sum(t.max_points for t in tests),
                "earned_points": sum(t.earned_points for t in tests),
                "passed": sum(1 for t in tests if t.passed),
                "failed": sum(1 for t in tests if not t.passed),
            }

        # Individual test results
        for test in self.session.tests.values():
            report["tests"].append({
                "node_id": test.node_id,
                "name": test.name,
                "file": test.file_path,
                "max_points": test.max_points,
                "earned_points": test.earned_points,
                "passed": test.passed,
                "category": test.category,
                "duration": test.duration,
                "error": test.error_message[:200] if test.error_message else None
            })

        Path(path).write_text(json.dumps(report, indent=2))

    def _write_html_report(self, path: str) -> None:
        """Generate HTML score report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SAIV Test Score Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .grade {{ font-size: 48px; font-weight: bold; }}
        .grade-A {{ color: #22c55e; }}
        .grade-B {{ color: #84cc16; }}
        .grade-C {{ color: #eab308; }}
        .grade-D {{ color: #f97316; }}
        .grade-F {{ color: #ef4444; }}
        .progress {{ background: #e5e5e5; height: 24px; border-radius: 4px; overflow: hidden; }}
        .progress-bar {{ background: #3b82f6; height: 100%; transition: width 0.3s; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e5e5; }}
        th {{ background: #f9fafb; font-weight: 600; }}
        .pass {{ color: #22c55e; }}
        .fail {{ color: #ef4444; }}
        .file-header {{ background: #f0f9ff; font-weight: 600; }}
    </style>
</head>
<body>
    <h1>SAIV Capstone - Test Score Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <div class="grade grade-{self.session.letter_grade}">{self.session.letter_grade}</div>
        <h2>{self.session.earned_points:.1f} / {self.session.total_points:.1f} points ({self.session.score_percentage:.1f}%)</h2>
        <div class="progress">
            <div class="progress-bar" style="width: {min(self.session.score_percentage, 100):.1f}%"></div>
        </div>
    </div>

    <h2>Results by File</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Status</th>
            <th>Points</th>
            <th>Category</th>
        </tr>
"""

        for filepath, tests in sorted(self.session.by_file().items()):
            file_max = sum(t.max_points for t in tests)
            file_earned = sum(t.earned_points for t in tests)
            html += f'<tr class="file-header"><td colspan="4">{filepath} - {file_earned:.1f}/{file_max:.1f} pts</td></tr>\n'

            for test in tests:
                status = "pass" if test.passed else "fail"
                icon = "✓" if test.passed else "✗"
                html += f"""<tr>
            <td>{test.name}</td>
            <td class="{status}">{icon}</td>
            <td>{test.earned_points:.0f}/{test.max_points:.0f}</td>
            <td>{test.category or '-'}</td>
        </tr>\n"""

        html += """
    </table>
</body>
</html>"""

        Path(path).write_text(html)


def pytest_addoption(parser: Any) -> None:
    """Add scoring-related command line options."""
    group = parser.getgroup("scoring", "SAIV Scoring Options")
    group.addoption(
        "--score-report",
        action="store",
        dest="score_report",
        default=None,
        metavar="PATH",
        help="Output JSON score report to specified path"
    )
    group.addoption(
        "--score-html",
        action="store",
        dest="score_html",
        default=None,
        metavar="PATH",
        help="Output HTML score report to specified path"
    )
    group.addoption(
        "--no-score-summary",
        action="store_true",
        dest="no_score_summary",
        default=False,
        help="Disable terminal score summary"
    )


def pytest_configure(config: Any) -> None:
    """Register the scoring plugin."""
    plugin = ScoringPlugin(config)
    config._scoring_plugin = plugin
    config.pluginmanager.register(plugin, "saiv_scoring")

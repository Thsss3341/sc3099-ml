"""
Helper functions for graceful test failures with educational messages.

These helpers provide clear feedback to students about what's not implemented
instead of cryptic import errors or assertion failures.
"""
import pytest
from typing import Any, Dict, List, Optional, Type


def require_endpoint(response, endpoint: str, expected_status: int = 200) -> None:
    """
    Check that an endpoint exists and returns the expected status code.
    Provides helpful error messages on failure.

    Args:
        response: The HTTP response object
        endpoint: The endpoint path (for error messages)
        expected_status: Expected HTTP status code

    Raises:
        pytest.fail: With helpful message if endpoint doesn't work correctly
    """
    if response.status_code == 404:
        pytest.fail(
            f"\n\nEndpoint {endpoint} not found (404).\n"
            f"To fix: Implement this endpoint in your API router.\n"
            f"Make sure the router is included in app/main.py."
        )

    if response.status_code == 500:
        error_detail = ""
        try:
            error_detail = response.json().get("detail", response.text[:500])
        except Exception:
            error_detail = response.text[:500] if response.text else "No error details"

        pytest.fail(
            f"\n\nServer error (500) at {endpoint}.\n"
            f"Error: {error_detail}\n"
            f"To fix: Check your server logs for the full stack trace."
        )

    if response.status_code == 422:
        error_detail = ""
        try:
            error_detail = response.json().get("detail", response.text[:500])
        except Exception:
            error_detail = response.text[:500]

        pytest.fail(
            f"\n\nValidation error (422) at {endpoint}.\n"
            f"Error: {error_detail}\n"
            f"To fix: Check your Pydantic request schema matches the expected format."
        )

    if response.status_code != expected_status:
        pytest.fail(
            f"\n\nUnexpected status code at {endpoint}.\n"
            f"Expected: {expected_status}, Got: {response.status_code}\n"
            f"Response: {response.text[:300]}"
        )


def require_field(data: Dict[str, Any], field: str, field_type: Optional[Type] = None) -> Any:
    """
    Check that a response contains a required field with optional type checking.

    Args:
        data: The response data dictionary
        field: The field name to check
        field_type: Optional expected type for the field

    Returns:
        The field value if present

    Raises:
        pytest.fail: With helpful message if field is missing or wrong type
    """
    if field not in data:
        pytest.fail(
            f"\n\nResponse missing required field '{field}'.\n"
            f"Received fields: {list(data.keys())}\n"
            f"To fix: Update your Pydantic response schema to include '{field}'."
        )

    value = data[field]

    if field_type is not None:
        # Handle special case for UUID strings
        if field_type == str and isinstance(value, str):
            return value
        # Handle lists
        if field_type == list and isinstance(value, list):
            return value
        # Handle dicts
        if field_type == dict and isinstance(value, dict):
            return value
        # Handle None
        if value is None:
            return value
        # Standard type check
        if not isinstance(value, field_type):
            pytest.fail(
                f"\n\nField '{field}' has wrong type.\n"
                f"Expected: {field_type.__name__}, Got: {type(value).__name__}\n"
                f"To fix: Check your schema type annotations for '{field}'."
            )

    return value


def require_fields(data: Dict[str, Any], fields: List[str]) -> None:
    """
    Check that a response contains multiple required fields.

    Args:
        data: The response data dictionary
        fields: List of field names to check
    """
    missing = [f for f in fields if f not in data]
    if missing:
        pytest.fail(
            f"\n\nResponse missing required fields: {missing}\n"
            f"Received fields: {list(data.keys())}\n"
            f"To fix: Update your Pydantic response schema to include these fields."
        )


def require_module(module_path: str, purpose: str) -> bool:
    """
    Check if a module is importable, skip test with helpful message if not.

    Args:
        module_path: The module path to import (e.g., 'app.models.user')
        purpose: Human-readable description of what this module is needed for

    Returns:
        True if module is available

    Note:
        This function calls pytest.skip() if module is not available
    """
    try:
        __import__(module_path)
        return True
    except ImportError as e:
        pytest.skip(
            f"\n\nModule '{module_path}' not implemented.\n"
            f"This module is needed for: {purpose}\n"
            f"Import error: {e}"
        )
        return False


def require_app(app_path: str = "app.main") -> Any:
    """
    Check if the FastAPI app is available, skip with helpful message if not.

    Args:
        app_path: The module path where the app is defined

    Returns:
        The FastAPI app if available
    """
    try:
        import importlib
        module = importlib.import_module(app_path)
        return getattr(module, 'app', None)
    except ImportError as e:
        pytest.skip(
            f"\n\nFastAPI app not available.\n"
            f"To fix: Create {app_path}.py with a FastAPI application:\n\n"
            f"    from fastapi import FastAPI\n"
            f"    app = FastAPI()\n\n"
            f"Import error: {e}"
        )
        return None


def check_list_response(response, endpoint: str, min_items: int = 0) -> list:
    """
    Validate a list response from an endpoint.

    Args:
        response: The HTTP response
        endpoint: The endpoint path (for error messages)
        min_items: Minimum expected items in the list

    Returns:
        The list from the response
    """
    require_endpoint(response, endpoint, 200)

    data = response.json()

    if not isinstance(data, list):
        pytest.fail(
            f"\n\nEndpoint {endpoint} should return a list.\n"
            f"Got: {type(data).__name__}\n"
            f"To fix: Ensure your endpoint returns a list response."
        )

    if len(data) < min_items:
        pytest.fail(
            f"\n\nEndpoint {endpoint} returned too few items.\n"
            f"Expected at least: {min_items}, Got: {len(data)}"
        )

    return data


def check_auth_response(response, endpoint: str) -> dict:
    """
    Validate an authentication response (login/register).

    Args:
        response: The HTTP response
        endpoint: The endpoint path (for error messages)

    Returns:
        The response data with access_token
    """
    require_endpoint(response, endpoint, expected_status=200)

    data = response.json()
    require_field(data, "access_token", str)

    return data


def assert_within_range(value: float, min_val: float, max_val: float, field_name: str) -> None:
    """
    Assert that a numeric value is within a specified range.

    Args:
        value: The value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of the field (for error messages)
    """
    if not (min_val <= value <= max_val):
        pytest.fail(
            f"\n\nField '{field_name}' out of expected range.\n"
            f"Expected: {min_val} <= value <= {max_val}\n"
            f"Got: {value}"
        )

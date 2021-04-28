"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(np.array(expected), daily_mean(np.array(test)))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [1, 2]),
    ])
def test_daily_min(test, expected):
    """Test that min function works for an array of zeros and positive integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(np.array(expected), daily_min(test))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),
    ])
def test_daily_min(test, expected):
    """Test that max function works for an array of zeros and positive integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(np.array(expected), daily_max(test))


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test, expected, raises",
    [
        (
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None
        ),
        (
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None
        ),
        (
            np.array([[-1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            None,
            ValueError,
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            [[0.33, 0.66, 1], [0.66, 0.83, 1], [0.77, 0.88, 1]],
            None,
        ),
        (
            list([[0, 0.66, 1], 'foo', True]),
            None,
            AssertionError,
        ),
    ])
def test_patient_normalise(test, expected, raises):
    """Test normalisation works for arrays of one and positive integers."""
    from inflammation.models import patient_normalise
    if raises:
        with pytest.raises(raises):
            npt.assert_almost_equal(np.array(expected), patient_normalise(test), decimal=2)

    else:
        npt.assert_almost_equal(np.array(expected), patient_normalise(test), decimal=2)

# TODO(lesson-robust) Implement tests for the other statistical functions

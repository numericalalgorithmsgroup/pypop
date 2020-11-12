#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

__all__ = ["SimpleAsserter", "ListofStrAsserter"]


class BaseAsserter:
    """Parent class for type asserter objects

    Asserter classes provide a callable which can be used to assert the validity of an
    object.

    Subclasses should implement a check_type function returning true/false as needed.
    """

    _expected = None

    def __call__(self, testobj):
        if not self.check_type(testobj):
            raise TypeError(
                "Expected {} not {}".format(self._expected, testobj.__class__.__name__)
            )
        return True

    def check_type(self, testobj):
        """Return true if object is valid

        Parameters
        ----------
        testobj: object
            Object to be validated.
        """
        raise NotImplementedError()


class SimpleAsserter(BaseAsserter):
    """Assert that an object is of a given type

    Provides a callable object which asserts the provided object is of the required type.
    """

    def __init__(self, assert_class):
        """Define class to be validated

        Parameters
        ----------
        assert_class: class
            Reference class to be compared to.
        """
        self._assert_cls = assert_class
        self._expected = assert_class.__name__

    def check_type(self, testobj):
        if isinstance(testobj, self._assert_cls):
            return True


class ListofStrAsserter(BaseAsserter):
    """Assert that an object is a list of strings

    Provides a callable object which asserts a provided object is a list of strings.
    """

    _expected = "list of str"

    def check_type(self, testobj):
        if isinstance(testobj, list) and all(isinstance(x, str) for x in testobj):
            return True

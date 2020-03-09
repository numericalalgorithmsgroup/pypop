#!/usr/bin/env python3

import pandas


class HDFStoreContext:
    """A simple contextmanager that manages opening and closing of a pandas HDFStore
    object including closing correctly in the case that exceptions are thrown during
    read/write.
    """

    def __init__(self, filename, **kwargs):
        self._filename = filename
        self._kwargs = kwargs

    def __enter__(self):
        try:
            self._hdfstore = pandas.HDFStore(self._filename, **self._kwargs)
        except OSError as err:
            # Try to interpret and reraise the OSError as something actionable to the
            # despatcher
            if "does not exist" in str(err):
                raise FileNotFoundError('"{}" does not exist'.format(self._filename))
            if "HDF5 error" in str(err) or "not a regular file" in str(err):
                raise ValueError('"{}" is not a valid HDF5 file'.format(self._filename))
            # Reraise anything we don't catch explicitly
            raise
        return self._hdfstore

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._hdfstore.close()

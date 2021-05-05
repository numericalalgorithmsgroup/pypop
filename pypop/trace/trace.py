#!/usr/bin/env python3

from warnings import warn

import numpy
import pandas

from ..utils.pandas import HDFStoreContext
from ..utils.exceptions import WrongLoaderError, UnknownLoaderError

from .tracemetadata import TraceMetadata


class Trace:
    """Base class providing a skeleton for implementation of different trace classes.

    Includes support for summary files to allow fast resumption of analysis using cached
    data.
    """

    _loader_registry = {}
    _metadatakey = "PyPOPTraceMetadata"
    _statisticskey = "PyPOPTraceStatisticsKey"

    _formatversionkey = "PyPOPSummaryFormatVersion"
    _formatversion = 1

    def __new__(cls, filename, **kwargs):
        """Use a factory paradigm to return relevant subclass instance.

        Results should be agnostic to whether the trace or the summary file is passed,
        and return the appropriate subclass in either case.  Do this by caching the
        loader name in the metadata of the summary file. We can then peek at the
        summaryfile as we load it.
        """
        # If were are in a subclass __new__ then create a subclass instance
        if cls is Trace:
            raise NotImplementedError(
                "Trace should not be instantiated directly. Please use Trace.load()"
            )

        return super().__new__(cls)

    @staticmethod
    def load(filename, force_recalculation=False, tag=None, **kwargs):

        if force_recalculation is False:
            # First try to open file as if it *is* a summary file
            try:
                return Trace._load_from_summary_file(
                    filename, tag=tag, **kwargs
                )
            except WrongLoaderError:
                # Otherwise, see if it has a summaryfile present
                try:
                    return Trace._load_from_summary_file(
                        Trace.get_summary_filename(filename), tag=tag, **kwargs
                    )
                # Fallback - assume valid trace with no summary
                except (WrongLoaderError, FileNotFoundError):
                    pass

        # Otherwise iterate to find a suitable loader
        for loader in Trace._loader_registry.values():
            try:
                return loader(
                    filename, force_recalculation=force_recalculation, tag=tag, **kwargs
                )
            except WrongLoaderError:
                pass

        raise ValueError("Invalid or unsupported trace {}".format(filename))

    @staticmethod
    def _load_from_summary_file(filename, tag=None, **kwargs):

        # This will raise a WrongLoader error already so no need to trap and reraise
        metadata, _ = Trace._parse_summary_file(filename)

        try:
            trace = Trace._loader_registry[metadata.trace_subclass_name](
                filename, **kwargs
            )
        except KeyError:
            # Loader not known is a special case error
            raise UnknownLoaderError(
                "Unknown Loader {}. PyPOP version outdated or missing plugin?"
                "".format(metadata.trace_subclass_name)
            )

        if tag is not None:
            warn("Loading from summary, tag argument ignored")

        return trace

    def __init__(self, filename, force_recalculation=False, tag=None, **kwargs):
        # Save keyword args for later
        self._kwargs = kwargs

        self._analysis_failed = False

        if force_recalculation is False:
            # First check to see if we have been passed a summary file. If so, load from
            # that and skip parsing the whole trace again
            try:
                self._metadata, self._statistics = Trace._parse_summary_file(filename)
                # If this succeeded we were passed a summary file, so:
                self._tracefile = self._metadata.tracefile_name
                self._summaryfile = filename
                return
                # Otherwise, see if it has a summaryfile present
            except WrongLoaderError:
                try:
                    self._metadata, self._statistics = Trace._parse_summary_file(
                        Trace.get_summary_filename(filename)
                    )
                    # If this succeeded we were passed the trace itself but there is a
                    # valid summary file too:
                    self._tracefile = filename
                    self._summaryfile = Trace.get_summary_filename(filename)
                    return
                # Fallback - drop into full tracefile loader
                except (WrongLoaderError, FileNotFoundError):
                    pass

        self._metadata = TraceMetadata(self)
        if tag:
            self._metadata.tag = tag
        self._statistics = None

        # Loading from a tracefile so
        self._tracefile = filename
        self._summaryfile = None  # This will get set when we create it
        # And actually load the trace
        self._load_trace()

    def _load_trace(self):
        self._gather_metadata()

        if self._kwargs.get("eager_load", False):
            self._gather_statistics()
            self.write_summary_file()

    def _gather_metadata(self):
        raise NotImplementedError

    def _gather_statistics(self):
        raise NotImplementedError

    def ensure_loaded(self):
        # Can ensure load by evaluating self.statistics
        if self.statistics is None:
            raise RuntimeError("Internal Error: This should be unreachable code")

    @property
    def statistics(self):
        if self._statistics is not None:
            return self._statistics
        if self._analysis_failed is False:
            try:
                self._gather_statistics()
                return self._statistics
            except Exception as err:
                self._analysis_failed = str(err)
                raise
            self.write_summary_file()
        else:
            raise RuntimeError(
                "Analysis previously failed ({})".format(self._analysis_failed)
            )

    @property
    def stats(self):
        warn("Trace.stats is deprecated, please use Trace.statistics")
        return self.statistics

    @property
    def metadata(self):
        return self._metadata

    @staticmethod
    def get_summary_filename(filename):

        if filename.endswith(".pypopsummary"):
            return filename

        return ".".join([filename, "pypopsummary"])

    def write_summary_file(self):
        """Save the trace statistics to a summary file for fast reload next time.
        """
        self._summaryfile = Trace.get_summary_filename(self._tracefile)

        packed_metadata = self.metadata.pack_dataframe()

        packed_metadata[Trace._formatversionkey] = pandas.Series(
            data=Trace._formatversion, dtype=numpy.int32
        )

        with HDFStoreContext(self._summaryfile, mode="w") as hdfstore:
            hdfstore.put(Trace._metadatakey, packed_metadata, format="t")
            hdfstore.put(Trace._statisticskey, self.statistics, format="t")

    @staticmethod
    def _parse_summary_file(filename):
        """Load from HDF5 for reading
        """
        try:
            with HDFStoreContext(filename, mode="r") as hdfstore:
                try:
                    file_metadata = hdfstore[Trace._metadatakey]
                    format_version = file_metadata[Trace._formatversionkey][0]
                except KeyError:
                    raise WrongLoaderError(
                        '"{}" is not a PyPOP summary file'.format(filename)
                    )

                if format_version > Trace._formatversion:
                    warn(
                        "Trace summary was written with a newer PyPOP version. The "
                        "format is intended to be backward compatible but you may wish "
                        "to upgrade your installed PyPOP version to support all "
                        "features."
                    )

                tracemetadata = TraceMetadata.unpack_dataframe(file_metadata)
                statistics = hdfstore[Trace._statisticskey]
        except:
            raise WrongLoaderError('"{}" is not a PyPOP summary file'.format(filename))

        return (tracemetadata, statistics)

    @staticmethod
    def _pack_trace_metadata(unpacked):
        raise NotImplementedError

    @staticmethod
    def _unpack_trace_metadata(packed):
        raise NotImplementedError

    @classmethod
    def register_loader(cls):
        if issubclass(cls, Trace) and cls is not Trace:
            Trace._loader_registry[cls.__name__] = cls
            return
        raise TypeError("Can only register subclasses of Trace (passed {})".format(cls))

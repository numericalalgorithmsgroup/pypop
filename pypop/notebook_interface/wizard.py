#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from ipywidgets import Tab, Layout, VBox, Text

import pypop.metrics.metricset as pm
from ..traceset import TraceSet
from .fileselector import FileSelector
from .plotting import MetricTable, ScalingPlot
from .bokeh_widget import BokehWidgetWrapper
from .reporting import ReportGenerator
from .tqdm_widget import tqdm_notebook_noauto

from pypop.utils.exceptions import ExtraePRVNoOnOffEventsError

class TypeAsserter:
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


class SimpleAsserter(TypeAsserter):
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
        self._exepceted = assert_class.__name__


    def check_type(self, testobj):
        if isinstance(testobj, self._assert_cls):
            return True


class ListofStrAsserter(TypeAsserter):
    """Assert that an object is a list of strings

    Provides a callable object which asserts a provided object is a list of strings.
    """
    _expected = "list of str"

    def check_type(self, testobj):
        if isinstance(testobj, list) and all(isinstance(x, str) for x in testobj):
            return True


class AnalysisState:
    """Object containing analysis settings

    This is an internal class and not intended for direct use.
    """

    _required_params = {
        "trace_files": ListofStrAsserter(),
        "metrics_object": SimpleAsserter(pm.MetricSet),
    }

    _optional_params = {}

    _all_params = {**_required_params, **_optional_params}

    def __init__(self, **kwargs):

        self._params = {}
        for key, value in kwargs.items():
            if key not in AnalysisState._all_params:
                continue

            asserter = AnalysisState._all_params[key]
            try:
                asserter(value)
            except TypeError as err:
                raise TypeError("{}: {}".format(key, err))

            self._params[key] = value

    def validate(self, error=True):

        for key in AnalysisState._required_params:
            if key not in self._params:
                if error:
                    raise ValueError("Missing required parameter {}".format(key))
                return False

        return True

    def __getitem__(self, key):

        if key in self._all_params:
            try:
                return self._params[key]
            except KeyError:
                return None

    def __setitem__(self, key, value):

        if key not in AnalysisState._all_params:
            raise KeyError('Invalid parameter key: "{}"'.format(key))

        asserter = AnalysisState._all_params[key]
        try:
            asserter(value)
        except TypeError as err:
            raise TypeError("{}: {}".format(key, err))

        self._params[key] = value


class MetricsWizard(Tab):
    """IPyWidgets GUI Element providing an analysis Wizard.
    """
    def __init__(self, metric_calc="auto", base_dir=".", starting_files=None, **kwargs):
        """

        Parameters
        ----------

        metric_calc: class MetricSet or 'auto'
            Provide a specific metric calculator class. If 'auto' is passed then try to
            choose a suitable default.

        base_dir: str
            Starting directory for the filechooser. Defaults to result of `os.getcwd()`.

        starting_files: list of str
            List of files that should be preselected in the file chooser.
        """

        self._metrics_display = None
        self._scaling_plot = None
        self._report_generator = None
        self._metric_calculator = metric_calc
        self._base_dir = base_dir

        self._analysis_state = AnalysisState()

        self._fileselector = FileSelector(
            base_dir=self._base_dir,
            starting_files=starting_files,
            calculation_callback=self._calculate_callback_hook,
            analysis_state=self._analysis_state,
        )

        self._status_box = VBox()

        super().__init__(
            children=[VBox([self._fileselector, self._status_box])],
            layout=Layout(width="auto", max_width="1280px"),
            **kwargs
        )

        self.set_title(0, "Trace Files")

    def _calculate_callback_hook(self, callback_reference=None):

        advanced_config = self._fileselector._advanced_config_controls

        fileprogress = tqdm_notebook_noauto(self._fileselector.filenames, leave=False)
        self._status_box.children = [fileprogress.container]

        try:
            statistics = TraceSet(
                fileprogress,
                force_recalculation=advanced_config["Delete Cache"].value,
                chop_to_roi=advanced_config["Chop to ROI"].value,
                no_progress=True,
                chop_fail_is_error=True
            )
        except ExtraePRVNoOnOffEventsError as err:
            fileprogress = tqdm_notebook_noauto(self._fileselector.filenames, leave=False)
            warnstr = "Warning: Disabling Chopping to ROI ({})".format(err)
            self._status_box.children = [fileprogress.container,
                                         Text(warnstr, layout=Layout(width='auto'))]

            statistics = TraceSet(
                fileprogress,
                force_recalculation=advanced_config["Delete Cache"].value,
                chop_to_roi=False,
                no_progress=True,
            )

        if self._metric_calculator in ("auto", None):
            self._metric_calculate = statistics.suggested_metrics

        metrics = self._metric_calculator(statistics)
        self._analysis_state["metrics_object"] = metrics

        metrics_display = BokehWidgetWrapper(
            MetricTable(metrics, analysis_state=self._analysis_state)
        )

        if self._metrics_display is None:
            self.children = self.children + (metrics_display,)
            self.set_title(1, "Metrics Table")
        else:
            self._metrics_display.close()
            new_children = list(self.children)
            new_children[1] = metrics_display
            self.children = new_children

        self._metrics_display = metrics_display

        scaling_plot = BokehWidgetWrapper(
            ScalingPlot(metrics, analysis_state=self._analysis_state)
        )

        if self._scaling_plot is None:
            self.children = self.children + (scaling_plot,)
            self.set_title(2, "Scaling Plot")
        else:
            self._scaling_plot.close()
            new_children = list(self.children)
            new_children[2] = scaling_plot
            self.children = new_children

        self._scaling_plot = scaling_plot

        report_generator = ReportGenerator(analysis_state=self._analysis_state)

        if self._report_generator is None:
            self.children = self.children + (report_generator,)
            self.set_title(3, "Report Generation")
        else:
            self._report_generator.close()
            new_children = list(self.children)
            new_children[3] = report_generator
            self.children = new_children

        self._report_generator = report_generator

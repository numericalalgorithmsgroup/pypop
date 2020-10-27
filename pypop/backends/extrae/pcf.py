# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""\
PCF File Parser
---------------

Provides lightweight support for interrogating the contents of a PRV file, including all
event types and id->string lookups for value types.
"""

from collections import namedtuple

import pandas

Event = namedtuple("Event", ["ID", "Name", "State", "Values"])
Value = namedtuple("Value", ["ID", "Name"])
State = namedtuple("State", ["ID", "Name"])

class PCF:

    def __init__(self, pcf_path):

        self._pcf_path = pcf_path

        self._state_names = {}
        self._event_names = {}
        self._event_states = {}
        self._event_vals = {}

        self._states = None
        self._events = None
        self._parse_pcf()

    @property
    def filename(self):
        return self._pcf_path

    @property
    def event_names(self):
        return self._event_names

    @property
    def event_values(self):
        return self._event_vals

    @property
    def events(self):
        return self._events


    def _parse_pcf(self):

        states = []
        events = []
        values = []

        try:
            with open(self._pcf_path, "rt") as fh:
                block_mode = None
                for line in fh:
                    if not line.strip():
                        continue
                    if not line[0].isdigit():
                        block_mode = line.split()[0]
                        continue

                    if block_mode == "EVENT_TYPE":
                        elem = line.strip().split(maxsplit=2)
                        event_id = int(elem[1])
                        events.append(Event(event_id, elem[2], elem[0], {}))
                        values[event_id] = []
                        continue

                    if block_mode == "VALUES":
                        elem = line.strip().split(maxsplit=1)
                        value_id = int(elem[0])
                        values[event_id].append(Value(value_id, elem[1]))

                    if block_mode == "STATES":
                        elem = line.strip().split(maxsplit=1)
                        state_id = int(elem[0])
                        states.append(State(state_id, elem[1]))

        except FileNotFoundError:
            pass

        self._events = pandas.DataFrame(events)
        self._states = pandas.DataFrame(states)
        self._event_values = {k:pandas.DataFrame(v) for k,v in values.items}



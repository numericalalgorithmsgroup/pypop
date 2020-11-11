#!/usr/bin/env python3


class WrongLoaderError(RuntimeError):
    pass


class UnknownLoaderError(RuntimeError):
    pass


class ExtraePRVNoOnOffEventsError(RuntimeError):
    pass


class NoWebDriverError(RuntimeError):
    pass

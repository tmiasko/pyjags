# Copyright (C) 2016 Tomasz Miasko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

__all__ = ['stepwise', 'progress_bar']

import contextlib
import sys
import time
from datetime import timedelta


default_timer = getattr(time, 'monotonic', time.time)


def stepwise(f, steps, period, timer=default_timer):
    """
    Execute function in a stepwise manner.

    Parameters
    ----------
    f : callable
        A function taking single positive int as an argument.
    steps : int
        A positive integer specifying total number of steps to execute.
    period : float
        A positive float number describing desired period between calls to `f`.
    timer : callable, optional
        Monotonic clock, i.e., function returning number of elapsed seconds
        since some arbitrary point in time. Uses ``time.monotonic`` by default,
        if not available falls back to ``time.time``.
    """
    start = timer()
    left = steps
    next = 1

    while left > 0:
        f(next)
        elapsed = timer() - start
        left -= next
        done = steps - left
        next = int(period * done / elapsed) if elapsed > 0 else 2 * next
        if next < 1:
            next = 1
        if next > left:
            next = left


class ProgressBar:

    FORMAT = 'iterations {self.iterations_done} ' \
             'of {self.iterations_total}, ' \
             'elapsed {self.elapsed}, ' \
             'remaining {self.remaining}'

    def __init__(self, f, steps, header='', file=sys.stdout, timer=default_timer):
        self.function = f
        self.format = header + self.FORMAT
        self.file = file
        self.isatty = file.isatty()
        self.timer = timer
        self.start_seconds = self.timer()
        self.last_seconds = self.start_seconds
        self.iterations_done = 0
        self.iterations_total = steps
        self.previous_length = 0

    def __enter__(self):
        def update(steps):
            self.function(steps)
            self.iterations_done += steps
            self.last_seconds = self.timer()
            self.write(self.render())
        return update

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.isatty:
            self.file.write('\n')
            self.file.flush()

    def render(self):
        return self.format.format(self=self)

    def write(self, line):
        if self.isatty:
            # 1. Move to the beginning of the line
            # 2. Overwrite previous content (necessary when new line is shorter)
            # 3. Move to the beginning again.
            n = self.previous_length
            self.file.write('\b' * n + ' ' * n + '\b' * n)
            self.file.write(line)
            self.previous_length = len(line)
        else:
            self.file.write(line)
            self.file.write('\n')
        self.file.flush()

    @property
    def iterations_remaining(self):
        return self.iterations_total - self.iterations_done

    @property
    def percentage(self):
        if self.iterations_total:
            return 100 * self.iterations_done / self.iterations_total
        else:
            return 100

    @property
    def elapsed(self):
        elapsed_seconds = self.last_seconds - self.start_seconds
        return timedelta(seconds=round(elapsed_seconds, 0))

    @property
    def time_per_iteration(self):
        elapsed_seconds = self.last_seconds - self.start_seconds
        return elapsed_seconds / self.iterations_done

    @property
    def remaining(self):
        remaining_seconds = self.iterations_remaining * self.time_per_iteration
        return timedelta(seconds=round(remaining_seconds, 0))


class EmptyProgressBar:

    def __init__(self, func, *args, **kwargs):
        self.func = func

    def __enter__(self):
        return self.func

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def progress_bar(enable):
    return ProgressBar if enable else EmptyProgressBar

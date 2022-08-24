import os
import sys


class HideAllOutput():
    """
    Block all output from a block of code

    Notes
    -----
    HDBSCAN.fit() prints output that comes from its Cython libraries, which means it can't
    be suppressed by normal methods. It's something which annoyed me for MONTHS and now this
    will finally fix it
    """
    def __init__(self):
        sys.stdout.flush()

        # Save original stdout
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())

        # We want to send output to devnull
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        """
        Send output to the void, even if it comes from Cython :))
        """
        self._newstdout = os.dup(1)

        os.dup2(self._devnull, 1)
        os.close(self._devnull)

        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, type, value, tb):
        """
        Restore previous stdout when finished

        Attributes are boilerplate
        """
        sys.stdout = self._origstdout
        sys.stdout.flush()

        os.dup2(self._oldstdout_fno, 1)

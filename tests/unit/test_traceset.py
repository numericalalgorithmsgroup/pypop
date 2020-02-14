#!/usr/bin/env python3

import os
import pypop.traceset


class TestPreReqs:
    """Test config files all present/path functionality works, etc.
    """

    def test_configs_present(self):
        """Check all config paths resolve to a file
        """
        missing_files = []

        for config_set in (
            pypop.traceset.base_configs,
            pypop.traceset.omp_configs,
            pypop.traceset.ideal_configs,
        ):
            for configs in config_set.values():
                missing_files.extend([f for f in configs if not os.path.exists(f)])

        assert len(missing_files) == 0

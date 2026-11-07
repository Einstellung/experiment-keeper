import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"HOME": home}):
    spec = importlib.util.spec_from_file_location(
        "gpu_idle_monitor", Path(__file__).resolve().parents[1] / "gpu-idle-monitor.py"
    )
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)


class MonitorTest(unittest.TestCase):
    def test_allocation_respects_target_chunk_and_free_margin(self):
        self.assertEqual(monitor.allocation_mib(10000, 9700, 4096), 256)
        self.assertEqual(monitor.allocation_mib(10000, 9950, 4096), 50)
        self.assertEqual(monitor.allocation_mib(10000, 10000, 4096), 0)
        self.assertEqual(monitor.allocation_mib(10000, 100, 1050), 26)
        self.assertEqual(monitor.allocation_mib(10000, 100, 1024), 0)

    def test_idle_requires_no_process_zero_utilization_and_bounded_memory(self):
        self.assertTrue(monitor.idle({"pids": [], "util": 0, "mem": 256}))
        self.assertFalse(monitor.idle({"pids": [123], "util": 0, "mem": 20}))
        self.assertFalse(monitor.idle({"pids": [], "util": 1, "mem": 20}))
        self.assertFalse(monitor.idle({"pids": [], "util": 0, "mem": 257}))

    def test_snapshot_maps_compute_processes_to_the_correct_device(self):
        def query(fields, kind):
            if kind == 'gpu':
                return [['0', 'GPU-first', '0', '15'], ['1', 'GPU-second', '80', '500']]
            return [['GPU-second', '42']]
        with mock.patch.object(monitor, 'query', side_effect=query):
            devices = monitor.snapshot()
        self.assertEqual(devices['GPU-first']['pids'], [])
        self.assertEqual(devices['GPU-second']['pids'], [42])
        self.assertTrue(monitor.idle(devices['GPU-first']))
        self.assertFalse(monitor.idle(devices['GPU-second']))


if __name__ == '__main__':
    unittest.main()

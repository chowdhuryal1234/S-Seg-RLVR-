import unittest
from collections import Counter

from nucleus_rl.sft import exposure_for_index


class D4ScheduleTest(unittest.TestCase):
    def test_epoch_schedule_unchanged(self):
        got = [exposure_for_index(i, 5, 42, "d4") for i in range(20)]
        self.assertEqual(got, [exposure_for_index(i, 5, 42, "d4", "epoch") for i in range(20)])
        self.assertEqual({t for _, t in got[:5]}, {0})

    def test_rotating_balances_views_in_short_schedule(self):
        rows, steps = 1146, 2048
        views = Counter(exposure_for_index(i, rows, 42, "d4", "rotating")[1] for i in range(steps))
        self.assertEqual(set(views), set(range(8)))
        self.assertLess(max(views.values()) - min(views.values()), 40)
        epoch_views = Counter(exposure_for_index(i, rows, 42, "d4", "epoch")[1] for i in range(steps))
        self.assertEqual(set(epoch_views), {0, 1})

    def test_rotating_cycles_all_views_per_row_and_keeps_order(self):
        n = 6
        per_row = {}
        for i in range(8 * n):
            row, view = exposure_for_index(i, n, 7, "d4", "rotating")
            self.assertEqual(row, exposure_for_index(i, n, 7, "d4", "epoch")[0])
            per_row.setdefault(row, []).append(view)
        for views in per_row.values():
            self.assertEqual(sorted(views), list(range(8)))

    def test_no_augmentation_ignores_schedule(self):
        self.assertTrue(all(exposure_for_index(i, 4, 1, "none", "rotating")[1] == 0 for i in range(16)))

    def test_bad_schedule_rejected(self):
        with self.assertRaises(ValueError):
            exposure_for_index(0, 4, 1, "d4", "random")


if __name__ == "__main__":
    unittest.main()

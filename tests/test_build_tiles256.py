"""CPU tests for the 256 px all-nuclei tile builder (scripts/build_tiles256.py)."""
import random
import unittest

import numpy as np

from scripts.build_tiles256 import gt_summary, histogram, interleave, sliver_ids, stratified, tile_stats


class TileStatsTests(unittest.TestCase):
    def setUp(self):
        # 8x8 image, 4x4 tiles. Nucleus 1 crosses the vertical tile border (3 px left, 1 px right), nucleus 2 sits
        # inside the top-right tile, nucleus 3 inside the bottom-right tile.
        self.labels = np.zeros((8, 8), dtype=np.int32)
        self.labels[1, 1:5] = 1
        self.labels[2, 6] = 2
        self.labels[6, 6] = 3

    def test_grid_cells_keep_all_visible_ids_and_mark_truncation(self):
        cells = {(x, y): rest for x, y, *rest in tile_stats(self.labels, 4)}
        self.assertEqual(set(cells), {(0, 0), (4, 0), (4, 4)})  # the empty bottom-left cell is skipped
        ids, areas, truncated, border = cells[(0, 0)]
        self.assertEqual((ids, areas, truncated), ([1], [3], [1]))
        ids, areas, truncated, border = cells[(4, 0)]
        self.assertEqual((ids, areas, truncated), ([1, 2], [1, 1], [1]))
        self.assertEqual(border, [1])

    def test_tail_that_does_not_fill_a_tile_is_dropped(self):
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[5, 5] = 1
        self.assertEqual(tile_stats(labels, 4), [])

    def test_sliver_needs_truncation_and_small_area(self):
        self.assertEqual(sliver_ids([1, 2, 3], [50, 50, 150], [1, 3]), [1])


class SelectionTests(unittest.TestCase):
    def test_stratified_takes_one_per_count_bin(self):
        rows = [{'id': str(i), 'instance_count': i} for i in range(100)]
        chosen = stratified(rows, 10, random.Random(0))
        self.assertEqual(len(chosen), 10)
        self.assertEqual(sorted(r['instance_count'] // 10 for r in chosen), list(range(10)))
        self.assertEqual(stratified(rows[:5], 10, random.Random(0)), rows[:5])

    def test_interleave_alternates_datasets(self):
        out = interleave({'a': [{'id': 'a1'}, {'id': 'a2'}], 'b': [{'id': 'b1'}]}, random.Random(0))
        self.assertEqual([r['id'][0] for r in out], ['a', 'b', 'a'])

    def test_summary_fractions(self):
        summary = gt_summary([(10, 4, 1), (30, 6, 3)])
        self.assertEqual((summary['nuclei'], summary['count_max']), (40, 30))
        self.assertAlmostEqual(summary['edge_sliver_fraction'], 0.1)
        self.assertEqual(histogram([1, 8, 9, 200])['1-8'], 2)


if __name__ == '__main__':
    unittest.main()

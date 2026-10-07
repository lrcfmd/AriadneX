"""Regression checks for roundoff ties and stable-interval selection."""
import unittest
from pathlib import Path

import numpy as np

from ariadnex.library.artifact import CifMetadata
from ariadnex.library.filtering import deduplicate_records


def record(accession, volume):
    return CifMetadata(
        source_id=accession, source_path=Path(accession + '.cif'),
        formula='SiO2', space_group='P1', cell_volume=volume,
        metadata={'source_accession': accession},
    )


class SelectionRegressionTests(unittest.TestCase):
    def test_pair_roundoff_uses_stable_source_order(self):
        # The mean rounds toward the second member; lexical accession order
        # still picks 100, including after input order is reversed.
        first = np.nextafter(100.0, np.inf)
        pair = [record('100', first), record('20', np.nextafter(first, np.inf))]
        for records in (pair, pair[::-1]):
            kept, status = deduplicate_records(records)
            self.assertEqual([r.source_id for r in kept], ['100'])
            self.assertEqual(status['20']['dedup_canonical_source_id'], '100')

    def test_larger_group_keeps_closest_mean(self):
        kept, _ = deduplicate_records([
            record('100', 100.0), record('20', 101.0), record('30', 103.0),
        ])
        self.assertEqual([r.source_id for r in kept], ['20'])

    def test_pair_outside_absolute_tolerance_keeps_closest_mean(self):
        kept, _ = deduplicate_records([
            record('100', np.nextafter(1e6, np.inf)),
            record('20', np.nextafter(np.nextafter(1e6, np.inf), np.inf)),
        ])
        self.assertEqual([r.source_id for r in kept], ['20'])

    def test_penultimate_of_rightmost_interval(self):
        import pandas as pd
        from ariadnex.library.selection import StableRegimeConfig, identify_stable_regime
        thresholds = [.001, .002, .003, .004, .021, .0249, .0257]
        scan = pd.DataFrame({
            'threshold': thresholds, 'n_clusters': [20]*7,
            'compression_ratio': [.5]*7, 'singleton_ratio': [.5]*7,
            'largest_cluster_ratio': [.1]*7, 'coverage_90': [.01]*7,
            'compactness_separation_ratio': [.01]*7,
        })
        stability = pd.DataFrame({
            'threshold': thresholds, 'ari_mean': [.99,.99,.99,.5,.99,.99,.99],
            'nmi_mean': [1.]*7, 'medoid_exact_jaccard_mean': [1.]*7,
            'medoid_matched_distance_mean': [.01]*7, 'repeat_n_clusters_cv': [.01]*7,
        })
        old = identify_stable_regime(scan, stability, StableRegimeConfig())
        new = identify_stable_regime(scan, stability,
                                    StableRegimeConfig(interval_selection='penultimate'))
        self.assertEqual(old.selected_threshold, .0257)
        self.assertEqual(new.selected_threshold, .0249)
        self.assertEqual(new.stable_intervals, old.stable_intervals)
        self.assertEqual(new.selected_interval_index, 1)
        self.assertEqual(int(new.summary.selected_threshold_flag.sum()), 1)
        with self.assertRaises(ValueError):
            identify_stable_regime(scan, stability,
                                  StableRegimeConfig(interval_selection='penultimate',
                                                     min_consecutive_thresholds=1))


if __name__ == '__main__':
    unittest.main()

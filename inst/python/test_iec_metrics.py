"""
Unit Tests for Inter-Event Concordance (IEC) Metrics

Tests pure calculation logic with synthetic risk scores, verifying:
  1. Single-event concordance calculation
  2. Batch processing
  3. Stratification by event type
  4. Error handling for invalid inputs
  5. Ranking direction (higher score → better prediction)
"""

import numpy as np
import pytest
from iec_metrics import (
    compute_iec_single,
    compute_iec_batch,
    compute_iec_stratified
)


class TestComputeIecSingle:
    """Test compute_iec_single() function"""
    
    def test_highest_risk_event(self):
        """Observed event has highest risk → IEC = 1.0"""
        risk_scores = [10, 5, 20, 8]
        result = compute_iec_single(risk_scores, observed_event_idx=3)
        
        assert result['iec'] == 1.0, "Highest-risk event should have IEC=1.0"
        assert result['observed_rank'] == 4, "Highest-risk should be rank 4/4"
        assert result['observed_rank_from_top'] == 1, "Rank from top should be 1"
        assert result['vocab_size'] == 4
    
    def test_lowest_risk_event(self):
        """Observed event has lowest risk → IEC = 0.0"""
        risk_scores = [10, 5, 20, 8]
        result = compute_iec_single(risk_scores, observed_event_idx=2)
        
        assert result['iec'] == 0.0, "Lowest-risk event should have IEC=0.0"
        assert result['observed_rank'] == 1, "Lowest-risk should be rank 1/4"
        assert result['observed_rank_from_top'] == 4, "Rank from top should be 4"
    
    def test_middle_rank_event(self):
        """Observed event ranks in middle → IEC between 0 and 1"""
        # risk_scores: [10, 5, 20, 8]
        # Ascending order by score: event 2 (5), event 4 (8), event 1 (10), event 3 (20)
        # For observed_event_idx=4 (score=8): position=1 (0-indexed), IEC=1/3≈0.333
        risk_scores = [10, 5, 20, 8]
        result = compute_iec_single(risk_scores, observed_event_idx=4)
        
        assert np.isclose(result['iec'], 1/3), f"Expected IEC≈0.333, got {result['iec']}"
        assert result['observed_rank'] == 2  # 1-indexed position
        assert result['observed_rank_from_top'] == 3  # 4 - 1 = 3
    
    def test_identical_scores(self):
        """All events have same risk → deterministic but arbitrary ranking"""
        risk_scores = [5, 5, 5, 5]
        result = compute_iec_single(risk_scores, observed_event_idx=1)
        
        # argsort on identical values gives arbitrary (but deterministic) order
        # The event should get some rank between 1 and 4
        assert 0.0 <= result['iec'] <= 1.0
        assert 1 <= result['observed_rank'] <= 4
    
    def test_single_event(self):
        """Only one event in vocabulary → IEC should be 0 (division by 0-1)"""
        risk_scores = [42.0]
        result = compute_iec_single(risk_scores, observed_event_idx=1)
        
        assert result['iec'] == 0.0, "Single event should have IEC=0"
        assert result['observed_rank'] == 1
    
    def test_numpy_array_input(self):
        """Test with numpy array instead of list"""
        risk_scores = np.array([10, 5, 20, 8])
        result = compute_iec_single(risk_scores, observed_event_idx=3)
        
        assert result['iec'] == 1.0
    
    def test_invalid_event_index_too_high(self):
        """Event index beyond vocab_size raises IndexError"""
        risk_scores = [10, 5, 20, 8]
        with pytest.raises(IndexError):
            compute_iec_single(risk_scores, observed_event_idx=5)
    
    def test_invalid_event_index_zero(self):
        """Event index must be 1-indexed, not 0"""
        risk_scores = [10, 5, 20, 8]
        with pytest.raises(ValueError, match="1-indexed"):
            compute_iec_single(risk_scores, observed_event_idx=0)
    
    def test_invalid_event_index_negative(self):
        """Negative event index raises ValueError"""
        risk_scores = [10, 5, 20, 8]
        with pytest.raises(ValueError, match="1-indexed"):
            compute_iec_single(risk_scores, observed_event_idx=-1)
    
    def test_empty_risk_scores(self):
        """Empty risk_scores array raises ValueError"""
        with pytest.raises(ValueError, match="empty"):
            compute_iec_single([], observed_event_idx=1)
    
    def test_return_keys(self):
        """Verify all expected keys in return dict"""
        result = compute_iec_single([10, 5, 20, 8], observed_event_idx=2)
        
        expected_keys = {'iec', 'observed_rank', 'observed_rank_from_top', 'vocab_size'}
        assert set(result.keys()) == expected_keys


class TestComputeIecBatch:
    """Test compute_iec_batch() function"""
    
    def test_basic_batch(self):
        """Process multiple transitions"""
        risk_matrices = [
            [10, 5, 20, 8],
            [15, 3, 12, 9],
            [7, 18, 6, 25]
        ]
        observed = [3, 2, 4]  # Expected indices (1-indexed)
        
        result = compute_iec_batch(risk_matrices, observed)
        
        assert result['n_total'] == 3
        assert result['n_valid'] == 3
        assert len(result['iec_values']) == 3
        assert all(0.0 <= iec <= 1.0 for iec in result['iec_values'])
    
    def test_batch_with_errors(self):
        """Batch with some invalid indices → errors captured"""
        risk_matrices = [
            [10, 5, 20, 8],
            [15, 3, 12, 9],  # Will error: index 5 > vocab_size 4
            [7, 18, 6, 25]
        ]
        observed = [3, 5, 4]
        
        result = compute_iec_batch(risk_matrices, observed, suppress_errors=True)
        
        assert result['n_total'] == 3
        assert result['n_valid'] == 2
        assert len(result['errors']) == 1
        assert len(result['error_indices']) == 1
        assert result['error_indices'][0] == 1
    
    def test_batch_error_no_suppress(self):
        """With suppress_errors=False, first error is raised"""
        risk_matrices = [[10, 5, 20, 8], [15, 3, 12, 9]]
        observed = [3, 5]  # Index 5 invalid
        
        with pytest.raises(IndexError):
            compute_iec_batch(risk_matrices, observed, suppress_errors=False)
    
    def test_length_mismatch(self):
        """Mismatched lengths of risk_matrices and observed raises ValueError"""
        risk_matrices = [[10, 5, 20, 8]]
        observed = [3, 2]  # Length mismatch
        
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_iec_batch(risk_matrices, observed)
    
    def test_empty_batch(self):
        """Empty batch returns empty results"""
        result = compute_iec_batch([], [])
        
        assert result['n_total'] == 0
        assert result['n_valid'] == 0
        assert result['iec_values'] == []
        assert result['errors'] == []
    
    def test_return_keys(self):
        """Verify all expected keys in return dict"""
        result = compute_iec_batch([[10, 5, 20]], [3])
        
        expected_keys = {
            'iec_values', 'observed_ranks', 'observed_ranks_from_top',
            'n_valid', 'n_total', 'errors', 'error_indices'
        }
        assert set(result.keys()) == expected_keys
    
    def test_consistency_with_single(self):
        """Batch results should match individual single() calls"""
        risk_matrices = [[10, 5, 20, 8], [15, 3, 12, 9]]
        observed = [3, 2]
        
        batch_result = compute_iec_batch(risk_matrices, observed)
        single_results = [
            compute_iec_single(risk_matrices[i], observed[i])
            for i in range(len(observed))
        ]
        
        for i, single_res in enumerate(single_results):
            assert batch_result['iec_values'][i] == single_res['iec']
            assert batch_result['observed_ranks'][i] == single_res['observed_rank']


class TestComputeIecStratified:
    """Test compute_iec_stratified() function"""
    
    def test_basic_stratification(self):
        """Group IEC by event type"""
        risk_matrices = [
            [10, 5, 20],
            [15, 3, 12],
            [7, 18, 6],
            [20, 5, 10]
        ]
        observed = [3, 2, 1, 3]
        vocab = ['HTN', 'DM', 'MI']
        
        result = compute_iec_stratified(risk_matrices, observed, event_vocabulary=vocab)
        
        assert result['n_total'] == 4
        assert result['n_valid'] == 4
        assert result['n_events'] == 3
        assert set(result['by_event'].keys()) == {'HTN', 'DM', 'MI'}
        
        # Event 3 (MI) appears twice
        assert result['by_event']['MI']['n_obs'] == 2
        # Event 1 (HTN) appears once
        assert result['by_event']['HTN']['n_obs'] == 1
    
    def test_stratification_without_vocab(self):
        """Stratification keyed by event index if no vocabulary provided"""
        risk_matrices = [[10, 5, 20], [15, 3, 12]]
        observed = [3, 2]
        
        result = compute_iec_stratified(risk_matrices, observed, event_vocabulary=None)
        
        assert set(result['by_event'].keys()) == {3, 2}
    
    def test_stratified_mean_iec(self):
        """Mean IEC per event is correctly computed"""
        risk_matrices = [
            [10, 5, 20],  # Event 3: rank=3, IEC=1.0
            [10, 5, 20],  # Event 3: rank=3, IEC=1.0
            [20, 10, 5]   # Event 2: rank=2, IEC=0.5
        ]
        observed = [3, 3, 2]
        
        result = compute_iec_stratified(risk_matrices, observed, vocab_size=3)
        
        # Event 3 should have mean IEC = 1.0
        assert result['by_event'][3]['mean_iec'] == 1.0
        # Event 2 should have mean IEC = 0.5
        assert result['by_event'][2]['mean_iec'] == 0.5
    
    def test_global_mean_iec(self):
        """Global mean IEC matches batch mean"""
        risk_matrices = [
            [10, 5, 20, 8],
            [15, 3, 12, 9],
            [7, 18, 6, 25]
        ]
        observed = [3, 2, 4]
        
        batch_result = compute_iec_batch(risk_matrices, observed)
        stratified_result = compute_iec_stratified(risk_matrices, observed)
        
        assert np.isclose(
            stratified_result['mean_iec'],
            np.mean(batch_result['iec_values'])
        )
    
    def test_stratified_with_errors(self):
        """Stratified computation with some errors"""
        risk_matrices = [
            [10, 5, 20],
            [15, 3, 12],  # Error: index 5 > vocab_size 3
            [7, 18, 6]
        ]
        observed = [3, 5, 1]
        
        result = compute_iec_stratified(
            risk_matrices, observed, vocab_size=3, suppress_errors=True
        )
        
        assert result['n_total'] == 3
        assert result['n_valid'] == 2  # One error skipped
    
    def test_return_keys(self):
        """Verify all expected keys in return dict"""
        result = compute_iec_stratified([[10, 5, 20]], [3])
        
        expected_keys = {
            'mean_iec', 'by_event', 'n_valid', 'n_total', 'n_events', 'event_indices'
        }
        assert set(result.keys()) == expected_keys
    
    def test_event_by_keys(self):
        """Verify keys in by_event sub-dicts"""
        result = compute_iec_stratified([[10, 5, 20]], [3], vocab_size=3)
        
        event_key = list(result['by_event'].keys())[0]
        event_stats = result['by_event'][event_key]
        
        expected_event_keys = {'iec_values', 'mean_iec', 'n_obs', 'n_valid'}
        assert set(event_stats.keys()) == expected_event_keys


class TestIntegration:
    """Integration tests across functions"""
    
    def test_realistic_workflow(self):
        """Simulate realistic evaluation: batch → stratify → interpret"""
        # Simulate 100 patient transitions, 10 possible events
        np.random.seed(42)
        n_transitions = 100
        n_events = 10
        
        # Random risk scores
        risk_matrices = [
            np.random.exponential(scale=2.0, size=n_events)
            for _ in range(n_transitions)
        ]
        
        # Random observed events (1-indexed)
        observed = np.random.randint(1, n_events + 1, size=n_transitions).tolist()
        
        # Compute stratified IEC
        result = compute_iec_stratified(risk_matrices, observed, vocab_size=n_events)
        
        # Verify plausible results
        assert 0.0 <= result['mean_iec'] <= 1.0
        assert result['n_valid'] == n_transitions
        assert len(result['by_event']) <= n_events
    
    def test_perfect_ranking(self):
        """Perfect model: always ranks true event as highest"""
        # Create risk scores where true event always has max score
        risk_matrices = []
        observed = []
        
        for true_event_idx in range(1, 5):  # 4 events
            # Create risk scores with true event having highest score
            risk_scores = [i for i in range(4)]
            risk_scores[true_event_idx - 1] = 100  # Set true event to max
            risk_matrices.append(risk_scores)
            observed.append(true_event_idx)
        
        result = compute_iec_batch(risk_matrices, observed)
        
        # All should have IEC = 1.0 (perfect ranking)
        assert all(iec == 1.0 for iec in result['iec_values'])
        assert result['n_valid'] == 4
    
    def test_worst_ranking(self):
        """Worst model: always ranks true event as lowest"""
        risk_matrices = []
        observed = []
        
        for true_event_idx in range(1, 5):  # 4 events
            # Create risk scores with true event having lowest score
            risk_scores = [100, 100, 100, 100]
            risk_scores[true_event_idx - 1] = 0  # Set true event to min
            risk_matrices.append(risk_scores)
            observed.append(true_event_idx)
        
        result = compute_iec_batch(risk_matrices, observed)
        
        # All should have IEC = 0.0 (worst ranking)
        assert all(iec == 0.0 for iec in result['iec_values'])
        assert result['n_valid'] == 4


if __name__ == '__main__':
    # Run tests with: pytest test_iec_metrics.py -v
    pytest.main([__file__, '-v'])

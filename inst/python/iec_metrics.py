"""
Inter-Event Concordance (IEC) Metrics Module

This module implements the original SurvivEHR event-concordance calculation for ranking-based
evaluation of competing risks predictions. IEC measures how well the model ranks competing next 
events at each transition in patient history.

Core Logic:
    - Rank all possible next events by their risk scores (ascending: low to high risk)
    - Find the zero-indexed position of the observed (true) next event in that ranking
    - Normalize: IEC = position_in_ranking / (vocab_size - 1)
    
Interpretation:
    - IEC close to 1.0: observed event ranked among highest-risk (good prediction)
    - IEC close to 0.0: observed event ranked among lowest-risk (poor prediction)
    - Paper interpretation: mean rank from top ≈ n_events - mean(position) 
      (i.e., on average, true next event was among top X predicted events)

References:
    Original implementation: SurvivEHR/src/models/survival/custom_callbacks/causal_eval.py
    Method: PerformanceMetrics._compute_and_log_concordance() lines 45-68
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple


def compute_iec_single(
    risk_scores: Union[List[float], np.ndarray],
    observed_event_idx: int,
    vocab_size: Optional[int] = None
) -> Dict[str, Union[float, int]]:
    """
    Compute Inter-Event Concordance for a single transition.
    
    This implements the core SurvivEHR concordance calculation from causal_eval.py:
    rank all events by risk (ascending), find the position of the observed event, normalize.
    
    Args:
        risk_scores: Array of risk scores for each event type. Shape: (n_events,).
        observed_event_idx: Index (1-indexed) of the observed/true next event.
        vocab_size: Optional override of vocabulary size. If None, uses len(risk_scores).
        
    Returns:
        Dictionary with keys:
            - 'iec': float in [0, 1], IEC score = zero_indexed_position / (vocab_size - 1)
            - 'observed_rank': int in [1, vocab_size], rank in ascending-risk order (1-indexed)
            - 'observed_rank_from_top': int, complementary rank (n_events - zero_indexed_position)
            - 'vocab_size': int, total number of events
            
    Raises:
        IndexError: If observed_event_idx is not in the valid range or not found in risk_scores
        ValueError: If risk_scores is empty, observed_event_idx <= 0, or lengths don't match
        
    Example:
        >>> risk_scores = [10, 5, 20, 8]  # Event scores at indices 1,2,3,4 (1-indexed)
        >>> result = compute_iec_single(risk_scores, observed_event_idx=3)
        >>> # Event 3 has score 20 (highest risk) -> position 3 in ascending order [5,8,10,20]
        >>> print(result['iec'])  # 3 / 3 = 1.0
        1.0
        >>> print(result['observed_rank'])  # Position + 1 = 4
        4
        >>> print(result['observed_rank_from_top'])  # How many events rank higher? 0
        1
    """
    
    # Validate inputs
    if isinstance(risk_scores, list):
        risk_scores = np.array(risk_scores)
    
    if len(risk_scores) == 0:
        raise ValueError("risk_scores cannot be empty")
    
    if observed_event_idx <= 0:
        raise ValueError(f"observed_event_idx must be 1-indexed (>= 1), got {observed_event_idx}")
    
    if vocab_size is None:
        vocab_size = len(risk_scores)
    
    if len(risk_scores) != vocab_size:
        raise ValueError(
            f"risk_scores length ({len(risk_scores)}) does not match vocab_size ({vocab_size})"
        )
    
    if observed_event_idx > vocab_size:
        raise IndexError(
            f"observed_event_idx ({observed_event_idx}) exceeds vocab_size ({vocab_size})"
        )
    
    # Core ranking logic (from SurvivEHR causal_eval.py lines 45-68)
    # argsort returns indices ordered from lowest to highest score
    # +1 converts these indices to 1-indexed event IDs in ascending risk order
    ordered_events = np.argsort(risk_scores) + 1
    
    # Find WHERE the observed event appears in that ascending-risk ordering
    # This is the key: we search for the observed_event_idx within the ordered list
    matches = np.where(ordered_events == observed_event_idx)[0]
    
    if len(matches) == 0:
        raise IndexError(
            f"observed_event_idx {observed_event_idx} not found in ranked events. "
            f"Valid range: [1, {vocab_size}]"
        )
    
    # Zero-indexed position in the ascending-risk ordering
    position_zero_indexed = int(matches[0])
    
    # Normalize: IEC = zero_indexed_position / (vocab_size - 1)
    # Original SurvivEHR: event_concordance = np.where(...)[0][0] / (len(risk_scores) - 1)
    iec = position_zero_indexed / (vocab_size - 1) if vocab_size > 1 else 0.0
    
    # 1-indexed rank (for display)
    observed_rank = position_zero_indexed + 1
    
    # Rank from top interpretation (for paper: "true event among top X")
    observed_rank_from_top = vocab_size - position_zero_indexed
    
    return {
        'iec': float(iec),
        'observed_rank': int(observed_rank),
        'observed_rank_from_top': int(observed_rank_from_top),
        'vocab_size': int(vocab_size)
    }


def compute_iec_batch(
    risk_scores_list: List[Union[List[float], np.ndarray]],
    observed_events: List[int],
    vocab_size: Optional[int] = None,
    suppress_errors: bool = True
) -> Dict[str, Union[List, int, List[str]]]:
    """
    Compute Inter-Event Concordance for a batch of transitions.
    
    Applies compute_iec_single to each (risk_scores, observed_event) pair.
    
    Args:
        risk_scores_list: List of risk score arrays. Each shape: (n_events,).
        observed_events: List of observed event indices (1-indexed). Length must match risk_scores_list.
        vocab_size: Optional override of vocabulary size. If None, uses size of first risk_scores array.
        suppress_errors: If True, capture IndexErrors and return them in 'errors' list.
                        If False, raise on first error.
        
    Returns:
        Dictionary with keys:
            - 'iec_values': list of floats, IEC for each transition
            - 'observed_ranks': list of ints, rank positions
            - 'observed_ranks_from_top': list of ints, ranks from top
            - 'n_valid': int, count of valid calculations
            - 'n_total': int, total transitions processed
            - 'errors': list of strings, error messages for invalid transitions
            - 'error_indices': list of ints, indices where errors occurred
            
    Raises:
        ValueError: If observed_events length does not match risk_scores_list length
        IndexError: If suppress_errors=False and an error occurs
        
    Example:
        >>> risk_matrices = [[10, 5, 20, 8], [15, 3, 12, 9]]
        >>> observed = [3, 2]
        >>> result = compute_iec_batch(risk_matrices, observed)
        >>> print(result['iec_values'])
        [1.0, 0.67]  # Approximate values
    """
    
    if len(risk_scores_list) != len(observed_events):
        raise ValueError(
            f"Length mismatch: risk_scores_list ({len(risk_scores_list)}) "
            f"!= observed_events ({len(observed_events)})"
        )
    
    if len(risk_scores_list) == 0:
        return {
            'iec_values': [],
            'observed_ranks': [],
            'observed_ranks_from_top': [],
            'n_valid': 0,
            'n_total': 0,
            'errors': [],
            'error_indices': []
        }
    
    if vocab_size is None:
        # Infer from first risk_scores array
        vocab_size = len(risk_scores_list[0])
    
    iec_values = []
    observed_ranks = []
    observed_ranks_from_top = []
    errors = []
    error_indices = []
    
    for idx, (risk_scores, observed_event) in enumerate(zip(risk_scores_list, observed_events)):
        try:
            result = compute_iec_single(risk_scores, observed_event, vocab_size=vocab_size)
            iec_values.append(result['iec'])
            observed_ranks.append(result['observed_rank'])
            observed_ranks_from_top.append(result['observed_rank_from_top'])
        except (IndexError, ValueError) as e:
            if not suppress_errors:
                raise
            errors.append(str(e))
            error_indices.append(idx)
    
    return {
        'iec_values': iec_values,
        'observed_ranks': observed_ranks,
        'observed_ranks_from_top': observed_ranks_from_top,
        'n_valid': len(iec_values),
        'n_total': len(risk_scores_list),
        'errors': errors,
        'error_indices': error_indices
    }


def compute_iec_stratified(
    risk_scores_list: List[Union[List[float], np.ndarray]],
    observed_events: List[int],
    event_vocabulary: Optional[List[str]] = None,
    vocab_size: Optional[int] = None,
    suppress_errors: bool = True
) -> Dict[str, Union[float, int, Dict]]:
    """
    Compute Inter-Event Concordance with stratification by event type.
    
    Groups IEC values and statistics by the observed event type. Useful for identifying
    whether concordance varies significantly across different event types (e.g., high
    prevalence events may have different concordance than rare events).
    
    Args:
        risk_scores_list: List of risk score arrays. Each shape: (n_events,).
        observed_events: List of observed event indices (1-indexed).
        event_vocabulary: Optional list of event names/IDs. If provided, results are keyed by name.
                         If None, results are keyed by event index.
        vocab_size: Optional override of vocabulary size.
        suppress_errors: If True, capture errors; if False, raise on first error.
        
    Returns:
        Dictionary with keys:
            - 'mean_iec': float, global mean IEC across all valid transitions
            - 'by_event': dict mapping event (name or index) to statistics:
                - 'iec_values': list of IEC values for this event
                - 'mean_iec': float, mean IEC for this event
                - 'n_obs': int, count of observations for this event
                - 'n_valid': int, count of valid (non-error) calculations
            - 'n_valid': int, total valid transitions
            - 'n_total': int, total transitions
            - 'n_events': int, number of unique event types observed
            - 'event_indices': list of unique event indices observed
            
    Example:
        >>> risk_matrices = [[10, 5, 20], [15, 3, 12], [8, 25, 6]]
        >>> observed = [3, 2, 1]
        >>> vocab = ['HTN', 'DM', 'MI']
        >>> result = compute_iec_stratified(risk_matrices, observed, vocab)
        >>> print(result['by_event']['HTN']['mean_iec'])
        0.5
    """
    
    # Get batch IEC values first
    batch_result = compute_iec_batch(
        risk_scores_list, 
        observed_events, 
        vocab_size=vocab_size,
        suppress_errors=suppress_errors
    )
    
    # Extract only valid indices
    valid_indices = [
        i for i in range(batch_result['n_total']) 
        if i not in batch_result['error_indices']
    ]
    
    if len(valid_indices) == 0:
        return {
            'mean_iec': 0.0,
            'by_event': {},
            'n_valid': 0,
            'n_total': batch_result['n_total'],
            'n_events': 0,
            'event_indices': []
        }
    
    # Build stratified dictionary
    by_event = {}
    unique_events = set()
    
    for list_idx, valid_idx in enumerate(valid_indices):
        observed_event = observed_events[valid_idx]
        iec_value = batch_result['iec_values'][list_idx]
        unique_events.add(observed_event)
        
        # Use event name if vocabulary provided, else use index
        event_key = event_vocabulary[observed_event - 1] if event_vocabulary else observed_event
        
        if event_key not in by_event:
            by_event[event_key] = {
                'iec_values': [],
                'mean_iec': 0.0,
                'n_obs': 0,
                'n_valid': 0
            }
        
        by_event[event_key]['iec_values'].append(iec_value)
        by_event[event_key]['n_obs'] += 1
    
    # Compute means for each event
    for event_key in by_event:
        values = by_event[event_key]['iec_values']
        by_event[event_key]['mean_iec'] = np.mean(values) if values else 0.0
        by_event[event_key]['n_valid'] = len(values)
    
    # Global mean
    all_iec_values = batch_result['iec_values']
    mean_iec = np.mean(all_iec_values) if all_iec_values else 0.0
    
    return {
        'mean_iec': float(mean_iec),
        'iec_values': batch_result['iec_values'],
        'observed_ranks': batch_result['observed_ranks'],
        'observed_ranks_from_top': batch_result['observed_ranks_from_top'],
        'by_event': by_event,
        'n_valid': batch_result['n_valid'],
        'n_total': batch_result['n_total'],
        'errors': batch_result['errors'],
        'error_indices': batch_result['error_indices'],
        'n_events': len(unique_events),
        'event_indices': sorted(list(unique_events))
    }

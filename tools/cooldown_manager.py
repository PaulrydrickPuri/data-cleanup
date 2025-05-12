#!/usr/bin/env python3
"""
Simple cooldown timer for object interaction states to prevent alert fatigue.
"""

class CooldownManager:
    """Manages cooldown periods for interaction pairs to prevent alert spam."""
    
    def __init__(self, cooldown_time=1.0):
        """
        Initialize cooldown manager.
        
        Args:
            cooldown_time: Default cooldown time in seconds
        """
        self.cooldown_times = {}  # Pair ID -> timestamp
        self.default_cooldown = cooldown_time
    
    def in_cooldown(self, id1, id2, current_time, cooldown_time=None):
        """
        Check if a pair of objects is in cooldown period.
        
        Args:
            id1: First object ID
            id2: Second object ID
            current_time: Current timestamp (seconds)
            cooldown_time: Optional custom cooldown time
        
        Returns:
            True if the pair is still in cooldown, False if allowed
        """
        pair_key = self._make_key(id1, id2)
        cooldown_time = cooldown_time or self.default_cooldown
        
        # If no previous record, definitely not in cooldown
        if pair_key not in self.cooldown_times:
            return False
        
        # Check if enough time has passed
        last_time = self.cooldown_times[pair_key]
        elapsed = current_time - last_time
        
        return elapsed < cooldown_time
    
    def set(self, id1, id2, current_time):
        """
        Set cooldown timestamp for a pair.
        
        Args:
            id1: First object ID
            id2: Second object ID
            current_time: Current timestamp (seconds)
        """
        pair_key = self._make_key(id1, id2)
        self.cooldown_times[pair_key] = current_time
    
    def _make_key(self, id1, id2):
        """Create a consistent key for a pair of IDs regardless of order."""
        return f"{min(id1, id2)}_{max(id1, id2)}"

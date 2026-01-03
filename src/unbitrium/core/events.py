"""
Event System for decoupling simulation components.
"""

from enum import Enum
from typing import Any, Callable, Dict, List

class EventType(str, Enum):
    SIMULATION_START = "simulation_start"
    SIMULATION_END = "simulation_end"
    SIMULATION_ERROR = "simulation_error"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    CLIENT_SELECTED = "client_selected"
    CLIENT_TRAIN_START = "client_train_start"
    CLIENT_TRAIN_END = "client_train_end"
    AGGREGATION_START = "aggregation_start"
    AGGREGATION_END = "aggregation_end"

class EventSystem:
    """
    Simple synchronous publish-subscribe system.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable[[Dict[str, Any]], None]]] = {}

    def subscribe(self, event_type: EventType, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def emit(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Trigger all callbacks for an event."""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    # Prevent one listener from crashing the system
                    print(f"Error in event listener for {event_type}: {e}")

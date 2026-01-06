"""Event System for Federated Learning Simulation.

Provides a decoupled publish-subscribe event system for
simulation components to communicate state changes.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Enumeration of simulation event types.

    Attributes:
        SIMULATION_START: Emitted when simulation begins.
        SIMULATION_END: Emitted when simulation completes.
        SIMULATION_ERROR: Emitted on simulation error.
        ROUND_START: Emitted at the start of each round.
        ROUND_END: Emitted at the end of each round.
        CLIENT_SELECTED: Emitted when a client is selected.
        CLIENT_TRAIN_START: Emitted when client training begins.
        CLIENT_TRAIN_END: Emitted when client training ends.
        AGGREGATION_START: Emitted when aggregation begins.
        AGGREGATION_END: Emitted when aggregation completes.
    """

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
    """Synchronous publish-subscribe event system.

    Enables decoupled communication between simulation components
    through event-based messaging.

    Example:
        >>> events = EventSystem()
        >>> events.subscribe(EventType.ROUND_END, lambda d: print(d))
        >>> events.emit(EventType.ROUND_END, {"round": 5})
    """

    def __init__(self) -> None:
        """Initialize the event system with empty subscriber registry."""
        self._subscribers: Dict[EventType, List[Callable[[Dict[str, Any]], None]]] = {}

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a callback for an event type.

        Args:
            event_type: The event type to subscribe to.
            callback: Function to call when event is emitted.

        Example:
            >>> def on_round_end(data: dict) -> None:
            ...     print(f"Round {data['round']} finished")
            >>> events.subscribe(EventType.ROUND_END, on_round_end)
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[Dict[str, Any]], None],
    ) -> bool:
        """Remove a callback from an event type.

        Args:
            event_type: The event type to unsubscribe from.
            callback: The callback function to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                return True
            except ValueError:
                pass
        return False

    def emit(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Emit an event to all subscribers.

        Args:
            event_type: The type of event to emit.
            data: Event payload dictionary.

        Note:
            Exceptions in callbacks are caught and logged to prevent
            one listener from crashing the entire system.
        """
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in event listener for {event_type}: {e}")

    def clear(self, event_type: EventType | None = None) -> None:
        """Clear all subscribers for an event type or all events.

        Args:
            event_type: Specific event type to clear, or None for all.
        """
        if event_type is None:
            self._subscribers.clear()
        elif event_type in self._subscribers:
            self._subscribers[event_type].clear()

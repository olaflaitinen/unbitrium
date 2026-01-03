"""
System metrics (Latency, Energy).
"""

def estimate_latency(
    model_size_mb: float,
    bandwidth_mbps: float,
    rtt_ms: float
) -> float:
    """
    Estimates communication latency.
    """
    transfer_time = (model_size_mb * 8) / bandwidth_mbps
    return transfer_time + (rtt_ms / 1000.0)

def estimate_energy(flops: float, joules_per_flop: float = 1e-12) -> float:
    """
    Estimates energy consumption.
    """
    return flops * joules_per_flop

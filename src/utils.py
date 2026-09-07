import time

def calculate_fps(previous_time: float) -> tuple[float, float]:
    """Calculate FPS based on the previous frame timestamp."""
    current_time = time.monotonic()
    elapsed = max(current_time - previous_time, 1e-6)
    fps = 1 / elapsed

    return fps, current_time
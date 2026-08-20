"""Small lifecycle helpers that stay importable without processor dependencies."""

import threading


def schedule_retirement(delay_seconds, retire, timer_factory=threading.Timer):
    """Run ``retire`` after a final metrics-scrape grace period.

    The returned timer is daemonized so a failed Kubernetes self-delete cannot
    keep a local processor process alive. A zero delay preserves the immediate
    retirement behavior used by local tests and opt-outs.
    """
    delay_seconds = max(0.0, float(delay_seconds))
    if delay_seconds == 0:
        retire()
        return None

    timer = timer_factory(delay_seconds, retire)
    timer.daemon = True
    timer.start()
    return timer

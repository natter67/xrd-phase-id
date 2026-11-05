"""
Automated XRD phase identification for high-entropy alloys.

    from xrd_phase_id import analyze
    result = analyze(two_theta, counts)
    print(result.summary())

See README.md for the physics behind every step.
"""

from .pipeline import AnalysisResult, analyze  # noqa: F401

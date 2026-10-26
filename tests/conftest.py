import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    max_examples=int(os.environ.get("TOOLGRAPH_EXAMPLES", "150")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("ci", max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

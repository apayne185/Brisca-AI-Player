from hypothesis import settings

# Property tests play whole games per example, so wall-clock deadlines only
# measure machine load (e.g. a busy CI runner), not regressions.
settings.register_profile("default", deadline=None)
settings.load_profile("default")

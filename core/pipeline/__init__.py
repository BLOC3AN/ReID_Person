# Lazy import PersonReIDPipeline (uses ByteTrackWrapper → torch)
# from .pipeline import PersonReIDPipeline

from .registration import register_person_mot17, register_person_from_images

def __getattr__(name):
    if name == 'PersonReIDPipeline':
        from .pipeline import PersonReIDPipeline
        return PersonReIDPipeline
    raise AttributeError(f"module 'core.pipeline' has no attribute '{name}'")

__all__ = ['PersonReIDPipeline', 'register_person_mot17', 'register_person_from_images']

import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class PipelineConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "pipeline"

    def ready(self):
        try:
            from pipeline.core.ensemble import (
                _load_image_bundle,
                _load_sequence_bundle,
                _load_tabular_bundle,
            )
            _load_tabular_bundle()
            _load_image_bundle()
            _load_sequence_bundle()
        except Exception as exc:
            # Non-fatal: models will load lazily on first request instead.
            # Happens in dev/test environments where TF is not installed.
            logger.warning("Model pre-warm skipped: %s", exc)

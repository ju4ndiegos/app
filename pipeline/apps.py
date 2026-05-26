import logging
import sys

from django.apps import AppConfig

logger = logging.getLogger(__name__)

_SKIP_CMDS = {
    "migrate", "makemigrations", "collectstatic", "check",
    "shell", "dbshell", "test", "showmigrations", "sqlmigrate",
    "createsuperuser", "changepassword",
}


class PipelineConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "pipeline"

    def ready(self):
        # Don't load heavy ML models during management commands that don't
        # serve requests — migrate and collectstatic each add ~30s otherwise.
        if len(sys.argv) > 1 and sys.argv[1] in _SKIP_CMDS:
            return
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
            logger.warning("Model pre-warm skipped: %s", exc)

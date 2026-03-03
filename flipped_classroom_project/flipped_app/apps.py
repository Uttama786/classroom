from django.apps import AppConfig


class FlippedAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'flipped_app'

    def ready(self):
        """
        Called once Django is fully loaded.
        Register real-time dataset signals.
        RAG retriever loads lazily on first chat request.
        """

        # ── 1. Register signals ──────────────────────────────────
        try:
            import flipped_app.signals  # noqa: F401  (side-effect import)
            print("[FlipLearn] Real-time dataset signals registered ✓")
        except Exception as e:
            print(f"[FlipLearn] Signal registration skipped: {e}")

        # ── 2. Pre-warm RAG retriever ────────────────────────────
        # Disabled: model download (~90MB) at startup causes health check
        # timeouts on Railway. The retriever loads lazily on first chat request.
        pass

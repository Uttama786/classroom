from django.apps import AppConfig


class FlippedAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'flipped_app'

    def ready(self):
        """
        Called once Django is fully loaded.
        1. Register real-time dataset signals (StudentPerformance, QuizAttempt,
           AssignmentSubmission, VideoWatchHistory → auto-upsert dataset.csv
           and trigger background model retraining when enough new rows arrive).
        2. Pre-warm the RAG retriever in a background thread.
        """
        import threading

        # ── 1. Register signals ──────────────────────────────────
        try:
            import flipped_app.signals  # noqa: F401  (side-effect import)
            print("[FlipLearn] Real-time dataset signals registered ✓")
        except Exception as e:
            print(f"[FlipLearn] Signal registration skipped: {e}")

        # ── 2. Pre-warm RAG retriever ────────────────────────────
        # Skip pre-warming during management commands (migrate, collectstatic, etc.)
        # to avoid downloading models at build time and causing timeouts.
        import sys
        _management_cmds = {
            'migrate', 'collectstatic', 'makemigrations', 'build_rag_index',
            'shell', 'createsuperuser', 'check', 'inspectdb',
        }
        _is_management = any(cmd in sys.argv for cmd in _management_cmds)

        if not _is_management:
            def _prewarm():
                try:
                    from rag_engine.retriever import _load_resources
                    _load_resources()
                    print("[FlipLearn] RAG retriever pre-warmed ✓")
                except Exception as e:
                    print(f"[FlipLearn] RAG pre-warm skipped: {e}")

            t = threading.Thread(target=_prewarm, daemon=True)
            t.start()
        else:
            print("[FlipLearn] RAG pre-warm skipped (management command)")

from django.apps import AppConfig


class FlippedAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'flipped_app'

    def ready(self):
        """Pre-warm the RAG retriever (loads FAISS index + embedding model)
        in a background thread so the first user request isn't slow."""
        import threading

        def _prewarm():
            try:
                from rag_engine.retriever import _load_resources
                _load_resources()
                print("[FlipLearn] RAG retriever pre-warmed ✓")
            except Exception as e:
                print(f"[FlipLearn] RAG pre-warm skipped: {e}")

        t = threading.Thread(target=_prewarm, daemon=True)
        t.start()

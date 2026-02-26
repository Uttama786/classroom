"""Management command to build the RAG FAISS index."""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Build the RAG FAISS index from knowledge files, PDFs, and quiz questions."

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Rebuild index even if it already exists.',
        )

    def handle(self, *args, **options):
        import pathlib
        save_dir = pathlib.Path(__file__).resolve().parents[4] / 'rag_engine' / 'saved_index'

        if save_dir.exists() and (save_dir / 'index.faiss').exists() and not options['force']:
            self.stdout.write(
                self.style.WARNING(
                    f'Index already exists at {save_dir}. Use --force to rebuild.'
                )
            )
            return

        self.stdout.write(self.style.HTTP_INFO('Building RAG index...'))
        try:
            from rag_engine.indexer import build_index
            build_index()
            self.stdout.write(self.style.SUCCESS('RAG index built successfully!'))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'Failed to build index: {e}'))
            raise

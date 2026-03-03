"""
Django management command wrapper for generate_dataset.py
Usage: python manage.py generate_dataset [--total 200] [--min-real 20]
"""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Generate dataset.csv from real DB activity (student/teacher/admin) + synthetic augmentation'

    def add_arguments(self, parser):
        parser.add_argument(
            '--total', type=int, default=200,
            help='Target total rows in the dataset (default: 200)'
        )
        parser.add_argument(
            '--min-real', type=int, default=20,
            help='Minimum real DB records before augmenting with synthetic data (default: 20)'
        )

    def handle(self, *args, **options):
        from ml_model.generate_dataset import generate_dataset
        generate_dataset(
            target_total=options['total'],
            min_real=options['min_real']
        )
        self.stdout.write(self.style.SUCCESS('Dataset generation complete.'))

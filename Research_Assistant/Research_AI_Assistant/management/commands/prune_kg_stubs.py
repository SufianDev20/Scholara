"""
Django management command to delete old, weakly-cited stub nodes from the
knowledge graph.
Run: python manage.py prune_kg_stubs [--min-citers 2] [--older-than-days 14] [--dry-run]
"""

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from django.utils import timezone

from Research_AI_Assistant.models import KGEdge, KGNode

# Postgres caps a statement at 65535 bind parameters (old SQLite: 999), so
# delete in chunks rather than with one huge pk__in list.
DELETE_CHUNK_SIZE = 1000


class Command(BaseCommand):
    help = (
        "Delete stub KGNode rows older than the cutoff that have fewer than "
        "--min-citers incoming CITES edges (their edges cascade)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--min-citers", type=int, default=2)
        parser.add_argument("--older-than-days", type=int, default=14)
        parser.add_argument("--dry-run", action="store_true")

    def _counts(self):
        return (
            KGNode.objects.count(),
            KGNode.objects.filter(is_stub=True).count(),
            KGEdge.objects.count(),
        )

    def _write_counts(self, label, counts):
        nodes, stubs, edges = counts
        self.stdout.write(f"{label}: {nodes} nodes, {stubs} stubs, {edges} edges")

    def handle(self, *args, **options):
        cutoff = timezone.now() - timedelta(days=options["older_than_days"])
        candidates = KGNode.objects.filter(
            is_stub=True, updated_at__lt=cutoff
        ).annotate(
            citers=Count("in_edges", filter=Q(in_edges__edge_type="CITES"))
        ).filter(citers__lt=options["min_citers"])

        self._write_counts("Before", self._counts())
        to_delete = candidates.count()

        if options["dry_run"]:
            self.stdout.write(f"Dry run: would delete {to_delete} stub nodes.")
            return

        # Evaluate ids first: delete() on an annotated queryset is not allowed.
        pks = list(candidates.values_list("pk", flat=True))
        for i in range(0, len(pks), DELETE_CHUNK_SIZE):
            KGNode.objects.filter(pk__in=pks[i : i + DELETE_CHUNK_SIZE]).delete()
        self._write_counts("After", self._counts())
        self.stdout.write(self.style.SUCCESS(f"Deleted {to_delete} stub nodes."))

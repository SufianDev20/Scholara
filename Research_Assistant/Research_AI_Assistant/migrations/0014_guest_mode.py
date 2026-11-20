# Guest Mode tables (models.GuestAllowance / GuestIssuedPaper / GuestDailyUsage).
# Additive only: creates three new tables and touches no existing table.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("Research_AI_Assistant", "0013_paperextraction_paperqachunks_delete_paperpdf"),
    ]

    operations = [
        migrations.CreateModel(
            name="GuestAllowance",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("guest_id", models.CharField(max_length=64, unique=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("expires_at", models.DateTimeField(db_index=True)),
                ("issued_ip_hash", models.CharField(blank=True, default="", max_length=64)),
                ("searches_used", models.PositiveIntegerField(default=0)),
                ("summaries_used", models.PositiveIntegerField(default=0)),
                ("questions_used", models.PositiveIntegerField(default=0)),
            ],
            options={
                "indexes": [models.Index(fields=["issued_ip_hash", "created_at"], name="guest_ip_created_idx")],
            },
        ),
        migrations.CreateModel(
            name="GuestDailyUsage",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("day", models.DateField(unique=True)),
                ("ai_calls", models.PositiveIntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name="GuestIssuedPaper",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("openalex_id", models.CharField(max_length=100)),
                ("query", models.CharField(max_length=500)),
                ("paper", models.JSONField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "allowance",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="issued_papers",
                        to="Research_AI_Assistant.guestallowance",
                    ),
                ),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(fields=("allowance", "openalex_id", "query"), name="uniq_guest_issued_paper")
                ],
            },
        ),
    ]

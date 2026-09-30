"""
Seed Multi-Tenant Analytics & Anomaly Data.

Generates 90 days of realistic daily metrics for 3 clinical tenants:
1. Metro Eye Institute & Research Hospital (default-metro-eye-hospital)
2. Apex Retina Center (apex-retina)
3. St. Jude Eye Clinic (st-jude-eye)

Injects realistic clinical events mirroring ad-tech telemetry patterns:
- Volume Surge (Mobile screening camp <-> Viral campaign impression surge)
- Quality Drop (Camera lens dust/defocus <-> Creative fatigue CTR decline)
- SLA Latency Breach (Server GPU saturation <-> RTB bid response auction timeout)
- Distribution Shift (High DR screening cohort <-> Programmatic audience composition drift)
"""

from __future__ import annotations

import datetime
from datetime import date, datetime as dt, timedelta, timezone
import math
import os
import random
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.analytics import TenantDailyMetric
from backend.db import Base, SessionLocal, Tenant, create_tables, engine


def seed_analytics():
    print("=" * 70)
    print("OphthalmoAI - Multi-Tenant Analytics & Monitoring Data Seeder")
    print("=" * 70)

    create_tables()
    session = SessionLocal()

    # 1. Define and ensure the 3 tenants exist
    tenants_config = [
        {
            "id": "default-metro-eye-hospital",
            "name": "Metro Eye Institute & Research Hospital",
            "slug": "metro-eye-hospital",
            "tier": "tertiary_care",
            "base_screenings": 135,
            "vol_std": 14,
            "base_conf": 0.945,
            "base_lat": 82.0,
        },
        {
            "id": "apex-retina",
            "name": "Apex Retina Center",
            "slug": "apex-retina",
            "tier": "specialty_clinic",
            "base_screenings": 75,
            "vol_std": 8,
            "base_conf": 0.952,
            "base_lat": 78.5,
        },
        {
            "id": "st-jude-eye",
            "name": "St. Jude Eye Clinic",
            "slug": "st-jude-eye",
            "tier": "community_hospital",
            "base_screenings": 45,
            "vol_std": 6,
            "base_conf": 0.938,
            "base_lat": 94.0,
        },
    ]

    for t_conf in tenants_config:
        tenant = session.query(Tenant).filter(Tenant.id == t_conf["id"]).first()
        if not tenant:
            tenant = session.query(Tenant).filter(Tenant.slug == t_conf["slug"]).first()
        if not tenant:
            tenant = Tenant(
                id=t_conf["id"],
                name=t_conf["name"],
                slug=t_conf["slug"],
                tier=t_conf["tier"],
                is_active=True,
            )
            session.add(tenant)
            print(f"[*] Registered clinic tenant: {t_conf['name']} ({t_conf['id']})")
        else:
            tenant.name = t_conf["name"]
            tenant.id = t_conf["id"]
            tenant.tier = t_conf["tier"]
    session.commit()

    # 2. Seed 90 days of metrics
    today = dt.now(timezone.utc).date()
    random.seed(42)

    total_inserted = 0
    total_updated = 0

    print("\n[*] Synthesizing 90 days of time-series screening metrics...")

    for t_idx, t_conf in enumerate(tenants_config):
        tenant_id = t_conf["id"]
        base_vol = t_conf["base_screenings"]
        vol_std = t_conf["vol_std"]
        base_conf = t_conf["base_conf"]
        base_lat = t_conf["base_lat"]

        for day_offset in range(89, -1, -1):
            metric_date = today - timedelta(days=day_offset)

            # Weekly seasonality (slightly lower on Sundays)
            day_of_week = metric_date.weekday()
            seasonality = 0.65 if day_of_week == 6 else (0.85 if day_of_week == 5 else 1.0)

            # Natural Gaussian variance
            daily_vol = int(max(10, random.gauss(base_vol * seasonality, vol_std)))
            daily_conf = min(0.985, max(0.85, random.gauss(base_conf, 0.015)))
            daily_lat = max(45.0, random.gauss(base_lat, 6.0))

            # Standard disease distribution
            # 55% Normal, 22% DR, 11% Glaucoma, 8% Cataract, 4% AMD
            n_normal = int(daily_vol * 0.55)
            n_dr = int(daily_vol * 0.22)
            n_glauc = int(daily_vol * 0.11)
            n_cat = int(daily_vol * 0.08)
            n_amd = max(0, daily_vol - (n_normal + n_dr + n_glauc + n_cat))

            diagnoses = {
                "Normal": n_normal,
                "Diabetic Retinopathy": n_dr,
                "Glaucoma": n_glauc,
                "Cataract": n_cat,
                "Age-Related Macular Degeneration": n_amd,
            }

            # INJECT ANOMALIES into Tenant 1 ("default-metro-eye-hospital")
            if tenant_id == "default-metro-eye-hospital":
                # 1. Volume Surge (Day -12: Mobile screening camp)
                if day_offset == 12:
                    daily_vol = 430
                    diagnoses["Normal"] = int(daily_vol * 0.55)
                    diagnoses["Diabetic Retinopathy"] = int(daily_vol * 0.23)
                    diagnoses["Glaucoma"] = int(daily_vol * 0.11)
                    diagnoses["Cataract"] = int(daily_vol * 0.07)
                    diagnoses["Age-Related Macular Degeneration"] = daily_vol - sum(diagnoses.values()) + diagnoses["Age-Related Macular Degeneration"]

                # 2. Quality Drop (Day -6: Faulty camera lens smudge / defocus)
                elif day_offset == 6:
                    daily_conf = 0.732  # 22.5% drop from baseline 0.945

                # 3. SLA Latency Breach (Day -2: Server GPU saturation / high ensemble queue)
                elif day_offset == 2:
                    daily_lat = 278.5  # Exceeds 200ms SLA by 39%

                # 4. Distribution Shift (Day 0: Diabetic Camp cohort surge)
                elif day_offset == 0:
                    daily_vol = 160
                    diagnoses = {
                        "Diabetic Retinopathy": 122,  # 76.2% DR vs normal 22%
                        "Normal": 22,
                        "Glaucoma": 8,
                        "Cataract": 5,
                        "Age-Related Macular Degeneration": 3,
                    }

            # Derived fields
            patients = int(daily_vol * random.uniform(0.92, 0.98))
            high_risk = sum(
                v for k, v in diagnoses.items() if k != "Normal"
            )
            # High risk count = confident diseased detections
            high_risk_count = int(high_risk * random.uniform(0.75, 0.90))

            edge_ratio = random.uniform(0.30, 0.45)
            edge_screenings = int(daily_vol * edge_ratio)
            server_screenings = daily_vol - edge_screenings

            fpr = round(random.uniform(0.015, 0.035), 4)
            agreement = round(random.uniform(0.93, 0.98), 4)

            # Upsert into database
            row = (
                session.query(TenantDailyMetric)
                .filter(
                    TenantDailyMetric.tenant_id == tenant_id,
                    TenantDailyMetric.date == metric_date,
                )
                .first()
            )

            if not row:
                row = TenantDailyMetric(
                    tenant_id=tenant_id,
                    date=metric_date,
                    total_screenings=daily_vol,
                    total_patients=patients,
                    avg_confidence=daily_conf,
                    avg_inference_time_ms=daily_lat,
                    diagnoses_by_class=diagnoses,
                    high_risk_count=high_risk_count,
                    edge_screenings=edge_screenings,
                    server_screenings=server_screenings,
                    false_positive_rate=fpr,
                    model_agreement_rate=agreement,
                )
                session.add(row)
                total_inserted += 1
            else:
                row.total_screenings = daily_vol
                row.total_patients = patients
                row.avg_confidence = daily_conf
                row.avg_inference_time_ms = daily_lat
                row.diagnoses_by_class = diagnoses
                row.high_risk_count = high_risk_count
                row.edge_screenings = edge_screenings
                row.server_screenings = server_screenings
                row.false_positive_rate = fpr
                row.model_agreement_rate = agreement
                total_updated += 1

    session.commit()
    session.close()

    print(f"\n[+] Seeding complete: {total_inserted} inserted, {total_updated} updated.")
    print("=" * 70)
    print("INJECTED ANOMALY SUMMARY (Demonstrating Ad-Tech Engineering Equivalents):")
    print("1. Volume Spike (Day -12): 430 screenings vs ~135 baseline.")
    print("   -> Ad-Tech Parallel: Sudden impression volume surges in ad exchanges.")
    print("2. Quality Drop (Day -6): Confidence fell to 73.2% vs ~94.5% baseline.")
    print("   -> Ad-Tech Parallel: CTR / conversion rate decline from creative fatigue.")
    print("3. SLA Breach (Day -2): Latency surged to 278.5ms (breaching 200ms limit).")
    print("   -> Ad-Tech Parallel: Bid response latency exceeding RTB auction SLA.")
    print("4. Distribution Shift (Day 0): DR jumped to 76% vs ~22% baseline.")
    print("   -> Ad-Tech Parallel: Audience composition drift in programmatic targeting.")
    print("=" * 70)


if __name__ == "__main__":
    seed_analytics()

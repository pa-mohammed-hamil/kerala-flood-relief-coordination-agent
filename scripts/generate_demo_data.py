"""Placeholder module: scripts/generate_demo_data.py; implementation required."""

#!/usr/bin/env python3
"""
Kerala Flood Relief Coordination Agent
======================================
generate_demo_data.py

Generate fictional demonstration datasets for frontend dashboards,
backend development, automated tests, and portfolio demonstrations.

SAFETY
------
- All records are synthetic.
- No real persons, emergency calls, dispatches, or alerts are created.
- Geographic coordinates are not generated.
- No official IMD, CWC, or KSDMA feeds are accessed.
- All measurements and risk scores are illustrative, not forecasts.
- Never use these datasets for real emergency decisions.

Usage
-----
python backend/scripts/generate_demo_data.py

python backend/scripts/generate_demo_data.py \
    --output data/demo \
    --seed 42 \
    --days 14 \
    --incidents 50 \
    --volunteers 100 \
    --shelters 40

python backend/scripts/generate_demo_data.py --format json
python backend/scripts/generate_demo_data.py --format csv
python backend/scripts/generate_demo_data.py --format both
python backend/scripts/generate_demo_data.py --validate-only

Python: 3.10+
Dependencies: Standard library only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_NAME = "Kerala Flood Relief Coordination Agent"
GENERATOR_VERSION = "1.0.0"
DEMO_DATE = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)

DISTRICTS = [
    "Thiruvananthapuram",
    "Kollam",
    "Pathanamthitta",
    "Alappuzha",
    "Kottayam",
    "Idukki",
    "Ernakulam",
    "Thrissur",
    "Palakkad",
    "Malappuram",
    "Kozhikode",
    "Wayanad",
    "Kannur",
    "Kasaragod",
]

SKILLS = [
    "medical",
    "logistics",
    "shelter",
    "communication",
    "transport",
]

INCIDENT_TYPES = [
    "Water Rescue",
    "Flood Entrapment",
    "Evacuation Support",
    "Medical Assistance",
    "Road Isolation",
    "Shelter Transport",
    "Emergency Supplies",
    "Landslide Risk",
]

RESOURCES = {
    "drinking_water": "litres",
    "food_kits": "kits",
    "blankets": "units",
    "hygiene_kits": "kits",
    "first_aid_kits": "kits",
    "tarpaulins": "units",
}

TEAM_TYPES = [
    "Water Rescue",
    "Evacuation",
    "Medical",
    "Logistics",
    "General Rescue",
]

HOSPITAL_SPECIALTIES = [
    "General Care",
    "Emergency Support",
    "Medical Stabilization",
]

ALERT_TYPES = [
    "Rainfall Scenario",
    "River Level Scenario",
    "Shelter Capacity Scenario",
    "Supply Shortage Scenario",
    "Infrastructure Risk Scenario",
]

PRIORITIES = ["low", "medium", "high", "critical"]

DATASETS: dict[str, list[dict[str, Any]]] = {}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def timestamp(rng: random.Random, days: int = 7) -> str:
    minutes = rng.randint(0, max(1, days * 24 * 60))
    return iso(DEMO_DATE - timedelta(minutes=minutes))


def identifier(prefix: str, index: int) -> str:
    return f"{prefix}-{index:03d}"


def choose_district(rng: random.Random) -> str:
    return rng.choice(DISTRICTS)


def risk_level(score: float) -> str:
    if score >= 85:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 40:
        return "moderate"
    return "low"


def occupancy_status(occupied: int, capacity: int) -> str:
    if capacity <= 0:
        return "closed"
    ratio = occupied / capacity
    if ratio >= 1:
        return "full"
    if ratio >= 0.85:
        return "high"
    return "available"


def random_priority(rng: random.Random) -> str:
    return rng.choices(
        PRIORITIES,
        weights=[20, 40, 30, 10],
        k=1,
    )[0]


def demo_metadata() -> dict[str, Any]:
    return {
        "synthetic": True,
        "source": "generate_demo_data.py",
        "authoritative": False,
        "operational_use": False,
    }


def add_dataset(
    name: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    DATASETS[name] = records
    return records


# ============================================================
# 1. DISTRICT OVERVIEW
# ============================================================

def generate_districts(
    rng: random.Random,
) -> list[dict[str, Any]]:
    records = []

    for i, district in enumerate(DISTRICTS, start=1):
        rainfall = round(rng.uniform(15, 190), 1)
        river_pressure = round(rng.uniform(5, 95), 1)
        exposure = round(rng.uniform(10, 95), 1)
        terrain = round(rng.uniform(5, 95), 1)

        score = round(
            0.35 * min(100, rainfall / 2)
            + 0.30 * river_pressure
            + 0.20 * terrain
            + 0.15 * exposure,
            1,
        )

        records.append({
            "district_id": identifier("DST", i),
            "name": district,
            "rainfall_24h_mm": rainfall,
            "river_pressure_index": river_pressure,
            "terrain_index": terrain,
            "exposure_index": exposure,
            "risk_score": score,
            "risk_level": risk_level(score),
            "updated_at": timestamp(rng, 1),
            **demo_metadata(),
        })

    return add_dataset("districts", records)


# ============================================================
# 2. RAINFALL HISTORY
# ============================================================

def generate_rainfall(
    rng: random.Random,
    days: int,
) -> list[dict[str, Any]]:
    records = []
    index = 1

    for district in DISTRICTS:
        baseline = rng.uniform(15, 95)

        for offset in range(days):
            day = (
                DEMO_DATE.date()
                - timedelta(days=days - offset - 1)
            )

            seasonal_wave = 24 * math.sin(offset / 2.8)
            rainfall = max(
                0,
                baseline + seasonal_wave + rng.gauss(0, 22),
            )

            records.append({
                "rainfall_id": identifier("RAIN", index),
                "district": district,
                "date": day.isoformat(),
                "rainfall_mm": round(rainfall, 1),
                "intensity": (
                    "heavy" if rainfall >= 100
                    else "moderate" if rainfall >= 40
                    else "light"
                ),
                **demo_metadata(),
            })
            index += 1

    return add_dataset("rainfall", records)


# ============================================================
# 3. RIVER MONITORING
# ============================================================

def generate_rivers(
    rng: random.Random,
    count: int = 14,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        warning = round(rng.uniform(4, 12), 2)
        danger = round(warning + rng.uniform(0.5, 2.5), 2)
        current = round(rng.uniform(1.5, danger + 1.2), 2)

        status = (
            "danger" if current >= danger
            else "warning" if current >= warning
            else "normal"
        )

        records.append({
            "gauge_id": identifier("RIV", i),
            "station_name": f"Demo River Gauge {i:02d}",
            "district": choose_district(rng),
            "current_level_m": current,
            "warning_level_m": warning,
            "danger_level_m": danger,
            "change_3h_m": round(rng.uniform(-0.8, 1.2), 2),
            "status": status,
            "last_updated": timestamp(rng, 1),
            **demo_metadata(),
        })

    return add_dataset("rivers", records)


# ============================================================
# 4. DAM MONITORING
# ============================================================

def generate_dams(
    rng: random.Random,
    count: int = 8,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        storage = round(rng.uniform(35, 99), 1)

        records.append({
            "dam_id": identifier("DAM", i),
            "name": f"Demo Reservoir {i:02d}",
            "district": choose_district(rng),
            "storage_percent": storage,
            "inflow_cumecs": round(rng.uniform(10, 500), 1),
            "outflow_cumecs": round(rng.uniform(5, 350), 1),
            "scenario_status": (
                "high" if storage >= 90
                else "watch" if storage >= 75
                else "normal"
            ),
            "updated_at": timestamp(rng, 1),
            **demo_metadata(),
        })

    return add_dataset("dams", records)


# ============================================================
# 5. INCIDENTS AND SOS
# ============================================================

def generate_incidents(
    rng: random.Random,
    count: int,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        priority = random_priority(rng)
        people = rng.randint(1, 15)

        records.append({
            "incident_id": identifier("INC", i),
            "district": choose_district(rng),
            "incident_type": rng.choice(INCIDENT_TYPES),
            "priority": priority,
            "people_affected": people,
            "status": rng.choice([
                "pending",
                "under_review",
                "assigned",
                "resolved",
            ]),
            "description": (
                f"Fictional flood response scenario {i:03d}. "
                "Not a real incident."
            ),
            "created_at": timestamp(rng, 5),
            "human_review_required": priority in ("high", "critical"),
            **demo_metadata(),
        })

    return add_dataset("incidents", records)


def generate_sos(
    rng: random.Random,
    incidents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []

    for i, incident in enumerate(incidents, start=1):
        if i % 2 == 0:
            continue

        records.append({
            "sos_id": identifier("SOS", i),
            "incident_id": incident["incident_id"],
            "district": incident["district"],
            "priority": incident["priority"],
            "people_count": incident["people_affected"],
            "status": "pending",
            "contact_code": f"DEMO-CONTACT-{i:03d}",
            "location_label": f"Fictional Area {i:03d}",
            "message": "Synthetic SOS request for UI testing.",
            "created_at": incident["created_at"],
            **demo_metadata(),
        })

    return add_dataset("sos_requests", records)


# ============================================================
# 6. RESCUE TEAMS
# ============================================================

def generate_rescue_teams(
    rng: random.Random,
    count: int = 18,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        records.append({
            "team_id": identifier("TEAM", i),
            "name": f"Demo Response Team {i:02d}",
            "district": choose_district(rng),
            "team_type": rng.choice(TEAM_TYPES),
            "members": rng.randint(3, 12),
            "availability": rng.choice([
                "available",
                "assigned",
                "resting",
            ]),
            "equipment_readiness_percent": rng.randint(65, 100),
            "communication_status": rng.choice([
                "simulated_online",
                "simulated_offline",
            ]),
            "last_updated": timestamp(rng, 1),
            **demo_metadata(),
        })

    return add_dataset("rescue_teams", records)


# ============================================================
# 7. SHELTERS
# ============================================================

def generate_shelters(
    rng: random.Random,
    count: int,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        capacity = rng.randint(80, 550)
        occupied = rng.randint(0, capacity)

        records.append({
            "shelter_id": identifier("SH", i),
            "name": f"Demo Relief Shelter {i:03d}",
            "district": choose_district(rng),
            "capacity": capacity,
            "occupied": occupied,
            "available_beds": capacity - occupied,
            "occupancy_percent": round(
                100 * occupied / capacity, 1
            ),
            "status": occupancy_status(occupied, capacity),
            "water_status": rng.choice(["ready", "limited"]),
            "food_status": rng.choice(["ready", "limited"]),
            "medical_status": rng.choice(["ready", "limited"]),
            "accessible": rng.choice([True, False]),
            "updated_at": timestamp(rng, 1),
            **demo_metadata(),
        })

    return add_dataset("shelters", records)


# ============================================================
# 8. HOSPITAL CAPACITY
# ============================================================

def generate_hospitals(
    rng: random.Random,
    count: int = 16,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        total_beds = rng.randint(50, 350)
        occupied = rng.randint(0, total_beds)
        ambulances = rng.randint(0, 8)

        records.append({
            "hospital_id": identifier("HOSP", i),
            "name": f"Demo Hospital {i:02d}",
            "district": choose_district(rng),
            "specialty_label": rng.choice(HOSPITAL_SPECIALTIES),
            "total_beds": total_beds,
            "occupied_beds": occupied,
            "available_beds": total_beds - occupied,
            "ambulances_available": ambulances,
            "oxygen_readiness": rng.choice([
                "ready",
                "limited",
            ]),
            "status": occupancy_status(
                occupied, total_beds
            ),
            "updated_at": timestamp(rng, 1),
            **demo_metadata(),
        })

    return add_dataset("hospitals", records)


# ============================================================
# 9. VOLUNTEERS
# ============================================================

def generate_volunteers(
    rng: random.Random,
    count: int,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        skill_count = rng.randint(1, 3)
        skills = rng.sample(SKILLS, skill_count)

        review_status = rng.choices(
            ["pending", "approved", "rejected"],
            weights=[25, 65, 10],
            k=1,
        )[0]

        records.append({
            "volunteer_id": identifier("VOL", i),
            "registration_code": f"DEMO-VOL-{i:03d}",
            "display_name": f"Demo Volunteer {i:03d}",
            "district": choose_district(rng),
            "skills": skills,
            "experience_label": rng.choice([
                "new",
                "trained",
                "experienced",
            ]),
            "preferred_shift": rng.choice([
                "morning",
                "afternoon",
                "evening",
                "flexible",
            ]),
            "available": (
                review_status == "approved"
                and rng.random() < 0.75
            ),
            "review_status": review_status,
            "qualification_verified": False,
            "registered_at": timestamp(rng, 20),
            **demo_metadata(),
        })

    return add_dataset("volunteers", records)


# ============================================================
# 10. VOLUNTEER ASSIGNMENTS
# ============================================================

def generate_volunteer_assignments(
    rng: random.Random,
    volunteers: list[dict[str, Any]],
    count: int = 30,
) -> list[dict[str, Any]]:
    records = []

    candidates = [
        v for v in volunteers
        if v["review_status"] == "approved"
        and v["available"]
    ]

    rng.shuffle(candidates)
    allocated = set()

    for i in range(1, count + 1):
        skill = rng.choice(SKILLS)
        district = choose_district(rng)
        status = "pending"
        volunteer_id = ""

        matching = [
            v for v in candidates
            if v["volunteer_id"] not in allocated
            and skill in v["skills"]
            and v["district"] == district
        ]

        if matching and rng.random() < 0.65:
            volunteer = matching[0]
            volunteer_id = volunteer["volunteer_id"]
            allocated.add(volunteer_id)
            status = rng.choice(["assigned", "active"])

        records.append({
            "assignment_id": identifier("ASN", i),
            "task": f"Demo {skill} support task {i:03d}",
            "district": district,
            "required_skill": skill,
            "priority": random_priority(rng),
            "volunteer_id": volunteer_id,
            "status": status,
            "human_review": (
                "simulated_approved"
                if volunteer_id else "pending"
            ),
            "created_at": timestamp(rng, 5),
            **demo_metadata(),
        })

    return add_dataset("volunteer_assignments", records)


# ============================================================
# 11. RELIEF INVENTORY
# ============================================================

def generate_inventory(
    rng: random.Random,
    warehouses: int = 4,
) -> list[dict[str, Any]]:
    records = []
    index = 1

    for warehouse_no in range(1, warehouses + 1):
        warehouse_id = identifier("WH", warehouse_no)
        district = choose_district(rng)

        for resource, unit in RESOURCES.items():
            quantity = rng.randint(150, 5000)
            reserved = rng.randint(0, quantity // 3)

            records.append({
                "inventory_id": identifier("INV", index),
                "warehouse_id": warehouse_id,
                "district": district,
                "resource": resource,
                "unit": unit,
                "quantity": quantity,
                "reserved": reserved,
                "available": quantity - reserved,
                "stock_status": (
                    "low" if quantity - reserved < 400
                    else "adequate"
                ),
                "updated_at": timestamp(rng, 1),
                **demo_metadata(),
            })

            index += 1

    return add_dataset("relief_inventory", records)


# ============================================================
# 12. SUPPLY-CHAIN TRACKING
# ============================================================

def generate_supply_chain(
    rng: random.Random,
    inventory: list[dict[str, Any]],
    count: int = 35,
) -> list[dict[str, Any]]:
    records = []

    statuses = [
        "draft",
        "pending_approval",
        "approved",
        "in_transit",
        "delivered",
        "cancelled",
    ]

    for i in range(1, count + 1):
        item = rng.choice(inventory)
        quantity = rng.randint(5, 250)
        status = rng.choice(statuses)

        records.append({
            "shipment_id": identifier("SHIP", i),
            "warehouse_id": item["warehouse_id"],
            "origin_district": item["district"],
            "destination_district": choose_district(rng),
            "resource": item["resource"],
            "unit": item["unit"],
            "quantity": quantity,
            "status": status,
            "vehicle_code": f"DEMO-VEH-{i:03d}",
            "driver_code": f"DEMO-DRV-{i:03d}",
            "tracking_mode": "simulation",
            "eta_label": (
                "Illustrative ETA only"
                if status == "in_transit"
                else "Not applicable"
            ),
            "human_review_required": True,
            "created_at": timestamp(rng, 7),
            **demo_metadata(),
        })

    return add_dataset("supply_chain", records)


# ============================================================
# 13. RELIEF REQUESTS
# ============================================================

def generate_relief_requests(
    rng: random.Random,
    count: int = 35,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        resource = rng.choice(list(RESOURCES))

        records.append({
            "request_id": identifier("REL", i),
            "district": choose_district(rng),
            "resource": resource,
            "unit": RESOURCES[resource],
            "quantity_requested": rng.randint(10, 500),
            "priority": random_priority(rng),
            "status": rng.choice([
                "pending",
                "under_review",
                "approved",
                "fulfilled",
            ]),
            "requester_code": f"DEMO-ORG-{i:03d}",
            "created_at": timestamp(rng, 6),
            **demo_metadata(),
        })

    return add_dataset("relief_requests", records)


# ============================================================
# 14. MISSING PERSONS / REUNIFICATION
# ============================================================

def generate_reunification_cases(
    rng: random.Random,
    count: int = 20,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        records.append({
            "case_id": identifier("FAM", i),
            "person_code": f"DEMO-PERSON-{i:03d}",
            "family_code": f"DEMO-FAMILY-{i:03d}",
            "district": choose_district(rng),
            "case_type": rng.choice([
                "family_reunification",
                "welfare_check",
                "location_confirmation",
            ]),
            "status": rng.choice([
                "pending",
                "under_review",
                "matched_for_review",
                "closed",
            ]),
            "privacy_classification": "synthetic",
            "human_verification_required": True,
            "created_at": timestamp(rng, 10),
            **demo_metadata(),
        })

    return add_dataset("reunification_cases", records)


# ============================================================
# 15. DAMAGE ASSESSMENT
# ============================================================

def generate_damage_assessments(
    rng: random.Random,
    count: int = 25,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        severity = rng.choice([
            "minor",
            "moderate",
            "severe",
            "critical",
        ])

        estimated_cost = {
            "minor": (5000, 30000),
            "moderate": (30000, 150000),
            "severe": (150000, 600000),
            "critical": (600000, 1800000),
        }[severity]

        records.append({
            "assessment_id": identifier("DMG", i),
            "district": choose_district(rng),
            "asset_code": f"DEMO-ASSET-{i:03d}",
            "asset_type": rng.choice([
                "building",
                "road",
                "bridge",
                "public_facility",
            ]),
            "severity": severity,
            "illustrative_cost_inr": rng.randint(
                *estimated_cost
            ),
            "review_status": rng.choice([
                "pending",
                "under_review",
                "simulated_approved",
            ]),
            "professional_inspection_performed": False,
            "created_at": timestamp(rng, 10),
            **demo_metadata(),
        })

    return add_dataset("damage_assessments", records)


# ============================================================
# 16. ALERTS
# ============================================================

def generate_alerts(
    rng: random.Random,
    count: int = 25,
) -> list[dict[str, Any]]:
    records = []

    for i in range(1, count + 1):
        priority = random_priority(rng)

        records.append({
            "alert_id": identifier("ALT", i),
            "district": choose_district(rng),
            "alert_type": rng.choice(ALERT_TYPES),
            "priority": priority,
            "title": f"Fictional alert scenario {i:03d}",
            "message": (
                "Demonstration alert. "
                "Not an official public warning."
            ),
            "status": rng.choice([
                "draft",
                "pending_review",
                "archived",
            ]),
            "sent": False,
            "official_warning": False,
            "created_at": timestamp(rng, 4),
            **demo_metadata(),
        })

    return add_dataset("alerts", records)


# ============================================================
# 17. AUDIT LOGS
# ============================================================

def generate_audit_logs(
    rng: random.Random,
    count: int = 100,
) -> list[dict[str, Any]]:
    records = []

    actions = [
        "demo_record_created",
        "demo_review_requested",
        "demo_review_approved",
        "demo_review_rejected",
        "demo_status_changed",
        "demo_report_generated",
        "demo_data_exported",
    ]

    for i in range(1, count + 1):
        records.append({
            "event_id": identifier("AUD", i),
            "actor_code": f"DEMO-USER-{rng.randint(1, 8):03d}",
            "action": rng.choice(actions),
            "entity_type": rng.choice([
                "incident",
                "shelter",
                "volunteer",
                "shipment",
                "relief_request",
                "alert",
            ]),
            "entity_code": f"DEMO-ENTITY-{rng.randint(1, 80):03d}",
            "result": rng.choice([
                "success",
                "review_required",
                "rejected",
            ]),
            "timestamp": timestamp(rng, 15),
            "simulation_event": True,
            **demo_metadata(),
        })

    return add_dataset("audit_logs", records)


# ============================================================
# 18. DATA VALIDATION
# ============================================================

def validate_datasets(
    datasets: dict[str, list[dict[str, Any]]],
) -> list[str]:
    errors = []

    if len(datasets.get("districts", [])) != 14:
        errors.append("Expected exactly 14 Kerala districts.")

    for dataset_name, records in datasets.items():
        if not isinstance(records, list):
            errors.append(f"{dataset_name}: not a list.")
            continue

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(
                    f"{dataset_name}[{index}]: not an object."
                )
                continue

            if record.get("synthetic") is not True:
                errors.append(
                    f"{dataset_name}[{index}]: missing synthetic marker."
                )

            if record.get("operational_use") is not False:
                errors.append(
                    f"{dataset_name}[{index}]: invalid operational marker."
                )

    for shelter in datasets.get("shelters", []):
        if not 0 <= shelter["occupied"] <= shelter["capacity"]:
            errors.append(
                f"{shelter['shelter_id']}: invalid occupancy."
            )

        if (
            shelter["available_beds"]
            != shelter["capacity"] - shelter["occupied"]
        ):
            errors.append(
                f"{shelter['shelter_id']}: incorrect available beds."
            )

    for hospital in datasets.get("hospitals", []):
        if not 0 <= hospital["occupied_beds"] <= hospital["total_beds"]:
            errors.append(
                f"{hospital['hospital_id']}: invalid bed counts."
            )

    for item in datasets.get("relief_inventory", []):
        if not 0 <= item["reserved"] <= item["quantity"]:
            errors.append(
                f"{item['inventory_id']}: invalid stock reservation."
            )

    volunteers = {
        v["volunteer_id"]: v
        for v in datasets.get("volunteers", [])
    }

    busy_volunteers = set()

    for assignment in datasets.get(
        "volunteer_assignments", []
    ):
        volunteer_id = assignment["volunteer_id"]

        if not volunteer_id:
            if assignment["status"] != "pending":
                errors.append(
                    f"{assignment['assignment_id']}: "
                    "unassigned task must be pending."
                )
            continue

        volunteer = volunteers.get(volunteer_id)

        if not volunteer:
            errors.append(
                f"{assignment['assignment_id']}: "
                "unknown volunteer."
            )
            continue

        if volunteer["review_status"] != "approved":
            errors.append(
                f"{assignment['assignment_id']}: "
                "volunteer is not demo-approved."
            )

        if not volunteer["available"]:
            errors.append(
                f"{assignment['assignment_id']}: "
                "volunteer is unavailable."
            )

        if (
            assignment["required_skill"]
            not in volunteer["skills"]
        ):
            errors.append(
                f"{assignment['assignment_id']}: "
                "required skill missing."
            )

        if (
            assignment["district"]
            != volunteer["district"]
        ):
            errors.append(
                f"{assignment['assignment_id']}: "
                "district mismatch."
            )

        if volunteer_id in busy_volunteers:
            errors.append(
                f"{assignment['assignment_id']}: "
                "duplicate active allocation."
            )

        busy_volunteers.add(volunteer_id)

    return errors


# ============================================================
# 19. JSON AND CSV EXPORT
# ============================================================

def write_json(
    output_dir: Path,
    name: str,
    records: list[dict[str, Any]],
) -> None:
    path = output_dir / f"{name}.json"

    path.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def csv_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    elif value is None:
        text = ""
    else:
        text = str(value)

    # Prevent spreadsheet formula interpretation.
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text

    return text


def write_csv(
    output_dir: Path,
    name: str,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    path = output_dir / f"{name}.csv"

    columns = list(
        dict.fromkeys(
            key
            for record in records
            for key in record.keys()
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()

        for record in records:
            writer.writerow({
                key: csv_value(record.get(key))
                for key in columns
            })


# ============================================================
# 20. MANIFEST AND SUMMARY
# ============================================================

def create_manifest(
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = {
        name: len(records)
        for name, records in DATASETS.items()
    }

    return {
        "project": PROJECT_NAME,
        "generator_version": GENERATOR_VERSION,
        "generated_at": iso(DEMO_DATE),
        "random_seed": args.seed,
        "days": args.days,
        "format": args.format,
        "synthetic": True,
        "authoritative": False,
        "operational_use": False,
        "warning": (
            "Fictional demonstration data only. "
            "Not suitable for emergency response."
        ),
        "dataset_counts": counts,
        "total_records": sum(counts.values()),
    }


def print_summary(
    manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    print()
    print("=" * 64)
    print(" KERALA FLOOD AI — SYNTHETIC DATA GENERATOR")
    print("=" * 64)
    print(f" Output directory : {output_dir}")
    print(f" Random seed      : {manifest['random_seed']}")
    print(f" Data format      : {manifest['format']}")
    print(f" Dataset count    : {len(DATASETS)}")
    print("-" * 64)

    for name, count in manifest["dataset_counts"].items():
        print(f" {name:<29} {count:>8} records")

    print("-" * 64)
    print(
        f" TOTAL RECORDS{'':<16}"
        f"{manifest['total_records']:>8}"
    )
    print("=" * 64)
    print(" SYNTHETIC DATA ONLY — NOT FOR REAL EMERGENCY USE")
    print()


# ============================================================
# 21. GENERATOR ORCHESTRATION
# ============================================================

def generate_all(
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    DATASETS.clear()
    rng = random.Random(args.seed)

    generate_districts(rng)
    generate_rainfall(rng, args.days)
    generate_rivers(rng)
    generate_dams(rng)

    incidents = generate_incidents(
        rng,
        args.incidents,
    )

    generate_sos(rng, incidents)
    generate_rescue_teams(rng)

    generate_shelters(
        rng,
        args.shelters,
    )

    generate_hospitals(rng)

    volunteers = generate_volunteers(
        rng,
        args.volunteers,
    )

    generate_volunteer_assignments(
        rng,
        volunteers,
        args.assignments,
    )

    inventory = generate_inventory(rng)

    generate_supply_chain(
        rng,
        inventory,
        args.shipments,
    )

    generate_relief_requests(rng)
    generate_reunification_cases(rng)
    generate_damage_assessments(rng)
    generate_alerts(rng)
    generate_audit_logs(rng)

    return DATASETS


# ============================================================
# 22. COMMAND-LINE INTERFACE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible synthetic Kerala Flood AI "
            "demo datasets."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/demo"),
        help="Output directory (default: data/demo)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Number of rainfall history days",
    )

    parser.add_argument(
        "--incidents",
        type=int,
        default=50,
        help="Number of fictional incidents",
    )

    parser.add_argument(
        "--volunteers",
        type=int,
        default=100,
        help="Number of fictional volunteers",
    )

    parser.add_argument(
        "--shelters",
        type=int,
        default=40,
        help="Number of fictional shelters",
    )

    parser.add_argument(
        "--assignments",
        type=int,
        default=30,
        help="Number of volunteer assignments",
    )

    parser.add_argument(
        "--shipments",
        type=int,
        default=35,
        help="Number of fictional shipments",
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate generated datasets without writing files",
    )

    args = parser.parse_args()

    for field in [
        "days",
        "incidents",
        "volunteers",
        "shelters",
        "assignments",
        "shipments",
    ]:
        value = getattr(args, field)

        if not 1 <= value <= 10000:
            parser.error(
                f"--{field} must be between 1 and 10000"
            )

    return args


def main() -> int:
    args = parse_args()

    print("Generating fictional Kerala Flood AI datasets...")

    datasets = generate_all(args)

    errors = validate_datasets(datasets)

    if errors:
        print("DATA VALIDATION FAILED")
        for error in errors[:30]:
            print(" -", error)
        return 1

    print("Validation passed.")

    manifest = create_manifest(args)

    if not args.validate_only:
        args.output.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name, records in datasets.items():
            if args.format in ("json", "both"):
                write_json(
                    args.output,
                    name,
                    records,
                )

            if args.format in ("csv", "both"):
                write_csv(
                    args.output,
                    name,
                    records,
                )

        manifest_path = args.output / "manifest.json"

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("Files generated successfully.")
    else:
        print("Validation-only mode: no files written.")

    print_summary(manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

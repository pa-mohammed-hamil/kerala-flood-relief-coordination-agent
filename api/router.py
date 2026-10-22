"""
backend/api/router.py

Kerala Flood Relief Coordination Agent - Central FastAPI Router
"""
from __future__ import annotations

import logging
from fastapi import APIRouter

from backend.api import (
    agents,
    analytics,
    auth,
    blockchain,
    crowdsource,
    damage,
    dams,
    districts,
    evacuation,
    family_safety,
    flood_alerts,
    flood_forecast,
    healthcare,
    incidents,
    missing_persons,
    notifications,
    rag,
    rainfall,
    recovery,
    relief,
    rescue,
    rivers,
    shelters,
    sos,
    users,
    voice_sos,
    volunteers,
)

logger = logging.getLogger(__name__)

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

api_router = APIRouter()


# ============================================================
# ROUTER REGISTRY
# ============================================================

ROUTER_REGISTRY = (
    ("Authentication", auth.router),
    ("Users", users.router),
    ("Districts", districts.router),

    ("Flood Alerts", flood_alerts.router),
    ("Flood Forecast", flood_forecast.router),
    ("Rainfall", rainfall.router),
    ("Rivers", rivers.router),
    ("Dams", dams.router),

    ("Emergency SOS", sos.router),
    ("Voice SOS", voice_sos.router),
    ("Crowdsourced Reporting", crowdsource.router),
    ("Family Safety", family_safety.router),
    ("Incidents", incidents.router),
    ("Rescue", rescue.router),
    ("Evacuation", evacuation.router),

    ("Shelters", shelters.router),
    ("Relief", relief.router),
    ("Healthcare", healthcare.router),
    ("Volunteers", volunteers.router),

    ("Missing Persons", missing_persons.router),
    ("Damage Assessment", damage.router),
    ("Recovery", recovery.router),
    
    ("Blockchain Supply Chain", blockchain.router),

    ("AI Agents", agents.router),
    ("RAG", rag.router),
    ("Analytics", analytics.router),
    ("Notifications", notifications.router),
)


def register_api_routers() -> None:
    """
    Register every REST API router.

    Each module defines its own prefix and tags.

    Example:
        rainfall.router.prefix = "/rainfall"

    Final endpoint:
        /api/v1/rainfall/stations
    """

    registered_prefixes: set[str] = set()

    for name, module_router in ROUTER_REGISTRY:
        prefix = module_router.prefix

        if not prefix:
            raise RuntimeError(
                f"Router '{name}' has no prefix"
            )

        if prefix in registered_prefixes:
            raise RuntimeError(
                f"Duplicate API prefix detected: {prefix}"
            )

        registered_prefixes.add(prefix)

        api_router.include_router(module_router)

        logger.info(
            "Registered API router: %s (%s)",
            name,
            prefix,
        )


register_api_routers()


# ============================================================
# API ROOT
# ============================================================

@api_router.get(
    "/",
    tags=["API Information"],
    summary="API information",
)
async def api_information() -> dict:
    return {
        "name": "Kerala Flood Relief Coordination Agent",
        "version": API_VERSION,
        "status": "available",
        "api_prefix": API_PREFIX,
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "registered_modules": [
            {
                "name": name,
                "prefix": module_router.prefix,
            }
            for name, module_router in ROUTER_REGISTRY
        ],
    }


# ============================================================
# API HEALTH
# ============================================================

@api_router.get(
    "/health",
    tags=["API Information"],
    summary="API router health",
)
async def api_health() -> dict:
    """
    Indicates that the API router is responding.

    This does not verify database connectivity,
    external services, sensors, or AI model health.
    """

    return {
        "status": "ok",
        "service": "api-router",
        "version": API_VERSION,
        "registered_router_count": len(ROUTER_REGISTRY),
    }


__all__ = [
    "api_router",
    "API_VERSION",
    "API_PREFIX",
    "ROUTER_REGISTRY",
]

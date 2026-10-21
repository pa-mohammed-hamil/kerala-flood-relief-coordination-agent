"""
Kerala Flood Relief Coordination Agent - Standalone Server
Simple entry point that avoids module conflicts
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Create app
app = FastAPI(
    title="Kerala Flood Relief Coordination Agent",
    description="AI-powered flood relief and emergency coordination system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend - serves CSS, JS, images, and HTML files
frontend_dir = project_root / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    # Also serve at root for direct HTML access
    app.mount("/pages", StaticFiles(directory=str(frontend_dir), html=True), name="pages")

@app.get("/")
async def home():
    """Serve index page"""
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("""
        <html>
            <head>
                <title>Kerala Flood Relief</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 50px auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }
                    .container {
                        background: white;
                        padding: 40px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }
                    h1 { color: #2c3e50; }
                    .status { color: #27ae60; font-weight: bold; }
                    .features {
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        gap: 15px;
                        margin-top: 30px;
                    }
                    .feature {
                        padding: 15px;
                        background: #ecf0f1;
                        border-radius: 5px;
                        border-left: 4px solid #3498db;
                    }
                    .links {
                        margin-top: 30px;
                        padding: 20px;
                        background: #e8f5e9;
                        border-radius: 5px;
                    }
                    a {
                        color: #2980b9;
                        text-decoration: none;
                        margin-right: 15px;
                    }
                    a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🌊 Kerala Flood Relief Coordination Agent</h1>
                    <p class="status">✓ Server is running successfully!</p>
                    
                    <h2>Features Implemented:</h2>
                    <div class="features">
                        <div class="feature">🎤 Voice SOS System</div>
                        <div class="feature">📸 Crowdsourced Reporting</div>
                        <div class="feature">🚁 Drone Fleet Management</div>
                        <div class="feature">👨‍👩‍👧‍👦 Family Safety Network</div>
                        <div class="feature">⛓️ Blockchain Supply Chain</div>
                        <div class="feature">📱 Community Alerts</div>
                        <div class="feature">🚨 Auto Alert Escalation</div>
                        <div class="feature">🗺️ Real-time Monitoring</div>
                    </div>
                    
                    <div class="links">
                        <h3>Quick Links:</h3>
                        <a href="/docs">📚 API Documentation</a>
                        <a href="/health">❤️ Health Check</a>
                        <a href="/static/command-center.html">🎯 Command Center</a>
                        <a href="/static/voice-sos.html">🎤 Voice SOS</a>
                        <a href="/static/blockchain-supply-chain.html">⛓️ Blockchain</a>
                        <a href="/static/family-safety-dashboard.html">👨‍👩‍👧‍👦 Family Safety</a>
                    </div>
                    
                    <p style="margin-top: 30px; color: #7f8c8d;">
                        <strong>Note:</strong> This is the minimal server version. 
                        Full backend services will be loaded once database is configured.
                    </p>
                </div>
            </body>
        </html>
    """)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "application": "Kerala Flood Relief Coordination Agent",
        "version": "1.0.0",
        "server": "running",
        "features": {
            "voice_sos": True,
            "crowdsource": True,
            "drone_coordination": True,
            "family_safety": True,
            "blockchain": True,
            "community_alerts": True,
            "automated_escalation": True
        }
    }

@app.get("/api/status")
async def api_status():
    """API status"""
    return {
        "api": "online",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "home": "/",
            "static": "/static/*",
            "pages": "/pages/*"
        }
    }

# ============================================
# Mock API Endpoints for Frontend Pages
# ============================================

# Pydantic models for request/response
class AlertData(BaseModel):
    district: str
    severity: str
    message: str

class SOSRequest(BaseModel):
    name: str
    phone: str
    location: str
    emergency_type: str

# Districts API
@app.get("/api/districts")
async def get_districts():
    """Get all Kerala districts"""
    return {
        "districts": [
            {"id": 1, "name": "Thiruvananthapuram", "population": 3301427},
            {"id": 2, "name": "Kollam", "population": 2635375},
            {"id": 3, "name": "Pathanamthitta", "population": 1197412},
            {"id": 4, "name": "Alappuzha", "population": 2127789},
            {"id": 5, "name": "Kottayam", "population": 1974551},
            {"id": 6, "name": "Idukki", "population": 1108974},
            {"id": 7, "name": "Ernakulam", "population": 3282388},
            {"id": 8, "name": "Thrissur", "population": 3121200},
            {"id": 9, "name": "Palakkad", "population": 2809934},
            {"id": 10, "name": "Malappuram", "population": 4112920},
            {"id": 11, "name": "Kozhikode", "population": 3086293},
            {"id": 12, "name": "Wayanad", "population": 817420},
            {"id": 13, "name": "Kannur", "population": 2523003},
            {"id": 14, "name": "Kasaragod", "population": 1307375}
        ]
    }

# Flood Alerts API
@app.get("/api/flood-alerts")
async def get_flood_alerts():
    """Get active flood alerts"""
    return {
        "alerts": [
            {
                "id": 1,
                "district": "Ernakulam",
                "severity": "high",
                "level": "orange",
                "message": "Heavy rainfall expected. Water level rising.",
                "issued_at": datetime.now().isoformat(),
                "active": True
            },
            {
                "id": 2,
                "district": "Idukki",
                "severity": "medium",
                "level": "yellow",
                "message": "Moderate rainfall. Stay alert.",
                "issued_at": datetime.now().isoformat(),
                "active": True
            }
        ]
    }

@app.post("/api/flood-alerts")
async def create_flood_alert(alert: AlertData):
    """Create new flood alert"""
    return {
        "success": True,
        "message": "Alert created",
        "alert_id": 123,
        "data": alert.dict()
    }

# SOS API
@app.get("/api/sos")
async def get_sos_requests():
    """Get all SOS requests"""
    return {
        "requests": [
            {
                "id": 1,
                "name": "John Doe",
                "phone": "+919876543210",
                "location": "Aluva, Ernakulam",
                "emergency_type": "flood",
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
        ]
    }

@app.post("/api/sos")
async def create_sos(request: SOSRequest):
    """Submit SOS request"""
    return {
        "success": True,
        "message": "SOS request received",
        "request_id": 456,
        "estimated_response_time": "15 minutes"
    }

# Shelters API
@app.get("/api/shelters")
async def get_shelters():
    """Get all shelters"""
    return {
        "shelters": [
            {
                "id": 1,
                "name": "Community Hall Shelter",
                "district": "Ernakulam",
                "capacity": 500,
                "occupied": 320,
                "available": 180,
                "facilities": ["food", "medical", "water"],
                "latitude": 10.1050,
                "longitude": 76.3523
            },
            {
                "id": 2,
                "name": "School Shelter",
                "district": "Thrissur",
                "capacity": 300,
                "occupied": 150,
                "available": 150,
                "facilities": ["food", "water"],
                "latitude": 10.5276,
                "longitude": 76.2144
            }
        ]
    }

# Rescue Teams API
@app.get("/api/rescue-teams")
async def get_rescue_teams():
    """Get all rescue teams"""
    return {
        "teams": [
            {
                "id": 1,
                "name": "NDRF Team Alpha",
                "status": "active",
                "location": "Ernakulam",
                "members": 12,
                "vehicles": 3,
                "equipment": ["boats", "ropes", "medical_kits"]
            },
            {
                "id": 2,
                "name": "Fire Force Unit 1",
                "status": "standby",
                "location": "Kochi",
                "members": 8,
                "vehicles": 2,
                "equipment": ["boats", "rescue_equipment"]
            }
        ]
    }

# Weather API
@app.get("/api/weather")
async def get_weather():
    """Get weather data"""
    return {
        "current": {
            "temperature": 28,
            "humidity": 85,
            "rainfall": 45,
            "wind_speed": 15,
            "condition": "Heavy Rain"
        },
        "forecast": [
            {"time": "12:00", "rainfall": 50, "condition": "Heavy Rain"},
            {"time": "15:00", "rainfall": 60, "condition": "Very Heavy Rain"},
            {"time": "18:00", "rainfall": 40, "condition": "Moderate Rain"}
        ]
    }

# River Levels API
@app.get("/api/rivers")
async def get_river_levels():
    """Get river water levels"""
    return {
        "rivers": [
            {
                "id": 1,
                "name": "Periyar",
                "station": "Aluva",
                "current_level": 5.2,
                "danger_level": 6.0,
                "status": "warning",
                "trend": "rising"
            },
            {
                "id": 2,
                "name": "Bharathapuzha",
                "station": "Pattambi",
                "current_level": 4.8,
                "danger_level": 5.5,
                "status": "normal",
                "trend": "stable"
            }
        ]
    }

# Dam Status API
@app.get("/api/dams")
async def get_dam_status():
    """Get dam water levels"""
    return {
        "dams": [
            {
                "id": 1,
                "name": "Idukki Dam",
                "capacity": 1996.0,
                "current_storage": 1750.0,
                "percentage": 87.7,
                "status": "warning",
                "gates_open": 3,
                "discharge_rate": 500
            },
            {
                "id": 2,
                "name": "Mullaperiyar",
                "capacity": 443.0,
                "current_storage": 380.0,
                "percentage": 85.8,
                "status": "alert",
                "gates_open": 2,
                "discharge_rate": 200
            }
        ]
    }

# Analytics API
@app.get("/api/analytics")
async def get_analytics():
    """Get analytics data"""
    return {
        "summary": {
            "total_alerts": 45,
            "active_sos": 12,
            "shelters_active": 28,
            "people_evacuated": 3450,
            "rescue_operations": 67
        },
        "by_district": [
            {"district": "Ernakulam", "alerts": 12, "sos": 5, "evacuated": 890},
            {"district": "Idukki", "alerts": 8, "sos": 3, "evacuated": 560}
        ]
    }

# Volunteers API
@app.get("/api/volunteers")
async def get_volunteers():
    """Get volunteer data"""
    return {
        "volunteers": [
            {
                "id": 1,
                "name": "Rajesh Kumar",
                "phone": "+919876543210",
                "skills": ["first_aid", "driving"],
                "status": "active",
                "assignments": 5
            },
            {
                "id": 2,
                "name": "Priya Menon",
                "phone": "+919876543211",
                "skills": ["cooking", "childcare"],
                "status": "available",
                "assignments": 3
            }
        ]
    }

# Relief Inventory API
@app.get("/api/inventory")
async def get_inventory():
    """Get relief inventory"""
    return {
        "items": [
            {"id": 1, "name": "Food Packets", "quantity": 5000, "unit": "packets"},
            {"id": 2, "name": "Water Bottles", "quantity": 10000, "unit": "bottles"},
            {"id": 3, "name": "Blankets", "quantity": 2000, "unit": "pieces"},
            {"id": 4, "name": "Medical Kits", "quantity": 500, "unit": "kits"}
        ]
    }

# Missing Persons API
@app.get("/api/missing-persons")
async def get_missing_persons():
    """Get missing persons reports"""
    return {
        "reports": [
            {
                "id": 1,
                "name": "Missing Person Name",
                "age": 45,
                "last_seen": "Aluva, Ernakulam",
                "description": "Male, 5'8\", wearing blue shirt",
                "reported_by": "Family member",
                "status": "searching",
                "photo_url": None
            }
        ]
    }

# Notifications API
@app.get("/api/notifications")
async def get_notifications():
    """Get notifications"""
    return {
        "notifications": [
            {
                "id": 1,
                "type": "alert",
                "title": "New Flood Alert",
                "message": "Orange alert issued for Ernakulam",
                "timestamp": datetime.now().isoformat(),
                "read": False
            },
            {
                "id": 2,
                "type": "sos",
                "title": "New SOS Request",
                "message": "Emergency request from Aluva",
                "timestamp": datetime.now().isoformat(),
                "read": False
            }
        ]
    }

# Page alias/redirect mapping for common page names
PAGE_ALIASES = {
    "shelters": "shelter-dashboard",
    "volunteers": "volunteer-dashboard",
    "relief": "relief-dashboard",
    "rescue": "rescue-dashboard",
    "sos": "emergency-sos",
    "alerts": "flood-alerts",
    "floods": "flood-alerts",
    "weather": "rainfall-monitoring",
    "dams": "dam-monitoring",
    "rivers": "river-monitoring",
    "hospital": "hospital-capacity",
    "medical": "medical-emergency",
    "missing": "missing-persons",
    "evacuation": "evacuation-planning",
    "damage": "damage-assessment",
    "prediction": "flood-prediction",
    "map": "live-flood-map",
    "agents": "multi-agent-monitor",
    "voice": "voice-sos",
    "blockchain": "blockchain-supply-chain",
    "family": "family-safety-dashboard",
    "ambulance": "ambulance-dispatch",
    "satellite": "satellite-monitoring"
}

# Serve individual frontend pages
@app.get("/{page_name}.html", include_in_schema=False)
async def serve_page(page_name: str):
    """Serve any HTML page from frontend (with alias support)"""
    # Check for alias
    actual_page = PAGE_ALIASES.get(page_name, page_name)
    
    page_file = frontend_dir / f"{actual_page}.html"
    if page_file.exists():
        return FileResponse(str(page_file))
    
    # If not found, return helpful 404
    raise HTTPException(
        status_code=404, 
        detail=f"Page '{page_name}.html' not found. Try: {', '.join(list(PAGE_ALIASES.keys())[:10])}"
    )

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    print("=" * 70)
    print("Kerala Flood Relief Coordination Agent")
    print("=" * 70)
    print("Starting server...")
    print()
    print("Server will be available at:")
    print("   * http://localhost:8000")
    print("   * http://localhost:8000/docs")
    print("   * http://localhost:8000/health")
    print()
    print("Press CTRL+C to stop")
    print("=" * 70)
    print()
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )

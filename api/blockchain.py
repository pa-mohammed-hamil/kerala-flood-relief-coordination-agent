"""
Blockchain Supply Chain API
REST endpoints for transparent relief material tracking
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models.user import User
from blockchain.blockchain_service import BlockchainService, IPFSService, BatchStatus

router = APIRouter(prefix="/api/blockchain", tags=["Blockchain Supply Chain"])

# Initialize services (would come from config in production)
blockchain_service = None  # Initialized in startup
ipfs_service = IPFSService()


# ========== Request/Response Schemas ==========

class DonationRequest(BaseModel):
    donor_name: str = Field(..., min_length=1, max_length=200)
    donor_type: str = Field(..., regex="^(individual|organization|government)$")
    material_type: str = Field(..., regex="^(food|water|medicine|clothing|cash|other)$")
    quantity: int = Field(..., gt=0)
    unit: str = Field(..., max_length=50)
    location: str = Field(..., max_length=500)


class BatchCreateRequest(BaseModel):
    donation_ids: List[int] = Field(..., min_items=1)
    batch_number: str = Field(..., min_length=1, max_length=100)
    material_type: str
    total_quantity: int = Field(..., gt=0)
    unit: str
    warehouse_location: str


class DistributionRequest(BaseModel):
    batch_id: int = Field(..., gt=0)
    distributor_name: str
    recipient_type: str = Field(..., regex="^(shelter|individual|community)$")
    recipient_id: str = Field(..., min_length=1)  # Aadhaar or shelter ID
    recipient_name: str
    recipient_location: str
    quantity: int = Field(..., gt=0)
    unit: str
    beneficiary_count: int = Field(..., gt=0)


class DonationResponse(BaseModel):
    donation_id: int
    donor: str
    donor_name: str
    donor_type: str
    material_type: str
    quantity: int
    unit: str
    timestamp: datetime
    location: str
    verified: bool
    transaction_hash: str
    certificate_url: Optional[str]


class BatchResponse(BaseModel):
    batch_id: int
    batch_number: str
    material_type: str
    total_quantity: int
    unit: str
    created_at: datetime
    warehouse_location: str
    status: int
    status_name: str
    transaction_hash: str
    qr_code_url: Optional[str]
    checkpoint_count: int


class DistributionResponse(BaseModel):
    distribution_id: int
    batch_id: int
    distributor_name: str
    recipient_id: str
    recipient_name: str
    recipient_location: str
    quantity: int
    unit: str
    timestamp: datetime
    beneficiary_count: int
    verified: bool
    transaction_hash: str
    photo_url: Optional[str]


# ========== Donation Endpoints ==========

@router.post("/donations", response_model=DonationResponse)
async def record_donation(
    request: DonationRequest,
    certificate: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Record a donation on blockchain with certificate"""
    if not blockchain_service or not blockchain_service.is_connected():
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        # Read certificate file
        certificate_data = await certificate.read()
        
        # Upload certificate to IPFS
        certificate_hash = ipfs_service.upload_file(certificate_data, certificate.filename)
        
        # Record on blockchain
        result = blockchain_service.record_donation(
            donor_name=request.donor_name,
            donor_type=request.donor_type,
            material_type=request.material_type,
            quantity=request.quantity,
            unit=request.unit,
            location=request.location,
            certificate_data=certificate_data
        )
        
        # Get full donation details
        donation = blockchain_service.get_donation(result['donation_id'])
        
        return DonationResponse(
            donation_id=donation['id'],
            donor=donation['donor'],
            donor_name=donation['donor_name'],
            donor_type=donation['donor_type'],
            material_type=donation['material_type'],
            quantity=donation['quantity'],
            unit=donation['unit'],
            timestamp=donation['timestamp'],
            location=donation['location'],
            verified=donation['verified'],
            transaction_hash=result['transaction_hash'],
            certificate_url=ipfs_service.get_file_url(certificate_hash)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record donation: {str(e)}")


@router.post("/donations/{donation_id}/verify")
async def verify_donation(
    donation_id: int,
    current_user: User = Depends(get_current_user)
):
    """Verify a donation (verifier role required)"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    # Check user has verifier role
    if current_user.role not in ["admin", "verifier"]:
        raise HTTPException(status_code=403, detail="Verifier role required")
    
    try:
        result = blockchain_service.verify_donation(donation_id)
        return {
            "message": "Donation verified",
            "donation_id": donation_id,
            "transaction_hash": result['transaction_hash']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.get("/donations/{donation_id}", response_model=DonationResponse)
async def get_donation(donation_id: int):
    """Get donation details from blockchain"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        donation = blockchain_service.get_donation(donation_id)
        
        return DonationResponse(
            donation_id=donation['id'],
            donor=donation['donor'],
            donor_name=donation['donor_name'],
            donor_type=donation['donor_type'],
            material_type=donation['material_type'],
            quantity=donation['quantity'],
            unit=donation['unit'],
            timestamp=donation['timestamp'],
            location=donation['location'],
            verified=donation['verified'],
            transaction_hash="N/A",  # Would need to store/query
            certificate_url=ipfs_service.get_file_url(donation['certificate_hash'])
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Donation not found: {str(e)}")


@router.get("/donations/donor/{donor_address}")
async def get_donor_donations(donor_address: str):
    """Get all donations by a donor"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        donation_ids = blockchain_service.get_donor_donations(donor_address)
        
        donations = []
        for donation_id in donation_ids:
            donation = blockchain_service.get_donation(donation_id)
            donations.append(donation)
        
        return {
            "donor_address": donor_address,
            "donation_count": len(donations),
            "donations": donations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ========== Batch Endpoints ==========

@router.post("/batches", response_model=BatchResponse)
async def create_batch(
    request: BatchCreateRequest,
    qr_code: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Create supply batch from donations"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    # Check warehouse role
    if current_user.role not in ["admin", "warehouse"]:
        raise HTTPException(status_code=403, detail="Warehouse role required")
    
    try:
        # Read QR code
        qr_data = await qr_code.read()
        
        # Upload QR to IPFS
        qr_hash = ipfs_service.upload_file(qr_data, qr_code.filename)
        
        # Create batch on blockchain
        result = blockchain_service.create_batch(
            donation_ids=request.donation_ids,
            batch_number=request.batch_number,
            material_type=request.material_type,
            total_quantity=request.total_quantity,
            unit=request.unit,
            warehouse_location=request.warehouse_location,
            qr_code_data=qr_data
        )
        
        # Get full batch details
        batch = blockchain_service.get_batch(result['batch_id'])
        
        status_names = {
            0: "Created", 1: "In Transit", 2: "At Warehouse",
            3: "Ready for Distribution", 4: "Partially Distributed", 5: "Fully Distributed"
        }
        
        return BatchResponse(
            batch_id=batch['id'],
            batch_number=batch['batch_number'],
            material_type=batch['material_type'],
            total_quantity=batch['total_quantity'],
            unit=batch['unit'],
            created_at=batch['created_at'],
            warehouse_location=batch['warehouse_location'],
            status=batch['status'],
            status_name=status_names.get(batch['status'], "Unknown"),
            transaction_hash=result['transaction_hash'],
            qr_code_url=ipfs_service.get_file_url(qr_hash),
            checkpoint_count=0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create batch: {str(e)}")


@router.patch("/batches/{batch_id}/status")
async def update_batch_status(
    batch_id: int,
    new_status: int,
    current_user: User = Depends(get_current_user)
):
    """Update batch status"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    if current_user.role not in ["admin", "warehouse"]:
        raise HTTPException(status_code=403, detail="Warehouse role required")
    
    try:
        result = blockchain_service.update_batch_status(batch_id, new_status)
        return {
            "message": "Batch status updated",
            "batch_id": batch_id,
            "new_status": new_status,
            "transaction_hash": result['transaction_hash']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.post("/batches/{batch_id}/checkpoints")
async def add_checkpoint(
    batch_id: int,
    checkpoint: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Add GPS checkpoint during transit"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        result = blockchain_service.add_checkpoint(batch_id, checkpoint)
        return {
            "message": "Checkpoint added",
            "batch_id": batch_id,
            "checkpoint": checkpoint,
            "transaction_hash": result['transaction_hash']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add checkpoint: {str(e)}")


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: int):
    """Get batch details with checkpoints"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        batch = blockchain_service.get_batch(batch_id)
        checkpoints = blockchain_service.get_batch_checkpoints(batch_id)
        
        status_names = {
            0: "Created", 1: "In Transit", 2: "At Warehouse",
            3: "Ready for Distribution", 4: "Partially Distributed", 5: "Fully Distributed"
        }
        
        return {
            **batch,
            'status_name': status_names.get(batch['status'], "Unknown"),
            'checkpoints': checkpoints,
            'checkpoint_count': len(checkpoints)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Batch not found: {str(e)}")


# ========== Distribution Endpoints ==========

@router.post("/distributions", response_model=DistributionResponse)
async def record_distribution(
    request: DistributionRequest,
    photo: UploadFile = File(...),
    signature: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Record distribution to beneficiary"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    if current_user.role not in ["admin", "distributor"]:
        raise HTTPException(status_code=403, detail="Distributor role required")
    
    try:
        # Read files
        photo_data = await photo.read()
        signature_data = await signature.read()
        
        # Upload to IPFS
        photo_hash = ipfs_service.upload_file(photo_data, photo.filename)
        signature_hash = ipfs_service.upload_file(signature_data, signature.filename)
        
        # Record distribution
        result = blockchain_service.record_distribution(
            batch_id=request.batch_id,
            distributor_name=request.distributor_name,
            recipient_type=request.recipient_type,
            recipient_id=request.recipient_id,
            recipient_name=request.recipient_name,
            recipient_location=request.recipient_location,
            quantity=request.quantity,
            unit=request.unit,
            photo_data=photo_data,
            signature_data=signature_data,
            beneficiary_count=request.beneficiary_count
        )
        
        # Get full distribution details
        distribution = blockchain_service.get_distribution(result['distribution_id'])
        
        return DistributionResponse(
            distribution_id=distribution['id'],
            batch_id=distribution['batch_id'],
            distributor_name=distribution['distributor_name'],
            recipient_id=distribution['recipient_id'],
            recipient_name=distribution['recipient_name'],
            recipient_location=distribution['recipient_location'],
            quantity=distribution['quantity'],
            unit=distribution['unit'],
            timestamp=distribution['timestamp'],
            beneficiary_count=distribution['beneficiary_count'],
            verified=distribution['verified'],
            transaction_hash=result['transaction_hash'],
            photo_url=ipfs_service.get_file_url(photo_hash)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record distribution: {str(e)}")


@router.post("/distributions/{distribution_id}/verify")
async def verify_distribution(
    distribution_id: int,
    current_user: User = Depends(get_current_user)
):
    """Verify a distribution"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    if current_user.role not in ["admin", "verifier"]:
        raise HTTPException(status_code=403, detail="Verifier role required")
    
    try:
        result = blockchain_service.verify_distribution(distribution_id)
        return {
            "message": "Distribution verified",
            "distribution_id": distribution_id,
            "transaction_hash": result['transaction_hash']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.get("/distributions/{distribution_id}")
async def get_distribution(distribution_id: int):
    """Get distribution details"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        distribution = blockchain_service.get_distribution(distribution_id)
        return distribution
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Distribution not found: {str(e)}")


@router.get("/beneficiaries/{beneficiary_id}")
async def get_beneficiary(beneficiary_id: str):
    """Get beneficiary details and history"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        beneficiary = blockchain_service.get_beneficiary(beneficiary_id)
        return beneficiary
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Beneficiary not found: {str(e)}")


@router.post("/beneficiaries/{beneficiary_id}/flag")
async def flag_beneficiary(
    beneficiary_id: str,
    reason: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Flag beneficiary for duplicate/fraud"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    if current_user.role not in ["admin", "verifier"]:
        raise HTTPException(status_code=403, detail="Verifier role required")
    
    try:
        result = blockchain_service.flag_beneficiary(beneficiary_id, reason)
        return {
            "message": "Beneficiary flagged",
            "beneficiary_id": beneficiary_id,
            "reason": reason,
            "transaction_hash": result['transaction_hash']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flag beneficiary: {str(e)}")


# ========== Analytics Endpoints ==========

@router.get("/analytics/overview")
async def get_supply_chain_overview():
    """Get overall supply chain statistics"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        stats = blockchain_service.get_supply_chain_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/analytics/material/{material_type}")
async def get_material_stats(material_type: str):
    """Get statistics for specific material type"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        donated = blockchain_service.get_total_donations_by_type(material_type)
        distributed = blockchain_service.get_total_distributed_by_type(material_type)
        
        return {
            "material_type": material_type,
            "total_donated": donated,
            "total_distributed": distributed,
            "in_pipeline": donated - distributed,
            "distribution_percentage": (distributed / donated * 100) if donated > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/health")
async def blockchain_health():
    """Check blockchain service health"""
    if not blockchain_service:
        return {
            "status": "unavailable",
            "connected": False,
            "message": "Blockchain service not initialized"
        }
    
    return {
        "status": "healthy" if blockchain_service.is_connected() else "disconnected",
        "connected": blockchain_service.is_connected(),
        "current_block": blockchain_service.get_block_number() if blockchain_service.is_connected() else None
    }


# Initialize blockchain service on startup
def init_blockchain_service(
    provider_url: str,
    contract_address: str,
    contract_abi_path: str,
    private_key: Optional[str] = None
):
    """Initialize blockchain service (called from main.py)"""
    global blockchain_service
    blockchain_service = BlockchainService(
        provider_url=provider_url,
        contract_address=contract_address,
        contract_abi_path=contract_abi_path,
        private_key=private_key
    )

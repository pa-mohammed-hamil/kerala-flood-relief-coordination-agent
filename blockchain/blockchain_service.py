"""
Blockchain Service for Relief Supply Chain
Manages interaction with Ethereum/Polygon blockchain for transparent tracking
"""
from web3 import Web3
from web3.middleware import geth_poa_middleware
from typing import Dict, List, Optional, Tuple
import json
import hashlib
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BlockchainService:
    """Main service for blockchain interactions"""
    
    def __init__(
        self,
        provider_url: str,
        contract_address: str,
        contract_abi_path: str,
        private_key: Optional[str] = None
    ):
        """
        Initialize blockchain service
        
        Args:
            provider_url: RPC endpoint (e.g., Infura, Alchemy, local Ganache)
            contract_address: Deployed smart contract address
            contract_abi_path: Path to contract ABI JSON
            private_key: Private key for signing transactions (optional for read-only)
        """
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        
        # For Polygon/BSC compatibility
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Load contract
        with open(contract_abi_path, 'r') as f:
            contract_abi = json.load(f)
        
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=contract_abi
        )
        
        self.private_key = private_key
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
        else:
            self.account = None
        
        logger.info(f"Blockchain service initialized. Connected: {self.w3.is_connected()}")
    
    def is_connected(self) -> bool:
        """Check blockchain connection"""
        return self.w3.is_connected()
    
    def get_block_number(self) -> int:
        """Get current block number"""
        return self.w3.eth.block_number
    
    # ==================== Donation Functions ====================
    
    def record_donation(
        self,
        donor_name: str,
        donor_type: str,
        material_type: str,
        quantity: int,
        unit: str,
        location: str,
        certificate_data: bytes
    ) -> Dict:
        """
        Record a donation on blockchain
        
        Args:
            donor_name: Name of donor
            donor_type: "individual", "organization", "government"
            material_type: "food", "water", "medicine", "clothing", "cash"
            quantity: Amount donated
            unit: Unit of measurement
            location: Source location
            certificate_data: Donation certificate (PDF, image)
        
        Returns:
            Transaction receipt with donation ID
        """
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        # Generate certificate hash (would upload to IPFS in production)
        certificate_hash = self._generate_hash(certificate_data)
        
        # Build transaction
        function = self.contract.functions.recordDonation(
            donor_name,
            donor_type,
            material_type,
            quantity,
            unit,
            location,
            certificate_hash
        )
        
        tx_receipt = self._send_transaction(function)
        
        # Extract donation ID from event logs
        donation_id = self._get_event_value(
            tx_receipt,
            'DonationRecorded',
            'donationId'
        )
        
        logger.info(f"Donation recorded: ID {donation_id}")
        
        return {
            'donation_id': donation_id,
            'transaction_hash': tx_receipt['transactionHash'].hex(),
            'block_number': tx_receipt['blockNumber'],
            'gas_used': tx_receipt['gasUsed']
        }
    
    def verify_donation(self, donation_id: int) -> Dict:
        """Verify a donation (verifier only)"""
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        function = self.contract.functions.verifyDonation(donation_id)
        tx_receipt = self._send_transaction(function)
        
        logger.info(f"Donation {donation_id} verified")
        
        return {
            'donation_id': donation_id,
            'transaction_hash': tx_receipt['transactionHash'].hex(),
            'verified': True
        }
    
    def get_donation(self, donation_id: int) -> Dict:
        """Get donation details"""
        donation = self.contract.functions.getDonation(donation_id).call()
        
        return {
            'id': donation[0],
            'donor': donation[1],
            'donor_name': donation[2],
            'donor_type': donation[3],
            'amount': donation[4],
            'material_type': donation[5],
            'quantity': donation[6],
            'unit': donation[7],
            'timestamp': datetime.fromtimestamp(donation[8]),
            'location': donation[9],
            'certificate_hash': donation[10].hex(),
            'verified': donation[11],
            'verified_by': donation[12]
        }
    
    def get_donor_donations(self, donor_address: str) -> List[int]:
        """Get all donations by a donor"""
        return self.contract.functions.getDonorDonations(
            Web3.to_checksum_address(donor_address)
        ).call()
    
    # ==================== Batch Functions ====================
    
    def create_batch(
        self,
        donation_ids: List[int],
        batch_number: str,
        material_type: str,
        total_quantity: int,
        unit: str,
        warehouse_location: str,
        qr_code_data: bytes
    ) -> Dict:
        """
        Create supply batch from donations
        
        Args:
            donation_ids: List of source donation IDs
            batch_number: Unique batch identifier (e.g., "BATCH-2024-001")
            material_type: Type of material in batch
            total_quantity: Total quantity in batch
            unit: Unit of measurement
            warehouse_location: Current warehouse location
            qr_code_data: QR code image data
        
        Returns:
            Transaction receipt with batch ID
        """
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        qr_hash = self._generate_hash(qr_code_data)
        
        function = self.contract.functions.createBatch(
            donation_ids,
            batch_number,
            material_type,
            total_quantity,
            unit,
            warehouse_location,
            qr_hash
        )
        
        tx_receipt = self._send_transaction(function)
        
        batch_id = self._get_event_value(tx_receipt, 'BatchCreated', 'batchId')
        
        logger.info(f"Batch created: ID {batch_id}, Number {batch_number}")
        
        return {
            'batch_id': batch_id,
            'batch_number': batch_number,
            'transaction_hash': tx_receipt['transactionHash'].hex(),
            'block_number': tx_receipt['blockNumber']
        }
    
    def update_batch_status(self, batch_id: int, new_status: int) -> Dict:
        """
        Update batch status
        
        Status codes:
        0 = Created
        1 = InTransit
        2 = AtWarehouse
        3 = ReadyForDistribution
        4 = PartiallyDistributed
        5 = FullyDistributed
        """
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        function = self.contract.functions.updateBatchStatus(batch_id, new_status)
        tx_receipt = self._send_transaction(function)
        
        logger.info(f"Batch {batch_id} status updated to {new_status}")
        
        return {
            'batch_id': batch_id,
            'new_status': new_status,
            'transaction_hash': tx_receipt['transactionHash'].hex()
        }
    
    def add_checkpoint(self, batch_id: int, checkpoint: str) -> Dict:
        """Add GPS checkpoint to batch transit"""
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        function = self.contract.functions.addCheckpoint(batch_id, checkpoint)
        tx_receipt = self._send_transaction(function)
        
        logger.info(f"Checkpoint added to batch {batch_id}: {checkpoint}")
        
        return {
            'batch_id': batch_id,
            'checkpoint': checkpoint,
            'transaction_hash': tx_receipt['transactionHash'].hex()
        }
    
    def get_batch(self, batch_id: int) -> Dict:
        """Get batch details"""
        batch = self.contract.functions.getBatch(batch_id).call()
        
        return {
            'id': batch[0],
            'donation_ids': batch[1],
            'batch_number': batch[2],
            'material_type': batch[3],
            'total_quantity': batch[4],
            'unit': batch[5],
            'created_at': datetime.fromtimestamp(batch[6]),
            'warehouse': batch[7],
            'warehouse_location': batch[8],
            'status': batch[9],
            'qr_code_hash': batch[10].hex()
        }
    
    def get_batch_checkpoints(self, batch_id: int) -> List[Dict]:
        """Get all checkpoints for a batch"""
        checkpoints, timestamps = self.contract.functions.getBatchCheckpoints(
            batch_id
        ).call()
        
        return [
            {
                'checkpoint': checkpoint,
                'timestamp': datetime.fromtimestamp(timestamp)
            }
            for checkpoint, timestamp in zip(checkpoints, timestamps)
        ]
    
    # ==================== Distribution Functions ====================
    
    def record_distribution(
        self,
        batch_id: int,
        distributor_name: str,
        recipient_type: str,
        recipient_id: str,
        recipient_name: str,
        recipient_location: str,
        quantity: int,
        unit: str,
        photo_data: bytes,
        signature_data: bytes,
        beneficiary_count: int
    ) -> Dict:
        """
        Record distribution to beneficiary
        
        Args:
            batch_id: Source batch ID
            distributor_name: Name of distributor
            recipient_type: "shelter", "individual", "community"
            recipient_id: Unique ID (Aadhaar, shelter ID)
            recipient_name: Recipient name
            recipient_location: Distribution location
            quantity: Quantity distributed
            unit: Unit of measurement
            photo_data: Photo evidence
            signature_data: Digital signature
            beneficiary_count: Number of people benefited
        
        Returns:
            Transaction receipt with distribution ID
        """
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        photo_hash = self._generate_hash(photo_data)
        signature_hash = self._generate_hash(signature_data)
        
        function = self.contract.functions.recordDistribution(
            batch_id,
            distributor_name,
            recipient_type,
            recipient_id,
            recipient_name,
            recipient_location,
            quantity,
            unit,
            photo_hash,
            signature_hash,
            beneficiary_count
        )
        
        tx_receipt = self._send_transaction(function)
        
        distribution_id = self._get_event_value(
            tx_receipt,
            'DistributionRecorded',
            'distributionId'
        )
        
        logger.info(f"Distribution recorded: ID {distribution_id}")
        
        return {
            'distribution_id': distribution_id,
            'batch_id': batch_id,
            'recipient_id': recipient_id,
            'transaction_hash': tx_receipt['transactionHash'].hex(),
            'beneficiary_count': beneficiary_count
        }
    
    def verify_distribution(self, distribution_id: int) -> Dict:
        """Verify a distribution (verifier only)"""
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        function = self.contract.functions.verifyDistribution(distribution_id)
        tx_receipt = self._send_transaction(function)
        
        logger.info(f"Distribution {distribution_id} verified")
        
        return {
            'distribution_id': distribution_id,
            'transaction_hash': tx_receipt['transactionHash'].hex(),
            'verified': True
        }
    
    def get_distribution(self, distribution_id: int) -> Dict:
        """Get distribution details"""
        dist = self.contract.functions.getDistribution(distribution_id).call()
        
        return {
            'id': dist[0],
            'batch_id': dist[1],
            'distributor': dist[2],
            'distributor_name': dist[3],
            'recipient_type': dist[4],
            'recipient_id': dist[5],
            'recipient_name': dist[6],
            'recipient_location': dist[7],
            'quantity': dist[8],
            'unit': dist[9],
            'timestamp': datetime.fromtimestamp(dist[10]),
            'photo_hash': dist[11].hex(),
            'signature_hash': dist[12].hex(),
            'verified': dist[13],
            'verified_by': dist[14],
            'beneficiary_count': dist[15]
        }
    
    def get_beneficiary(self, beneficiary_id: str) -> Dict:
        """Get beneficiary details"""
        ben = self.contract.functions.getBeneficiary(beneficiary_id).call()
        
        return {
            'id': ben[0],
            'name': ben[1],
            'location': ben[2],
            'distribution_ids': ben[3],
            'total_items_received': ben[4],
            'last_received_at': datetime.fromtimestamp(ben[5]) if ben[5] > 0 else None,
            'flagged_duplicate': ben[6]
        }
    
    def flag_beneficiary(self, beneficiary_id: str, reason: str) -> Dict:
        """Flag beneficiary for duplicate/fraud detection"""
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        function = self.contract.functions.flagBeneficiary(beneficiary_id, reason)
        tx_receipt = self._send_transaction(function)
        
        logger.warning(f"Beneficiary {beneficiary_id} flagged: {reason}")
        
        return {
            'beneficiary_id': beneficiary_id,
            'reason': reason,
            'transaction_hash': tx_receipt['transactionHash'].hex()
        }
    
    # ==================== Analytics Functions ====================
    
    def get_total_donations_by_type(self, material_type: str) -> int:
        """Get total verified donations by material type"""
        return self.contract.functions.getTotalDonationsByType(material_type).call()
    
    def get_total_distributed_by_type(self, material_type: str) -> int:
        """Get total distributed by material type"""
        return self.contract.functions.getTotalDistributedByType(material_type).call()
    
    def get_supply_chain_stats(self) -> Dict:
        """Get overall supply chain statistics"""
        donation_count = self.contract.functions.donationCount().call()
        batch_count = self.contract.functions.batchCount().call()
        distribution_count = self.contract.functions.distributionCount().call()
        
        material_types = ["food", "water", "medicine", "clothing", "cash"]
        
        donations_by_type = {}
        distributed_by_type = {}
        
        for mat_type in material_types:
            donations_by_type[mat_type] = self.get_total_donations_by_type(mat_type)
            distributed_by_type[mat_type] = self.get_total_distributed_by_type(mat_type)
        
        return {
            'total_donations': donation_count,
            'total_batches': batch_count,
            'total_distributions': distribution_count,
            'donations_by_type': donations_by_type,
            'distributed_by_type': distributed_by_type,
            'current_block': self.get_block_number()
        }
    
    # ==================== Helper Functions ====================
    
    def _generate_hash(self, data: bytes) -> bytes:
        """Generate keccak256 hash (Ethereum-compatible)"""
        return Web3.keccak(data)
    
    def _send_transaction(self, function) -> Dict:
        """Send transaction to blockchain"""
        if not self.account:
            raise ValueError("No account configured for transactions")
        
        # Build transaction
        tx = function.build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 500000,  # Adjust based on function
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
        
        # Send transaction
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt['status'] != 1:
            raise Exception("Transaction failed")
        
        return tx_receipt
    
    def _get_event_value(self, tx_receipt: Dict, event_name: str, param_name: str):
        """Extract value from event logs"""
        event = getattr(self.contract.events, event_name)()
        logs = event.process_receipt(tx_receipt)
        
        if logs:
            return logs[0]['args'][param_name]
        return None
    
    def get_transaction_url(self, tx_hash: str, network: str = "polygon") -> str:
        """Get block explorer URL for transaction"""
        explorers = {
            "ethereum": f"https://etherscan.io/tx/{tx_hash}",
            "polygon": f"https://polygonscan.com/tx/{tx_hash}",
            "bsc": f"https://bscscan.com/tx/{tx_hash}",
            "ganache": f"Local: {tx_hash}"
        }
        return explorers.get(network, f"Unknown network: {tx_hash}")


class IPFSService:
    """Service for storing documents on IPFS"""
    
    def __init__(self, ipfs_gateway: str = "https://ipfs.io/ipfs/"):
        """
        Initialize IPFS service
        
        Args:
            ipfs_gateway: IPFS gateway URL
        """
        self.gateway = ipfs_gateway
        # In production, would use ipfshttpclient library
    
    def upload_file(self, file_data: bytes, filename: str) -> str:
        """
        Upload file to IPFS
        
        Returns:
            IPFS hash (CID)
        """
        # Placeholder - would use actual IPFS upload
        # import ipfshttpclient
        # client = ipfshttpclient.connect('/dns/ipfs.infura.io/tcp/5001/https')
        # res = client.add(file_data)
        # return res['Hash']
        
        # For demo, return mock hash
        file_hash = hashlib.sha256(file_data).hexdigest()
        logger.info(f"IPFS upload simulated: {filename} -> {file_hash}")
        return f"Qm{file_hash[:44]}"  # Mock IPFS CID
    
    def get_file_url(self, ipfs_hash: str) -> str:
        """Get public URL for IPFS file"""
        return f"{self.gateway}{ipfs_hash}"


# Batch status enum for convenience
class BatchStatus:
    CREATED = 0
    IN_TRANSIT = 1
    AT_WAREHOUSE = 2
    READY_FOR_DISTRIBUTION = 3
    PARTIALLY_DISTRIBUTED = 4
    FULLY_DISTRIBUTED = 5

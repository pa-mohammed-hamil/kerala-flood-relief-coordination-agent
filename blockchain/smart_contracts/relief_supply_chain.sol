// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Kerala Flood Relief Supply Chain Smart Contract
 * Transparent tracking of donations, supplies, and distributions
 */

contract ReliefSupplyChain {
    
    // ==================== Structs ====================
    
    struct Donation {
        uint256 id;
        address donor;
        string donorName;
        string donorType; // "individual", "organization", "government"
        uint256 amount; // In Wei for ETH donations
        string materialType; // "food", "water", "medicine", "clothing", "cash"
        uint256 quantity;
        string unit; // "kg", "liters", "pieces", "rupees"
        uint256 timestamp;
        string location; // Source location
        bytes32 certificateHash; // IPFS hash of donation certificate
        bool verified;
        address verifiedBy;
    }
    
    struct SupplyBatch {
        uint256 id;
        uint256[] donationIds; // Source donations
        string batchNumber;
        string materialType;
        uint256 totalQuantity;
        string unit;
        uint256 createdAt;
        address warehouse;
        string warehouseLocation;
        BatchStatus status;
        bytes32 qrCodeHash; // QR code for batch tracking
        string[] checkpoints; // GPS waypoints during transit
        uint256[] checkpointTimestamps;
    }
    
    struct Distribution {
        uint256 id;
        uint256 batchId;
        address distributor;
        string distributorName;
        string recipientType; // "shelter", "individual", "community"
        string recipientId; // Shelter ID or individual ID
        string recipientName;
        string recipientLocation;
        uint256 quantity;
        string unit;
        uint256 timestamp;
        bytes32 photoHash; // IPFS hash of distribution photo
        bytes32 signatureHash; // Digital signature of recipient
        bool verified;
        address verifiedBy;
        uint256 beneficiaryCount; // Number of people benefited
    }
    
    struct Beneficiary {
        string id; // Aadhaar or unique ID
        string name;
        string location;
        uint256[] distributionIds;
        uint256 totalItemsReceived;
        uint256 lastReceivedAt;
        bool flaggedDuplicate;
    }
    
    enum BatchStatus {
        Created,
        InTransit,
        AtWarehouse,
        ReadyForDistribution,
        PartiallyDistributed,
        FullyDistributed
    }
    
    // ==================== State Variables ====================
    
    address public admin;
    mapping(address => bool) public verifiers;
    mapping(address => bool) public warehouses;
    mapping(address => bool) public distributors;
    
    uint256 public donationCount;
    uint256 public batchCount;
    uint256 public distributionCount;
    
    mapping(uint256 => Donation) public donations;
    mapping(uint256 => SupplyBatch) public batches;
    mapping(uint256 => Distribution) public distributions;
    mapping(string => Beneficiary) public beneficiaries;
    
    // Indexing for queries
    mapping(address => uint256[]) public donorDonations;
    mapping(string => uint256[]) public materialTypeBatches;
    mapping(string => uint256[]) public beneficiaryDistributions;
    
    // ==================== Events ====================
    
    event DonationRecorded(
        uint256 indexed donationId,
        address indexed donor,
        string materialType,
        uint256 quantity,
        uint256 timestamp
    );
    
    event DonationVerified(
        uint256 indexed donationId,
        address indexed verifier,
        uint256 timestamp
    );
    
    event BatchCreated(
        uint256 indexed batchId,
        string batchNumber,
        string materialType,
        uint256 totalQuantity,
        uint256 timestamp
    );
    
    event BatchStatusChanged(
        uint256 indexed batchId,
        BatchStatus oldStatus,
        BatchStatus newStatus,
        uint256 timestamp
    );
    
    event CheckpointAdded(
        uint256 indexed batchId,
        string checkpoint,
        uint256 timestamp
    );
    
    event DistributionRecorded(
        uint256 indexed distributionId,
        uint256 indexed batchId,
        string recipientId,
        uint256 quantity,
        uint256 beneficiaryCount,
        uint256 timestamp
    );
    
    event DistributionVerified(
        uint256 indexed distributionId,
        address indexed verifier,
        uint256 timestamp
    );
    
    event BeneficiaryFlagged(
        string indexed beneficiaryId,
        string reason,
        uint256 timestamp
    );
    
    // ==================== Modifiers ====================
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }
    
    modifier onlyVerifier() {
        require(verifiers[msg.sender], "Only verifiers can perform this action");
        _;
    }
    
    modifier onlyWarehouse() {
        require(warehouses[msg.sender], "Only warehouses can perform this action");
        _;
    }
    
    modifier onlyDistributor() {
        require(distributors[msg.sender], "Only distributors can perform this action");
        _;
    }
    
    // ==================== Constructor ====================
    
    constructor() {
        admin = msg.sender;
        verifiers[msg.sender] = true;
        warehouses[msg.sender] = true;
        distributors[msg.sender] = true;
    }
    
    // ==================== Admin Functions ====================
    
    function addVerifier(address _verifier) external onlyAdmin {
        verifiers[_verifier] = true;
    }
    
    function removeVerifier(address _verifier) external onlyAdmin {
        verifiers[_verifier] = false;
    }
    
    function addWarehouse(address _warehouse) external onlyAdmin {
        warehouses[_warehouse] = true;
    }
    
    function addDistributor(address _distributor) external onlyAdmin {
        distributors[_distributor] = true;
    }
    
    // ==================== Donation Functions ====================
    
    function recordDonation(
        string memory _donorName,
        string memory _donorType,
        string memory _materialType,
        uint256 _quantity,
        string memory _unit,
        string memory _location,
        bytes32 _certificateHash
    ) external returns (uint256) {
        donationCount++;
        
        donations[donationCount] = Donation({
            id: donationCount,
            donor: msg.sender,
            donorName: _donorName,
            donorType: _donorType,
            amount: msg.value,
            materialType: _materialType,
            quantity: _quantity,
            unit: _unit,
            timestamp: block.timestamp,
            location: _location,
            certificateHash: _certificateHash,
            verified: false,
            verifiedBy: address(0)
        });
        
        donorDonations[msg.sender].push(donationCount);
        
        emit DonationRecorded(
            donationCount,
            msg.sender,
            _materialType,
            _quantity,
            block.timestamp
        );
        
        return donationCount;
    }
    
    function verifyDonation(uint256 _donationId) external onlyVerifier {
        require(_donationId > 0 && _donationId <= donationCount, "Invalid donation ID");
        require(!donations[_donationId].verified, "Already verified");
        
        donations[_donationId].verified = true;
        donations[_donationId].verifiedBy = msg.sender;
        
        emit DonationVerified(_donationId, msg.sender, block.timestamp);
    }
    
    // ==================== Batch Functions ====================
    
    function createBatch(
        uint256[] memory _donationIds,
        string memory _batchNumber,
        string memory _materialType,
        uint256 _totalQuantity,
        string memory _unit,
        string memory _warehouseLocation,
        bytes32 _qrCodeHash
    ) external onlyWarehouse returns (uint256) {
        // Verify all donations exist and are verified
        for (uint256 i = 0; i < _donationIds.length; i++) {
            require(donations[_donationIds[i]].verified, "All donations must be verified");
        }
        
        batchCount++;
        
        batches[batchCount] = SupplyBatch({
            id: batchCount,
            donationIds: _donationIds,
            batchNumber: _batchNumber,
            materialType: _materialType,
            totalQuantity: _totalQuantity,
            unit: _unit,
            createdAt: block.timestamp,
            warehouse: msg.sender,
            warehouseLocation: _warehouseLocation,
            status: BatchStatus.Created,
            qrCodeHash: _qrCodeHash,
            checkpoints: new string[](0),
            checkpointTimestamps: new uint256[](0)
        });
        
        materialTypeBatches[_materialType].push(batchCount);
        
        emit BatchCreated(
            batchCount,
            _batchNumber,
            _materialType,
            _totalQuantity,
            block.timestamp
        );
        
        return batchCount;
    }
    
    function updateBatchStatus(
        uint256 _batchId,
        BatchStatus _newStatus
    ) external onlyWarehouse {
        require(_batchId > 0 && _batchId <= batchCount, "Invalid batch ID");
        
        BatchStatus oldStatus = batches[_batchId].status;
        batches[_batchId].status = _newStatus;
        
        emit BatchStatusChanged(_batchId, oldStatus, _newStatus, block.timestamp);
    }
    
    function addCheckpoint(
        uint256 _batchId,
        string memory _checkpoint
    ) external onlyWarehouse {
        require(_batchId > 0 && _batchId <= batchCount, "Invalid batch ID");
        
        batches[_batchId].checkpoints.push(_checkpoint);
        batches[_batchId].checkpointTimestamps.push(block.timestamp);
        
        emit CheckpointAdded(_batchId, _checkpoint, block.timestamp);
    }
    
    // ==================== Distribution Functions ====================
    
    function recordDistribution(
        uint256 _batchId,
        string memory _distributorName,
        string memory _recipientType,
        string memory _recipientId,
        string memory _recipientName,
        string memory _recipientLocation,
        uint256 _quantity,
        string memory _unit,
        bytes32 _photoHash,
        bytes32 _signatureHash,
        uint256 _beneficiaryCount
    ) external onlyDistributor returns (uint256) {
        require(_batchId > 0 && _batchId <= batchCount, "Invalid batch ID");
        require(
            batches[_batchId].status == BatchStatus.ReadyForDistribution ||
            batches[_batchId].status == BatchStatus.PartiallyDistributed,
            "Batch not ready for distribution"
        );
        
        distributionCount++;
        
        distributions[distributionCount] = Distribution({
            id: distributionCount,
            batchId: _batchId,
            distributor: msg.sender,
            distributorName: _distributorName,
            recipientType: _recipientType,
            recipientId: _recipientId,
            recipientName: _recipientName,
            recipientLocation: _recipientLocation,
            quantity: _quantity,
            unit: _unit,
            timestamp: block.timestamp,
            photoHash: _photoHash,
            signatureHash: _signatureHash,
            verified: false,
            verifiedBy: address(0),
            beneficiaryCount: _beneficiaryCount
        });
        
        // Update beneficiary record
        if (bytes(beneficiaries[_recipientId].id).length == 0) {
            beneficiaries[_recipientId] = Beneficiary({
                id: _recipientId,
                name: _recipientName,
                location: _recipientLocation,
                distributionIds: new uint256[](0),
                totalItemsReceived: _quantity,
                lastReceivedAt: block.timestamp,
                flaggedDuplicate: false
            });
        } else {
            beneficiaries[_recipientId].totalItemsReceived += _quantity;
            beneficiaries[_recipientId].lastReceivedAt = block.timestamp;
        }
        
        beneficiaries[_recipientId].distributionIds.push(distributionCount);
        beneficiaryDistributions[_recipientId].push(distributionCount);
        
        // Update batch status
        batches[_batchId].status = BatchStatus.PartiallyDistributed;
        
        emit DistributionRecorded(
            distributionCount,
            _batchId,
            _recipientId,
            _quantity,
            _beneficiaryCount,
            block.timestamp
        );
        
        return distributionCount;
    }
    
    function verifyDistribution(uint256 _distributionId) external onlyVerifier {
        require(
            _distributionId > 0 && _distributionId <= distributionCount,
            "Invalid distribution ID"
        );
        require(!distributions[_distributionId].verified, "Already verified");
        
        distributions[_distributionId].verified = true;
        distributions[_distributionId].verifiedBy = msg.sender;
        
        emit DistributionVerified(_distributionId, msg.sender, block.timestamp);
    }
    
    function flagBeneficiary(
        string memory _beneficiaryId,
        string memory _reason
    ) external onlyVerifier {
        require(bytes(beneficiaries[_beneficiaryId].id).length > 0, "Beneficiary not found");
        
        beneficiaries[_beneficiaryId].flaggedDuplicate = true;
        
        emit BeneficiaryFlagged(_beneficiaryId, _reason, block.timestamp);
    }
    
    // ==================== Query Functions ====================
    
    function getDonation(uint256 _donationId) external view returns (Donation memory) {
        return donations[_donationId];
    }
    
    function getBatch(uint256 _batchId) external view returns (SupplyBatch memory) {
        return batches[_batchId];
    }
    
    function getDistribution(uint256 _distributionId) external view returns (Distribution memory) {
        return distributions[_distributionId];
    }
    
    function getBeneficiary(string memory _beneficiaryId) external view returns (Beneficiary memory) {
        return beneficiaries[_beneficiaryId];
    }
    
    function getDonorDonations(address _donor) external view returns (uint256[] memory) {
        return donorDonations[_donor];
    }
    
    function getBatchCheckpoints(uint256 _batchId) external view returns (
        string[] memory checkpoints,
        uint256[] memory timestamps
    ) {
        return (batches[_batchId].checkpoints, batches[_batchId].checkpointTimestamps);
    }
    
    function getBeneficiaryDistributions(string memory _beneficiaryId) external view returns (uint256[] memory) {
        return beneficiaryDistributions[_beneficiaryId];
    }
    
    // ==================== Analytics Functions ====================
    
    function getTotalDonationsByType(string memory _materialType) external view returns (uint256 total) {
        for (uint256 i = 1; i <= donationCount; i++) {
            if (
                keccak256(bytes(donations[i].materialType)) == keccak256(bytes(_materialType)) &&
                donations[i].verified
            ) {
                total += donations[i].quantity;
            }
        }
        return total;
    }
    
    function getTotalDistributedByType(string memory _materialType) external view returns (uint256 total) {
        for (uint256 i = 1; i <= distributionCount; i++) {
            if (
                keccak256(bytes(batches[distributions[i].batchId].materialType)) == keccak256(bytes(_materialType)) &&
                distributions[i].verified
            ) {
                total += distributions[i].quantity;
            }
        }
        return total;
    }
    
    function getTotalBeneficiaries() external view returns (uint256 count) {
        // Note: This is inefficient for large datasets. In production, use off-chain indexing.
        // Placeholder for demonstration
        return 0; // Would need separate counter in production
    }
}

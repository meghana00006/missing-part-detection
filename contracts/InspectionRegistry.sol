// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract InspectionRegistry {
    struct Inspection {
        string imageHash;
        string result;
        uint256 timestamp;
        address inspector;
    }
    
    Inspection[] public inspections;
    
    event InspectionRecorded(
        uint256 indexed id,
        string imageHash,
        string result,
        uint256 timestamp,
        address indexed inspector
    );

    function addInspection(string memory _imageHash, string memory _result) public {
        inspections.push(Inspection({
            imageHash: _imageHash,
            result: _result,
            timestamp: block.timestamp,
            inspector: msg.sender
        }));
        
        emit InspectionRecorded(inspections.length - 1, _imageHash, _result, block.timestamp, msg.sender);
    }
    
    function getTotalInspections() public view returns (uint256) {
        return inspections.length;
    }

    // ⚠️ THIS IS THE NEW FUNCTION WE NEEDED
    function getInspection(uint256 _id) public view returns (string memory, string memory, uint256, address) {
        require(_id < inspections.length, "Inspection does not exist");
        Inspection memory i = inspections[_id];
        return (i.imageHash, i.result, i.timestamp, i.inspector);
    }
}
import json
import os
from web3 import Web3
from solcx import compile_standard, install_solc

import os

ALCHEMY_URL = os.getenv("ALCHEMY_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")  # <--- PASTE YOUR PRIVATE KEY HERE
# -----------------------------

print("Installing solc...")
install_solc('0.8.0')

print("Compiling contract...")
with open("./contracts/InspectionRegistry.sol", "r", encoding="utf-8") as file:
    contract_source = file.read()

compiled_sol = compile_standard({
    "language": "Solidity",
    "sources": {"InspectionRegistry.sol": {"content": contract_source}},
    "settings": {"outputSelection": {"*": {"*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]}}},
}, solc_version="0.8.0")

abi = compiled_sol['contracts']['InspectionRegistry.sol']['InspectionRegistry']['abi']
bytecode = compiled_sol['contracts']['InspectionRegistry.sol']['InspectionRegistry']['evm']['bytecode']['object']

# Connect to Sepolia
w3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))
account = w3.eth.account.from_key(PRIVATE_KEY)
w3.eth.default_account = account.address

print(f"Deploying to Sepolia from account: {account.address}...")
InspectionRegistry = w3.eth.contract(abi=abi, bytecode=bytecode)

# Build and Sign Transaction
tx = InspectionRegistry.constructor().build_transaction({
    'from': account.address,
    'nonce': w3.eth.get_transaction_count(account.address),
    'gas': 3000000,
    'gasPrice': w3.eth.gas_price
})

signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
print("Waiting for deployment...")
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print(f"✅ Contract deployed at address: {tx_receipt.contractAddress}")

os.makedirs("blockchain", exist_ok=True)
with open("blockchain/contract_info.json", "w") as f:
    json.dump({"address": tx_receipt.contractAddress, "abi": abi}, f, indent=4)
print("✅ Saved contract info to blockchain/contract_info.json")
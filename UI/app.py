import streamlit as st
import cv2
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import os
import hashlib
from pathlib import Path
from web3 import Web3
import datetime
import pandas as pd

# ==================== PAGE CONFIG (MUST BE FIRST) ====================
st.set_page_config(page_title="Industrial Component Inspector", page_icon="🔧", layout="wide")

# ==================== PATHS & CONFIG ====================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best (2).pt"
REFERENCE_PATH = BASE_DIR / "references" / "expected_parts.json"
CONTRACT_INFO_PATH = BASE_DIR / "blockchain" / "contract_info.json"

# --- LOCAL KEYS ---
ALCHEMY_URL = st.secrets.get("ALCHEMY_URL", "https://eth-sepolia.g.alchemy.com/v2/alch_aSI0qQQuXeEt3nJFG2KNh")
PRIVATE_KEY = st.secrets.get("PRIVATE_KEY", "0x894d1ba595847f7562c4ca324a285c85655e3a3ca4de74e593c...") 

CLASS_NAMES = {0: "bolt", 1: "bearing", 2: "plate", 3: "pin"}
CLASS_COLORS_BGR = {
    "bolt": (0, 255, 0),
    "bearing": (0, 165, 255),
    "plate": (255, 255, 0),
    "pin": (255, 0, 255)
}

if "audit_history" not in st.session_state:
    st.session_state.audit_history = []

# ==================== LOADERS ====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): 
        return None
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_reference():
    if not os.path.exists(REFERENCE_PATH): 
        return {}
    with open(REFERENCE_PATH, "r") as f: 
        return json.load(f)

model = load_model()
REFERENCE = load_reference()

# ==================== BLOCKCHAIN SETUP ====================
@st.cache_resource
def init_blockchain():
    try:
        w3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))
        if not w3.is_connected():
            return None, None, "Could not connect to Sepolia"
            
        if not os.path.exists(CONTRACT_INFO_PATH):
            return None, None, "Contract info not found."
            
        with open(CONTRACT_INFO_PATH, "r") as f:
            contract_info = json.load(f)
            
        contract = w3.eth.contract(address=contract_info["address"], abi=contract_info["abi"])
        return w3, contract, None
    except Exception as e:
        return None, None, str(e)

w3, contract, bc_error = init_blockchain()

def save_to_blockchain(image_array, result_status):
    if not w3 or not contract:
        return None, "Blockchain not connected"
    
    try:
        image_bytes = image_array.tobytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        account = w3.eth.account.from_key(PRIVATE_KEY)
        
        tx = contract.functions.addInspection(image_hash, result_status).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 200000,
            'gasPrice': w3.eth.gas_price
        })
        
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.transactionHash.hex(), None
    except Exception as e:
        return None, str(e)

# ==================== HELPER FUNCTIONS ====================
def detect_parts_yolo(image_np, conf_threshold=0.35):
    results = model(image_np, conf=conf_threshold, verbose=False)[0]
    detected_counts = Counter()
    detected_details = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            class_id = int(cls)
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            detected_counts[class_name] += 1
            x1, y1, x2, y2 = map(int, box[:4])
            detected_details.append({"class_id": class_id, "class_name": class_name, "bbox": [x1, y1, x2, y2], "confidence": float(conf)})
    return detected_counts, detected_details

def draw_detections(image_rgb, detected_details):
    output = image_rgb.copy()
    h, w = output.shape[:2]
    for det in detected_details:
        x1, y1, x2, y2 = det["bbox"]
        bgr = CLASS_COLORS_BGR.get(det["class_name"], (0, 255, 0))
        color = (bgr[2], bgr[1], bgr[0])
        x1, y1 = max(0, min(int(x1), w - 1)), max(0, min(int(y1), h - 1))
        x2, y2 = max(0, min(int(x2), w - 1)), max(0, min(int(y2), h - 1))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        label = f'{det["class_name"]} ({det["confidence"]:.0%})'
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(y1, h_text + 10)
        cv2.rectangle(output, (x1, label_y - h_text - 10), (x1 + w_text, label_y), color, -1)
        cv2.putText(output, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return output

# ==================== UI / CSS ====================
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* ==================== HEADERS & TEXT ANIMATIONS ==================== */
    .main-header { 
        font-size: 3.5rem; 
        text-align: center; 
        font-weight: 900; 
        margin-bottom: 0.2rem; 
        background: -webkit-linear-gradient(45deg, #60a5fa, #a78bfa, #f472b6); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
    }
    .sub-header { 
        font-size: 1.2rem; 
        text-align: center; 
        margin-bottom: 3rem; 
        font-weight: 500;
        background: linear-gradient(270deg, #94a3b8, #60a5fa, #a78bfa, #94a3b8);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .card-title-gradient {
        font-size: 1.3rem; 
        font-weight: 700; 
        margin-bottom: 1rem; 
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #475569;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s linear infinite;
        display: inline-block;
    }

    /* ==================== SLIM HORIZONTAL TABS ==================== */
    div[data-testid="stRadio"] > label { display: none !important; }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        flex-direction: row !important;
        justify-content: center !important;
        gap: 20px !important;
        margin-top: 1rem !important;
        margin-bottom: 2.5rem !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display: none !important; }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 10px 30px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        color: #94a3b8 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        transition: color 0.3s ease !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: #0f172a !important;
        border-color: #60a5fa !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.2) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover p { color: #ffffff !important; }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        border-color: #a78bfa !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p { color: #ffffff !important; }

    /* ==================== UPLOAD BUTTON STYLING ==================== */
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: 1px solid #a78bfa !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
    }
    div[data-testid="stFileUploader"] section {
        border: 1px dashed #475569 !important;
        transition: border-color 0.3s ease !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #60a5fa !important;
        background-color: rgba(96, 165, 250, 0.05) !important;
    }

    /* ==================== SIDEBAR INTERACTIVITY ==================== */
    /* Style the sidebar cards */
    [data-testid="stSidebar"] .card {
        transition: all 0.3s ease;
        border-left: 3px solid transparent;
    }
    [data-testid="stSidebar"] .card:hover {
        transform: translateX(5px);
        border-left: 3px solid #3b82f6;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.15);
    }
    /* Animate the sidebar title text */
    [data-testid="stSidebar"] .card-title {
        transition: color 0.3s ease;
    }
    [data-testid="stSidebar"] .card:hover .card-title {
        color: #60a5fa;
    }

    /* ==================== INTERACTIVE WELCOME HERO ==================== */
    .welcome-hero { 
        text-align: center; 
        padding: 4rem 2rem; 
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-radius: 20px; 
        border: 2px dashed #334155; 
        margin-top: 2rem; 
        transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    .welcome-hero:hover {
        border-color: #60a5fa;
        border-style: solid;
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 15px 35px rgba(96, 165, 250, 0.15), 
                    inset 0 0 20px rgba(96, 165, 250, 0.05);
    }
    .welcome-icon { 
        font-size: 5rem; 
        margin-bottom: 1rem; 
        transition: transform 0.4s ease;
        display: inline-block;
    }
    .welcome-hero:hover .welcome-icon {
        transform: translateY(-10px) scale(1.1);
        animation: floatIcon 2s ease-in-out infinite;
    }
    @keyframes floatIcon {
        0%, 100% { transform: translateY(-10px); }
        50% { transform: translateY(-20px); }
    }
    .welcome-title { 
        font-size: 2.2rem; 
        font-weight: 800; 
        margin-bottom: 1rem; 
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #60a5fa);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: background-position 0.5s ease;
    }
    .welcome-hero:hover .welcome-title {
        animation: shine 3s linear infinite;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    .welcome-text { 
        color: #94a3b8; 
        max-width: 600px; 
        margin: 0 auto; 
        line-height: 1.6; 
        transition: color 0.3s ease;
    }
    .welcome-hero:hover .welcome-text {
        color: #cbd5e1;
    }

    /* ==================== STANDARD CARDS & BOXES ==================== */
    .card { background: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid #334155; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .card-title { font-size: 1.3rem; color: #f1f5f9; font-weight: 600; border-bottom: 2px solid #475569; margin-bottom: 1rem; padding-bottom: 0.5rem; }
    .success-box { background: linear-gradient(135deg, #064e3b 0%, #047857 100%); padding: 1.5rem; border-radius: 12px; border-left: 6px solid #10b981; margin: 1rem 0; box-shadow: 0 0 15px rgba(16, 185, 129, 0.2); }
    .error-box { background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%); padding: 1.5rem; border-radius: 12px; border-left: 6px solid #ef4444; margin: 1rem 0; box-shadow: 0 0 15px rgba(239, 68, 68, 0.2); }
    .part-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: 600; margin: 4px; color: white !important; font-size: 0.95rem; transition: transform 0.2s ease; }
    .part-badge:hover { transform: scale(1.1); }
    .bolt-badge { background-color: #065f46; border: 1px solid #10b981; }
    .bearing-badge { background-color: #92400e; border: 1px solid #f59e0b; }
    .plate-badge { background-color: #1e40af; border: 1px solid #3b82f6; }
    .pin-badge { background-color: #6b21a8; border: 1px solid #a855f7; }
    .divider { height: 2px; background: linear-gradient(to right, transparent, #3b82f6, transparent); margin: 2rem 0; }

    /* ==================== STANDARD BUTTONS ==================== */
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white !important;
        border: 1px solid #a78bfa;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(139, 92, 246, 0.5);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏭 Industrial Component Inspector</h1><p class="sub-header">AI-Powered Quality Control with Immutable Blockchain Audit Trail</p>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown('<div class="card"><div class="card-title">⚙️ AI Settings</div></div>', unsafe_allow_html=True)
    confidence_threshold = st.slider("**Detection Sensitivity**", 0.1, 0.9, 0.35, 0.05)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">⛓️ Web3 Status</div></div>', unsafe_allow_html=True)
    
    if w3 and contract:
        try:
            account = w3.eth.account.from_key(PRIVATE_KEY)
            balance_wei = w3.eth.get_balance(account.address)
            balance_eth = w3.from_wei(balance_wei, 'ether')
            st.success("✅ **Connected to Sepolia**")
            st.markdown(f"""
            <div style="background-color: #064e3b; padding: 12px; border-radius: 8px; font-size: 0.85rem; color: #d1fae5; border: 1px solid #10b981;">
                <b>Wallet:</b> {account.address[:6]}...{account.address[-4:]}<br>
                <b>Balance:</b> {balance_eth:.4f} ETH<br>
                <b>Contract:</b> {contract.address[:6]}...{contract.address[-4:]}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.success("✅ **Connected (Key hidden)**")
    else:
        st.error(f"❌ **Offline**\n\n{bc_error}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">🤖 AI Target Parts</div></div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center;">
            <span class="part-badge bolt-badge">🔩 Bolt</span>
            <span class="part-badge bearing-badge">🏗️ Bearing</span>
            <span class="part-badge plate-badge">📏 Plate</span>
            <span class="part-badge pin-badge">📍 Pin</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">📊 System Status</div></div>', unsafe_allow_html=True)
    if model:
        st.success("✅ **YOLOv8 Model Loaded**")
    else:
        st.error("❌ **Model Missing**")
    if REFERENCE:
        st.success(f"✅ **{len(REFERENCE)} Reference Images Loaded**")
    else:
        st.error("❌ **Refs Missing**")


# ==================== CUSTOM HORIZONTAL TABS ====================
selected_tab = st.radio(
    "Navigation", 
    ["🔍 Inspection Dashboard", "📜 Immutable Audit Log"], 
    label_visibility="collapsed",
    horizontal=True
)

# ==================== TAB 1: INSPECTION DASHBOARD ====================
if selected_tab == "🔍 Inspection Dashboard":
    st.markdown('<div class="card"><div class="card-title-gradient">📤 Upload Image for Inspection</div></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="card"><div class="card-title">📷 Uploaded Image</div></div>', unsafe_allow_html=True)
            st.image(image_np, width="stretch")
        
        if model:
            with st.spinner("🔍 **Running YOLOv8 Analysis...**"):
                detected_counts, detected_details = detect_parts_yolo(image_np, confidence_threshold)
            
            with col2:
                st.markdown('<div class="card"><div class="card-title">📊 Detection Metrics</div></div>', unsafe_allow_html=True)
                if detected_counts:
                    st.markdown("**Parts Distribution:**")
                    st.bar_chart(detected_counts, color="#3b82f6")
                    st.metric("Total Parts Found", sum(detected_counts.values()))
                else:
                    st.warning("⚠️ No parts detected")
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # --- SMART LOGIC (Allows 3 parts missing max) ---
            status = "FAIL"
            missing_parts = Counter()
            expected_counts = Counter()
            match_reason = ""

            if sum(detected_counts.values()) == 0:
                status = "FAIL"
                match_reason = "No valid parts detected in the image."
            else:
                best_ref = None
                best_diff = float('inf')
                
                for ref_name, ref_data in REFERENCE.items():
                    ref_counts = Counter(ref_data.get("part_counts", {}))
                    diff = 0
                    for part in set(list(ref_counts.keys()) + list(detected_counts.keys())):
                        diff += abs(ref_counts.get(part, 0) - detected_counts.get(part, 0))
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_ref = ref_name
                        expected_counts = ref_counts

                total_missing = 0
                for part, exp_count in expected_counts.items():
                    det_count = detected_counts.get(part, 0)
                    if exp_count > det_count:
                        missing_parts[part] = exp_count - det_count
                        total_missing += (exp_count - det_count)
                
                if total_missing <= 3:
                    status = "PASS"
                    match_reason = f"Good match with reference: {best_ref}"
                else:
                    status = "FAIL"
                    match_reason = f"Missing components compared to reference: {best_ref}"

            # ============= DISPLAY RESULTS =============
            st.markdown('<div class="card"><div class="card-title">📝 Inspection Results</div></div>', unsafe_allow_html=True)
            result_image = draw_detections(image_np, detected_details)
            
            rc1, rc2 = st.columns([3, 2])
            with rc1:
                st.image(result_image, width="stretch")
                
            with rc2:
                if status == "PASS":
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### ✅ **INSPECTION PASSED**")
                    st.markdown(f"*{match_reason}*")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("**Verified Components:**")
                    for part, count in detected_counts.items():
                        st.success(f"**{part.title()}:** {count} verified")
                else:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown("### ❌ **INSPECTION FAILED**")
                    st.markdown("**Missing components detected!**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.info(f"**Analysis:** {match_reason}")
                    
                    if missing_parts:
                        st.markdown("**Missing Components:**")
                        for part, count in missing_parts.items():
                            st.error(f"✗ **{part.title()}:** {count} missing")
                    elif not detected_counts:
                        st.error("The AI did not recognize any parts. Please upload a valid assembly image.")

            # ============= BLOCKCHAIN SAVING =============
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### ⛓️ Blockchain Audit Trail")
            
            if not w3 or not contract:
                st.warning("⚠️ Blockchain offline. Cannot save immutable record.")
            else:
                if st.button("💾 Save Inspection to Blockchain"):
                    with st.spinner("🔗 Recording immutable record to Sepolia..."):
                        tx_hash, error = save_to_blockchain(image_np, status)
                        
                    if tx_hash:
                        st.success("✅ **Inspection permanently secured on the blockchain!**")
                        st.code(f"Transaction Hash: {tx_hash}", language="text")
                        st.markdown(f"**[🔍 View Immutable Record on Sepolia Etherscan](https://sepolia.etherscan.io/tx/{tx_hash})**")
                        
                        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        st.session_state.audit_history.insert(0, {
                            "timestamp": current_time,
                            "status": status,
                            "tx_hash": tx_hash
                        })
                    else:
                        st.error(f"❌ Failed to record: {error}")

    else:
        st.markdown("""
        <div class="welcome-hero">
            <div class="welcome-icon">🏭</div>
            <div class="welcome-title">System Ready for Inspection</div>
            <div class="welcome-text">
                Upload an image of an industrial assembly above to begin automated quality control.<br><br>
                Our YOLOv8 AI model will detect bolts, bearings, plates, and pins, compare the results against the reference database, and permanently record the inspection results on the Ethereum Sepolia blockchain.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==================== TAB 2: IMMUTABLE AUDIT LOG ====================
elif selected_tab == "📜 Immutable Audit Log":
    st.markdown('<div class="card"><div class="card-title">📜 Session Immutable Audit Log</div></div>', unsafe_allow_html=True)
    
    if len(st.session_state.audit_history) == 0:
        st.info("No inspections recorded in this session yet. Go to the Dashboard, upload an image, and save it to the blockchain!")
    else:
        df = pd.DataFrame(st.session_state.audit_history)
        df['dt'] = pd.to_datetime(df['timestamp'])
        df['Date'] = df['dt'].dt.strftime('%Y-%m-%d')
        df['Time'] = df['dt'].dt.strftime('%H:%M:%S')
        df['Short Hash'] = df['tx_hash'].apply(lambda x: f"{x[:10]}...{x[-6:]}")
        df = df[['Date', 'Time', 'status', 'Short Hash', 'tx_hash']]
        
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Time": st.column_config.TextColumn("Time", width="small"),
                "status": st.column_config.TextColumn("Inspection Result", width="medium"),
                "Short Hash": st.column_config.TextColumn("Tx Hash (Short)", width="medium"),
                "tx_hash": st.column_config.LinkColumn(
                    "Explorer Link",
                    display_text="🔍 View on Etherscan",
                    width="medium"
                )
            }
        )
from flask import Flask, request, jsonify
from flask_cors import CORS
from models import visits_collection
from datetime import datetime
from plate_ocr import extract_plate_from_base64
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
import numpy as np

app = Flask(__name__)
CORS(app)


# -------------------------------
# Helper: Convert numpy types to native Python
# -------------------------------
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

# -------------------------------
# Helper: Convert ObjectId to str
# -------------------------------
def convert_objectid(obj):
    if isinstance(obj, dict):
        return {k: convert_objectid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid(v) for v in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj
# -------------------------------
# POST: Create new visit (Visitor Form)
# -------------------------------
@app.route("/api/visits", methods=["POST"])
def create_visit():
    data = request.json or {}

    # ------------------------------
    # 1️⃣ REQUIRED FIELD VALIDATION
    # ------------------------------
    category = data.get("category")
    arrival_at = data.get("arrivalAt")
    driver_name = data.get("formData", {}).get("driverName")

    if not category or not arrival_at or not driver_name:
        return jsonify({
            "error": "category, formData.driverName and arrivalAt are required"
        }), 400

    # ------------------------------
    # 2️⃣ VEHICLE NUMBER PLATE EXTRACTION
    # ------------------------------
    image_b64 = data.get("vehicleNoPhoto")
    plate_info = None

    if image_b64:
        try:
            plate_info = extract_plate_from_base64(image_b64)
        except Exception as e:
            app.logger.warning(f"Plate extraction failed: {e}")
            plate_info = None

    if plate_info:
        data["vehiclePlate"] = plate_info

    # ------------------------------
    # 3️⃣ METADATA
    # ------------------------------
    data["status"] = "approved"
    data["submittedAt"] = datetime.utcnow()  # PyMongo can store datetime objects

    # ------------------------------
    # 4️⃣ INSERT INTO MONGO WITH FULL ERROR LOGGING
    # ------------------------------
    try:
        result = visits_collection.insert_one(data)
        app.logger.info(f"Inserted visit with ID: {result.inserted_id}")
    except DuplicateKeyError:
        app.logger.warning("Duplicate visit detected")
        return jsonify({
            "error": "Duplicate visit detected for same category, driver and arrival time"
        }), 409
    except Exception as e:
        app.logger.error(f"Mongo insert failed: {e}")
        return jsonify({"error": "Server error"}), 500

    # ------------------------------
    # 5️⃣ SUCCESS RESPONSE
    # ------------------------------
    return jsonify({
        "message": "Visit submitted successfully",
        "receiptId": data.get("receiptId")
    }), 201


# -------------------------------
# GET: Fetch all visits (Admin Dashboard)
# -------------------------------
#@app.route("/api/visits", methods=["GET"])
#def get_visits():
    visits = []

    for v in visits_collection.find({}, {"_id": 0}):
        visits.append(v)

    return jsonify(visits), 200

#@app.route("/api/receipts", methods=["GET"])
#def get_receipts():
    receipts = list(
        visits_collection.find(
            {"receiptId": {"$exists": True}},
            {"_id": 0}
        )
    )

    return jsonify(receipts), 200

@app.route("/api/visits", methods=["GET"])
def get_visits():
    try:
        visits = list(visits_collection.find({}))
        visits = convert_objectid(visits)  # Convert ObjectId to string
        return jsonify(visits), 200
    except Exception as e:
        app.logger.error(f"Failed to fetch visits: {e}")
        return jsonify({"error": "Server error"}), 500
# -------------------------------
# GET: Fetch receipts
# -------------------------------
@app.route("/api/receipts", methods=["GET"])
def get_receipts():
    try:
        receipts = list(visits_collection.find(
            {"receiptId": {"$exists": True}}
        ))
        receipts = convert_objectid(receipts)
        return jsonify(receipts), 200
    except Exception as e:
        app.logger.error(f"Failed to fetch receipts: {e}")
        return jsonify({"error": "Server error"}), 500
# -------------------------------
# PUT: Update status (Approve / Reject)
# -------------------------------
@app.route("/api/visits/<receipt_id>/status", methods=["PUT"])
def update_status(receipt_id):
    data = request.json
    new_status = data.get("status")

    if new_status not in ["approved", "rejected", "pending"]:
        return jsonify({"error": "Invalid status"}), 400

    result = visits_collection.update_one(
        {"receiptId": receipt_id},
        {"$set": {"status": new_status}}
    )

    if result.matched_count == 0:
        return jsonify({"error": "Record not found"}), 404

    return jsonify({"message": "Status updated"}), 200


# -------------------------------
# DELETE: Delete visit (Admin Dashboard)
# -------------------------------
@app.route("/api/visits/<receipt_id>", methods=["DELETE"])
def delete_visit(receipt_id):
    result = visits_collection.delete_one({"receiptId": receipt_id})

    if result.deleted_count == 0:
        return jsonify({"error": "Record not found"}), 404

    return jsonify({"message": "Deleted successfully"}), 200


# -------------------------------
# Health Check
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return "Backend is running"


if __name__ == "__main__":
    app.run(debug=True)

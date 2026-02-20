// Use explicit DB
db = db.getSiblingDB("entry_shield");

// 1️⃣ Create unique index (LOGISTICS-specific)
db.gate_passes.createIndex(
  {
    category: 1,
    "formData.driverName": 1,
    "arrivalAt": 1
  },
  { unique: true }
);

// 2️⃣ Prepare document
const gatePass = {
  category: "LOGISTICS",
  source: "MANUAL",
  status: "PENDING",

  arrivalAt: ISODate("2026-01-08T06:00:00Z"),
  departureAt: ISODate("2026-01-08T09:00:00Z"),

  transport: {
    mode: "TRUCK",
    vehicleNumber: "HR55TR9090"
  },

  createdBy: {
    userId: ObjectId(),
    role: "SECURITY"
  },

  approvedBy: null,

  formData: {
    driverName: "Mahesh",
    serviceType: "Water Tanker",
    vendorName: "Shree Water Supply"
  },

  createdAt: new Date(),
  updatedAt: new Date()
};

// 3️⃣ Idempotent insert (UPSERT)
db.gate_passes.updateOne(
  {
    category: gatePass.category,
    "formData.driverName": gatePass.formData.driverName,
    arrivalAt: gatePass.arrivalAt
  },
  { $setOnInsert: gatePass },
  { upsert: true }
);

print("✅ LOGISTICS gate_pass seed completed");
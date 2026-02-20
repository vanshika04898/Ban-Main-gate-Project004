// Use explicit DB
db = db.getSiblingDB("entry_shield");

// 1️⃣ Create unique index (prevents duplicates)
db.gate_passes.createIndex(
  { "formData.studentId": 1 }
);

// 2️⃣ Prepare document
const gatePass = {
  category: "STUDENT",
  source: "MANUAL",
  status: "APPROVED",

  arrivalAt: ISODate("2026-01-07T08:30:00Z"),
  departureAt: null,

  transport: {
    mode: "BUS",
    vehicleNumber: "BUS-22"
  },

  createdBy: {
    userId: ObjectId(),
    role: "SECURITY"
  },

  approvedBy: {
    userId: ObjectId(),
    role: "MANAGEMENT",
    type: "PREAPPROVED",
    reason: "STUDENT_ENTRY",
    approvedAt: new Date()
  },

  formData: {
    fullName: "Rahul Singh",
    studentId: "STU789",
    course: "MBA",
    hostel: "Hostel A",
    idPhotoUrl: "s3://studentid.jpg"
  },

  createdAt: new Date(),
  updatedAt: new Date()
};

// 3️⃣ Idempotent insert (UPSERT)
db.gate_passes.updateOne(
  { "formData.studentId": gatePass.formData.studentId },
  { $setOnInsert: gatePass },
  { upsert: true }
);

print("✅ gate_passes seed completed");
// Use explicit DB
db = db.getSiblingDB("entry_shield");

// 1️⃣ Create unique index
// Natural uniqueness: category + arrivalAt + primary person name
db.gate_passes.createIndex(
  {
    category: 1,
    arrivalAt: 1,
    "formData.primaryPerson.fullName": 1
  },
  { unique: true }
);

// 2️⃣ Prepare gate pass document
const gatePass = {
  category: "RESIDENTIAL_GUEST",
  source: "MANUAL",
  status: "PENDING",

  arrivalAt: ISODate("2026-01-06T09:00:00Z"),
  departureAt: ISODate("2026-01-06T20:00:00Z"),

  transport: {
    mode: "CAR",
    vehicleNumber: "RJ14CD5678"
  },

  createdBy: {
    userId: ObjectId(),
    role: "SECURITY"
  },

  approvedBy: null,

  formData: {
    totalPeople: 2,
    totalMale: 1,
    totalFemale: 1,

    primaryPerson: {
      fullName: "Amit Verma",
      idType: "PAN",
      idNumber: "ABCDE1234F"
    },

    members: [
      {
        name: "Neha Verma",
        idPhotoUrl: "s3://id2.jpg"
      }
    ]
  },

  createdAt: new Date(),
  updatedAt: new Date()
};

// 3️⃣ Idempotent insert (UPSERT)
db.gate_passes.updateOne(
  {
    category: gatePass.category,
    arrivalAt: gatePass.arrivalAt,
    "formData.primaryPerson.fullName": gatePass.formData.primaryPerson.fullName
  },
  { $setOnInsert: gatePass },
  { upsert: true }
);

print("✅ RESIDENTIAL_GUEST gate pass seed completed");
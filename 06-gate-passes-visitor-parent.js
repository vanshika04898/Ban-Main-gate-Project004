// Use explicit DB
db = db.getSiblingDB("entry_shield");

// 1️⃣ Create unique index to prevent duplicate visitor passes
// Chosen natural key: category + arrivalAt + primary person name
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
  _id: ObjectId("695c0a8e745f87b990ab2836"),

  category: "VISITOR_PARENT",
  source: "MANUAL",
  status: "PENDING",

  arrivalAt: ISODate("2026-01-05T10:00:00Z"),
  departureAt: ISODate("2026-01-05T18:00:00Z"),

  transport: {
    mode: "CAR",
    vehicleNumber: "KA01AB1234",
    vehiclePhotoUrl: "s3://vehicle.jpg"
  },

  createdBy: {
    userId: ObjectId("695c0a8e745f87b990ab2835"),
    role: "SECURITY"
  },

  approvedBy: null,

  formData: {
    totalPeople: 3,
    totalMale: 2,
    totalFemale: 1,

    primaryPerson: {
      fullName: "Ramesh Kumar",
      idType: "AADHAR",
      idPhotoUrl: "s3://vehicle.jpg"
    },

    members: [
      {
        name: "Suresh Kumar",
        idPhotoUrl: "s3://id1.jpg"
      }
    ],

    guestHouseNo: "GH-12",

    hostStudent: {
      studentName: "Ankit Sharma",
      studentId: "STU123",
      course: "B.Tech"
    }
  },

  createdAt: ISODate("2026-01-05T19:01:34.616Z"),
  updatedAt: ISODate("2026-01-05T19:01:34.616Z")
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

print("✅ VISITOR_PARENT gate pass seed completed");
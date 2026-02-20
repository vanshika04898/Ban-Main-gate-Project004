// Use explicit DB
db = db.getSiblingDB("entry_shield");

// Ensure unique page index exists
db.page_permissions.createIndex(
  { page: 1 },
  { unique: true }
);

// Prepare permission document
const pagePermission = {
  page: "APPROVE_GATE_PASS",
  allowedRoles: ["ADMIN", "MANAGEMENT"],
  createdAt: new Date()
};

// Idempotent insert (UPSERT)
db.page_permissions.updateOne(
  { page: pagePermission.page },
  { $setOnInsert: pagePermission },
  { upsert: true }
);

print("✅ APPROVE_GATE_PASS page permission seed completed");
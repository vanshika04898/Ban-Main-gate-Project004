// Use explicit DB
db = db.getSiblingDB("entry_shield");

// 1️⃣ Create unique index (one permission set per page)
db.page_permissions.createIndex(
  { page: 1 },
  { unique: true }
);

// 2️⃣ Prepare permission document
const pagePermission = {
  page: "CREATE_GATE_PASS",
  allowedRoles: ["SECURITY", "ADMIN"],
  createdAt: new Date()
};

// 3️⃣ Idempotent insert (UPSERT)
db.page_permissions.updateOne(
  { page: pagePermission.page },
  { $setOnInsert: pagePermission },
  { upsert: true }
);

print("✅ CREATE_GATE_PASS page permission seed completed");
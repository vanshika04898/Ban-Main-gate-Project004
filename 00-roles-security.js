// Use explicit DB
db = db.getSiblingDB("entry_shield");

// 1️⃣ Create unique index (role name must be unique)
db.roles.createIndex(
  { role: 1 },
  { unique: true }
);

// 2️⃣ Prepare role document
const role = {
  role: "SECURITY",
  description: "Security guard entry access",
  createdAt: new Date()
};

// 3️⃣ Idempotent insert (UPSERT)
db.roles.updateOne(
  { role: role.role },
  { $setOnInsert: role },
  { upsert: true }
);

print("✅ SECURITY role seed completed");
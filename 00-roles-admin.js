// Use explicit DB
db = db.getSiblingDB("entry_shield");

// Ensure unique role index exists (safe if already created)
db.roles.createIndex(
  { role: 1 },
  { unique: true }
);

// Prepare role document
const role = {
  role: "ADMIN",
  description: "Full system access",
  createdAt: new Date()
};

// Idempotent insert (UPSERT)
db.roles.updateOne(
  { role: role.role },
  { $setOnInsert: role },
  { upsert: true }
);

print("✅ ADMIN role seed completed");
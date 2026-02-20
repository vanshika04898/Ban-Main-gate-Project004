// Use explicit DB
db = db.getSiblingDB("entry_shield");

// 1️⃣ Create unique index (email must be unique)
db.users.createIndex(
  { email: 1 },
  { unique: true }
);

// 2️⃣ Prepare user document
const user = {
  name: "Security Guard A",
  email: "guard@college.edu",
  passwordHash: "hashed_password_here", // pre-hashed
  role: "SECURITY",
  isActive: true,
  createdAt: new Date()
};

// 3️⃣ Idempotent insert (UPSERT)
db.users.updateOne(
  { email: user.email },
  { $setOnInsert: user },
  { upsert: true }
);

print("✅ SECURITY user seed completed");
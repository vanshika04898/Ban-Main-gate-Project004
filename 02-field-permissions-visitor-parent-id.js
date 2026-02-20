// Use explicit DB
db = db.getSiblingDB("entry_shield");

// 1️⃣ Create unique index
// One permission rule per (category + field)
db.field_permissions.createIndex(
  { category: 1, field: 1 },
  { unique: true }
);

// 2️⃣ Prepare field permission document
const fieldPermission = {
  category: "VISITOR_PARENT",
  field: "formData.primaryPerson.idNumber",
  visibleTo: ["SECURITY", "ADMIN"],
  editableBy: ["SECURITY"],
  readOnlyFor: ["MANAGEMENT"],
  createdAt: new Date()
};

// 3️⃣ Idempotent insert (UPSERT)
db.field_permissions.updateOne(
  {
    category: fieldPermission.category,
    field: fieldPermission.field
  },
  { $setOnInsert: fieldPermission },
  { upsert: true }
);

print("✅ VISITOR_PARENT field permission seed completed");
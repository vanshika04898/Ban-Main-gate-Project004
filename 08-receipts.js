db = db.getSiblingDB("entry_shield");

// 1️⃣ Ensure unique receipt number
db.receipts.createIndex(
  { receiptNumber: 1 },
  { unique: true }
);

// 2️⃣ Find an existing gate pass
const gatePass = db.gate_passes.findOne(
  { status: "PENDING" },
  { _id: 1 }
);

if (!gatePass) {
  print("⚠️ No gate pass found. Receipt not created.");
} else {
  const receipt = {
    gatePassId: gatePass._id,
    receiptNumber: "GP-2026-000123",
    category: "VISITOR_PARENT",

    generatedAt: new Date(),

    generatedBy: {
      userId: ObjectId(),
      role: "SECURITY"
    }
  };

  db.receipts.updateOne(
    { receiptNumber: receipt.receiptNumber },
    { $setOnInsert: receipt },
    { upsert: true }
  );

  print("✅ Receipt seeded successfully");
}
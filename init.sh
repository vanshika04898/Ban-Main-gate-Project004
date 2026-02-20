#!/bin/bash
set -e

echo "🚀 MongoDB initialization started..."

# Wait for MongoDB to be ready
until mongosh --eval "db.adminCommand('ping')" >/dev/null 2>&1; do
  echo "⏳ Waiting for MongoDB..."
  sleep 2
done

echo "✅ MongoDB is up. Running init scripts..."

# Execute all JS files in order
for file in /docker-entrypoint-initdb.d/*.js; do
  echo "▶ Running $file"
  mongosh "$MONGO_INITDB_DATABASE" "$file"
done

echo "🎉 MongoDB initialization completed successfully!"
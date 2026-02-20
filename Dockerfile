FROM node:24.12.0-alpine

WORKDIR /app

# Copy only package files first (better caching)
COPY package*.json ./
RUN npm ci

# Copy the rest of the app (without node_modules thanks to .dockerignore)
COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]

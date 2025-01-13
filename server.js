const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const mongoose = require('mongoose');  // ✅ FIX: Import mongoose
const connectDB = require('./config/database');  // MongoDB connection
const authRoutes = require('./routes/auth');
const promptRoutes = require('./routes/prompt');
const { exec } = require('child_process');

dotenv.config();

// ✅ Install Python Dependencies (Optional in Production)
console.log("🔄 Checking Python dependencies...");
exec('python -m pip install --upgrade pip && python -m pip install --break-system-packages -r requirements.txt', (error, stdout, stderr) => {
    if (error) {
        console.error(`❌ Error installing Python dependencies: ${error.message}`);
        return;
    }
    if (stderr) {
        console.error(`⚠️ stderr: ${stderr}`);
        return;
    }
    console.log(`✅ Python dependencies installed:\n${stdout}`);
});

const app = express();
// app.use(cors());


app.use(cors({
    origin: ['https://promptopia-frontend.onrender.com', 'http://localhost:3000'],
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
    credentials: true
}));
app.use(express.json());

// ✅ Routes
app.use('/api/auth', authRoutes);
app.use('/api/prompts', promptRoutes);

// ✅ Connect to MongoDB
connectDB();

// ✅ LOGGING isEvaluated Changes
mongoose.set('debug', function (collectionName, method, query, doc) {
    if (doc && doc.isEvaluated !== undefined) {
        const stackTrace = new Error().stack;
        console.log(`🔄 isEvaluated is being changed in collection "${collectionName}"`);
        console.log(`📝 Method: ${method}`);
        console.log(`📄 Query:`, JSON.stringify(query, null, 2));
        console.log(`📦 Document:`, JSON.stringify(doc, null, 2));
        console.log(`📍 Stack Trace: \n${stackTrace}`);
    }
});

// ✅ Centralized Error Handling
app.use((err, req, res, next) => {
    console.error(`❌ Error: ${err.message}`);
    res.status(500).json({ error: 'Internal Server Error' });
});

// ✅ Start Server
const PORT = process.env.PORT || 5000;
const server = app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));

// ✅ Prevent server from timing out for long tasks
server.timeout = 600 * 60 * 1000;  // 600 minutes || 10 hours

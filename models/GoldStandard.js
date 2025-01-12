const mongoose = require('mongoose');

const GoldStandardSchema = new mongoose.Schema({
    model: { type: String, required: true },
    latency: { type: Number, required: true },
    tokenEfficiencyRatio: { type: Number, required: true },
    accuracy: { type: Number, required: true },
    response: { type: String, required: true }
}, { timestamps: true });

module.exports = mongoose.model('GoldStandard', GoldStandardSchema);
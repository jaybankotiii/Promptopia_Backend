const mongoose = require('mongoose');

const DecisionSchema = new mongoose.Schema({
    bestModel: { type: String, required: true },
    preferredForLatency: { type: String, required: true },
    preferredForTokenEfficiency: { type: String, required: true }
}, { timestamps: true });

module.exports = mongoose.model('Decision', DecisionSchema);

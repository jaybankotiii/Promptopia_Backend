const mongoose = require('mongoose');

const PromptSchema = new mongoose.Schema({
    user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    prompt: { type: String, required: true },
    goldenModel: { type: String, required: true },
    suggestedModel: { type: String, default: null },
    goldStandard: { type: mongoose.Schema.Types.ObjectId, ref: 'GoldStandard' },
    comparisons: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Comparison' }],
    decision: { type: mongoose.Schema.Types.ObjectId, ref: 'Decision' },
}, { timestamps: true });

module.exports = mongoose.model('Prompt', PromptSchema);

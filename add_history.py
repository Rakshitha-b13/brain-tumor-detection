from pathlib import Path

path = Path("templates/index.html")
html = path.read_text(encoding="utf-8")

history_html = """
<!-- =====================================================
     PREDICTION HISTORY
     ===================================================== -->

<section id="predictionHistory" style="margin-top:32px;">

    <div class="section-title">
        <h2>Recent Prediction History</h2>
        <p>Previous AI-assisted classification results stored locally in this browser.</p>
    </div>

    <div id="historyList">
        <p>No previous predictions yet.</p>
    </div>

    <button
        type="button"
        id="clearHistory"
        style="margin-top:16px; padding:10px 18px; border:0; border-radius:8px; cursor:pointer;">
        Clear History
    </button>

</section>
"""

history_js = """
// =====================================================
// LOCAL PREDICTION HISTORY
// =====================================================

const HISTORY_KEY = "brainTumorPredictionHistory";
const HISTORY_LIMIT = 5;

function loadPredictionHistory() {

    const list = document.getElementById("historyList");

    if (!list) return;

    let history = [];

    try {
        history = JSON.parse(
            localStorage.getItem(HISTORY_KEY) || "[]"
        );
    } catch (error) {
        history = [];
    }

    if (history.length === 0) {
        list.innerHTML = "<p>No previous predictions yet.</p>";
        return;
    }

    list.innerHTML = history.map(item => `
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:16px;
            padding:14px 16px;
            margin-bottom:10px;
            border:1px solid rgba(128,128,128,.25);
            border-radius:10px;
        ">
            <div>
                <strong>${item.prediction}</strong>
                <div style="font-size:13px; opacity:.75; margin-top:4px;">
                    ${item.time}
                </div>
            </div>

            <div style="font-weight:600;">
                ${item.confidence}%
            </div>
        </div>
    `).join("");
}

function addPredictionToHistory(prediction, confidence) {

    let history = [];

    try {
        history = JSON.parse(
            localStorage.getItem(HISTORY_KEY) || "[]"
        );
    } catch (error) {
        history = [];
    }

    history.unshift({
        prediction: prediction,
        confidence: Number(confidence).toFixed(2),
        time: new Date().toLocaleString()
    });

    history = history.slice(0, HISTORY_LIMIT);

    localStorage.setItem(
        HISTORY_KEY,
        JSON.stringify(history)
    );

    loadPredictionHistory();
}

document.addEventListener("DOMContentLoaded", function() {

    loadPredictionHistory();

    const clearButton =
        document.getElementById("clearHistory");

    if (clearButton) {

        clearButton.addEventListener("click", function() {

            localStorage.removeItem(HISTORY_KEY);

            loadPredictionHistory();

        });

    }

});
"""
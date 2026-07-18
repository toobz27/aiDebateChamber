"""
app.py
------
The REST gateway. This file intentionally contains almost NO logic of its own --
it just receives HTTP requests from the frontend, delegates to the two service
classes, and returns JSON. Keeping logic out of the routes is why aiService.py
and mlJudge.py exist as separate modules (separation of concerns).
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from services.aiService import DebateConductor
from services.mlJudge import DebateRegressionJudge

app = Flask(__name__)
CORS(app)  # Allow Cross-Origin Requests from the UI (served from a different port/file://)

# System Modules -- single shared instances for the life of the server process.
conductor = DebateConductor()
ml_judge = DebateRegressionJudge()

# Simple in-memory turn tracker so /next-turn knows whose turn it is without
# the frontend having to specify it every time.
MAX_ROUNDS = 10
turn_state = {"round": 0, "next_agent": "A"}


@app.route('/api/debate/start', methods=['POST'])
def start_debate():
    """Initializes a fresh debate: resets history and turn counters."""
    data = request.json or {}
    topic = data.get('topic', 'Untitled Debate Topic')

    conductor.start_debate(topic)
    turn_state["round"] = 0
    turn_state["next_agent"] = "A"

    return jsonify({"status": "active", "topic": topic})


@app.route('/api/debate/next-turn', methods=['POST'])
def next_turn():
    """
    Triggers whichever agent is next in sequence to generate a response via
    the local Ollama model, alternating A -> B -> A -> B ...
    """
    if conductor.topic is None:
        return jsonify({"error": "No debate has been started yet."}), 400

    if turn_state["round"] >= MAX_ROUNDS:
        return jsonify({"status": "complete", "message": "Debate has reached max rounds."})

    agent = turn_state["next_agent"]

    if agent == "A":
        message = conductor.generate_agent_a_response(conductor.topic)
    else:
        message = conductor.generate_agent_b_response(conductor.topic)

    turn_state["round"] += 1
    turn_state["next_agent"] = "B" if agent == "A" else "A"

    return jsonify({
        "agent": agent,
        "message": message,
        "round": turn_state["round"],
        "max_rounds": MAX_ROUNDS,
    })


@app.route('/api/machine-learning/train', methods=['POST'])
def trigger_training():
    """Triggers the SciKit-Learn Regression Model Training Loop."""
    try:
        accuracy_metrics = ml_judge.train_model("historical_debates.csv")
        return jsonify({"status": "Training Completed", "metrics": accuracy_metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/machine-learning/evaluate', methods=['POST'])
def evaluate_debate():
    """Uses the trained ML model to score both agents' full arguments and picks a winner."""
    data = request.json or {}
    advocate_text = data.get('advocate_text', '')
    challenger_text = data.get('challenger_text', '')

    try:
        advocate_score = ml_judge.predict_score(advocate_text)
        challenger_score = ml_judge.predict_score(challenger_text)
    except Exception as e:
        # Most common cause: /train wasn't called before /evaluate.
        return jsonify({"error": str(e)}), 400

    if advocate_score > challenger_score:
        winner = "Agent A (Advocate)"
    elif challenger_score > advocate_score:
        winner = "Agent B (Challenger)"
    else:
        winner = "Draw"

    return jsonify({
        "winner": winner,
        "advocate_score": advocate_score,
        "challenger_score": challenger_score,
    })


if __name__ == '__main__':
    print("🚀 AI Server running on http://127.0.0.1:5000")
    print("Ensure Ollama is running locally on port 11434!")
    app.run(debug=True, port=5000)
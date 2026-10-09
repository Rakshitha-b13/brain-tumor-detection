from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import sqlite3
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from PIL import Image
from matplotlib import colormaps
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage,
    PageBreak
)

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "brain-tumor-ai-development-secret-key"
)
# ============================================================
# AUTHENTICATION DATABASE
# ============================================================

DATABASE_PATH = "users.db"


def init_auth_database():
    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    connection.commit()
    connection.close()

init_auth_database()

# ============================================================
# AUTHENTICATION HELPERS
# ============================================================

def create_user(username, password):
    password_hash = generate_password_hash(password)

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        connection.execute(
            """
            INSERT INTO users
            (username, password_hash, created_at)
            VALUES (?, ?, ?)
            """,
            (
                username,
                password_hash,
                datetime.now().isoformat()
            )
        )

        connection.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()


def verify_user(username, password):
    connection = sqlite3.connect(DATABASE_PATH)

    user = connection.execute(
        """
        SELECT id, username, password_hash
        FROM users
        WHERE username = ?
        """,
        (username,)
    ).fetchone()

    connection.close()

    if user and check_password_hash(
        user[2],
        password
    ):
        return {
            "id": user[0],
            "username": user[1]
        }

    return None


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "ai_model",
    "models",
    "resnet50_best.keras"
)

UPLOAD_FOLDER = os.path.join(
    "static",
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# ============================================================
# PRIVACY: CLEAN TEMPORARY MRI FILES
# ============================================================

def cleanup_temporary_mri_files():
    """
    Remove previously uploaded MRI images and Grad-CAM files.
    Generated PDF reports are stored separately and are not removed.
    """

    for file_name in os.listdir(UPLOAD_FOLDER):

        file_path = os.path.join(
            UPLOAD_FOLDER,
            file_name
        )

        if os.path.isfile(file_path):

            extension = os.path.splitext(
                file_name
            )[1].lower()

            if extension in [".jpg", ".jpeg", ".png"]:
                try:
                    os.remove(file_path)
                except OSError:
                    pass


# ============================================================
# LOAD FEDERATED RESNET50 MODEL
# ============================================================

print("Loading ResNet50 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("ResNet50 model loaded successfully!")


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    img_array,
    predicted_index,
    original_path,
    output_path
):

    print("Generating Grad-CAM...")

    # ============================================================
    # FIND RESNET50 BACKBONE
    # ============================================================

    backbone_index = None
    backbone = None

    for i, layer in enumerate(model.layers):

        if isinstance(layer, tf.keras.Model):

            if layer.output_shape[-1] == 2048:
                backbone_index = i
                backbone = layer
                break

    if backbone is None:
        raise ValueError(
            "ResNet50 backbone could not be found."
        )

    print(
        "Grad-CAM backbone:",
        backbone.name
    )

    # ============================================================
    # FORWARD PASS
    # ============================================================

    with tf.GradientTape() as tape:

        conv_outputs = backbone(
            img_array,
            training=False
        )

        tape.watch(conv_outputs)

        x = conv_outputs

        # Apply every layer after ResNet50
        # automatically, without depending on layer names.
        for layer in model.layers[
            backbone_index + 1:
        ]:

            x = layer(
                x,
                training=False
            )

        predictions = x

        class_score = predictions[
            :,
            predicted_index
        ]

    # ============================================================
    # CALCULATE GRADIENTS
    # ============================================================

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    # ============================================================
    # GLOBAL AVERAGE OF GRADIENTS
    # ============================================================

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    # ============================================================
    # CREATE HEATMAP
    # ============================================================

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value.numpy()) > 0:

        heatmap = (
            heatmap / max_value
        )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224)
    )

    heatmap = (
        heatmap
        .numpy()
        .squeeze()
    )

    # ============================================================
    # COLORIZE HEATMAP
    # ============================================================

    colored_heatmap = colormaps["jet"](
        heatmap
    )[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(
        np.uint8
    )

    # ============================================================
    # LOAD ORIGINAL IMAGE
    # ============================================================

    original = Image.open(
        original_path
    ).convert(
        "RGB"
    )

    original = original.resize(
        (224, 224)
    )

    original_array = np.asarray(
        original,
        dtype=np.float32
    )

    # ============================================================
    # CREATE OVERLAY
    # ============================================================

    overlay = (
        0.60 * original_array
        +
        0.40 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    # ============================================================
    # SAVE GRAD-CAM
    # ============================================================

    Image.fromarray(
        overlay
    ).save(
        output_path,
        quality=95
    )

    print(
        "Grad-CAM saved successfully:",
        output_path
    )

    return output_path
# ============================================================
# HOME PAGE
# ============================================================
# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = verify_user(
            username,
            password
        )

        if user:

            session["user_id"] = user["id"]
            session["username"] = user["username"]

            return redirect(
                url_for("home")
            )

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if len(username) < 3:
            return render_template(
                "register.html",
                error="Username must contain at least 3 characters."
            )

        if len(password) < 8:
            return render_template(
                "register.html",
                error="Password must contain at least 8 characters."
            )

        if password != confirm_password:
            return render_template(
                "register.html",
                error="Passwords do not match."
            )

        if create_user(
            username,
            password
        ):
            return redirect(
                url_for("login")
            )

        return render_template(
            "register.html",
            error="Username already exists."
        )

    return render_template("register.html")
@app.route("/my-reports")
def my_reports():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "error": "Authentication required."
        }), 401

    connection = sqlite3.connect(DATABASE_PATH)

    reports = connection.execute(
        """
        SELECT
            users.username,
            reports.filename,
            reports.created_at
        FROM reports
        INNER JOIN users
            ON reports.user_id = users.id
        ORDER BY reports.id DESC
        """
    ).fetchall()

    connection.close()

    report_list = []

    for report in reports:

        report_list.append({
            "username": report[0],
            "time": report[2],
            "report": "/report/" + report[1]
        })

    return jsonify({
        "success": True,
        "reports": report_list
    })

@app.route("/delete-report", methods=["POST"])
def delete_report():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "error": "Authentication required."
        }), 401

    data = request.get_json(silent=True) or {}

    report_filename = data.get("filename", "").strip()

    if not report_filename:
        return jsonify({
            "success": False,
            "error": "Report filename is required."
        }), 400

    from werkzeug.utils import secure_filename

    safe_filename = secure_filename(report_filename)

    connection = sqlite3.connect(DATABASE_PATH)

    report = connection.execute(
        """
        SELECT filename
        FROM reports
        WHERE filename = ?
        """,
        (safe_filename,)
    ).fetchone()

    if not report:
        connection.close()

        return jsonify({
            "success": False,
            "error": "Report not found."
        }), 404

    connection.execute(
        """
        DELETE FROM reports
        WHERE filename = ?
        """,
        (safe_filename,)
    )

    connection.commit()
    connection.close()

    report_folder = os.path.join(
        app.root_path,
        "reports"
    )

    report_path = os.path.join(
        report_folder,
        safe_filename
    )

    if os.path.isfile(report_path):
        try:
            os.remove(report_path)
        except OSError:
            pass

    return jsonify({
        "success": True,
        "message": "Report deleted successfully."
    })

@app.route("/verify-report-password", methods=["POST"])
def verify_report_password():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "error": "Authentication required."
        }), 401

    data = request.get_json(silent=True) or {}

    password = data.get("password", "")
    report_url = data.get("report", "")

    if not password:
        return jsonify({
            "success": False,
            "error": "Password is required."
        }), 400

    if not report_url:
        return jsonify({
            "success": False,
            "error": "Report is required."
        }), 400

    report_filename = os.path.basename(report_url)

    connection = sqlite3.connect(DATABASE_PATH)

    report = connection.execute(
        """
        SELECT users.password_hash
        FROM reports
        INNER JOIN users
            ON reports.user_id = users.id
        WHERE reports.filename = ?
        """,
        (report_filename,)
    ).fetchone()

    connection.close()

    if not report:
        return jsonify({
            "success": False,
            "error": "Report not found."
        }), 404

    if not check_password_hash(
        report[0],
        password
    ):
        return jsonify({
            "success": False,
            "error": "Incorrect password."
        }), 403

    if "verified_reports" not in session:
        session["verified_reports"] = []

    verified_reports = session["verified_reports"]

    if report_filename not in verified_reports:
        verified_reports.append(report_filename)

    session["verified_reports"] = verified_reports

    return jsonify({
        "success": True
    })
@app.route("/profile")
def profile():

    if "user_id" not in session:
        return redirect(
            url_for("login")
        )

    return render_template(
        "profile.html",
        username=session.get("username")
    )



@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )
@app.route("/")
def home():

    if "user_id" not in session:
        return redirect(
            url_for("login")
        )

    return render_template(
        "index.html"
    )

# ============================================================
# MRI UPLOAD + PREDICTION
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "error": "Authentication required."
        }), 401

    cleanup_temporary_mri_files()
    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if "mri_image" not in request.files:

        return jsonify({
            "success": False,
            "error": "No MRI image uploaded."
        }), 400

    img_file = request.files[
        "mri_image"
    ]

    if img_file.filename == "":

        return jsonify({
            "success": False,
            "error": "Please select an MRI image."
        }), 400

    try:

        # ----------------------------------------------------
        # Secure filename
        # ----------------------------------------------------

        filename = secure_filename(
            img_file.filename
        )

        if not filename:

            return jsonify({
                "success": False,
                "error": "Invalid image filename."
            }), 400

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        img_file.save(
            file_path
        )

        print(
            "Image uploaded:",
            filename
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        img = image.load_img(
            file_path,
            target_size=(224, 224)
        )

        # ----------------------------------------------------
        # Convert to NumPy array
        # ----------------------------------------------------

        img_array = image.img_to_array(
            img
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # ----------------------------------------------------
        # ResNet50 preprocessing
        # ----------------------------------------------------

        img_array = preprocess_input(
            img_array
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            img_array,
            verbose=0
        )

        predictions = np.asarray(
            predictions[0],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Verify number of outputs
        # ----------------------------------------------------

        if predictions.shape[0] != len(
            CLASS_NAMES
        ):

            raise ValueError(
                f"Model returned "
                f"{predictions.shape[0]} outputs, "
                f"but CLASS_NAMES contains "
                f"{len(CLASS_NAMES)} classes."
            )

        # ----------------------------------------------------
        # Find predicted class
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(
                predictions
            )
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = (
            float(
                predictions[
                    predicted_index
                ]
            )
            * 100
        )

        # ----------------------------------------------------
        # Class name
        # ----------------------------------------------------

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        # ====================================================
        # GENERATE GRAD-CAM
        # ====================================================

        base_name = os.path.splitext(
            filename
        )[0]

        gradcam_filename = (
            base_name
            + "_gradcam.jpg"
        )

        gradcam_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            gradcam_filename
        )

        generate_gradcam(
            img_array,
            predicted_index,
            file_path,
            gradcam_path
        )

        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        probabilities = {

            CLASS_NAMES[i]: round(
                float(
                    predictions[i]
                ) * 100,
                2
            )

            for i in range(
                len(CLASS_NAMES)
            )
        }

        # ====================================================
        # RETURN RESULT
        # ====================================================
                # ====================================================
        # GENERATE AI MEDICAL PDF REPORT
        # ====================================================

        report_filename = generate_medical_report(
            predicted_class,
            confidence,
            probabilities,
            file_path,
            gradcam_path
        )

        connection = sqlite3.connect(DATABASE_PATH)

        connection.execute(
            """
            INSERT INTO reports
            (user_id, filename, prediction, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                report_filename,
                predicted_class,
                confidence,
                datetime.now().isoformat()
            )
        )

        connection.commit()
        connection.close()

        return jsonify({

            "success": True,

            "prediction":
                predicted_class,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "probabilities":
                probabilities,

            "image":
                "/static/uploads/"
                + filename,

            "gradcam":
                "/static/uploads/"
                + gradcam_filename,

            "report":
                "/report/"
                + report_filename
        })

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500

# ============================================================
# FEDERATED LEARNING RESULTS API
# ============================================================

@app.route("/federated-results", methods=["GET"])
def federated_results():

    history_path = os.path.join(
        "ai_model",
        "fusion_results",
        "privacy_federated_history.json"
    )

    if not os.path.exists(history_path):
        return jsonify({
            "success": False,
            "error": "Federated learning history file not found."
        }), 404

    try:

        with open(history_path, "r") as f:
            history = json.load(f)

        global_accuracy = history.get("global_accuracy", [])
        global_loss = history.get("global_loss", [])
        rounds = history.get("rounds", [])

        return jsonify({

            "success": True,

            "final_accuracy": round(
                global_accuracy[-1] * 100, 2
            ) if global_accuracy else None,

            "final_loss": round(
                global_loss[-1], 4
            ) if global_loss else None,

            "rounds": rounds,

            "global_accuracy": [
                round(value * 100, 2)
                for value in global_accuracy
            ],

            "global_loss": [
                round(value, 4)
                for value in global_loss
            ],

            "clients": [
                {
                    "client_id": 1,
                    "samples": 1400,
                    "distribution": {
                        "Glioma": 800,
                        "Meningioma": 400,
                        "No Tumor": 100,
                        "Pituitary Tumor": 100
                    }
                },
                {
                    "client_id": 2,
                    "samples": 2000,
                    "distribution": {
                        "Glioma": 300,
                        "Meningioma": 700,
                        "No Tumor": 300,
                        "Pituitary Tumor": 700
                    }
                },
                {
                    "client_id": 3,
                    "samples": 2200,
                    "distribution": {
                        "Glioma": 300,
                        "Meningioma": 300,
                        "No Tumor": 1000,
                        "Pituitary Tumor": 600
                    }
                }
            ],

            "privacy": {
                "clip_norm": 1.0,
                "noise_multiplier": 0.01,
                "method": "DP-style update perturbation",
                "formal_accountant": False
            },

            "model_comparison": [
                {
                    "model": "CNN Baseline",
                    "accuracy": 87.56
                },
                {
                    "model": "Centralized ResNet50",
                    "accuracy": 91.13
                },
                {
                    "model": "Centralized Optimized Fusion",
                    "accuracy": 88.88
                },
                {
                    "model": "Privacy-Aware Federated Fusion",
                    "accuracy": 83.44
                }
            ]
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============================================================
# AI MEDICAL REPORT GENERATOR
# ============================================================

def generate_medical_report(
    prediction,
    confidence,
    probabilities,
    image_path,
    gradcam_path
):

    report_id = datetime.now().strftime(
        "BT-AI-%Y%m%d-%H%M%S"
    )

    report_folder = os.path.join(
        app.root_path,
        "reports"
    )

    os.makedirs(
        report_folder,
        exist_ok=True
    )

    report_filename = report_id + ".pdf"

    report_path = os.path.join(
        report_folder,
        report_filename
    )

    document = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        leading=13,
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=6
    )

    normal_style = ParagraphStyle(
        "NormalReport",
        parent=styles["Normal"],
        fontSize=9,
        leading=14
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8,
        leading=11
    )

    story = []
    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "AI-ASSISTED BRAIN MRI ANALYSIS REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Privacy-Aware Federated Deep Learning Framework "
            "with Optimized Feature Fusion for Robust Brain "
            "Tumor Classification in Smart Healthcare Systems",
            subtitle_style
        )
    )

    report_info = [
        ["Report ID", report_id],
        [
            "Analysis Date",
            datetime.now().strftime(
                "%d %B %Y, %I:%M %p"
            )
        ],
        ["Analysis Type", "AI-Assisted MRI Classification"],
        ["Number of Classes", "4"]
    ]

    info_table = Table(
        report_info,
        colWidths=[45 * mm, 115 * mm]
    )

    info_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(info_table)
    story.append(Spacer(1, 8))

    # --------------------------------------------------------
    # MRI ANALYSIS RESULT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. MRI CLASSIFICATION RESULT",
            section_style
        )
    )

    result_table = Table(
        [
            ["Predicted Class", prediction],
            [
                "Prediction Confidence",
                f"{confidence:.2f}%"
            ]
        ],
        colWidths=[65 * mm, 95 * mm]
    )

    result_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(result_table)
    story.append(Spacer(1, 8))

    # --------------------------------------------------------
    # MRI IMAGE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Submitted MRI Image",
            styles["Heading3"]
        )
    )

    if os.path.exists(image_path):

        mri_image = ReportImage(
            image_path,
            width=75 * mm,
            height=75 * mm
        )

        story.append(mri_image)

    story.append(Spacer(1, 8))

    # --------------------------------------------------------
    # CLASS PROBABILITIES
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. CLASS PROBABILITY DISTRIBUTION",
            section_style
        )
    )

    probability_data = [
        ["Class", "Probability"]
    ]

    for class_name, probability in probabilities.items():

        probability_data.append([
            class_name,
            f"{float(probability):.2f}%"
        ])

    probability_table = Table(
        probability_data,
        colWidths=[100 * mm, 60 * mm]
    )

    probability_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6EAF8")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(probability_table)

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. EXPLAINABLE AI – GRAD-CAM",
            section_style
        )
    )

    story.append(
        Paragraph(
            "Grad-CAM is used to visualize image regions "
            "that contributed to the model's prediction. "
            "The visualization provides an explainability "
            "layer for the AI classification result.",
            normal_style
        )
    )

    story.append(Spacer(1, 6))

    if os.path.exists(gradcam_path):

        gradcam_image = ReportImage(
            gradcam_path,
            width=85 * mm,
            height=70 * mm
        )

        story.append(gradcam_image)

    # --------------------------------------------------------
    # FEATURE FUSION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. OPTIMIZED FEATURE FUSION",
            section_style
        )
    )

    fusion_data = [
        ["Feature Component", "Implementation"],
        ["Intensity Features", "32-dimensional representation"],
        ["Texture Features", "32-dimensional representation"],
        ["Spatial Features", "32-dimensional representation"],
        ["Deep Features", "ResNet50 representation"],
        ["Dimensionality Reduction", "PCA: 2048 → 128"],
        ["Fusion Strategy", "Gated feature fusion"]
    ]

    fusion_table = Table(
        fusion_data,
        colWidths=[65 * mm, 95 * mm]
    )

    fusion_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6EAF8")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
        ])
    )

    story.append(fusion_table)

    # --------------------------------------------------------
    # PRIVACY-AWARE FEDERATED LEARNING
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. PRIVACY-AWARE FEDERATED LEARNING",
            section_style
        )
    )

    privacy_data = [
        ["Component", "Configuration"],
        ["Simulated Healthcare Clients", "3"],
        ["Data Distribution", "Non-IID"],
        ["Federated Rounds", "5"],
        ["Aggregation", "Federated Averaging (FedAvg)"],
        ["Update Clipping Norm", "1.0"],
        ["Noise Multiplier", "0.01"],
        [
            "Privacy Mechanism",
            "DP-style update perturbation"
        ]
    ]

    privacy_table = Table(
        privacy_data,
        colWidths=[75 * mm, 85 * mm]
    )

    privacy_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6EAF8")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
        ])
    )

    story.append(privacy_table)

    story.append(Spacer(1, 6))

    story.append(
        Paragraph(
            "The federated implementation applies update "
            "clipping and Gaussian noise before aggregation. "
            "The mechanism is described as DP-style update "
            "perturbation; a formal differential-privacy "
            "accountant is not implemented in the current "
            "research prototype.",
            small_style
        )
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "6. FEDERATED MODEL PERFORMANCE",
            section_style
        )
    )

    performance_data = [
        ["Metric", "Result"],
        ["Test Accuracy", "83.44%"],
        ["Weighted Precision", "84.37%"],
        ["Weighted Recall", "83.44%"],
        ["Weighted F1-Score", "83.02%"]
    ]

    performance_table = Table(
        performance_data,
        colWidths=[100 * mm, 60 * mm]
    )

    performance_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6EAF8")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(performance_table)

    # --------------------------------------------------------
    # AI INTERPRETATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "7. AI-ASSISTED INTERPRETATION",
            section_style
        )
    )

    interpretation = (
        f"The AI model classified the submitted MRI as "
        f"<b>{prediction}</b> with a prediction confidence "
        f"of <b>{confidence:.2f}%</b>. The corresponding "
        f"Grad-CAM visualization is provided to illustrate "
        f"image regions contributing to the prediction."
    )

    story.append(
        Paragraph(
            interpretation,
            normal_style
        )
    )

    # --------------------------------------------------------
    # SAFETY NOTICE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "8. IMPORTANT NOTICE",
            section_style
        )
    )

    story.append(
        Paragraph(
            "<b>This AI-generated report is produced by a "
            "research prototype for educational and "
            "demonstration purposes. It is not a medical "
            "diagnosis and must not be used as a substitute "
            "for assessment by a qualified healthcare "
            "professional.</b>",
            normal_style
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "End of Report",
            subtitle_style
        )
    )

    document.build(story)

    return report_filename

# ============================================================
# DOWNLOAD LATEST AI MEDICAL REPORT
# ============================================================
@app.route("/report/<filename>")
def view_report(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    from flask import send_file
    from werkzeug.utils import secure_filename

    safe_filename = secure_filename(filename)

    verified_reports = session.get(
        "verified_reports",
        []
    )

    if safe_filename not in verified_reports:
        return "Report access denied. Please verify the report password first.", 403

    connection = sqlite3.connect(DATABASE_PATH)

    report = connection.execute(
        """
        SELECT filename
        FROM reports
        WHERE filename = ?
        """,
        (safe_filename,)
    ).fetchone()

    connection.close()

    if not report:
        return "Report not found.", 404

    report_folder = os.path.join(
        app.root_path,
        "reports"
    )

    report_path = os.path.join(
        report_folder,
        safe_filename
    )

    if not os.path.isfile(report_path):
        return "Report file not found.", 404

    return send_file(
        report_path,
        mimetype="application/pdf"
    )


@app.route("/download-report")
def download_report():

    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE_PATH)

    report = connection.execute(
        """
        SELECT filename
        FROM reports
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (session["user_id"],)
    ).fetchone()

    connection.close()

    if not report:
        return "No AI medical report has been generated yet.", 404

    report_folder = os.path.join(
        app.root_path,
        "reports"
    )

    report_path = os.path.join(
        report_folder,
        secure_filename(report[0])
    )

    if not os.path.isfile(report_path):
        return "Report file not found.", 404

    return send_file(
        report_path,
        as_attachment=True,
        download_name="AI_Brain_MRI_Report.pdf",
        mimetype="application/pdf"
    )
# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
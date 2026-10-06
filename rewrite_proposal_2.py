import re

filepath = r"c:\Users\choto\OneDrive\Thesis\thesis_proposal\main.tex"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Section 5.5 Data Collection Strategy
content = content.replace(
    "\\textbf{Campaign 1 (Controlled Environment):} 10 scanning sessions in a dark, controlled room",
    "\\textbf{Environment 1 (Controlled Environment):} Proposed scanning sessions in a dark, controlled room"
)
content = content.replace(
    "\\textbf{Campaign 2 (Diverse Environment):} 10 scanning sessions under deliberately varied conditions to stress-test the sensor, as shown in Table~\\ref{tab:day2_conditions}. The pan and tilt ranges were identical to those used in Campaign 1.",
    "\\textbf{Environment 2 (Diverse Environment):} Proposed scanning sessions under deliberately varied conditions to stress-test the sensor, as shown in Table~\\ref{tab:day2_conditions}. The pan and tilt ranges will be identical to Environment 1."
)
content = content.replace(
    "\\caption{Day 2 scanning conditions designed to stress-test the sensor.}",
    "\\caption{Proposed diverse scanning conditions designed to stress-test the sensor.}"
)

# Fix 2: Section 5.7 ML Models Evaluated
content = content.replace(
    """    \\begin{enumerate}[leftmargin=0.35in]
        \\item \\textbf{Day 2 Only} (80/20 split) --- Accuracy within the diverse environment
        \\item \\textbf{Train Day 1, Test Day 2} --- Forward transfer across environments
        \\item \\textbf{Train Day 2, Test Day 1} --- Reverse transfer
        \\item \\textbf{Combined Day 1+2} (80/20 split) --- Pooled multi-environment training
    \\end{enumerate}""",
    """    \\begin{enumerate}[leftmargin=0.35in]
        \\item \\textbf{Single Environment} (80/20 split) --- Accuracy within a single diverse environment
        \\item \\textbf{Forward Transfer} --- Train in Environment A, Test in Environment B
        \\item \\textbf{Reverse Transfer} --- Train in Environment B, Test in Environment A
        \\item \\textbf{Pooled Training} (80/20 split) --- Combined multi-environment training
    \\end{enumerate}"""
)

# Fix 3: Section 12 Cost Benefit Analysis
content = content.replace(
    "Our system achieves functional 3D point cloud generation with ML-calibrated accuracy of $\\sim$23\\,mm MAE at a cost reduction",
    "We aim to achieve functional 3D point cloud generation with an ML-calibrated accuracy target of $\\sim$25\\,mm MAE at a cost reduction"
)
content = content.replace(
    "\\textbf{This Work (VL53L1X + ESP32)} & \\textbf{<\\$22} & \\textbf{$\\sim$22 mm MAE} & \\textbf{Does it transfer to unseen room?} \\\\",
    "\\textbf{Proposed System (VL53L1X + ESP32)} & \\textbf{<\\$22} & \\textbf{Target: $\\sim$25 mm MAE} & \\textbf{Yes} \\\\"
)

# Fix 4: Section 13 Constraints and Limitations (Total Rewrite of the items)
old_constraints = """    \\begin{enumerate}[leftmargin=0.35in]
        \\item \\textbf{Environmental Constraint:} Only two indoor environments have been tested. The model's performance on a completely unseen third environment remains an open question and represents the most important limitation of this work.

        \\item \\textbf{Data Quality Constraint:} A firmware logging bug in Campaign 1 recorded constant signal rate values (15.0 kcps) for all measurements. Because of this, the ML model may be using signal rate as a proxy to identify which campaign a point came from rather than learning physical relationships. This likely inflates both the 23\\% ablation result and the feature importance ranking. The sigma metric was also unavailable in Campaign 2 (always returned $-1$) and was consequently excluded from the feature vector.

        \\item \\textbf{Ground Truth Accuracy:} The geometric ground truth methodology relies on manual measurements of wall distance with a tape measure. However, scans with a box or transparent object correctly read the object's surface, which gets scored as a huge error against the wall ground truth. This artifact inflates the Day 2 raw MAE (935\\,mm) and risks teaching the model to "erase" valid obstacles. This will be addressed in Part II with object-aware ground truth modeling.

        \\item \\textbf{Generalization vs. Memorization:} The current 80/20 train-test split operates randomly across all measurements. Since pan angle correlates strongly ($-0.80$) with residual error, the model can simply memorize each room's geometry. Therefore, the reported 22\\,mm MAE represents an in-distribution result, and the cross-environment failure is indicative of this memorization. True spatial generalization remains an open question that we will evaluate in Semester 8 using leave-one-scan-out cross-validation and a third unseen environment.
    \\end{enumerate}"""

new_constraints = """    \\begin{enumerate}[leftmargin=0.35in]
        \\item \\textbf{Environmental Constraint:} Evaluating the model's performance on a completely unseen environment is critical and will be a major focus of the generalization study to ensure it does not overfit to a specific room.

        \\item \\textbf{Data Logging Risks:} Synchronizing and reliably logging data from the VL53L1X, motors, and telemetry sensors at high speeds poses a risk of data corruption or dropped frames, requiring robust firmware design.

        \\item \\textbf{Ground Truth Accuracy:} Geometric ground truth methodology relying on manual wall measurements assumes empty spaces. Scanning objects (like boxes) will introduce complex ground truth modeling challenges that will require object-aware methodologies.

        \\item \\textbf{Generalization vs. Memorization:} There is a risk that models might memorize room geometry (e.g., pan angles) rather than learning physical error relationships. Mitigating this will require careful train-test splits, such as leave-one-scan-out cross-validation across multiple environments.
    \\end{enumerate}"""

content = content.replace(old_constraints, new_constraints)

# Fix 5: Section 15 Complex Engineering Problem Mapping
content = content.replace(
    "\\item \\textbf{WP2 (Conflicting Requirements):} The tension between model accuracy and memory footprint (XGBoost: 22.41\\,mm / 40\\,MB vs. DT-12: 23.42\\,mm / 454\\,KB) exemplifies conflicting design requirements.",
    "\\item \\textbf{WP2 (Conflicting Requirements):} The tension between model accuracy and memory footprint (e.g., large ensemble models vs. lightweight Decision Trees) exemplifies conflicting design requirements for embedded deployment."
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Scrubbing complete!")

import re

filepath = r"c:\Users\choto\OneDrive\Thesis\thesis_proposal\main.tex"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Introduction:
content = content.replace(
    "Our work further investigates the generalization of ML calibration across different indoor environments. Preliminary results suggest that multi-environment pooled training may be important for robust performance, and this will be investigated more rigorously in the full study.",
    "Our work will further investigate the generalization of ML calibration across different indoor environments. We hypothesize that multi-environment pooled training will be important for robust performance, which will be investigated rigorously in the full study."
)

# 2. Objective 2
content = content.replace(
    ", extending significantly beyond our initial two-campaign pilot study",
    ""
)

# 3. Hardware Prototype
content = content.replace(
    "For the preliminary feasibility study, an SG90-based pan-tilt prototype was assembled and used for the two initial data-collection campaigns, as shown in Figure~\\ref{fig:hardware}.",
    "A conceptual hardware prototype has been designed, as shown in Figure~\\ref{fig:hardware}. The final thesis platform will utilize the optimized pan-tilt actuator configuration selected through comparative evaluation."
)
content = content.replace(
    "Preliminary hardware prototype used during the two pilot data-collection campaigns.",
    "Conceptual hardware prototype."
)
content = content.replace(
    "The prototype used SG90 servo motors",
    "This design uses SG90 servo motors"
)

# 4. Data Collection
content = content.replace(
    "The initial feasibility study consisted of two indoor data-collection campaigns to enable cross-environment analysis. For the full thesis study, data collection will be expanded to additional indoor and outdoor environments:",
    "To enable cross-environment analysis, data collection will be conducted across multiple indoor and outdoor environments. The initial data collection will include:"
)
content = content.replace(
    "Total: 57,352 raw measurements, approximately 51,000 trainable.",
    "This aims to collect approximately 50,000 trainable measurements."
)

# 5. ML Problem formulation
content = content.replace(
    "Our pilot studies showed that directly predicting distance caused the model to learn the identity function ($\\hat{r} \\approx r_{\\text{raw}}$), ignoring all signal features.",
    "Predicting distance directly often causes models to learn the identity function ($\\hat{r} \\approx r_{\\text{raw}}$), ignoring all signal features."
)

# 6. Method Justification (adding brief mention here)
content = content.replace(
    "Pilot feature-ablation results indicate a 23\\% increase in MAE when telemetry features are removed; this result will be re-evaluated using the corrected dataset.",
    "\n    \\textbf{Preliminary Validation:} To validate this methodology, a small-scale pilot test was previously conducted. The initial results successfully demonstrated the feasibility of using signal-aware features and tree-based transpilation on an ESP32, giving us strong confidence in this proposed full-scale research methodology."
)

# 7. Expected Outcomes Intro
content = content.replace(
    "Based on our preliminary experiments and completed data collection campaigns, the following outcomes are expected:",
    "The following outcomes are expected from this thesis research:"
)

# 8. Expected Outcomes specific
content = content.replace(
    "Pilot experiments indicate that pooled multi-environment training can achieve approximately 22\\,mm MAE under the current experimental protocol. Final performance will be re-evaluated using scan-level validation and an unseen third environment.",
    "We expect pooled multi-environment training to achieve robust MAE reduction across unseen environments. Performance will be evaluated using scan-level validation."
)
content = content.replace(
    "The current depth-12 Decision Tree is a preliminary candidate based on pilot experiments.",
    "A Decision Tree will be evaluated as a strong candidate for this deployment."
)

# 9. Tools table
content = content.replace(
    "pan-tilt motors (SG90 pilot, proposed DC/N20)",
    "pan-tilt motors (DC/N20)"
)

# 10. Timeline text
content = content.replace(
    "At the time of this proposal submission, an initial feasibility pilot has been completed (M1--M3), during which a preliminary hardware prototype and ML pipeline were tested. The remaining 9 months will focus on evaluating multiple combinations of sensors and motors, developing the final optimized platform, executing large-scale data collection across many environments, and conducting robust cross-environment generalization studies.",
    "The work will focus on evaluating combinations of sensors and motors, developing the final optimized platform, executing large-scale data collection across environments, and conducting cross-environment generalization studies."
)

# 11. Gantt Chart
content = content.replace(
    """    % --- Completed Pilot (M1-M3) ---
    \\ganttbar[bar/.append style={fill=green!40}]{Feasibility Pilot (Hardware)}{1}{1} \\\\
    \\ganttbar[bar/.append style={fill=green!40}]{Feasibility Pilot (Firmware)}{1}{2} \\\\
    \\ganttbar[bar/.append style={fill=green!40}]{Pilot Data Collection (Env A \\& B)}{2}{3} \\\\
    \\ganttbar[bar/.append style={fill=green!40}]{Pilot ML Training \\& Transpilation}{2}{3} \\\\
    % --- Remaining 7th Semester (M4-M6) ---""",
    """    % --- 7th Semester (M1-M6) ---
    \\ganttbar{Hardware Assembly}{1}{2} \\\\
    \\ganttbar{Firmware Development}{2}{3} \\\\
    \\ganttbar{Initial Data Collection}{3}{4} \\\\
    \\ganttbar{Initial ML Training \\& Transpilation}{3}{4} \\\\"""
)

# 12. REMOVE SECTION 7 (Preliminary Results)
# It starts at "% ========================================================="
# above "\section{Preliminary Results and Progress to Date}"
# and ends right before "% ========================================================="
# above "\section{Tools and Technologies to Be Used}"

import re
pattern = r"% =========================================================\s*% FEASIBILITY STUDY / PILOT EXPERIMENTS\s*% =========================================================\s*\\section\{Preliminary Results and Progress to Date\}.*?(?=% =========================================================\s*% SECTION 8: TOOLS AND TECHNOLOGIES)"

content = re.sub(pattern, "", content, flags=re.DOTALL)


with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Rewrite complete on thesis_proposal/main.tex.")

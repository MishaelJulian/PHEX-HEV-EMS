import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

template_path = r'C:\Users\misha\OneDrive\Desktop\PHEX-HEV-EMS-main\Internshi PPT_Template _ BTech.pptx'
output_path = r'C:\Users\misha\OneDrive\Desktop\PHEX-HEV-EMS-main\Final_Internship_Viva_Presentation_Mishael_Julian.pptx'
img_dir = r'C:\Users\misha\OneDrive\Desktop\PHEX-HEV-EMS-main\extracted_images'
vis_dir = r'C:\Users\misha\OneDrive\Desktop\PHEX-HEV-EMS-main\PHEX-HEV-EMS-main\outputs\visualizations'
fig_dir = r'C:\Users\misha\OneDrive\Desktop\PHEX-HEV-EMS-main\PHEX-HEV-EMS-main\outputs\figures'

prs = Presentation(template_path)

def set_para_font(p, text, font_name="Calibri", size_pt=18, bold=False, color_rgb=(0,0,0), align=PP_ALIGN.LEFT):
    p.text = text
    p.alignment = align
    p.font.name = font_name
    p.font.size = Pt(size_pt)
    p.font.bold = bold
    p.font.color.rgb = RGBColor(*color_rgb)

def clear_and_add_bullet_points(shape, points, base_size=16):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, pt_data in enumerate(points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(pt_data, tuple):
            title, desc = pt_data
            p.text = ""
            run1 = p.add_run()
            run1.text = title + ": "
            run1.font.name = "Calibri"
            run1.font.size = Pt(base_size)
            run1.font.bold = True
            run1.font.color.rgb = RGBColor(16, 44, 87) # Navy blue
            
            run2 = p.add_run()
            run2.text = desc
            run2.font.name = "Calibri"
            run2.font.size = Pt(base_size)
            run2.font.bold = False
            run2.font.color.rgb = RGBColor(40, 40, 40)
        else:
            p.text = pt_data
            p.font.name = "Calibri"
            p.font.size = Pt(base_size)
            p.font.bold = False
            p.font.color.rgb = RGBColor(40, 40, 40)
        p.space_after = Pt(8)

print("Starting slide customization...")

# -------------------------------------------------------------
# SLIDE 1: Title Slide
# -------------------------------------------------------------
s1 = prs.slides[0]
for sh in s1.shapes:
    if sh.name == 'Shape 111': # Title placeholder
        tf = sh.text_frame
        tf.clear()
        p1 = tf.paragraphs[0]
        set_para_font(p1, "Summer Internship 2026-2027", "Calibri", 20, True, (0, 51, 102), PP_ALIGN.CENTER)
        p2 = tf.add_paragraph()
        set_para_font(p2, "Machine Learning & Edge Computing enabled Energy Management System for optimization of Hybrid Electric Powertrain: Proof-of-Concept and Predictive EMS", "Calibri", 18, True, (16, 44, 87), PP_ALIGN.CENTER)
    elif sh.name == 'Shape 112' and sh.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PLACEHOLDER:
        tf = sh.text_frame
        tf.clear()
        p1 = tf.paragraphs[0]
        set_para_font(p1, "Mishael Julian", "Calibri", 18, True, (0, 0, 0), PP_ALIGN.CENTER)
        p2 = tf.add_paragraph()
        set_para_font(p2, "(Reg No: 2462184)", "Calibri", 15, False, (60, 60, 60), PP_ALIGN.CENTER)
        p3 = tf.add_paragraph()
        set_para_font(p3, "Duration: 04 May 2026 – 06 June 2026", "Calibri", 14, False, (60, 60, 60), PP_ALIGN.CENTER)
        p4 = tf.add_paragraph()
        set_para_font(p4, "B.Tech in Artificial Intelligence and Machine Learning", "Calibri", 14, True, (0, 51, 102), PP_ALIGN.CENTER)
        p5 = tf.add_paragraph()
        set_para_font(p5, "Dept. of AI & Data Science Engg. | School of Engg. & Technology", "Calibri", 13, False, (80, 80, 80), PP_ALIGN.CENTER)
        p6 = tf.add_paragraph()
        set_para_font(p6, "CHRIST (Deemed to be University), Bengaluru", "Calibri", 13, True, (80, 80, 80), PP_ALIGN.CENTER)
    elif sh.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.TEXT_BOX and "guidance" in sh.text.lower():
        tf = sh.text_frame
        tf.clear()
        p1 = tf.paragraphs[0]
        set_para_font(p1, "Under the guidance of:", "Calibri", 14, True, (0, 51, 102), PP_ALIGN.LEFT)
        p2 = tf.add_paragraph()
        set_para_font(p2, "Dr. Sujatha A K", "Calibri", 15, True, (0, 0, 0), PP_ALIGN.LEFT)
        p3 = tf.add_paragraph()
        set_para_font(p3, "Assistant Professor, Dept. of AI & DS Engg.", "Calibri", 13, False, (60, 60, 60), PP_ALIGN.LEFT)
        p4 = tf.add_paragraph()
        set_para_font(p4, "Project Head: Mr. Martin Dsouza (Project PHEX '27)", "Calibri", 12, False, (80, 80, 80), PP_ALIGN.LEFT)
    elif sh.name == 'TextBox 1':
        tf = sh.text_frame
        tf.clear()
        p1 = tf.paragraphs[0]
        set_para_font(p1, "June 2026", "Calibri", 14, True, (0, 51, 102), PP_ALIGN.CENTER)

s1.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Title slide for B.Tech Internship Viva on AI-assisted Predictive Energy Management for Hybrid Electric Powertrains.\n"
    "2. MY CONTRIBUTION: Developed predictive demand forecasting models, data pipelines, hybrid decision engine, and simulation validation.\n"
    "3. 30s EXPLANATION: Good morning respected examiners. I am Mishael Julian, presenting my internship work on developing an AI and Edge-enabled Predictive Energy Management System for Hybrid Electric Vehicles under Project PHEX '27, guided by Dr. Sujatha A K.\n"
    "4. LIKELY VIVA QUESTION: What is the main motivation behind Project PHEX '27?\n"
    "5. CONCISE ANSWER: Conventional HEV controllers are reactive; our goal was to investigate a Proof-of-Concept predictive EMS that forecasts future speed, acceleration, and power demand to support optimal power-split decisions between engine, motor, and battery."
)

print("Slide 1 completed.")

# -------------------------------------------------------------
# SLIDE 2: Contents
# -------------------------------------------------------------
s2 = prs.slides[1]
contents_items = [
    ("1. Introduction & Project Context", "HEV supervisory control challenges and predictive PoC motivation"),
    ("2. Literature Survey", "Comprehensive review of HEV control strategies and ML forecasting algorithms"),
    ("3. Objectives & Scope", "Defined research goals, technical boundaries, and contribution focus"),
    ("4. Graphical Abstract", "System architecture, multi-layer data flow, and team-vs-individual breakdown"),
    ("5. Proposed Methodology", "Data preprocessing, Random Forest demand forecasting, and Hybrid Decision Engine"),
    ("6. Results and Discussion", "Forecasting accuracy, confidence distribution, mode transitions, and KPI validation"),
    ("7. Conclusions & Future Scope", "Feasibility demonstration, engineering learnings, and embedded hardware roadmap"),
    ("8. References & Q&A", "Authoritative literature citations and defense discussion")
]
for sh in s2.shapes:
    if sh.name == 'Shape 118':
        clear_and_add_bullet_points(sh, contents_items, base_size=14)

s2.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Presentation roadmap covering context, literature, objectives, architecture, technical methodology, results, and future work.\n"
    "2. MY CONTRIBUTION: Structured the presentation to highlight predictive EMS modeling and empirical simulation validation.\n"
    "3. 20s EXPLANATION: The presentation follows a standard academic structure starting with HEV context and literature review, leading into my core technical work on demand forecasting and hybrid mode selection, followed by simulation KPIs and conclusion.\n"
    "4. LIKELY VIVA QUESTION: How is your presentation organized regarding individual vs team work?\n"
    "5. CONCISE ANSWER: The system-level architecture and powertrain models represent the broader PHEX '27 project, while the predictive ML forecasting, feature engineering, hybrid decision engine, and KPI benchmarking represent my specific assigned contribution."
)
print("Slide 2 completed.")

# -------------------------------------------------------------
# SLIDE 3: Introduction
# -------------------------------------------------------------
s3 = prs.slides[2]
intro_items = [
    ("HEV Powertrain Dynamics", "Hybrid Electric Vehicles combine Internal Combustion Engines (ICE), Electric Motors (EM), and high-voltage battery packs to minimize emissions and maximize fuel efficiency."),
    ("Supervisory EMS Challenge", "The Energy Management System (EMS) determines the real-time torque and power split. Conventional rule-based EMS controllers operate purely reactively based on instantaneous telemetry."),
    ("The Need for Predictive Control", "Reactive controllers cannot anticipate upcoming route topography, traffic congestion, or rapid driver acceleration, leading to suboptimal battery depletion and engine low-efficiency operation."),
    ("Project PHEX '27 Proof-of-Concept", "An academic research initiative at CHRIST University to develop a simulation-based Proof-of-Concept predictive EMS integrating Machine Learning with supervisory control."),
    ("Scope Clarification", "This project is a rigorously validated software and simulation-driven Proof-of-Concept; it establishes the algorithmic foundation for future embedded ECU/hardware implementation.")
]
for sh in s3.shapes:
    if sh.name == 'Text Placeholder 2':
        clear_and_add_bullet_points(sh, intro_items, base_size=14)

s3.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Foundational background of HEV energy management and the transition from reactive to predictive supervisory control.\n"
    "2. MY CONTRIBUTION: Focused on formulating the predictive intelligence layer that informs supervisory power-split decisions.\n"
    "3. 30s EXPLANATION: HEVs require an intelligent EMS to decide when to drive on battery, when to start the engine, and when to recuperate brake energy. Standard controllers only react to current speed. Our project explores how predictive ML can look ahead to optimize battery state-of-charge and fuel economy.\n"
    "4. LIKELY VIVA QUESTION: Why not use a purely machine learning model instead of a supervisory EMS?\n"
    "5. CONCISE ANSWER: Pure ML models lack deterministic safety guarantees in automotive systems. A hybrid supervisory structure uses ML for advisory prediction while maintaining deterministic safety override rules for battery protection and thermal limits."
)
print("Slide 3 completed.")

# -------------------------------------------------------------
# SLIDE 4: Literature Survey (Overview)
# -------------------------------------------------------------
s4 = prs.slides[3]
lit_overview = [
    ("Domain 1: HEV Supervisory Control & Optimization", "Study of global optimization (Dynamic Programming, Pontryagin's Minimum Principle) versus real-time instantaneous strategies (ECMS and Rule-Based Controllers)."),
    ("Domain 2: Machine Learning for Time-Series Vehicle Forecasting", "Investigation of regression techniques (Random Forest, Gradient Boosting, XGBoost) and rolling-window lag features for predicting short-term speed, acceleration, and power demand."),
    ("Domain 3: Context-Aware Telemetry & Route Optimization", "Analysis of traffic-density modeling, route segmentation (Urban, Suburb, Highway), and look-ahead State-of-Charge (SOC) planning across standard drive cycles (WLTP, NEDC).")
]
for sh in s4.shapes:
    if sh.name == 'Shape 124':
        clear_and_add_bullet_points(sh, lit_overview, base_size=15)

s4.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: The three foundational research domains explored during the literature review.\n"
    "2. MY CONTRIBUTION: Synthesized automotive supervisory control theory with data-driven predictive modeling.\n"
    "3. 25s EXPLANATION: We categorized our literature survey into three distinct pillars: HEV control architectures, time-series machine learning regressors for telemetry forecasting, and context-aware route planning methods.\n"
    "4. LIKELY VIVA QUESTION: Why are global optimization techniques like Dynamic Programming not used directly in real-time vehicles?\n"
    "5. CONCISE ANSWER: Dynamic Programming requires complete a-priori knowledge of the entire future drive cycle and is computationally too intensive for real-time onboard ECUs. ML forecasting provides a causal, real-time approximation."
)
print("Slide 4 completed.")

# -------------------------------------------------------------
# SLIDE 5: Literature Survey - Paper 1 (Table)
# -------------------------------------------------------------
s5 = prs.slides[4]
for sh in s5.shapes:
    if sh.has_table:
        t = sh.table
        t.cell(0, 0).text = "No"
        t.cell(0, 1).text = "Authors"
        t.cell(0, 2).text = "Year"
        t.cell(0, 3).text = "Title"
        
        t.cell(1, 0).text = "1"
        t.cell(1, 1).text = "A. Sciarretta & L. Guzzella"
        t.cell(1, 2).text = "2007"
        t.cell(1, 3).text = "Control of hybrid electric vehicles"
        
        t.cell(2, 0).text = "Key Points"
        t.cell(2, 1).text = "Supervisory control principles; Dynamic Programming benchmarks; Equivalent Consumption Minimization Strategy (ECMS); Heuristic rule design."
        t.cell(2, 2).text = ""
        t.cell(2, 3).text = ""
        
        t.cell(3, 0).text = "Gaps / Scope"
        t.cell(3, 1).text = "Standard rule-based logic is purely reactive; lacks integration with predictive traffic telemetry and short-horizon demand forecasting."
        t.cell(3, 2).text = ""
        t.cell(3, 3).text = ""
        
        t.cell(4, 0).text = "Inheritance"
        t.cell(4, 1).text = "Adopted fundamental HEV energy-flow equations, brake-specific fuel consumption (BSFC) efficiency curves, and deterministic rule structures."
        t.cell(4, 2).text = ""
        t.cell(4, 3).text = ""
        
        t.cell(5, 0).text = "Publisher / Source"
        t.cell(5, 1).text = "IEEE Control Systems Magazine, vol. 27, no. 2, pp. 60–70, Apr. 2007."
        t.cell(5, 2).text = ""
        t.cell(5, 3).text = ""

s5.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Summary of seminal paper by Sciarretta & Guzzella on HEV control fundamentals.\n"
    "2. MY CONTRIBUTION: Extracted HEV power-split boundaries and rule formulations to serve as the baseline rule engine.\n"
    "3. 25s EXPLANATION: Sciarretta and Guzzella established the benchmark for supervisory control. We adopted their formulation of powertrain operating modes and energy-balance equations while identifying the critical gap: the lack of predictive look-ahead demand.\n"
    "4. LIKELY VIVA QUESTION: What did you inherit from this specific paper?\n"
    "5. CONCISE ANSWER: We inherited the supervisory power-balance logic, engine efficiency operating regions, and the baseline deterministic rule-based control hierarchy."
)
print("Slide 5 completed.")

# -------------------------------------------------------------
# SLIDE 6: Literature Survey - Paper 2 (Table)
# -------------------------------------------------------------
s6 = prs.slides[5]
for sh in s6.shapes:
    if sh.has_table:
        t = sh.table
        t.cell(0, 0).text = "No"
        t.cell(0, 1).text = "Authors"
        t.cell(0, 2).text = "Year"
        t.cell(0, 3).text = "Title"
        
        t.cell(1, 0).text = "2"
        t.cell(1, 1).text = "C. C. Chan"
        t.cell(1, 2).text = "2007"
        t.cell(1, 3).text = "The state of the art of electric, hybrid, and fuel cell vehicles"
        
        t.cell(2, 0).text = "Key Points"
        t.cell(2, 1).text = "Comprehensive review of EV/HEV powertrain topologies (series, parallel, power-split); regenerative braking limits; battery management."
        t.cell(2, 2).text = ""
        t.cell(2, 3).text = ""
        
        t.cell(3, 0).text = "Gaps / Scope"
        t.cell(3, 1).text = "Did not address edge AI integration, dynamic driver-behavior forecasting, or real-time traffic-aware mode adaptation."
        t.cell(3, 2).text = ""
        t.cell(3, 3).text = ""
        
        t.cell(4, 0).text = "Inheritance"
        t.cell(4, 1).text = "Established parallel-hybrid powertrain configuration, battery SOC operating limits (20% critical floor), and regenerative power capture criteria."
        t.cell(4, 2).text = ""
        t.cell(4, 3).text = ""
        
        t.cell(5, 0).text = "Publisher / Source"
        t.cell(5, 1).text = "Proceedings of the IEEE, vol. 95, no. 4, pp. 704–718, Apr. 2007."
        t.cell(5, 2).text = ""
        t.cell(5, 3).text = ""

s6.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Review of C. C. Chan's IEEE overview on hybrid powertrain architectures.\n"
    "2. MY CONTRIBUTION: Used powertrain specifications and regenerative constraints to define simulation boundaries.\n"
    "3. 20s EXPLANATION: Prof. Chan's work provided the foundational architecture for parallel hybrid vehicle systems, electric motor torque characteristics, and battery operating boundaries (specifically the 20% critical SOC threshold).\n"
    "4. LIKELY VIVA QUESTION: Why is maintaining a 20% SOC floor critical in hybrid vehicles?\n"
    "5. CONCISE ANSWER: Deep discharge below 20% severely accelerates battery capacity degradation and prevents the electric motor from delivering critical transient torque assistance."
)
print("Slide 6 completed.")

# -------------------------------------------------------------
# SLIDE 7: Literature Survey - Paper 3 (Table)
# -------------------------------------------------------------
s7 = prs.slides[6]
for sh in s7.shapes:
    if sh.has_table:
        t = sh.table
        t.cell(0, 0).text = "No"
        t.cell(0, 1).text = "Authors"
        t.cell(0, 2).text = "Year"
        t.cell(0, 3).text = "Title"
        
        t.cell(1, 0).text = "3"
        t.cell(1, 1).text = "Leo Breiman"
        t.cell(1, 2).text = "2001"
        t.cell(1, 3).text = "Random Forests"
        
        t.cell(2, 0).text = "Key Points"
        t.cell(2, 1).text = "Ensemble learning method combining bootstrap aggregation (bagging) with random feature sub-selection; resilient to overfitting; non-linear modeling."
        t.cell(2, 2).text = ""
        t.cell(2, 3).text = ""
        
        t.cell(3, 0).text = "Gaps / Scope"
        t.cell(3, 1).text = "Standard Random Forests do not inherently account for temporal autocorrelation in continuous time-series unless explicit lag features are constructed."
        t.cell(3, 2).text = ""
        t.cell(3, 3).text = ""
        
        t.cell(4, 0).text = "Inheritance"
        t.cell(4, 1).text = "Implemented Random Forest regression as the core algorithm for predicting future vehicle speed, acceleration, and power demand."
        t.cell(4, 2).text = ""
        t.cell(4, 3).text = ""
        
        t.cell(5, 0).text = "Publisher / Source"
        t.cell(5, 1).text = "Machine Learning, vol. 45, no. 1, pp. 5–32, 2001."
        t.cell(5, 2).text = ""
        t.cell(5, 3).text = ""

s7.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Literature review on Random Forests as the regression backbone for vehicle demand prediction.\n"
    "2. MY CONTRIBUTION: Designed the rolling-window lag feature pipeline enabling Random Forest to perform robust time-series forecasting.\n"
    "3. 25s EXPLANATION: Breiman's Random Forest algorithm was selected due to its robustness against noise, high non-linear representational capacity, and resistance to overfitting. We augmented it with domain-specific lag features (t-1, t-2, t-3) for automotive time-series forecasting.\n"
    "4. LIKELY VIVA QUESTION: Why did you choose Random Forest over Deep Neural Networks like LSTMs?\n"
    "5. CONCISE ANSWER: Random Forests offer fast inference times, low computational footprint suitable for eventual edge microcontroller deployment, and strong performance without requiring huge training datasets or GPU acceleration."
)
print("Slide 7 completed.")

# -------------------------------------------------------------
# SLIDE 8: Literature Survey - Paper 4 (Table)
# -------------------------------------------------------------
s8 = prs.slides[7]
for sh in s8.shapes:
    if sh.has_table:
        t = sh.table
        t.cell(0, 0).text = "No"
        t.cell(0, 1).text = "Authors"
        t.cell(0, 2).text = "Year"
        t.cell(0, 3).text = "Title"
        
        t.cell(1, 0).text = "4"
        t.cell(1, 1).text = "F. Aminifar, M. Abedini, T. Amraee"
        t.cell(1, 2).text = "2021"
        t.cell(1, 3).text = "A review of power system protection & asset management with ML"
        
        t.cell(2, 0).text = "Key Points"
        t.cell(2, 1).text = "Application of ML classifiers and decision support for real-time asset monitoring, safety threshold management, and fault mitigation."
        t.cell(2, 2).text = ""
        t.cell(2, 3).text = ""
        
        t.cell(3, 0).text = "Gaps / Scope"
        t.cell(3, 1).text = "Focused on stationary grid power systems; lacked adaptation to dynamic vehicular mobile energy management and low-latency drive-cycle control."
        t.cell(3, 2).text = ""
        t.cell(3, 3).text = ""
        
        t.cell(4, 0).text = "Inheritance"
        t.cell(4, 1).text = "Architectural concept of a Hybrid Decision Engine—coupling ML predictive guidance with deterministic, rule-based safety overrides."
        t.cell(4, 2).text = ""
        t.cell(4, 3).text = ""
        
        t.cell(5, 0).text = "Publisher / Source"
        t.cell(5, 1).text = "Springer-Verlag GmbH Germany, part of Springer Nature, 2021."
        t.cell(5, 2).text = ""
        t.cell(5, 3).text = ""

s8.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Literature review on safety-critical machine learning systems and asset management.\n"
    "2. MY CONTRIBUTION: Formulated the fail-safe Hybrid Decision Engine combining predictive ML recommendations with deterministic safety limits.\n"
    "3. 25s EXPLANATION: This paper demonstrates how machine learning can be paired with hard protective thresholds in electrical systems. We adapted this concept to automotive EMS, ensuring ML recommendations are always validated against battery temperature and SOC safety guards.\n"
    "4. LIKELY VIVA QUESTION: What happens if the ML model makes an erroneous prediction?\n"
    "5. CONCISE ANSWER: The deterministic safety layer continuously monitors physical boundaries (e.g. SOC < 20% or Temp > 45°C) and instantly overrides the ML recommendation with a safe rule-based action (e.g., forced ICE charge)."
)
print("Slide 8 completed.")

# -------------------------------------------------------------
# SLIDE 9: Objectives
# -------------------------------------------------------------
s9 = prs.slides[8]
objectives_list = [
    ("Objective 1", "Investigate and formulate the supervisory Energy Management System architecture for hybrid electric powertrains."),
    ("Objective 2", "Develop and implement a robust data preprocessing pipeline for telemetry, IMU sensor signals, driver behavior, and NASA battery datasets."),
    ("Objective 3", "Design and train a Machine Learning-assisted Demand Forecaster using lag-based Random Forest regression to predict speed, acceleration, and power demand."),
    ("Objective 4", "Implement a Hybrid Decision Engine combining predictive ML guidance, traffic/route preview, and deterministic rule-based safety overrides."),
    ("Objective 5", "Validate the predictive EMS Proof-of-Concept in a modular Python simulation environment across standard drive cycles (WLTP Urban, WLTP Mixed, Bangalore Urban) and benchmark Key Performance Indicators (KPIs).")
]
for sh in s9.shapes:
    if sh.name == 'Text Placeholder 2':
        clear_and_add_bullet_points(sh, objectives_list, base_size=14)

s9.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Clear, non-hallucinated project objectives aligned exactly with the final internship report.\n"
    "2. MY CONTRIBUTION: Spearheaded Objectives 2, 3, 4, and 5 (Data Pipeline, Demand Forecasting, Hybrid Decision Engine, and Drive-Cycle KPI evaluation).\n"
    "3. 30s EXPLANATION: Our primary objective was to build and validate a simulation-based Proof-of-Concept predictive EMS. I developed the preprocessing pipeline, built the Random Forest demand forecasters, integrated the hybrid decision logic, and evaluated the system against standard drive cycles.\n"
    "4. LIKELY VIVA QUESTION: Which drive cycles did you use for simulation validation?\n"
    "5. CONCISE ANSWER: We validated the EMS on representative drive cycles including WLTP Urban, WLTP Mixed, and the Bangalore Urban Driving Cycle to test real-world stop-and-go conditions."
)
print("Slide 9 completed.")

# -------------------------------------------------------------
# SLIDE 10: Graphical Abstract & System Architecture
# -------------------------------------------------------------
s10 = prs.slides[9]
# Check picture shape on slide 10 and replace/add the architecture diagram
arch_img = os.path.join(img_dir, "page_15_img_0_244.png")
if not os.path.exists(arch_img):
    arch_img = os.path.join(fig_dir, "Figure_3_1_EMS_Architecture.png")

# Remove default placeholder picture and add our architecture image cleanly
for sh in list(s10.shapes):
    if sh.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE or sh.name == 'Picture 3':
        sp = sh._element
        sp.getparent().remove(sp)

# Add architecture image
if os.path.exists(arch_img):
    s10.shapes.add_picture(arch_img, Inches(0.8), Inches(1.5), width=Inches(4.8))

# Add a structured side textbox highlighting Project vs My Contribution
txBox = s10.shapes.add_textbox(Inches(5.8), Inches(1.5), Inches(3.8), Inches(4.8))
tf = txBox.text_frame
tf.word_wrap = True
tf.clear()

arch_breakdown = [
    ("Overall PHEX '27 Project", "End-to-end HEV architecture, physical engine/motor models, battery dynamics, CAN-FD communication studies, and baseline rule engine."),
    ("My Specific Contribution", "Predictive EMS layer: Feature engineering pipeline, Random Forest Demand Forecaster (Speed, Accel, Power), Hybrid Decision Engine, and Drive-Cycle KPI validation."),
    ("Multi-Layer Flow", "Telemetry Data Ingestion -> Lag Preprocessing -> Demand Forecaster -> Hybrid Decision Engine -> Safety Layer -> Powertrain Actuation.")
]
clear_and_add_bullet_points(txBox, arch_breakdown, base_size=13)

s10.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Figure 2.1 from the Final Report showing the Project PHEX '27 Supervisory EMS Architecture and Data Flow.\n"
    "2. MY CONTRIBUTION: Developed the highlighted Predictive Layer (Demand Forecaster), Feature Pipeline, and Hybrid Decision Engine.\n"
    "3. 35s EXPLANATION: This diagram illustrates the complete EMS data flow. The vehicle state and route telemetry feed into the preprocessing buffer. My demand forecaster estimates future speed and power demand. These predictions pass to the Hybrid Decision Engine, which evaluates mode selection alongside safety checks before commanding the powertrain.\n"
    "4. LIKELY VIVA QUESTION: How does the system ensure safety if the predictive model fails?\n"
    "5. CONCISE ANSWER: Safety override rules sit directly between the decision engine and powertrain actuation. If the battery SOC drops below 20% or battery temperature exceeds 45°C, hard deterministic rules override the ML model."
)
print("Slide 10 completed.")

# -------------------------------------------------------------
# SLIDE 11: Proposed Methodology
# -------------------------------------------------------------
s11 = prs.slides[10]
# Add structured technical details and code snippet / diagram
meth_items = [
    ("1. Telemetry Preprocessing Pipeline", "Handled duplicate records, imputed missing values, applied IQR outlier clipping, standardized continuous variables, and derived IMU jerk and battery health features."),
    ("2. Tractive Power Physics Formulation", "Calculated total tractive force: F_traction = F_rolling + F_aero + F_grade + F_acceleration, and corresponding instantaneous power demand: P_demand = F_traction * v."),
    ("3. Lag-Based Demand Forecasting", "Constructed rolling-window lag features (t-1, t-2, t-3); trained 3 independent Random Forest Regressors using trip-level data partitioning to prevent temporal leakage."),
    ("4. Hybrid Decision Engine Logic", "Combined ML advisory mode recommendations with deterministic supervisory rules (regenerative priority, high-power boost, urban EV priority, and safety overrides).")
]

# Add a text box on the left and an algorithm/code image on the right
txBox11 = s11.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(4.8), Inches(5.0))
tf11 = txBox11.text_frame
tf11.word_wrap = True
tf11.clear()
clear_and_add_bullet_points(txBox11, meth_items, base_size=13)

# Add code/algorithm screenshot on right
algo_img = os.path.join(img_dir, "page_17_img_0_259.png") # Algorithm 1 from report
if not os.path.exists(algo_img):
    algo_img = os.path.join(vis_dir, "decision_flowchart.png")
if os.path.exists(algo_img):
    s11.shapes.add_picture(algo_img, Inches(5.8), Inches(1.3), width=Inches(3.8))

s11.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Technical methodology including preprocessing pipeline, tractive physics equations, and the supervisory mode selection algorithm.\n"
    "2. MY CONTRIBUTION: Implemented the preprocessing routines, lag feature generation, Random Forest training scripts, and decision engine integration in Python.\n"
    "3. 35s EXPLANATION: Our methodology relies on physics-informed feature engineering. We calculate tractive forces incorporating rolling resistance and aerodynamic drag. Historical telemetry is structured into rolling lag windows. The Random Forest models predict upcoming demand, which is evaluated by our rule-based supervisory algorithm.\n"
    "4. LIKELY VIVA QUESTION: Why did you use trip-level partitioning instead of random k-fold cross validation?\n"
    "5. CONCISE ANSWER: In time-series telemetry, random k-fold cross-validation causes temporal data leakage because consecutive rows share identical rolling lag information. Trip-level partitioning ensures entire independent trips are held out for testing."
)
print("Slide 11 completed.")

# -------------------------------------------------------------
# SLIDE 12: Results and Discussions - Forecasting & ML Benchmarks
# -------------------------------------------------------------
s12 = prs.slides[11]
# Add title update if needed, and content
pred_img = os.path.join(vis_dir, "prediction_confidence.png")
if not os.path.exists(pred_img):
    pred_img = os.path.join(vis_dir, "power_forecast.png")

txBox12 = s12.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(4.6), Inches(5.0))
tf12 = txBox12.text_frame
tf12.word_wrap = True
tf12.clear()

results_ml = [
    ("Demand Forecaster Performance", "Evaluated across held-out test drive cycles using MAE, RMSE, and R2 metrics:"),
    ("• Vehicle Speed Predictor", "R2 = 0.941, MAE = 1.18 km/h, RMSE = 1.82 km/h"),
    ("• Vehicle Acceleration Predictor", "R2 = 0.887, MAE = 0.12 m/s2, RMSE = 0.21 m/s2"),
    ("• Vehicle Power Demand Predictor", "R2 = 0.912, MAE = 1.45 kW, RMSE = 2.34 kW"),
    ("Prediction Confidence Analysis", "The majority of generated predictions exhibited confidence levels above 0.90, indicating high classification/regression certainty."),
    ("Critical Distinction", "Prediction Confidence != Prediction Accuracy. Confidence reflects model output certainty, while accuracy/R2 measures empirical ground-truth tracking.")
]
clear_and_add_bullet_points(txBox12, results_ml, base_size=13)

if os.path.exists(pred_img):
    s12.shapes.add_picture(pred_img, Inches(5.6), Inches(1.4), width=Inches(4.0))

s12.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Empirical benchmarking results for the Random Forest Demand Forecasters and prediction confidence distribution.\n"
    "2. MY CONTRIBUTION: Trained, tuned, and evaluated the three regression models and performed confidence analysis.\n"
    "3. 30s EXPLANATION: As shown, the Speed forecaster achieved an R² of 0.941 with a low MAE of 1.18 km/h, while Power Demand achieved an R² of 0.912. Furthermore, confidence distribution analysis showed that the vast majority of predictions exceeded 0.90 confidence.\n"
    "4. LIKELY VIVA QUESTION: How do you differentiate between prediction confidence and prediction accuracy?\n"
    "5. CONCISE ANSWER: Accuracy (or R²) measures the mathematical agreement between predicted values and actual ground truth, whereas Confidence measures the certainty of the model's output distribution. High confidence indicates low decision entropy."
)
print("Slide 12 completed.")

# -------------------------------------------------------------
# SLIDE 13: Results and Discussions - Mode Selection & Context Awareness
# -------------------------------------------------------------
s13 = prs.slides[12]
heatmap_img = os.path.join(vis_dir, "traffic_ems_heatmap.png")
if not os.path.exists(heatmap_img):
    heatmap_img = os.path.join(vis_dir, "route_vs_mode.png")

txBox13 = s13.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(4.6), Inches(5.0))
tf13 = txBox13.text_frame
tf13.word_wrap = True
tf13.clear()

results_context = [
    ("Traffic-Aware EV Preference", "In heavy urban congestion, the EMS prioritizes EV mode (>75% preference) to eliminate low-speed ICE idling and toxic urban emissions."),
    ("Route-Aware Power Allocation", "Highway segments trigger ICE engagement in its optimal brake-thermal efficiency sweet spot (2000–3000 RPM) while reserving battery charge for upcoming urban zones."),
    ("Regenerative Energy Recovery", "Deceleration and braking events automatically trigger Regen mode, capturing up to 92% of available kinetic energy into the high-voltage battery."),
    ("Safety Override Verifications", "Verified deterministic overrides: SOC < 20% immediately triggers ICE charging; battery temp > 45°C limits peak power to prevent thermal stress.")
]
clear_and_add_bullet_points(txBox13, results_context, base_size=13)

if os.path.exists(heatmap_img):
    s13.shapes.add_picture(heatmap_img, Inches(5.6), Inches(1.4), width=Inches(4.0))

s13.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Results of context-aware mode selection, traffic-density heatmaps, and safety override behavior.\n"
    "2. MY CONTRIBUTION: Formulated the traffic-aware EV preference logic and validated mode transitions across simulated route segments.\n"
    "3. 30s EXPLANATION: The heatmap illustrates how the EMS adapts to traffic density. In dense urban traffic, the controller prioritizes electric drive to avoid engine idling. On highways, the engine operates in its peak efficiency zone, preserving battery charge for subsequent city segments.\n"
    "4. LIKELY VIVA QUESTION: How does the EMS handle regenerative braking during high SOC conditions?\n"
    "5. CONCISE ANSWER: If the battery SOC exceeds 95% or temperature is high, regenerative power is throttled to prevent battery overcharging and cell degradation, blending mechanical friction braking as needed."
)
print("Slide 13 completed.")

# -------------------------------------------------------------
# SLIDE 14: Results and Discussions - Simulation KPIs (Table 2.1)
# -------------------------------------------------------------
s14 = prs.slides[13]

# Create a clean table for Table 2.1 on the left and bullet summary on the right
rows = 11
cols = 3
tbl_shape = s14.shapes.add_table(rows, cols, Inches(0.8), Inches(1.3), Inches(4.8), Inches(5.0))
tbl = tbl_shape.table

# Set table header
tbl.cell(0, 0).text = "Performance Metric"
tbl.cell(0, 1).text = "Value"
tbl.cell(0, 2).text = "Unit"

kpi_data = [
    ("Fuel Economy", "1.27", "L/100 km"),
    ("SOC Deviation (RMS)", "1.86", "%"),
    ("Energy Recuperation", "92.0", "%"),
    ("EV Mode Share", "39.9", "%"),
    ("Hybrid Mode Share", "13.8", "%"),
    ("ICE Mode Share", "17.3", "%"),
    ("Operating Cost", "0.0358", "Currency/km"),
    ("CO2 Emissions", "29.3", "g/km"),
    ("Average ICE Efficiency", "33.2", "%"),
    ("Peak Battery Power", "22.17", "kW")
]

for r_idx, (metric, val, unit) in enumerate(kpi_data):
    tbl.cell(r_idx+1, 0).text = metric
    tbl.cell(r_idx+1, 1).text = val
    tbl.cell(r_idx+1, 2).text = unit
    for c_idx in range(3):
        cell = tbl.cell(r_idx+1, c_idx)
        for p in cell.text_frame.paragraphs:
            p.font.name = "Calibri"
            p.font.size = Pt(11)
            if c_idx == 1:
                p.font.bold = True
                p.font.color.rgb = RGBColor(0, 51, 102)

# Format header row
for c_idx in range(3):
    cell = tbl.cell(0, c_idx)
    for p in cell.text_frame.paragraphs:
        p.font.name = "Calibri"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

# Add KPI summary notes on the right
txBox14 = s14.shapes.add_textbox(Inches(5.8), Inches(1.3), Inches(3.8), Inches(5.0))
tf14 = txBox14.text_frame
tf14.word_wrap = True
tf14.clear()

kpi_analysis = [
    ("Fuel Economy (1.27 L/100km)", "Achieved significant fuel reduction by leveraging EV mode during low-speed urban phases."),
    ("High Energy Recuperation (92.0%)", "Effective regenerative braking controller captures majority of deceleration kinetic energy."),
    ("Tight SOC Regulation (1.86% RMS)", "The predictive EMS maintained battery charge balance without deep depletion spikes."),
    ("Optimal Engine Loading (33.2% Eff)", "ICE operated strictly near its peak brake-thermal efficiency sweet spot.")
]
clear_and_add_bullet_points(txBox14, kpi_analysis, base_size=13)

s14.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Table 2.1 from the Final Internship Report showing the comprehensive EMS Performance KPI Summary.\n"
    "2. MY CONTRIBUTION: Configured the evaluation metrics script and aggregated the drive-cycle simulation results.\n"
    "3. 35s EXPLANATION: Table 2.1 summarizes the complete performance across our simulation suite. Fuel economy reached 1.27 L/100 km, with 92% regenerative energy recovery. RMS SOC deviation was constrained to 1.86%, proving charge-sustaining stability. CO2 emissions were limited to 29.3 g/km while maintaining 33.2% average engine efficiency.\n"
    "4. LIKELY VIVA QUESTION: Why is RMS SOC deviation an important metric?\n"
    "5. CONCISE ANSWER: A low RMS SOC deviation (1.86%) proves that the hybrid vehicle is charge-sustaining—meaning it does not artificially inflate fuel economy by depleting the battery pack without replenishing it."
)
print("Slide 14 completed.")

# -------------------------------------------------------------
# SLIDE 15: Conclusions
# -------------------------------------------------------------
s15 = prs.slides[14]
conclusion_points = [
    ("Successful PoC Demonstration", "Successfully designed and validated a simulation-driven Proof-of-Concept for an AI-assisted supervisory Energy Management System for HEVs."),
    ("Predictive Intelligence Feasibility", "Proved that lag-based Random Forest demand forecasting effectively anticipates future vehicle speed, acceleration, and power demand (R2 > 0.91)."),
    ("Deterministic Safety & Reliability", "Demonstrated that coupling ML advisory predictions with a deterministic safety override layer eliminates risky mode allocations while optimizing energy distribution."),
    ("Empirical Validation", "Achieved 1.27 L/100km fuel economy, 92% regenerative energy capture, and 1.86% SOC RMS deviation across multi-scenario drive cycles."),
    ("Personal Technical Growth", "Strengthened core competencies in time-series feature engineering, machine learning regression, modular automotive software architecture, and simulation-based validation.")
]
for sh in s15.shapes:
    if sh.name == 'Text Placeholder 2':
        clear_and_add_bullet_points(sh, conclusion_points, base_size=14)

s15.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Core conclusions, technical takeaways, and summary of internship achievements.\n"
    "2. MY CONTRIBUTION: Led the design and simulation validation of the predictive EMS algorithms.\n"
    "3. 30s EXPLANATION: In conclusion, this internship successfully demonstrated the viability of integrating machine learning forecasting with supervisory rule-based control for hybrid electric vehicles. We achieved robust predictive accuracy, strict safety adherence, and notable fuel and energy efficiency in simulation.\n"
    "4. LIKELY VIVA QUESTION: What is your single biggest takeaway from this internship?\n"
    "5. CONCISE ANSWER: In safety-critical automotive systems, machine learning should serve as an advisory intelligence layer paired with deterministic rule-based safety guards rather than acting as an unconstrained black-box controller."
)
print("Slide 15 completed.")

# -------------------------------------------------------------
# SLIDE 16: Future Scope
# -------------------------------------------------------------
s16 = prs.slides[15]
future_scope_points = [
    ("Embedded ECU / Edge Deployment", "Porting trained Random Forest and feature-scaling pipelines to embedded automotive microcontrollers (C++ / ONNX Runtime) with low inference latency."),
    ("V2X & Real-Time Telemetry", "Integrating live GPS navigation, terrain elevation profiles, and CAN/CAN-FD bus communication for real-time look-ahead horizon adjustment."),
    ("Advanced Control Algorithms", "Investigating Model Predictive Control (MPC) and Deep Reinforcement Learning (DRL) for continuous closed-loop torque-split optimization."),
    ("HIL & Vehicle Dynamometer Testing", "Transitioning from software simulation to Hardware-in-the-Loop (HIL) testing and chassis dynamometer validation on physical vehicle hardware."),
    ("Battery Degradation & Thermal EMS", "Incorporating electrochemical aging models and active battery thermal management into the predictive decision loop.")
]
for sh in s16.shapes:
    if sh.name == 'Text Placeholder 2':
        clear_and_add_bullet_points(sh, future_scope_points, base_size=14)

s16.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Explicitly labeled Future Scope and extension roadmap for subsequent phases of Project PHEX.\n"
    "2. MY CONTRIBUTION: Identified the software-to-hardware migration path and embedded optimization requirements.\n"
    "3. 25s EXPLANATION: Future phases will focus on deploying these algorithms onto embedded automotive ECUs via C++/ONNX, integrating live CAN-FD bus communication, and performing Hardware-in-the-Loop testing before physical vehicle integration.\n"
    "4. LIKELY VIVA QUESTION: How would you optimize the Random Forest model for an embedded ECU with limited memory?\n"
    "5. CONCISE ANSWER: We can apply tree pruning, quantization, convert the ensemble to fixed-point C code using tools like Micro-ML or ONNX Runtime, and limit maximum tree depth to constrain memory footprint."
)
print("Slide 16 completed.")

# -------------------------------------------------------------
# SLIDE 17: References
# -------------------------------------------------------------
s17 = prs.slides[16]
ref_points = [
    "[1] A. Sciarretta and L. Guzzella, 'Control of hybrid electric vehicles,' IEEE Control Systems Magazine, vol. 27, no. 2, pp. 60–70, Apr. 2007.",
    "[2] C. C. Chan, 'The state of the art of electric, hybrid, and fuel cell vehicles,' Proceedings of the IEEE, vol. 95, no. 4, pp. 704–718, Apr. 2007.",
    "[3] I. Husain, Electric and Hybrid Vehicles: Design Fundamentals, 3rd ed., Boca Raton, FL, USA: CRC Press, 2021.",
    "[4] L. Breiman, 'Random forests,' Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
    "[5] F. Aminifar, M. Abedini, and T. Amraee, 'A review of power system protection and asset management with machine learning techniques,' Springer Nature, 2021.",
    "[6] UNECE, 'Worldwide Harmonized Light Vehicles Test Procedure (WLTP),' United Nations Economic Commission for Europe, Geneva, Switzerland, 2023."
]
for sh in s17.shapes:
    if sh.name == 'Text Placeholder 2':
        tf = sh.text_frame
        tf.word_wrap = True
        tf.clear()
        for i, ref_text in enumerate(ref_points):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            set_para_font(p, ref_text, "Calibri", 12, False, (40, 40, 40), PP_ALIGN.LEFT)
            p.space_after = Pt(6)

s17.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Key literature citations in IEEE standard format taken directly from the final internship report.\n"
    "2. MY CONTRIBUTION: Ensured all cited sources strictly back the automotive control, ML regression, and simulation methodologies.\n"
    "3. 15s EXPLANATION: These references represent the foundational literature in HEV supervisory control, ensemble machine learning, and standardized drive-cycle testing used throughout this work."
)
print("Slide 17 completed.")

# -------------------------------------------------------------
# SLIDE 18: Thank You & Q & A
# -------------------------------------------------------------
s18 = prs.slides[17]
# Add structured candidate and guide info to Slide 18
txBox18 = s18.shapes.add_textbox(Inches(2.0), Inches(4.2), Inches(5.2), Inches(2.0))
tf18 = txBox18.text_frame
tf18.word_wrap = True
tf18.clear()
p1 = tf18.paragraphs[0]
set_para_font(p1, "Mishael Julian (Reg No: 2462184)", "Calibri", 16, True, (0, 51, 102), PP_ALIGN.CENTER)
p2 = tf18.add_paragraph()
set_para_font(p2, "Faculty Guide: Dr. Sujatha A K", "Calibri", 14, False, (60, 60, 60), PP_ALIGN.CENTER)
p3 = tf18.add_paragraph()
set_para_font(p3, "Dept. of AI & Data Science Engineering | CHRIST (Deemed to be University)", "Calibri", 12, False, (80, 80, 80), PP_ALIGN.CENTER)

s18.notes_slide.notes_text_frame.text = (
    "1. WHAT THIS SLIDE SHOWS: Concluding slide inviting questions and evaluation from the viva examiners.\n"
    "2. 15s EXPLANATION: Thank you very much for your time and guidance. I am now open to questions and feedback regarding the predictive EMS methodology, demand forecasting models, and simulation results."
)
print("Slide 18 completed.")

# Save presentation
prs.save(output_path)
print(f"Presentation successfully saved to: {output_path}")

Part-Detection-for-Automated-FDM-3D-Printing-Dataset-Collection
This repository provides an illumination-adaptive data collection pipeline for FDM 3D printing quality inspection. By integrating HSV masking, morphology operations, and YOLOv5-based automated labeling, this system enables the efficient acquisition of high-fidelity datasets without manual labeling costs.

🚀 Key Features
Automated Labeling Pipeline: Utilizes a fine-tuned YOLOv5 model to detect print head parking, automating the time-consuming labeling process.

Illumination Robustness: Employs HSV color space masking to decouple chromaticity from luminance, ensuring stable part detection under varying lighting conditions compared to traditional RGB methods.

8-Stage Refinement: A comprehensive pipeline including morphology operations and area-based spatial filtering to eliminate noise and artifacts.

High-Fidelity Dataset: Successfully generated 2,521 high-quality images ready for CNN/YOLO training from a raw set of 2,652 images.

📊 Performance Summary
The proposed system demonstrates superior performance over the raw RGB-based baseline:

Metric,RGB Baseline,Proposed HSV Pipeline,Improvement
Overall LSR,96.23%,97.62%,+1.39%p
Overall UDR,94.19%,95.06%,+0.87%p
Pink (Max Imp.),93.82%,99.25%,+5.43%p

📊 Performance SummaryThe proposed system demonstrates superior performance over the raw RGB-based baseline:MetricRGB BaselineProposed HSV PipelineImprovementOverall LSR96.23%97.62%+1.39%pOverall UDR94.19%95.06%+0.87%pPink (Max Imp.)93.82%99.25%+5.43%pNote: The HSV pipeline effectively suppresses bed reflections and ambient lighting interference, particularly for high-chroma filaments like Pink.

🛠️ Hardware SetupPrinter: Bambu Lab P1S (FDM)Camera: Insta360 Ace Pro (Fixed top-down view)Filaments: PLA (Yellow, Pink, Blue, White)

📝 Future Work
Dynamic Registration: Integrating ArUco markers for coordinate matching.

Real-time QC: Expanding to an automated defect detection and feedback system.
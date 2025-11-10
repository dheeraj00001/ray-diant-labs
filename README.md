# ray-diant-labs
Ray-Diant Labs is an AI project for disease prediction using X-rays, MRIs, and CT scans. It features a hierarchy of four models: a ResNet50 base via transfer learning, and three specialized models — X-Ray (Google ViT), MRI (Meta ViT), and CT (Microsoft Swin ViT).
┌───────────────┐      ┌────────────────────┐      ┌──────────────────────────────┐
│   Start /     │ ---> │ Validate & Convert │ ---> │  Apply Base Transform (RGB)  │
│ Upload Image  │      │    to RGB (PIL)    │      │   (Resize, Normalize, etc.)  │
└───────────────┘      └────────────────────┘      └──────────────────────────────┘
                                                            │
                                                            ▼
                                         ┌───────────────────────────────────────────────┐
                                         │ Base Model (ResNet50) → Predict Modality      │
                                         │ ['ct_scan', 'mri_scan', 'xray']              │
                                         └───────────────────────────────────────────────┘
                                                            │
                                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Modality Routing Decision                                      │
│---------------------------------------------------------------------------------------------│
│ if modality == 'xray' ───> Use ViT Model (X-Ray classes)  ─┐                               │
│ if modality == 'mri_scan' ─> Use DeiT Model (MRI classes) ─┼─> Specialized Model Inference │
│ if modality == 'ct_scan' ──> Use Swin Model (CT classes)  ─┘                               │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                              ┌─────────────────────────────────────────────┐
                              │ Apply Correct Transform (per model)         │
                              │ Forward pass → Obtain logits                │
                              │ Apply Softmax → Probabilities               │
                              └─────────────────────────────────────────────┘
                                                            │
                                                            ▼
                         ┌────────────────────────────────────────────────────────────┐
                         │ Extract Top-3 Predictions                                  │
                         │ { ClassName: Probability }                                 │
                         │ Format Modality String (with confidence)                   │
                         └────────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
     ┌──────────────────────────────────────────────────────────────────────────────────────┐
     │                  Display Results via Gradio Interface                                │
     │  Detected Modality → Textbox                                                        │
     │  Top-3 Disease Predictions → Label Component                                        │
     └──────────────────────────────────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                                              ┌────────────────────┐
                                              │      End / Ready   │
                                              │     for Next Input │
                                              └────────────────────┘

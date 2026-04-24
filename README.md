# Assignment Helper Bot

A desktop app that analyzes assignment prompts and rubrics using zero-shot NLI (Natural Language Inference) to automatically extract and categorize requirements. 

## Features   

- **Zero-shot NLI classification** via `cross-encoder/nli-deberta-v3-small`                                                   
- **PDF import** — load rubrics directly from PDF files with selectable text
- **6 requirement categories:**
- Main Task
- Output Requirements
- Supported Options / Features
- Allowed Languages
- Exceptions / Notes
- Submission
- **Confidence scores** shown for each classified line
- **Rule-based pre-classifier** for high-confidence lines (flags, deadlines, exceptions) to skip NLI and improve speed and accuracy
- **Copy Text / Save File** — export extracted requirements
- **Split-panel GUI** built with Tkinter

## Installation  

```
pip install -r requirements.txt                                
```                                                            
## Run                                                         
```                                                            
python App.py
```                                                       
On first run the app will download the NLI model (~180MB). Wait for the status bar to show **✓ ready** before analyzing.     
                                                              
## How to Use                                                  
                                                              
1. Paste your assignment rubric into the left panel, or click *
*Import PDF** to load a PDF                                    
2. Click **Analyze Requirements**                              
3. Extracted requirements appear on the right grouped by category                       

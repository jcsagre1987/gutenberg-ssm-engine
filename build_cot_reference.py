# ==========================================
# Gutenberg SSM Chain-of-Thought Builder
# Copyright (C) 2026 Juan Carlo R. Sagre
# ==========================================

import json
from pathlib import Path

COT_FILE = Path("chainofthoughtreference.json")

CORE_COT_TEMPLATES = {
    "comparison_and_contrast": {
        "trigger_keywords": ["difference", "different", "versus", "vs", "compare", "contrast", "distinguish"],
        "system_instruction": "Structure your response into clear step-by-step criteria: 1. Definition/Role of Item A, 2. Definition/Role of Item B, 3. Core Key Difference.",
        "few_shot_example": "Q: What is the difference between a presidential and parliamentary system?\nA: Step 1 (Executive Choice): In a presidential system, the president is elected independently of the legislature. In a parliamentary system, the prime minister is chosen directly from the elected parliament.\nStep 2 (Branch Separation): Presidential systems enforce strict separation of powers, while parliamentary systems fuse executive and legislative powers.\nStep 3 (Core Difference): The fundamental difference is whether the executive is independently elected or selected by parliament."
    },
    "math_and_logic": {
        "trigger_keywords": ["calculate", "solve", "how many", "logic", "proof", "evaluate", "total"],
        "system_instruction": "Decompose the problem step-by-step before stating the final result. Work through arithmetic sequentially.",
        "few_shot_example": "Q: A store sells items for $15 each with a $5 flat shipping fee. What is the total cost for 4 items?\nA: Step 1 - Item Cost: 4 items x $15 = $60.\nStep 2 - Add Shipping: $60 + $5 shipping = $65.\nStep 3 - Final Result: The total cost is $65."
    },
    "causal_and_analysis": {
        "trigger_keywords": ["why", "cause", "effect", "reason", "impact", "consequence", "how does"],
        "system_instruction": "Analyze cause and effect sequentially using clear step-by-step reasoning lines.",
        "few_shot_example": "Q: Why does water boil faster at higher altitudes?\nA: Step 1 (Atmospheric Pressure): Air pressure decreases at higher altitudes.\nStep 2 (Vapor Pressure): Lower atmospheric pressure means water molecules need less thermal energy to escape into vapor.\nStep 3 (Boiling Point): Consequently, water reaches its boiling point at a lower temperature, causing it to boil faster."
    },
    "troubleshooting_and_root_cause": {
        "trigger_keywords": ["fix", "error", "issue", "bug", "troubleshoot", "why is", "not working"],
        "system_instruction": "Diagnose the problem by evaluating probable root causes from most common to least common.",
        "few_shot_example": "Q: Why is my local Python server returning a Connection Refused error?\nA: Step 1 - Check Process: Verify if the server script is actively running in terminal.\nStep 2 - Check Port & Host: Ensure host is set to 127.0.0.1 or 0.0.0.0 and port 18080 is not blocked by firewall.\nStep 3 - Resolution: Restart the script with uvicorn or python server.py."
    }
}

def build_cot_reference():
    COT_FILE.write_text(json.dumps(CORE_COT_TEMPLATES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[COT BUILDER SUCCESS] Created {COT_FILE.name} with {len(CORE_COT_TEMPLATES)} reasoning frameworks.")

if __name__ == "__main__":
    build_cot_reference()
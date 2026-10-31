#!/usr/bin/env python3
with open('backend/conformal.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_block_1 = """        stratum = "Candidate-Stratified Risk Control (Multi-Tier)"
        guarantee = (1.0 - min(self.alpha_emergency, self.alpha_routine)) * 100.0
        return prediction_set, set_probs, stratum, guarantee"""

new_block_1 = """        top_urgency = URGENCY_TIERS.get(top_diagnosis, "Non-urgent")
        is_emergency = top_urgency in {"Emergency", "Urgent"}
        if is_emergency:
            stratum = "Emergency-Stratified (Sight-Threatening)"
            guarantee = (1.0 - self.alpha_emergency) * 100.0
        else:
            stratum = "Routine-Stratified (Elective/Adnexal)"
            guarantee = (1.0 - self.alpha_routine) * 100.0
        return prediction_set, set_probs, stratum, guarantee"""

text = text.replace(old_block_1, new_block_1)

old_block_2 = """        stratum = f"AW-CRC Candidate-Stratified (S={eff_s:.2f}, gamma={gamma_safe:.1f})"
        guarantee = (1.0 - min(self.alpha_emergency, self.alpha_routine)) * 100.0
        return prediction_set, set_probs, stratum, guarantee"""

new_block_2 = """        top_urgency = URGENCY_TIERS.get(top_diagnosis, "Non-urgent")
        is_emergency = top_urgency in {"Emergency", "Urgent"}
        if is_emergency:
            stratum = f"AW-CRC Emergency-Stratified (S={eff_s:.2f})"
            guarantee = (1.0 - self.alpha_emergency) * 100.0
        else:
            stratum = f"AW-CRC Routine-Stratified (S={eff_s:.2f})"
            guarantee = (1.0 - self.alpha_routine) * 100.0
        return prediction_set, set_probs, stratum, guarantee"""

text = text.replace(old_block_2, new_block_2)

with open('backend/conformal.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated conformal.py successfully.")

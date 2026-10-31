# Independent Peer Review Report 2: Clinical AI and Retinal Ophthalmology

**Reviewer Identity:** Reviewer 2 (Consulting Ophthalmologist & Clinical AI Safety Specialist)  
**Target Manuscript:** `research/paper_revised.tex`  
**Title:** *Quality-Aware Conformal Screening for Retinal Fundus Imaging Under Optical Domain Shift*  
**Venue / Track:** ICCECE 2027 (Biomedical Informatics, Cognitive Computing & Trustworthy AI)  
**Review Type:** Double-Blind Peer Review  
**Recommendation:** **ACCEPT (Score: 7 / 10)**  

---

## 1. Summary of the Work

The manuscript investigates an automated tele-ophthalmic screening pipeline designed for primary care clinics and remote medical outposts. The clinical objective is early triaging of sight-threatening retinal pathologies—specifically Diabetic Retinopathy (DR), Glaucomatous Optic Neuropathy, and Age-Related Macular Degeneration (AMD)—alongside physiological Normal eyes and Cataract-induced media opacity.

Rather than treating deep learning as an autonomous black-box diagnostician, the paper designs a **clinically cautious decision-support framework**. The core clinical mechanisms include:
1. Pre-inference rejection of non-fundus or severely corrupted scans via biophysical spectroscopy and aperture checks.
2. Clinical urgency stratification within conformal prediction sets ($\alpha = 0.01$ for emergency sight-threatening conditions vs. $\alpha = 0.05$ for routine conditions).
3. Explicit disclosure and evaluation of camera hardware domain shift, showing that while naive classification collapses, conformal prediction sets expand to trigger a $100.0\%$ human specialist referral rate.

---

## 2. Clinical AI Safety Strengths

1. **Exemplary Disclosure of External Domain Shift (A Rare and Commendable Clinical Practice):**
   In the clinical AI literature, authors almost universally hide or downplay poor out-of-distribution performance, presenting only flattering cross-validation numbers. This manuscript takes the opposite, ethically commendable approach:
   In Table VI, the authors transparently report that when tested on external camera hardware from different continents (IDRiD in India using a Kowa VX-10 camera, and RIM-ONE DL in Spain using a Nidek AFC-210 camera), naive argmax classification suffers a catastrophic specificity collapse down to **$16.42\%$** and **$1.05\%$**, respectively.
   Crucially, the authors demonstrate that their proposed **AW-CRC selective triage engine successfully detects this shift and triggers a $100.0\%$ referral escalation**. In a clinical deployment, this behavior represents an essential fail-closed safety net: rather than sending patients home with false reassurance or unreviewed diagnoses, the system recognizes that the image characteristics depart from its training distribution and demands human examination.

2. **Clinically Grounded Urgency Stratification:**
   Standard conformal prediction applies a single uniform miscoverage parameter $\alpha$ (e.g., $5\%$) across all diagnostic categories. Clinically, this is deeply problematic: a $5\%$ error rate on mild cataract is an acceptable minor inconvenience, whereas a $5\%$ error rate on proliferative diabetic retinopathy or neovascular AMD leads to irreversible blindness.
   The authors partition the disease taxonomy into an **Emergency Stratum** ($\mathcal{K}_{\mathrm{emerg}} = \{\text{DR, Glaucoma, AMD}\}$, with stringent $\alpha = 0.01$) and a **Routine Stratum** ($\mathcal{K}_{\mathrm{rout}} = \{\text{Normal, Cataract}\}$, with $\alpha = 0.05$). On the sealed test cohort, the system delivers an empirical realized coverage of **$99.31\%$** on emergency pathologies, providing genuine peace of mind to referring clinicians.

3. **Purging of Synthetic Demographics and Unverified Claims:**
   The paper avoids the trap of claiming "multimodal clinical AI" based on synthetic tabular EHR data. The authors explicitly state that their system is vision-only, that demographic data on web-scraped corpora was absent and therefore omitted, and that the tool is an experimental screening prototype rather than an FDA-cleared diagnostic device.

---

## 3. Clinical Limitations and Constructive Feedback

1. **Low Sample Prevalence for AMD ($n=29$ ground truth cases):**
   Following dataset forensic deduplication, the sealed clean test cohort contains only 29 verified AMD cases (2.09% prevalence). While the authors report high One-vs-Rest AUROC ($0.9946$), AMD sensitivity is **$79.31\%$** ($23 / 29$), yielding a wide 95% Wilson confidence interval of **$[61.61\%, 90.15\%]$**.
   *Ophthalmological Perspective:* AMD diagnosis on standard color fundus photographs is notoriously challenging in early dry stages (small drusen) compared to optical coherence tomography (OCT). The authors have candidly disclosed this limitation in Section VII-2, but the paper should emphasize even more strongly in the clinical discussion that AMD screening remains provisional until evaluated on prospective, high-prevalence macular degeneration cohorts.

2. **Differentiation of Cataract from True Media Opacity:**
   The inclusion of Cataract ($n=134$) as a distinct class is clinically interesting. In primary care, a common clinical conundrum is distinguishing poor image quality due to anterior segment lens opacity (cataract) from poor image quality due to uncooperative patient alignment or optical blur. The Tier-0 guardrail Laplacian variance metric $\mathcal{V}_{\mathrm{lap}}$ and AW-CRC optical quality scaling $\mathcal{S}(X)$ handle this gracefully: severe lens opacification lowers $\mathcal{S}(X)$, which dynamically expands the prediction set to include Cataract alongside other possibilities, appropriately guiding the patient toward anterior segment examination.

3. **Handling of the Argmax Fallback Alert:**
   Under raw conformal prediction, 8 test cases ($0.58\%$) produced empty prediction sets $\mathcal{C}(X) = \emptyset$. The authors' operational policy injects the top argmax prediction alongside an urgent clinical review alert. From a clinical risk management standpoint, this policy is acceptable *only if* the electronic medical record (EMR) interface renders a prominent warning banner preventing clinicians from mistaking the fallback prediction for a high-confidence conformal set. The authors should ensure this UI warning is prominently featured in deployment guidelines.

---

## 4. Final Recommendation

The authors have constructed a clinically responsible, honest, and innovative retinal screening pipeline. The transparent reporting of domain shift collapse and the implementation of a fail-closed conformal referral net are major steps forward for clinical AI safety.

**Recommendation: ACCEPT.**

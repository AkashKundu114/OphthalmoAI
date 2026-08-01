"""
Comprehensive Clinical Medical Knowledge Base for OphthalmoAI.

Provides doctor-level medical insights, pathophysiology explanations, visual findings,
recommended diagnostic workups (Slit-lamp, Tonometry, OCT), precautions, treatment protocols,
differential diagnoses, and clinical consultation notes for eye conditions.
"""
from __future__ import annotations

from typing import Any, Dict

MEDICAL_INFO: Dict[str, Dict[str, Any]] = {
    'Cataract': {
        'name': 'Cataract',
        'group': 'Anterior Segment',
        'color': '#00F5D4',
        'pathophysiology': (
            "A cataract is a progressive opacification of the crystalline lens of the eye. With aging, metabolic changes, "
            "or oxidative stress, crystallin proteins within the lens undergo aggregation and cross-linking, causing light rays "
            "to scatter rather than focus sharply onto the retina."
        ),
        'analysis': (
            "Visual examination reveals significant opacification of the crystalline lens, manifesting as a greyish-white or "
            "yellowish-brown clouding behind the pupil. The normal red reflex on distant direct ophthalmoscopy is blunted or absent."
        ),
        'description': (
            "A cataract is a major cause of reversible visual impairment globally. As the lens becomes progressively cloudy, "
            "patients experience gradual loss of visual acuity, increased glare sensitivity, and diminished contrast sensitivity."
        ),
        'symptoms': [
            "Painless gradual blurring or dimming of vision (like looking through a frosty window)",
            "Marked sensitivity to light and glare, especially from headlights at night",
            "Appearance of 'halos' surrounding light sources",
            "Frequent changes in eyeglass or contact lens prescription ('second sight' phenomenon)",
            "Yellowing or desaturation of color perception",
            "Monocular diplopia (double vision in a single eye)",
        ],
        'treatment': [
            "Prescription lens update & anti-glare coatings (for early mild cataract)",
            "Phacoemulsification Surgery: Standard micro-incisional ultrasonic emulsification and aspiration of the opaque lens",
            "Intraocular Lens (IOL) Placement: Implantation of monofocal, toric, or multifocal synthetic lens into the capsular bag",
        ],
        'precautions': [
            "Wear UV400-rated protective sunglasses outdoors to slow photic lens oxidation",
            "Cease tobacco use, as smoking significantly elevates cataract risk",
            "Maintain strict glycemic control if diabetic",
            "Avoid chronic non-prescribed corticosteroid eye drops",
            "Ensure adequate lighting for reading and detail work",
        ],
        'diagnostic_workup': [
            "Slit-Lamp Biomicroscopy: Direct cross-sectional evaluation of nuclear, cortical, or posterior subcapsular lens density",
            "Visual Acuity Testing: Distance and near Snellen/ETDRS chart assessment with glare testing (BAT)",
            "Dilated Fundus Examination: To rule out co-existing macular degeneration or diabetic retinopathy",
            "Optical Biometry / Keratometry: Pre-operative calculation of intraocular lens power",
        ],
        'differential_diagnoses': [
            "Refractive error shift",
            "Age-related macular degeneration (AMD)",
            "Open-angle glaucoma",
            "Corneal stromal dystrophy",
        ],
        'doctor_notes': (
            "Cataracts are highly treatable with modern outpatient phacoemulsification surgery. Surgical intervention is indicated "
            "when visual impairment interferes with essential daily activities like night driving or reading."
        ),
        'questions_for_doctor': [
            "Is my cataract causing enough visual impairment to warrant surgery now?",
            "What type of Intraocular Lens (monofocal, toric, or multifocal) is best suited for my lifestyle?",
            "How will cataract surgery affect my need for glasses afterwards?",
        ],
        'severity': "Moderate to Severe (Depending on visual density)",
        'advice': "Consult an ophthalmologist for a slit-lamp examination and optical biometry assessment.",
    },
    'Conjunctivitis': {
        'name': 'Conjunctivitis',
        'group': 'Ocular Surface',
        'color': '#10B981',
        'pathophysiology': (
            "Conjunctivitis is an acute or chronic inflammation of the palpebral and bulbar conjunctiva. "
            "Etiologies include viral pathogens (most commonly Adenovirus), bacterial invasion (Staphylococcus aureus, Streptococcus pneumoniae), "
            "IgE-mediated allergic reactions, or direct chemical irritants."
        ),
        'analysis': (
            "Diffusely hyperemic bulbar and palpebral conjunctival vessels with injection prominent towards the fornices. "
            "Purulent discharge (bacterial), serous watery discharge (viral), or cobblestone papillae (allergic) may be present."
        ),
        'description': (
            "Commonly known as 'Pink Eye', conjunctivitis causes marked redness, discharge, grittiness, and irritation. "
            "Bacterial and viral forms are highly contagious and spread rapidly via direct contact."
        ),
        'symptoms': [
            "Diffuse pink or intense red discoloration of one or both eyes",
            "Purulent yellow-green discharge that crusts over eyelashes overnight (Bacterial)",
            "Profuse watery tearing with preauricular lymphadenopathy (Viral)",
            "Intense bilateral itching and eyelid edema (Allergic)",
            "Foreign body sensation or gritty feeling under eyelids",
        ],
        'treatment': [
            "Topical Broad-Spectrum Antibiotic Drops (e.g., Fluoroquinolones or Tobramycin for bacterial cases)",
            "Topical Antihistamines & Mast Cell Stabilizers (e.g., Olopatadine for allergic cases)",
            "Frequent Preservative-Free Lubricating Tears (for viral or irritant cases)",
            "Cold Compresses applied over closed eyelids to relieve pruritus and edema",
        ],
        'precautions': [
            "Do NOT touch or rub the affected eye to prevent corneal abrasion and autoinoculation",
            "Wash hands thoroughly with soap and water before and after touching the face",
            "Immediately discontinue contact lens wear until full clinical resolution",
            "Replace eye makeup, mascara, and pillowcases used during active infection",
            "Do not share towels, washcloths, or eye drop dispensers with family members",
        ],
        'diagnostic_workup': [
            "Slit-Lamp Examination: Assessment of follicular vs. papillary conjunctival response and corneal epithelial integrity",
            "Fluorescein Staining: To rule out superficial punctate keratitis or herpes simplex dendritic ulcers",
            "Conjunctival Swab Culture: Indicated for hyperacute purulent cases or treatment-resistant infections",
        ],
        'differential_diagnoses': [
            "Anterior Uveitis / Iridocyclitis",
            "Acute Angle-Closure Glaucoma",
            "Corneal Abrasion or Ulcer",
            "Episcleritis",
        ],
        'doctor_notes': (
            "Viral conjunctivitis is self-limiting (7-14 days). However, bacterial cases require topical antibiotics, "
            "and allergic forms benefit from mast-cell stabilizers. Seek immediate evaluation if severe pain, photophobia, or vision drop occurs."
        ),
        'questions_for_doctor': [
            "Is my conjunctivitis viral, bacterial, or allergic in nature?",
            "How many days am I contagious, and when can I return to school/work?",
            "When can I safely resume wearing my contact lenses?",
        ],
        'severity': "Low to Moderate (Highly contagious)",
        'advice': "Maintain strict hand hygiene. Consult a primary care physician or optometrist for targeted eye drops.",
    },
    'Eyelid': {
        'name': 'Eyelid Conditions',
        'group': 'Adnexal/Oculoplastic',
        'color': '#06B6D4',
        'pathophysiology': (
            "Adnexal eyelid pathologies involve acute focal infection (Hordeolum/Stye of the glands of Zeis or Meibomian glands), "
            "granulomatous chronic inflammation (Chalazion), or diffusal eyelid margin inflammation (Blepharitis) associated with Meibomian gland dysfunction."
        ),
        'analysis': (
            "Focal focal erythematous nodule along the lid margin (Hordeolum), a firm painless tarsal nodule (Chalazion), or "
            "collarettes and oily debris along eyelash roots with lid margin telangiectasia (Blepharitis)."
        ),
        'description': (
            "Eyelid conditions range from acute painful styes to non-tender chalazia and chronic blepharitis. "
            "They frequently cause localized swelling, tenderness, ocular grittiness, and secondary tear film instability."
        ),
        'symptoms': [
            "Tender, localized red lump along the eyelash margin or within the eyelid body",
            "Firm, non-tender sub-epithelial nodule (Chalazion)",
            "Crusting, flaking debris ('collarettes') at the base of eyelashes upon awakening",
            "Eyelid margin redness, burning, and recurrent dry eye symptoms",
            "Sensation of a eyelash or grain of sand lodged under the eyelid",
        ],
        'treatment': [
            "Warm Compresses: 10-15 minutes applied 3-4 times daily (Essential for liquefying meibum)",
            "Commercial Eyelid Lid Wipes or Hypochlorous Acid Lid Scrubs",
            "Topical Antibiotic/Steroid Ointments (e.g., Tobradex for infected styes)",
            "Oral Doxycycline or Azithromycin (for recalcitrant Meibomian Gland Dysfunction)",
            "In-office Incision & Curettage for non-resolving chronic chalazia",
        ],
        'precautions': [
            "NEVER attempt to squeeze, pop, or pierce an eyelid lump",
            "Discontinue eyelid cosmetics and eyeliner while inflammation is active",
            "Perform daily warm compresses and hygiene rituals",
            "Avoid wearing contact lenses if the eyelid swelling rubs against the cornea",
        ],
        'diagnostic_workup': [
            "High-Magnification Slit-Lamp Lid Examination: Inspection of Meibomian gland orifices and lash follicles",
            "Meibography: Infrared imaging of Meibomian gland architecture",
            "Tear Break-Up Time (TBUT): Evaluation of secondary evaporative dry eye",
        ],
        'differential_diagnoses': [
            "Preseptal or Orbital Cellulitis",
            "Basal Cell Carcinoma / Sebaceous Gland Carcinoma of the lid",
            "Dacryocystitis",
        ],
        'doctor_notes': (
            "Consistent heat therapy is the cornerstone of resolution. If eyelid redness spreads to the surrounding cheek or eyebrow, "
            "or if the eye becomes swollen shut with fever, urgent evaluation for preseptal cellulitis is mandatory."
        ),
        'questions_for_doctor': [
            "Is this lesion an acute stye or a chronic chalazion?",
            "Would a prescription antibiotic ointment or in-office procedure help speed up recovery?",
            "What daily eyelid hygiene routine will prevent future recurrences?",
        ],
        'severity': "Low (Painful but manageable)",
        'advice': "Apply warm compresses consistently. Consult an eye specialist if swelling worsens or persists beyond 2 weeks.",
    },
    'Jaundice': {
        'name': 'Jaundice (Scleral Icterus)',
        'group': 'Ocular Surface',
        'color': '#F59E0B',
        'pathophysiology': (
            "Scleral icterus occurs when total serum bilirubin levels exceed 2.5-3.0 mg/dL. Bilirubin has a strong affinity "
            "for elastin-rich tissues such as the sclera, resulting in generalized yellow discoloration. It stems from pre-hepatic "
            "(hemolysis), hepatic (hepatitis, cirrhosis), or post-hepatic (biliary obstruction) etiology."
        ),
        'analysis': (
            "Diffuse, intense yellow pigmentation of the bulbar sclera bilaterally. The cornea and anterior chamber remain clear. "
            "This is a systemic physical examination sign rather than a primary ocular pathology."
        ),
        'description': (
            "Scleral icterus (yellow eyes) is an important systemic red-flag sign indicating hyperbilirubinemia. "
            "It requires immediate medical investigation of liver, gallbladder, pancreatic, or hematologic function."
        ),
        'symptoms': [
            "Distinct yellowing of the normally white sclera in both eyes",
            "Associated yellowish discoloration of the skin and mucosal membranes",
            "Dark brown or 'tea-colored' urine",
            "Pale, clay-colored stools",
            "Abdominal pain (right upper quadrant), fatigue, nausea, or fever",
            "Generalized cutaneous pruritus (itching)",
        ],
        'treatment': [
            "Emergency Medical Workup: Immediate referral to Internal Medicine / Gastroenterology",
            "Treatment of Underlying Etiology: Antiviral therapy (Hepatitis), Biliary Stenting/Gallbladder Surgery, or Cessation of Hepatotoxic Drugs",
        ],
        'precautions': [
            "Seek SAME-DAY medical evaluation at an emergency department or clinic",
            "Avoid alcohol consumption and all non-essential over-the-counter medications (especially Acetaminophen/Paracetamol)",
            "Do NOT attempt self-treatment with eye drops — yellow eyes cannot be cured with topical medication",
        ],
        'diagnostic_workup': [
            "Comprehensive Metabolic Panel (CMP): Serum Total/Direct Bilirubin, ALT, AST, Alkaline Phosphatase, Albumin",
            "Abdominal Ultrasound or CT Scan: Evaluation of liver parenchyma and biliary tree patency",
            "Complete Blood Count (CBC) & Reticulocyte Count: To evaluate for hemolytic anemia",
        ],
        'differential_diagnoses': [
            "Acute Viral Hepatitis",
            "Choledocholithiasis (Gallstones)",
            "Cirrhosis / Alcoholic Liver Disease",
            "Hemolytic Anemia",
            "Gilbert Syndrome",
        ],
        'doctor_notes': (
            "Scleral icterus is a medical warning sign requiring systemic evaluation. The priority is identifying whether the hyperbilirubinemia "
            "is obstructive, hepatocellular, or hemolytic through immediate liver function testing."
        ),
        'questions_for_doctor': [
            "What are my total and direct bilirubin levels?",
            "Does this indicate a liver, gallbladder, or blood disorder?",
            "What diagnostic imaging (ultrasound, CT, MRI) is required today?",
        ],
        'severity': "Critical Emergency (Systemic Warning Sign)",
        'advice': "EMERGENCY: Seek immediate medical evaluation at an emergency room or primary care clinic for liver function testing.",
    },
    'Pterygium': {
        'name': 'Pterygium',
        'group': 'Ocular Surface',
        'color': '#3B82F6',
        'pathophysiology': (
            "A pterygium is a benign, wing-shaped fibrovascular proliferation of the bulbar conjunctiva that encroaches onto the limbus and cornea. "
            "It is driven by chronic ultraviolet (UV-A/UV-B) radiation exposure, micro-trauma from wind/sand, and tear film dryness."
        ),
        'analysis': (
            "Fleshy, triangular fibrovascular tissue growth extending from the nasal bulbar conjunctiva across the limbus onto the superficial cornea. "
            "A pigmented iron line (Stockers line) may be visible leading the head of the pterygium."
        ),
        'description': (
            "Often referred to as 'Surfer's Eye', a pterygium is a non-cancerous growth. While slowly progressive, large pterygia "
            "can induce astigmatism, cause chronic redness, or obscure the visual axis."
        ),
        'symptoms': [
            "Triangular fleshy growth extending onto the clear part of the eye (usually from the nasal side)",
            "Persistent redness, ocular dryness, and localized grittiness",
            "Feeling of a constant foreign body in the eye",
            "Induced astigmatism causing blurred vision",
            "Visual field restriction if growth encroaches over the central pupil",
        ],
        'treatment': [
            "Preservative-Free Lubricating Eye Drops & Ointments (for mild dryness and irritation)",
            "Short-Course Topical Mild Corticosteroids (for acute inflamed pterygium)",
            "Pterygium Excision with Conjunctival Autograft (CAG): Surgical removal with autologous tissue grafting to prevent recurrence",
        ],
        'precautions': [
            "Wear wrap-around UV400 sunglasses outdoors at all times",
            "Wear a wide-brimmed hat outdoors in sunny or sandy environments",
            "Use lubricating eye drops regularly in dusty or windy conditions",
            "Avoid rubbing the eyes, which aggravates localized vascular congestion",
        ],
        'diagnostic_workup': [
            "Slit-Lamp Examination: Assessment of pterygium apex distance from the pupillary border",
            "Corneal Topography: Measurement of induced corneal astigmatism and irregular surface curvature",
            "Photographic Documentation: Baseline imaging to track growth speed over time",
        ],
        'differential_diagnoses': [
            "Pinguecula",
            "Pseudopterygium",
            "Ocular Surface Squamous Neoplasia (OSSN)",
            "Nodular Episcleritis",
        ],
        'doctor_notes': (
            "Pterygia are managed conservatively with UV protection and artificial tears unless they cause significant visual disruption, "
            "induced astigmatism, or severe cosmetic concern. Modern autograft surgery carries low recurrence rates."
        ),
        'questions_for_doctor': [
            "Is my pterygium growing towards my pupil or affecting my corneal curvature?",
            "Are lubricating drops sufficient, or should I consider surgical removal?",
            "What surgical technique (such as conjunctival autograft) offers the lowest recurrence rate?",
        ],
        'severity': "Elective / Low to Moderate",
        'advice': "Wear UV-blocking sunglasses. Consult an ophthalmologist to monitor growth and evaluate surgical options if vision is affected.",
    },
    'Uveitis': {
        'name': 'Uveitis',
        'group': 'Anterior Segment Pathology',
        'color': '#EF4444',
        'pathophysiology': (
            "Uveitis is inflammation of the uveal tract (iris, ciliary body, choroid). Anterior uveitis (iridocyclitis) involves breakdown of the "
            "blood-aqueous barrier, causing protein flare and inflammatory white blood cells (keratic precipitates, hypopyon) to collect in the anterior chamber."
        ),
        'analysis': (
            "Ciliary flush (circumcorneal hyperemic ring around the limbus), keratic precipitates on the posterior corneal endothelium, "
            "anterior chamber cell and flare (+1 to +4), miotic or sluggish pupil, and possible synechiae formation."
        ),
        'description': (
            "Uveitis is a serious, sight-threatening intraocular inflammatory condition. Without prompt corticosteroid treatment, "
            "it can lead to permanent vision loss from glaucoma, cystoid macular edema, or secondary cataracts."
        ),
        'symptoms': [
            "Severe deep aching eye pain, often radiating to the eyebrow or temple",
            "Marked photophobia (extreme pain when exposed to light)",
            "Sudden or progressive blurring of vision",
            "Deep red ring of inflammation around the iris (Ciliary Flush)",
            "Floating spots or shadows in vision",
            "Constricted, irregular, or painful pupil",
        ],
        'treatment': [
            "Topical Potent Corticosteroids (e.g., Prednisolone Acetate 1% drops heavily frequency-tapered)",
            "Topical Cycloplegic / Mydriatic Drops (e.g., Cyclopentolate or Atropine to paralyze ciliary body, relieve pain, and break synechiae)",
            "Systemic Immunosuppressive Therapy / Biologics (for chronic, bilateral, or systemic autoimmune uveitis)",
        ],
        'precautions': [
            "SEEK IMMEDIATE OPHTHALMIC CARE (Same-Day / Urgent Evaluation)",
            "Never stop corticosteroid drops abruptly without a doctor's tapering schedule",
            "Wear dark protective sunglasses to alleviate photophobia",
            "Do NOT wear contact lenses",
        ],
        'diagnostic_workup': [
            "Slit-Lamp Examination with SUN (Standardization of Uveitis Nomenclature) Cell & Flare Grading",
            "Goldmann Applanation Tonometry: To check for secondary uveitic glaucoma / elevated IOP",
            "Optical Coherence Tomography (OCT): Screening for cystoid macular edema (CME)",
            "Systemic Autoimmune Workup: HLA-B27, ANA, ACE, Chest X-ray/CT (Sarcoidosis), Syphilis/TB serology",
        ],
        'differential_diagnoses': [
            "Acute Angle-Closure Glaucoma",
            "Infectious Keratitis / Endophthalmitis",
            "Scleritis",
            "Traumatic Iritis",
        ],
        'doctor_notes': (
            "Anterior Uveitis is an ophthalmic emergency. Immediate high-potency steroid drops and cycloplegics are mandatory to prevent "
            "posterior synechiae, secondary glaucoma, and irreversible macular edema."
        ),
        'questions_for_doctor': [
            "What is the SUN cell and flare grade in my eye today?",
            "Is my intraocular pressure elevated?",
            "Should we order blood work or imaging to screen for an underlying autoimmune condition?",
        ],
        'severity': "Urgent Sight-Threatening Emergency",
        'advice': "URGENT: Uveitis requires immediate same-day evaluation by an ophthalmologist to prevent permanent vision loss.",
    },
    'Normal': {
        'name': 'Normal Ocular Examination',
        'group': 'Healthy Baseline',
        'color': '#10B981',
        'pathophysiology': "Healthy ocular structures with normal anatomical clarity, clear cornea and crystalline lens, intact conjunctiva, and clear anterior chamber.",
        'analysis': "Clear cornea and anterior chamber. White, non-injected sclera and healthy bulbar/palpebral conjunctiva. Normal iris architecture and clear optical media.",
        'description': "No visual indicators of acute anterior segment, adnexal, or ocular surface pathology detected.",
        'symptoms': [
            "Clear, comfortable vision",
            "Absence of pain, redness, discharge, or photophobia",
            "Normal tear film stability",
        ],
        'treatment': [
            "Routine preventive eye examinations per standard schedule (Every 1-2 years)",
            "Maintain ocular protective habits and healthy lifestyle",
        ],
        'precautions': [
            "Wear UV400 sunglasses outdoors",
            "Practice 20-20-20 rule during screen use (Every 20 mins, look 20 feet away for 20 seconds)",
            "Keep hands clean and practice safe contact lens hygiene",
        ],
        'diagnostic_workup': [
            "Routine Comprehensive Eye Exam (Visual Acuity, Tonometry, Dilated Fundus Exam)",
        ],
        'differential_diagnoses': [],
        'doctor_notes': "Your eye examination appears within normal limits. Continue routine preventive vision care.",
        'questions_for_doctor': [
            "When should I schedule my next routine comprehensive eye exam?",
            "Are there preventive steps I should take based on my age or screen habits?",
        ],
        'severity': "Healthy / Normal Baseline",
        'advice': "Maintain routine preventive vision check-ups with your optometrist or ophthalmologist.",
    },
}

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
    'Ptosis': {
        'name': 'Ptosis (Drooping Eyelid)',
        'group': 'Adnexal/Oculoplastic',
        'color': '#06B6D4',
        'pathophysiology': "Ptosis is the abnormal drooping of the upper eyelid margin, which can be congenital or acquired. It typically results from dysfunction of the levator palpebrae superioris muscle or its aponeurosis, or Muller's muscle.",
        'analysis': "Upper eyelid margin rests abnormally low, covering more than 2mm of the superior cornea. The margin-reflex distance 1 (MRD-1) is reduced.",
        'description': "A drooping of the upper eyelid that can sometimes obstruct the visual field if it falls below the pupil.",
        'symptoms': [
            "Drooping upper eyelid",
            "Obstructed superior visual field",
            "Need to tilt head back to see clearly",
            "Eyestrain or fatigue from lifting eyebrows"
        ],
        'treatment': [
            "Observation for mild, non-visually significant cases",
            "Surgical correction (Levator advancement, Muller's muscle resection, or Frontalis sling)"
        ],
        'precautions': [
            "Monitor for sudden onset ptosis, especially if accompanied by pupil changes (Horner's syndrome or CN III palsy)"
        ],
        'diagnostic_workup': [
            "Visual field testing (superior field)",
            "Measurement of MRD-1, levator function, and palpebral fissure height",
            "Neurological evaluation if acute onset"
        ],
        'differential_diagnoses': [
            "Dermatochalasis",
            "Horner's syndrome",
            "Myasthenia gravis",
            "Oculomotor nerve palsy"
        ],
        'doctor_notes': "Surgical intervention is indicated when ptosis causes a functional visual field defect or significant cosmetic concern.",
        'questions_for_doctor': [
            "Is my ptosis affecting my field of vision?",
            "Am I a candidate for surgical correction?"
        ],
        'severity': "Low to Moderate",
        'advice': "Consult an oculoplastic surgeon if the drooping obstructs your vision.",
    },
    'Blepharitis': {
        'name': 'Blepharitis',
        'group': 'Adnexal/Oculoplastic',
        'color': '#06B6D4',
        'pathophysiology': "Chronic inflammation of the eyelid margins, often associated with bacterial overgrowth (Staphylococcus) or skin conditions like rosacea and seborrheic dermatitis. Leads to meibomian gland dysfunction and tear film instability.",
        'analysis': "Eyelid margins appear erythematous and thickened. Collarettes or crusting may be visible at the base of the eyelashes. Inspissated meibomian glands are common.",
        'description': "A chronic inflammatory condition of the eyelids causing red, swollen, and itchy eyelids with crusty debris at the base of the eyelashes.",
        'symptoms': [
            "Red, swollen, or itchy eyelids",
            "Gritty or burning sensation in the eyes",
            "Crusting at the eyelash base, worse upon waking",
            "Flaking of skin around eyes"
        ],
        'treatment': [
            "Daily warm compresses and eyelid scrubs",
            "Artificial tears for associated dry eye",
            "Topical or oral antibiotics for severe or rosacea-associated cases (e.g., Doxycycline)"
        ],
        'precautions': [
            "Maintain strict daily eyelid hygiene",
            "Avoid eye makeup during flare-ups"
        ],
        'diagnostic_workup': [
            "Slit-lamp examination of eyelid margins and meibomian glands",
            "Assessment of tear break-up time (TBUT)"
        ],
        'differential_diagnoses': [
            "Dry eye syndrome",
            "Allergic conjunctivitis",
            "Ocular rosacea"
        ],
        'doctor_notes': "Blepharitis is a chronic condition that cannot be 'cured' but can be effectively managed with consistent eyelid hygiene.",
        'questions_for_doctor': [
            "What type of eyelid scrub or cleanser is best for me?",
            "Do I need an oral antibiotic for this inflammation?"
        ],
        'severity': "Low",
        'advice': "Start a daily routine of warm compresses and eyelid scrubs.",
    },
    'Chalazion': {
        'name': 'Chalazion',
        'group': 'Adnexal/Oculoplastic',
        'color': '#06B6D4',
        'pathophysiology': "A sterile, focal, lipogranulomatous inflammation of the eyelid resulting from an obstructed meibomian gland. Unlike a stye, it is typically non-infectious and chronic.",
        'analysis': "A firm, painless, immobile sub-epithelial nodule within the tarsal plate of the eyelid. No acute signs of infection (erythema, warmth) are typically present unless secondarily infected.",
        'description': "A painless, firm lump inside the eyelid caused by a blocked oil gland. It usually develops slowly over weeks.",
        'symptoms': [
            "Firm, painless lump in the eyelid",
            "Mild cosmetic disfigurement",
            "Possible blurred vision if the lump presses on the cornea (inducing astigmatism)"
        ],
        'treatment': [
            "Frequent warm compresses (10-15 mins, 4x daily) to liquefy meibum",
            "Gentle massage over the nodule",
            "In-office incision and curettage if unresolved after several weeks",
            "Intralesional steroid injection"
        ],
        'precautions': [
            "Do NOT attempt to pop or squeeze the lump"
        ],
        'diagnostic_workup': [
            "Slit-lamp examination to differentiate from malignant lesions",
            "Eversion of the eyelid to evaluate the palpebral conjunctiva"
        ],
        'differential_diagnoses': [
            "Hordeolum (Stye)",
            "Sebaceous gland carcinoma (if recurrent in the same spot)",
            "Epidermal inclusion cyst"
        ],
        'doctor_notes': "Most chalazia resolve with conservative heat therapy, but it can take weeks to months. Persistent lesions may require minor surgery.",
        'questions_for_doctor': [
            "Is this lump ready to be drained surgically?",
            "Could this be something more serious than a blocked gland?"
        ],
        'severity': "Low",
        'advice': "Apply warm compresses consistently. Consult an eye doctor if the lump persists for months.",
    },
    'Stye': {
        'name': 'Stye (Hordeolum)',
        'group': 'Adnexal/Oculoplastic',
        'color': '#06B6D4',
        'pathophysiology': "An acute focal pyogenic infection (usually Staphylococcal) of an eyelash follicle or gland of Zeis/Moll (external hordeolum) or a meibomian gland (internal hordeolum).",
        'analysis': "Focal, erythematous, tender, and warm nodule on the eyelid margin, often with a visible purulent punctum.",
        'description': "An acute, painful, red bump near the edge of the eyelid, similar to a pimple, caused by a bacterial infection.",
        'symptoms': [
            "Painful, tender, red bump on the eyelid margin",
            "Localized swelling and erythema",
            "Foreign body sensation or tearing"
        ],
        'treatment': [
            "Warm compresses to promote pointing and spontaneous drainage",
            "Topical antibiotic ointment (e.g., Erythromycin) if spreading",
            "Epilation of the involved eyelash (for external hordeolum)"
        ],
        'precautions': [
            "Do NOT squeeze or pop the stye, as this can spread the infection into the surrounding tissue (preseptal cellulitis)",
            "Discard old eye makeup and avoid wearing contacts until resolved"
        ],
        'diagnostic_workup': [
            "Clinical inspection via slit-lamp"
        ],
        'differential_diagnoses': [
            "Chalazion",
            "Preseptal cellulitis",
            "Acute blepharitis"
        ],
        'doctor_notes': "Styes are acute infections that typically drain spontaneously within a week with the aid of warm compresses. Watch for signs of spreading redness.",
        'questions_for_doctor': [
            "Do I need an antibiotic ointment for this infection?",
            "When should I be concerned about the infection spreading?"
        ],
        'severity': "Low to Moderate",
        'advice': "Apply warm compresses and keep the area clean. Seek care if the entire eyelid becomes swollen or red.",
    },
    'Keratitis': {
        'name': 'Keratitis',
        'group': 'Anterior Segment',
        'color': '#EF4444',
        'pathophysiology': "Inflammation of the cornea, which can be infectious (bacterial, viral [HSV, VZV], fungal, amoebic) or non-infectious (exposure, auto-immune, UV radiation). Infectious cases often involve epithelial defects and stromal infiltration.",
        'analysis': "Corneal epithelial defect visible with fluorescein staining. Stromal infiltrate (white opacity) may be present. Associated with conjunctival injection (ciliary flush) and anterior chamber reaction.",
        'description': "A serious inflammation or ulceration of the clear front surface of the eye (cornea). It is a sight-threatening emergency that can cause rapid tissue damage.",
        'symptoms': [
            "Severe eye pain and redness",
            "Extreme sensitivity to light (photophobia)",
            "Decreased or blurry vision",
            "Excessive tearing or discharge",
            "Sensation of something in the eye"
        ],
        'treatment': [
            "Intensive topical fortified antibiotics (for bacterial ulcers)",
            "Topical antivirals (e.g., Trifluridine or Ganciclovir for HSV)",
            "Discontinuation of all contact lens wear",
            "Cycloplegic drops for pain control"
        ],
        'precautions': [
            "SEEK IMMEDIATE OPHTHALMIC CARE",
            "Never sleep in contact lenses or expose them to tap water/swimming pools",
            "Do NOT use steroid drops unless specifically prescribed by an ophthalmologist for this condition, as they can worsen certain infections."
        ],
        'diagnostic_workup': [
            "Slit-lamp examination with fluorescein staining",
            "Corneal scraping for culture and sensitivity (in cases of large or central ulcers)"
        ],
        'differential_diagnoses': [
            "Corneal abrasion",
            "Acute angle-closure glaucoma",
            "Uveitis"
        ],
        'doctor_notes': "Infectious keratitis (especially Pseudomonas in contact lens wearers) can rapidly perforate the cornea. Urgent, aggressive antimicrobial therapy is required.",
        'questions_for_doctor': [
            "Is the infection close to the center of my vision?",
            "Will this leave a permanent scar on my cornea?"
        ],
        'severity': "Urgent Sight-Threatening Emergency",
        'advice': "URGENT: Remove contact lenses immediately and seek emergency eye care.",
    },
    'Subconjunctival Hemorrhage': {
        'name': 'Subconjunctival Hemorrhage',
        'group': 'Ocular Surface',
        'color': '#10B981',
        'pathophysiology': "Rupture of a small conjunctival or episcleral blood vessel, leading to the pooling of blood in the potential space between the conjunctiva and the episclera. It can be triggered by a Valsalva maneuver (coughing, sneezing), minor trauma, hypertension, or coagulopathy.",
        'analysis': "Sharply demarcated area of dense, bright red blood beneath the bulbar conjunctiva. The underlying sclera is obscured. No discharge, chemosis, or anterior chamber involvement.",
        'description': "A benign, painless accumulation of bright red blood under the clear membrane of the eye. It looks alarming but is typically harmless, akin to a bruise on the skin.",
        'symptoms': [
            "Bright red patch on the white of the eye",
            "Completely painless",
            "No change in vision",
            "No discharge"
        ],
        'treatment': [
            "Reassurance and observation (resolves spontaneously in 1-3 weeks)",
            "Artificial tears if mild surface irritation is present"
        ],
        'precautions': [
            "Avoid rubbing the eye",
            "Check blood pressure if recurrent",
            "Review use of blood thinners with primary care provider if recurrent"
        ],
        'diagnostic_workup': [
            "Clinical inspection",
            "Blood pressure measurement"
        ],
        'differential_diagnoses': [
            "Conjunctivitis (viral/bacterial)",
            "Episcleritis (has inflamed vessels, not pooled blood)",
            "Kaposi sarcoma (rare, raised)"
        ],
        'doctor_notes': "While visually striking, a subconjunctival hemorrhage is usually idiopathic and benign. It clears much like a cutaneous bruise, shifting from red to yellow over time.",
        'questions_for_doctor': [
            "Is there any sign of damage to the inside of my eye?",
            "Should I have my blood pressure or bleeding times checked?"
        ],
        'severity': "Low / Benign",
        'advice': "Reassurance. The blood will absorb naturally over 1-3 weeks. No specific eye drops are needed.",
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

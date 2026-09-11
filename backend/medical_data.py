from __future__ import annotations
from typing import Any, Dict

MEDICAL_INFO: Dict[str, Dict[str, Any]] = {
    'Diabetic Retinopathy': {
        'name': 'Diabetic Retinopathy',
        'group': 'Posterior Segment / Retinal Vascular',
        'color': '#FF0055',
        'pathophysiology': (
            "Diabetic retinopathy (DR) is a microangiopathy of retinal precapillary arterioles, capillaries, and venules "
            "induced by prolonged hyperglycemia. Hyperglycemia triggers pericyte apoptosis, basement membrane thickening, "
            "and loss of endothelial junctional integrity, precipitating microaneurysms, vascular leakage (macular edema), "
            "and widespread capillary drop-out. Subsequent retinal hypoxia upregulates vascular endothelial growth factor (VEGF), "
            "driving fragile preretinal neovascularization prone to vitreous hemorrhage and tractional retinal detachment."
        ),
        'analysis': (
            "Fundus evaluation exhibits multiple microaneurysms, blot/dot intraretinal hemorrhages, cotton-wool spots (infarctions of the "
            "nerve fiber layer), and lipid exudates. Preretinal neovascular fronds or fibrous proliferation indicate proliferative stage (PDR)."
        ),
        'description': (
            "Diabetic Retinopathy is the leading cause of preventable blindness in working-age adults worldwide. It progresses through "
            "non-proliferative (mild, moderate, severe NPDR) to proliferative (PDR) stages, often remaining asymptomatic until severe visual loss ensues."
        ),
        'symptoms': [
            "Early stages are completely asymptomatic (silent progressive microvascular damage)",
            "Gradual or fluctuating blurring of vision",
            "Sudden emergence of floaters, dark cobwebs, or reddish haze (preretinal/vitreous hemorrhage)",
            "Impaired color vision and decreased contrast sensitivity",
            "Central dark spots or blank areas (diabetic macular edema)",
        ],
        'treatment': [
            "Strict systemic optimization of HbA1c (<7.0%), blood pressure (<130/80 mmHg), and serum lipids",
            "Intravitreal Anti-VEGF Injections (Aflibercept, Ranibizumab, Faricimab) for center-involving diabetic macular edema",
            "Panretinal Photocoagulation (PRP Laser): Targeted ablation of ischemic peripheral retina to halt neovascularization",
            "Pars Plana Vitrectomy (PPV): Surgical clearance of non-clearing vitreous hemorrhage or repair of tractional detachment",
        ],
        'precautions': [
            "Undergo annual comprehensive dilated fundus examination, or every 3-6 months if retinopathy is established",
            "Continuously track blood glucose using continuous glucose monitors (CGM)",
            "Avoid strenuous Valsalva maneuvers or heavy lifting if active proliferative neovascularization is suspected",
            "Maintain smoking cessation to decrease microvascular oxidative insult",
        ],
        'diagnostic_workup': [
            "Ultra-Widefield Color Fundus Photography: Comprehensive 200-degree retinal imaging of peripheral ischemia",
            "Optical Coherence Tomography (OCT): High-resolution micron-level cross-sectional imaging for macular edema",
            "Optical Coherence Tomography Angiography (OCTA): Non-invasive mapping of the foveal avascular zone (FAZ) and capillary drop-out",
            "Fluorescein Angiography (FA): Dynamic transit mapping of microvascular leakage and retinal non-perfusion",
        ],
        'differential_diagnoses': [
            "Hypertensive Retinopathy",
            "Retinal Vein Occlusion (CRVO / BRVO)",
            "Ocular Ischemic Syndrome",
            "Radiation Retinopathy",
        ],
        'doctor_notes': (
            "Vision loss from diabetic retinopathy is largely preventable with early detection and prompt anti-VEGF or laser therapy. "
            "Patients with diabetes must never await visual symptoms before seeking ophthalmic screening."
        ),
        'questions_for_doctor': [
            "What stage of diabetic retinopathy is currently present in each eye?",
            "Is there any evidence of diabetic macular edema affecting central vision?",
            "When should my next dilated fundus examination and OCT scan take place?",
            "Are anti-VEGF intravitreal injections indicated for my condition?",
        ],
    },

    'Glaucoma': {
        'name': 'Glaucoma',
        'group': 'Optic Neuropathy',
        'color': '#7928CA',
        'pathophysiology': (
            "Glaucoma encompasses a group of progressive optic neuropathies characterized by morphological changes at the optic nerve head "
            "(optic disc cupping and neuroretinal rim thinning) and corresponding retinal ganglion cell (RGC) death and visual field loss. "
            "Elevated intraocular pressure (IOP) resulting from impaired trabecular meshwork aqueous humor outflow is the primary modifiable risk factor."
        ),
        'analysis': (
            "Fundus examination reveals enlargement of the optic cup-to-disc ratio (CDR > 0.6 or vertical asymmetry > 0.2), "
            "neuroretinal rim notch formation (violating the ISNT rule), disc hemorrhages (Drance hemorrhages), and peripapillary retinal nerve fiber layer (RNFL) bundle defects."
        ),
        'description': (
            "Frequently referred to as the 'sneak thief of sight', primary open-angle glaucoma progresses painlessly from peripheral to central visual loss, "
            "often escaping patient awareness until extensive and irreversible optic atrophy has occurred."
        ),
        'symptoms': [
            "Typically asymptomatic in early and moderate stages",
            "Subtle progressive constriction of peripheral visual field ('tunnel vision')",
            "Difficulty adjusting to dark environments",
            "In acute angle-closure: Sudden severe ocular and periorbital pain, halos around lights, corneal clouding, and nausea/vomiting",
        ],
        'treatment': [
            "Topical Hypotensive Eyedrops: Prostaglandin analogues (Latanoprost), Beta-blockers (Timolol), Alpha-2 agonists, Carbonic anhydrase inhibitors",
            "Selective Laser Trabeculoplasty (SLT): Safe non-invasive laser therapy targeting pigmented trabecular meshwork to improve outflow",
            "Micro-Invasive Glaucoma Surgery (MIGS): Trabecular micro-bypass stents (iStent, Hydrus) during cataract extraction",
            "Incisional Surgery: Trabeculectomy or Glaucoma Drainage Device (Ahmed/Baerveldt valve) for advanced uncontrolled IOP",
        ],
        'precautions': [
            "Strictly adhere to daily prescribed eyedrop timing; non-compliance is the leading cause of glaucomatous progression",
            "Attend routine scheduled visual field and OCT RNFL tests every 6 to 12 months",
            "Inform family members, as first-degree relatives carry a 4- to 9-fold increased lifetime risk",
            "Avoid prolonged head-down inverted postures (such as certain yoga positions) which spike episcleral venous pressure",
        ],
        'diagnostic_workup': [
            "Applanation Tonometry: Goldmann intraocular pressure measurement",
            "Spectral-Domain OCT (SD-OCT): Quantitative circumpapillary RNFL thickness and macular ganglion cell complex (GCC) analysis",
            "Automated Humphrey Visual Field (HVF 24-2 / 30-2): Standard automated perimetry to map functional field defects",
            "Gonioscopy: Four-mirror examination of the anterior chamber angle to differentiate open vs closed angle",
        ],
        'differential_diagnoses': [
            "Physiologic optic disc cupping (large normal optic nerves)",
            "Non-arteritic anterior ischemic optic neuropathy (NAION)",
            "Compressive optic chiasm lesions (pituitary adenoma)",
            "Optic neuritis / toxic optic neuropathy",
        ],
        'doctor_notes': (
            "Once optic nerve fibers are lost to glaucoma, they cannot be regenerated. All current therapies target lowering IOP to prevent further progression."
        ),
        'questions_for_doctor': [
            "What is my current intraocular pressure and what is my target IOP?",
            "What does my OCT RNFL progression trend show over the past year?",
            "Am I a candidate for laser trabeculoplasty (SLT) instead of or in addition to eye drops?",
            "Should my children and siblings be screened for glaucoma?",
        ],
    },

    'Age-related Macular Degeneration': {
        'name': 'Age-related Macular Degeneration (AMD)',
        'group': 'Maculopathy',
        'color': '#FF4D4D',
        'pathophysiology': (
            "AMD is a degenerative disorder of the central retina (macula), retinal pigment epithelium (RPE), and choriocapillaris. "
            "Accumulation of lipid and proteinaceous metabolic debris (lipofuscin and drusen) beneath the RPE impairs nutrient exchange, "
            "leading to geographic atrophy of photoreceptors ('dry' AMD) or pathological neovascularization penetrating Bruch's membrane "
            "into the subretinal space with exudation, hemorrhage, and disciform fibrotic scarring ('wet' AMD)."
        ),
        'analysis': (
            "Fundus imaging reveals confluent soft drusen in the central fovea and perifoveal zone, RPE mottling/hyperpigmentation, "
            "sharply circumscribed areas of RPE atrophy (dry AMD), or subretinal fluid, intraretinal hemorrhage, and grey-green neovascular membranes (wet AMD)."
        ),
        'description': (
            "AMD is the premier cause of irreversible central blindness in individuals over age 55 in developed nations. While peripheral navigation vision "
            "is typically spared, central reading, facial recognition, and driving abilities are profoundly disrupted."
        ),
        'symptoms': [
            "Metamorphopsia: Wavy or distorted appearance of straight lines (door frames, tile grids)",
            "Central scotoma: A dark, grey, or blind spot in the middle of the visual field",
            "Difficulty recognizing faces or reading fine print despite adequate magnification",
            "Need for brighter light when reading",
            "Reduced color vibrancy and brightness",
        ],
        'treatment': [
            "Intravitreal Anti-VEGF Therapy (Aflibercept, Faricimab, Bevacizumab) on Treat-and-Extend protocols for wet AMD",
            "Complement Inhibitor Injections (Pegcetacoplan, Avacincaptad pegol) to slow geographic atrophy progression",
            "AREDS2 Micronutrient Supplementation (Vitamin C, E, Zinc, Copper, Lutein, Zeaxanthin) for intermediate dry AMD",
            "Low-Vision Rehabilitation: Optical magnifiers, contrast filters, and digital assistive technology",
        ],
        'precautions': [
            "Perform daily home monocular self-monitoring using the Amsler Grid; report any new distortion immediately",
            "Strict smoking cessation; smoking multiplies AMD progression risk by up to 4-fold",
            "Adopt a Mediterranean diet rich in dark leafy greens, omega-3 fatty acids, and cold-water fish",
            "Wear UV/blue-light blocking sunglasses outdoors",
        ],
        'diagnostic_workup': [
            "High-Resolution Macular OCT: Primary imaging modality to identify subretinal fluid (SRF) and intraretinal fluid (IRF)",
            "OCT Angiography (OCTA): Non-invasive visualization of choroidal neovascular networks in the choriocapillaris",
            "Fluorescein / Indocyanine Green Angiography (FA / ICGA): To delineate occult or polypoidal choroidal vasculopathy",
            "Fundus Autofluorescence (FAF): To map hyperautofluorescent active borders of geographic atrophy",
        ],
        'differential_diagnoses': [
            "Central Serous Chorioretinopathy (CSCR)",
            "Pathological Myopic Choroidal Neovascularization",
            "Pattern Dystrophy of the RPE",
            "Macular Telangiectasia Type 2 (MacTel)",
        ],
        'doctor_notes': (
            "Metamorphopsia detected on an Amsler grid is an ophthalmic emergency indicative of acute wet conversion. Prompt anti-VEGF therapy within days rescues photoreceptor viability."
        ),
        'questions_for_doctor': [
            "Do I have the dry or wet form of macular degeneration in each eye?",
            "Should I be taking the AREDS2 formula supplements?",
            "How often should I be testing myself with the Amsler Grid at home?",
            "What signs require same-day emergency evaluation?",
        ],
    },

    'Cataract': {
        'name': 'Cataract',
        'group': 'Anterior Segment / Media Opacity',
        'color': '#00F5D4',
        'pathophysiology': (
            "Cataract is the opacification of the crystalline lens. On fundus photography, cataracts scatter and attenuate the illuminating "
            "and returning light rays, resulting in a dim, yellowish, or completely obscured view of the optic disc and retinal vascular tree."
        ),
        'analysis': (
            "Fundus image shows significant generalized attenuation of retinal details, loss of high-frequency vascular margins, "
            "and blunting of the retinal reflex due to anterior media opacity."
        ),
        'description': (
            "While primarily a disorder of the crystalline lens, cataracts are a core category in fundus screening as they impede retinal diagnostics "
            "and represent the single largest cause of reversible blindness worldwide."
        ),
        'symptoms': [
            "Gradual, painless blurring or fogging of vision",
            "Severe glare and starbursts around bright lights or headlights",
            "Faded or yellowed color vision",
            "Frequent glasses prescription changes",
        ],
        'treatment': [
            "Micro-incisional Phacoemulsification with Intraocular Lens (IOL) implantation",
            "Femtosecond Laser-Assisted Cataract Surgery (FLACS)",
            "Post-operative refractive optimization with toric or extended depth of focus (EDOF) lenses",
        ],
        'precautions': [
            "Protect eyes from UV radiation using UV400 sunglasses",
            "Control blood glucose to retard diabetic cortical cataractogenesis",
            "Undergo dilated examination to confirm retinal health prior to scheduled surgery",
        ],
        'diagnostic_workup': [
            "Slit-Lamp Biomicroscopy of lens nucleus, cortex, and posterior subcapsular region",
            "Optical Biometry (IOL Master) to calculate intraocular lens power",
            "B-Scan Ultrasound if media opacity precludes fundus visualization",
        ],
        'differential_diagnoses': [
            "Corneal opacification",
            "Vitreous hemorrhage",
            "Posterior capsular opacification (PCO)",
        ],
        'doctor_notes': (
            "Cataract extraction not only restores crystal clear visual acuity but also permits detailed therapeutic monitoring of underlying retinal and macular health."
        ),
        'questions_for_doctor': [
            "Is my cataract ready for surgical removal?",
            "Which intraocular lens (monofocal, toric, multifocal) is best suited to my lifestyle?",
            "Does my retina look completely healthy behind the cataract?",
        ],
    },

    'Hypertensive Retinopathy / Pathological Myopia': {
        'name': 'Hypertensive Retinopathy / Pathological Myopia',
        'group': 'Vascular & Structural Retinal Disorders',
        'color': '#F59E0B',
        'pathophysiology': (
            "Hypertensive retinopathy results from acute or chronic systemic blood pressure elevations triggering retinal arteriolar vasoconstriction, "
            "sclerosis ('copper/silver wiring'), arteriovenous crossing compression (AV nicking), and breakdown of the inner blood-retina barrier. "
            "Pathological myopia is marked by excessive axial elongation (>26mm) producing posterior staphyloma, chorioretinal lacquer cracks, and retinal thinning."
        ),
        'analysis': (
            "Fundus appearance features flame-shaped hemorrhages in the superficial retinal nerve fiber layer, hard exudate 'macular stars', "
            "cotton-wool patches, myopic crescents around the optic disc, or tigroid/tesselated fundus patterns."
        ),
        'description': (
            "This diagnostic class covers critical systemic microvascular damage and high-myopic structural degeneration visible directly in the retina."
        ),
        'symptoms': [
            "Often asymptomatic in mild to moderate hypertensive retinopathy",
            "Severe sudden headaches and visual dimming in malignant hypertension",
            "Significant distance blur, floaters, and peripheral flashes in high myopia",
            "Peripheral visual shadow if retinal detachment or tears complicate myopia",
        ],
        'treatment': [
            "Urgent medical management and titration of antihypertensive medications",
            "Dilated peripheral retinal screening and prophylactic laser photocoagulation for myopic retinal tears/lattice degeneration",
            "Anti-VEGF therapy for myopic choroidal neovascularization",
        ],
        'precautions': [
            "Monitor blood pressure regularly at home with an automated upper-arm cuff",
            "Avoid contact sports or head trauma if high axial myopia is present",
            "Seek immediate ophthalmic triage for sudden visual field curtains or lightning flashes",
        ],
        'diagnostic_workup': [
            "Dilated 360-Degree Peripheral Retinal Indirect Ophthalmoscopy with scleral depression",
            "Ultra-Widefield Fundus Photography and Fluorescein Angiography",
            "Optical Biometry for axial length tracking",
            "Cardiovascular and renal systemic risk assessment",
        ],
        'differential_diagnoses': [
            "Diabetic Retinopathy",
            "Branch Retinal Vein Occlusion",
            "Collagen vascular disease retinopathy",
        ],
        'doctor_notes': (
            "The retina provides the only non-invasive window in the human body to view microvascular vessels directly. Retinal vessel damage mirrors cerebrovascular and renal disease."
        ),
        'questions_for_doctor': [
            "Are the retinal blood vessels showing signs of high blood pressure?",
            "Do I have any peripheral retinal thinning, lattice degeneration, or holes from high myopia?",
            "What should my target blood pressure be to protect my vision?",
        ],
    },

    'Normal': {
        'name': 'Normal (Healthy Retina)',
        'group': 'Healthy Fundus',
        'color': '#10B981',
        'pathophysiology': (
            "A healthy posterior pole exhibits an intact, pink, well-perfused neuroretinal rim with a cup-to-disc ratio within normal biological limits (<0.5). "
            "The foveal avascular zone (FAZ) and foveal light reflex are sharp and distinct. Retinal arteries and veins display normal 2:3 caliber ratios "
            "without focal narrowing, microaneurysms, hemorrhages, exudates, or drusen."
        ),
        'analysis': (
            "Normal fundus presentation: Optic disc margins are sharp and flat, macula is lutein-pigmented and devoid of fluid or drusen, "
            "and the peripheral background retina shows uniform choroidal perfusion."
        ),
        'description': (
            "No signs of sight-threatening posterior eye disease detected. Routine preventive screenings maintain longitudinal eye health."
        ),
        'symptoms': [
            "Sharp, clear visual acuity",
            "No distortion of straight lines",
            "Intact peripheral visual field",
            "Absence of unexplained flashes, floaters, or central blind spots",
        ],
        'treatment': [
            "No active ophthalmic medical or surgical intervention required",
            "Maintain general wellness and balanced dietary nutrition",
        ],
        'precautions': [
            "Undergo dilated retinal screening every 1-2 years (or annually if diabetic)",
            "Wear sunglasses with 100% UVA/UVB protection",
            "Protect eyes with safety glasses during hazardous sports or DIY activities",
        ],
        'diagnostic_workup': [
            "Standard routine comprehensive eye examination including dilated ophthalmoscopy",
            "Baseline non-mydriatic or mydriatic fundus photography",
        ],
        'differential_diagnoses': [],
        'doctor_notes': (
            "Healthy retinal examination. Continue routine annual preventative eye care visits to monitor for subtle asymptomatic changes over time."
        ),
        'questions_for_doctor': [
            "How often should I return for routine dilated retinal screenings based on my age and medical history?",
        ],
    },
}

"""Label metadata for the 38 PlantVillage classes.

Every class carries the plant name, disease name, a symptom description and a
treatment recommendation, so the API can return a complete diagnosis card
without any external lookup at request time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

# Canonical PlantVillage directory names in a fixed order. Training writes the
# same order into the checkpoint, so index <-> class mapping stays stable.
CLASS_NAMES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)

# Plant folder token -> display name
PLANTS: Dict[str, str] = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn (Maize)",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pepper,_bell": "Bell Pepper",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}

# Disease folder token -> diagnosis card content.
DISEASES: Dict[str, Dict[str, str]] = {
    "healthy": {
        "name": "Healthy",
        "pathogen": "-",
        "description": "No disease symptoms detected. Leaf colour and texture look normal.",
        "treatment": "No treatment needed. Continue regular irrigation, balanced fertiliser and weekly scouting.",
    },
    "Apple_scab": {
        "name": "Apple Scab",
        "pathogen": "Venturia inaequalis (fungus)",
        "description": "Olive-green to black velvety spots that later crack the leaf surface; severe infection causes early defoliation.",
        "treatment": "Remove and destroy fallen leaves, prune for airflow, and spray captan or myclobutanil at bud break, repeating every 10-14 days in wet weather.",
    },
    "Black_rot": {
        "name": "Black Rot",
        "pathogen": "Botryosphaeria obtusa / Guignardia bidwellii (fungi)",
        "description": "Brown circular leaf lesions with concentric rings, accompanied by rotting, shrivelled fruit and branch cankers.",
        "treatment": "Prune out cankers and mummified fruit, keep the canopy dry, and apply captan or thiophanate-methyl on a protective schedule.",
    },
    "Cedar_apple_rust": {
        "name": "Cedar Apple Rust",
        "pathogen": "Gymnosporangium juniperi-virginianae (fungus)",
        "description": "Bright yellow-orange leaf spots produced by a rust that alternates between apple and juniper hosts.",
        "treatment": "Remove nearby juniper galls where possible and spray myclobutanil or mancozeb from pink bud through the early cover sprays.",
    },
    "Powdery_mildew": {
        "name": "Powdery Mildew",
        "pathogen": "Podosphaera / Erysiphe spp. (fungi)",
        "description": "White powdery fungal growth on the leaf surface that stunts shoots and reduces photosynthetic area.",
        "treatment": "Improve spacing and airflow, avoid excess nitrogen, and spray sulphur, potassium bicarbonate or a triazole fungicide at first sign.",
    },
    "Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Cercospora / Gray Leaf Spot",
        "pathogen": "Cercospora zeae-maydis (fungus)",
        "description": "Long rectangular grey-tan lesions running parallel to the corn leaf veins, severe in warm humid weather.",
        "treatment": "Rotate crops, bury residue, plant resistant hybrids, and apply a strobilurin or triazole fungicide at early tasseling when pressure is high.",
    },
    "Common_rust_": {
        "name": "Common Rust",
        "pathogen": "Puccinia sorghi (fungus)",
        "description": "Cinnamon-brown pustules scattered on both leaf surfaces, rupturing the epidermis as they mature.",
        "treatment": "Grow resistant hybrids; a triazole or strobilurin spray pays off only when pustules appear before silking.",
    },
    "Northern_Leaf_Blight": {
        "name": "Northern Leaf Blight",
        "pathogen": "Exserohilum turcicum (fungus)",
        "description": "Long cigar-shaped grey-green lesions that merge and kill leaf tissue, cutting grain fill when they reach the ear leaf.",
        "treatment": "Use resistant hybrids, rotate away from corn, and spray at VT-R1 when lesions reach the ear leaf.",
    },
    "Esca_(Black_Measles)": {
        "name": "Esca (Black Measles)",
        "pathogen": "Phaeomoniella chlamydospora and associated wood-decay fungi",
        "description": "Interveinal chlorosis with tiger-stripe drying on grapevine leaves, driven by a trunk wood-decay complex.",
        "treatment": "There is no curative spray. Prune late in dry weather, seal large cuts, and remove severely affected vines.",
    },
    "Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": "Leaf Blight (Isariopsis Leaf Spot)",
        "pathogen": "Pseudocercospora vitis (fungus)",
        "description": "Angular dark-brown blotches with yellow halos on grape leaves, causing early defoliation.",
        "treatment": "Remove infected debris, open the canopy, and apply mancozeb or a copper fungicide during wet spells.",
    },
    "Haunglongbing_(Citrus_greening)": {
        "name": "Huanglongbing (Citrus Greening)",
        "pathogen": "Candidatus Liberibacter asiaticus, vectored by Diaphorina citri",
        "description": "Blotchy asymmetric leaf mottling that crosses veins, with lopsided bitter fruit and progressive tree decline.",
        "treatment": "No cure exists. Remove infected trees, control psyllid vectors, and replant with certified disease-free stock.",
    },
    "Bacterial_spot": {
        "name": "Bacterial Spot",
        "pathogen": "Xanthomonas spp. (bacteria)",
        "description": "Small water-soaked spots that turn dark and angular, often with a yellow margin, spreading fast in warm wet conditions.",
        "treatment": "Use pathogen-free seed, avoid overhead irrigation and working wet foliage, and apply copper plus mancozeb preventively.",
    },
    "Early_blight": {
        "name": "Early Blight",
        "pathogen": "Alternaria solani (fungus)",
        "description": "Brown spots with concentric target-like rings, appearing on the older lower leaves first and moving upward.",
        "treatment": "Rotate crops, mulch, stake plants, remove lower infected leaves, and spray chlorothalonil or mancozeb every 7-10 days.",
    },
    "Late_blight": {
        "name": "Late Blight",
        "pathogen": "Phytophthora infestans (oomycete)",
        "description": "Greasy grey-green water-soaked patches with white sporulation on the underside; spreads very fast in cool wet weather.",
        "treatment": "Act immediately: destroy infected plants, stop overhead watering, and apply metalaxyl or cymoxanil combined with mancozeb.",
    },
    "Leaf_Mold": {
        "name": "Leaf Mold",
        "pathogen": "Passalora fulva (fungus)",
        "description": "Pale yellow patches on the upper leaf surface with olive velvety mould beneath; thrives in humid greenhouses.",
        "treatment": "Drop humidity below 85%, ventilate, space plants, and spray chlorothalonil or a copper product.",
    },
    "Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "pathogen": "Septoria lycopersici (fungus)",
        "description": "Many small circular spots with grey centres and dark borders, starting on the lowest leaves after soil splash.",
        "treatment": "Remove infected lower leaves, mulch to stop soil splash, and spray chlorothalonil, mancozeb or copper on a 7-10 day cycle.",
    },
    "Spider_mites Two-spotted_spider_mite": {
        "name": "Two-spotted Spider Mite",
        "pathogen": "Tetranychus urticae (arachnid pest)",
        "description": "Fine yellow stippling, bronzing and fine webbing on the leaf underside from sustained cell-content feeding.",
        "treatment": "Hose down foliage, raise humidity, release predatory mites, and use insecticidal soap, horticultural oil or a specific miticide - not broad-spectrum insecticides.",
    },
    "Target_Spot": {
        "name": "Target Spot",
        "pathogen": "Corynespora cassiicola (fungus)",
        "description": "Small brown spots that enlarge into target-like concentric rings and merge into large necrotic areas.",
        "treatment": "Improve airflow, remove crop debris, and rotate azoxystrobin and difenoconazole sprays to limit resistance.",
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "pathogen": "Begomovirus, vectored by Bemisia tabaci (whitefly)",
        "description": "Upward leaf curling with yellow margins, severe stunting and poor fruit set; symptoms appear 2-3 weeks after infection.",
        "treatment": "No cure. Remove infected plants, use resistant varieties, install insect nets and control whitefly early.",
    },
    "Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "pathogen": "Tobamovirus (ToMV), mechanically transmitted",
        "description": "Light-and-dark green mottling, fern-like distorted leaves and internally mottled fruit; highly stable on tools and hands.",
        "treatment": "No cure. Rogue out infected plants, disinfect hands and tools, avoid tobacco contact and use clean certified seed.",
    },
    "Leaf_scorch": {
        "name": "Leaf Scorch",
        "pathogen": "Diplocarpon earlianum (fungus)",
        "description": "Irregular purple blotches that coalesce and dry the leaf margins, weakening the strawberry crown.",
        "treatment": "Renovate beds after harvest, remove old infected leaves, improve drainage, and spray captan or myclobutanil.",
    },
}


@dataclass(frozen=True)
class LabelInfo:
    class_name: str
    index: int
    plant: str
    disease: str
    pathogen: str
    healthy: bool
    description: str
    treatment: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split(class_name: str):
    plant, _, disease = class_name.partition("___")
    return plant, disease


def _build() -> Dict[str, LabelInfo]:
    table: Dict[str, LabelInfo] = {}
    for index, class_name in enumerate(CLASS_NAMES):
        plant_token, disease_token = _split(class_name)
        disease = DISEASES[disease_token]
        table[class_name] = LabelInfo(
            class_name=class_name,
            index=index,
            plant=PLANTS[plant_token],
            disease=disease["name"],
            pathogen=disease["pathogen"],
            healthy=disease_token == "healthy",
            description=disease["description"],
            treatment=disease["treatment"],
        )
    return table


LABELS: Dict[str, LabelInfo] = _build()


def label_for_index(index: int) -> LabelInfo:
    return LABELS[CLASS_NAMES[index]]


def label_for_name(class_name: str) -> LabelInfo:
    return LABELS[class_name]


PLANT_NAMES: List[str] = sorted(set(PLANTS.values()))

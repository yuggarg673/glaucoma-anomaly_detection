"""Plain-language explanations shown to users. Single source for the website (via GET /explain).

docs/USER_GUIDE.md carries the same text for readers of the repository; keep them in step.
Wording rules: never say "diagnosis"; never call confidence a probability of disease.
"""

DISCLAIMER = (
    "This is a student research project and a screening aid. It has not been clinically validated and does "
    "not give a medical diagnosis. Only an eye doctor can diagnose glaucoma. If you are worried about your "
    "vision, see an ophthalmologist, whatever this tool says."
)

GLOSSARY = {
    "label": {
        "title": "Result label",
        "what": "Whether the combined score is above or below the cut-off the system chose on its validation images.",
        "read": "'Glaucoma suspected' means this image looks more like the glaucoma examples than the healthy ones. "
                "'Normal-like' means the opposite. Neither is a diagnosis.",
    },
    "anomaly_score": {
        "title": "Anomaly score",
        "what": "A neural network was trained only on healthy retinal images. It tries to redraw the image you upload. "
                "The anomaly score is the average difference between your image and its redraw.",
        "read": "Higher = the image looks less like the healthy eyes the model learned from. Lower = more familiar. "
                "Compare it with the typical cut-off shown beside it. It is not a percentage and its scale depends on "
                "this model, so only the comparison with the cut-off is meaningful.",
        "caution": "Blur, glare, dark images, a different camera, or other eye conditions can also raise the score.",
    },
    "cdr": {
        "title": "Cup-to-disc ratio (CDR)",
        "what": "The optic disc is where the optic nerve leaves the eye. Its pale centre is the 'cup'. The CDR is the "
                "height of the cup divided by the height of the disc, measured on outlines the system draws automatically.",
        "read": "A bigger cup relative to the disc (a higher CDR) is one sign eye doctors look for in glaucoma. "
                "Values of about 0.65 or more are often treated as suspicious. Many healthy eyes also have large cups, "
                "so a high CDR alone does not mean glaucoma.",
        "caution": "Check the outlines on the annotated image. If they are in the wrong place, ignore the CDR. "
                   "'Unavailable' means the system could not find a cup or disc.",
    },
    "fused_score": {
        "title": "Combined (fused) score",
        "what": "The anomaly score and the CDR are blended into one number between 0 and 1, using weights learned "
                "on validation images. If the CDR is unavailable, only the anomaly score is used.",
        "read": "Closer to 1 = more suspicious. The label comes from comparing this number with the cut-off.",
    },
    "confidence": {
        "title": "Confidence",
        "what": "How far the combined score is from the cut-off, shown as High, Medium or Low.",
        "read": "High means the score is well away from the cut-off, so the label is less likely to flip with small "
                "changes. Low means it is close to the line and the result is uncertain.",
        "caution": "This is NOT the probability that you have or do not have glaucoma. It only describes how clear-cut "
                   "the score is. The model can be confidently wrong, especially on images unlike its training data.",
    },
    "signals_agree": {
        "title": "Do the two signals agree?",
        "what": "Whether the anomaly score and the CDR each point the same way.",
        "read": "If they disagree, treat the result with extra caution and have the image reviewed.",
    },
    "regions": {
        "title": "Marked regions (A1, A2 ...)",
        "what": "Boxes drawn where the redrawn image differs most from your image.",
        "read": "Red dashed = high difference, amber dashed = moderate. They show where the model found the image "
                "unfamiliar. They are not confirmed lesions and are not a medical finding.",
    },
}

HOW_TO_USE = [
    "Use a colour retinal fundus photograph (JPG or PNG) centred on the optic disc. Photos of a screen or printed "
    "photos work poorly.",
    "Choose a sharp, well-lit image without heavy glare, dark corners over the disc, or eyelashes in front.",
    "Upload it on the Analyze page and wait a few seconds.",
    "Read the label first, then the scores, then look at the annotated image to see where the model looked.",
    "Check the disc (teal) and cup (white) outlines. If they are clearly wrong, do not trust the CDR.",
    "Treat Low confidence, or signals that disagree, as 'unclear'.",
    "Whatever the result, show the image and any symptoms to an eye doctor. Do not use this tool to decide on "
    "treatment or to ignore symptoms.",
]

LIMITS = [
    "Trained and tested on public datasets of limited size, so performance on other people, cameras or hospitals "
    "may be worse.",
    "Not clinically validated and not a medical device.",
    "The anomaly score also rises for image problems and for diseases other than glaucoma.",
    "Uploaded images are analysed in memory and not saved. Only the scores and file name are kept in the history.",
]

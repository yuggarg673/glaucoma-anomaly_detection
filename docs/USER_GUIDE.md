# User guide: how to use the tool and read the results

> **Screening aid only.** This is a student research project. It has not been clinically validated and does not
> give a medical diagnosis. Only an eye doctor can diagnose glaucoma. If you are worried about your vision, see an
> ophthalmologist, whatever this tool says.

The same text is shown inside the website (the **How to read results** page). It lives in `backend/glossary.py`.

## Using the tool

1. Use a colour retinal fundus photograph (JPG or PNG) centred on the optic disc. Photos of a screen or printed
   photos work poorly.
2. Choose a sharp, well-lit image without heavy glare, dark corners over the disc, or eyelashes in front.
3. Upload it on the **Analyze** page and wait a few seconds.
4. Read the label first, then the scores, then look at the annotated image to see where the model looked.
5. Check the disc (teal) and cup (white) outlines. If they are clearly wrong, do not trust the CDR.
6. Treat Low confidence, or signals that disagree, as "unclear".
7. Whatever the result, show the image and any symptoms to an eye doctor.

## What each number means

### Result label
Whether the combined score is above or below the cut-off the system chose on its validation images.
**Glaucoma suspected** means this image looks more like the glaucoma examples than the healthy ones. **Normal-like**
means the opposite. Neither is a diagnosis.

### Anomaly score
A neural network was trained **only on healthy** retinal images. It tries to redraw the image you upload. The anomaly
score is the average difference between your image and its redraw.

- **Higher** = the image looks less like the healthy eyes the model learned from. **Lower** = more familiar.
- Compare it with the typical cut-off shown beside it. It is **not a percentage**, and its scale depends on this
  model, so only the comparison with the cut-off is meaningful.
- Blur, glare, dark images, a different camera, or other eye conditions can also raise it.

### Cup-to-disc ratio (CDR)
The optic disc is where the optic nerve leaves the eye. Its pale centre is the "cup". The CDR is the height of the cup
divided by the height of the disc, measured on outlines the system draws automatically.

- A bigger cup relative to the disc (a **higher CDR**) is one sign eye doctors look for in glaucoma. About **0.65 or
  more** is often treated as suspicious. Many healthy eyes also have large cups, so a high CDR alone does not mean
  glaucoma.
- Check the outlines on the annotated image. If they are in the wrong place, ignore the CDR.
- **Unavailable** means the system could not find a cup or disc.

### Combined (fused) score
The anomaly score and the CDR are blended into one number between 0 and 1, using weights learned on validation
images. If the CDR is unavailable, only the anomaly score is used. Closer to 1 = more suspicious.

### Confidence
How far the combined score is from the cut-off, shown as **High, Medium or Low**.

- **High**: well away from the cut-off, so the label is less likely to flip with small changes.
- **Low**: close to the line, so the result is uncertain.
- It is **not** the probability that you have or do not have glaucoma. The model can be confidently wrong,
  especially on images unlike its training data.

### Do the two signals agree?
Whether the anomaly score and the CDR each point the same way. If they disagree, treat the result with extra caution.

### Marked regions (A1, A2, ...)
Boxes drawn where the redrawn image differs most from your image. Red dashed = high difference, amber dashed =
moderate. They show where the model found the image unfamiliar. **They are not confirmed lesions.**

## Limits

- Trained and tested on public datasets of limited size, so results on other people, cameras or hospitals may be worse.
- Not clinically validated and not a medical device.
- The anomaly score also rises for image problems and for diseases other than glaucoma.
- Uploaded images are analysed in memory and not saved. Only the scores and file name are kept in the history.

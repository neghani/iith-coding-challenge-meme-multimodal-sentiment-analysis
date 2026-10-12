# Viva notes — meme multimodal sentiment

Personal notes so I can answer if they ask about the notebook.

**What I actually submitted**

I built a two-branch multimodal classifier: frozen CLIP for the meme image and caption, frozen Twitter-RoBERTa for caption sentiment, a small gated fusion head, and a majority-class baseline. On this noisy 5-class Memotion-style set the model does not beat always predicting `positive` (~45% train, ~49% Kaggle). I am submitting that result honestly rather than a file that only looks more diverse.

---

## Facts I should not mix up

| Item | What I used / got |
| --- | --- |
| Notebook | `AIET_PlayGround_Challenge_Colab_Notebook_v11.ipynb` |
| Image encoder | Frozen `openai/clip-vit-large-patch14` |
| Text encoder | Frozen `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Fallback text model | `distilbert-base-uncased-finetuned-sst-2-english` (only if RoBERTa fails to load) |
| Trainable part | Small MLP fusion head (hundreds of thousands of params, not the backbones) |
| Train size | 6,992 labelled memes |
| Test size | 2,098 rows |
| Classes | `very_positive`, `positive`, `neutral`, `negative`, `very_negative` |
| Metric | Accuracy |
| Majority class | `positive` ≈ 44.7% of training labels |
| 5-fold OOF accuracy | **0.447** = majority baseline |
| Kaggle | **~0.49** |
| Test prediction mix | **100%** `positive` |
| Test–train caption overlap | **99.9%** of test captions already appear in `labels.csv` |

If they ask whether I used DistilBERT: earlier iteration, yes, as the main text tower. In the submitted notebook it is only a fallback. Main text model is Twitter-RoBERTa.

If they ask about ResNet18: that was my first serious run. It scored 0.22. I replaced it.

---

## What I tried and why I changed it

I used AI for APIs and Colab bugs (transformers versions, Drive cache, boilerplate). I picked the architecture from the scores I saw.

### First run — ResNet18 (Kaggle **0.22**)

Image branch was a small ImageNet CNN. 0.22 on 5 classes is basically chance (1/5 = 0.20), and worse than always saying `positive`.

ImageNet object classification is the wrong prior for memes. These are screenshot-like, heavy text overlay, cultural — not “cat vs dog”.

### Second run — CLIP-B/32 + DistilBERT, unfroze last layers (Kaggle **0.498**)

CLIP ViT-B/32 for image, DistilBERT for text. Fine-tuned the last CLIP block and last DistilBERT layers. Test file looked mixed (~88% `positive`, some `neutral` / `very_positive`). Score 0.498.

Validation accuracy was **0.412**, *below* the 0.447 majority baseline. Those extra Kaggle points were not real class skill.

### Submitted version — frozen CLIP-L + Twitter-RoBERTa + small head (Kaggle **~0.49**)

I stopped fine-tuning the huge backbones on 7k noisy labels — that was memorising noise. Extract CLIP image, CLIP text, and RoBERTa features once, cache them, train only a regularised fusion head.

Added 5-fold CV, logistic regression as a second head, class-offset tuning, majority baseline on every plot.

OOF locked at 0.447. Test collapsed to all `positive`. Kaggle stayed ~49%.

I did not revert to the second run for 0.008 on the leaderboard. Submitting the honest notebook.

---

## Why I made those choices

### Why not ResNet18?

ResNet18 is trained on ImageNet: everyday objects. A meme is not “detect a dog”. Sentiment lives in the joke between pixels and overlay text. ResNet also has no idea what the caption means.

CLIP is trained with a contrastive image–text loss on a huge noisy web set. Its vision tower already lives in the same space as language. For memes that is the right inductive bias.

ResNet-style run → **0.22**. CLIP-based runs → **~0.49**.

### Why CLIP and not a from-scratch CNN?

6,992 noisy labels cannot train a vision transformer from scratch. Transfer learning is the only option that made sense. CLIP is already multimodal, so I did not have to invent an image–text alignment layer from nothing.

I used **ViT-L/14** rather than B/32 in the final notebook because I was extracting features once (frozen). A larger frozen encoder is cheap at train time; a larger *fine-tuned* encoder is what hurt the second run.

### Why a separate text model if CLIP already has a text tower?

CLIP text is a matching encoder (“does this caption go with this image?”), max 77 tokens. It is not a sentiment classifier.

Memes are slangy, sarcastic, Twitter-like. I added Twitter-RoBERTa sentiment and concatenated:

1. CLIP image embedding
2. CLIP caption embedding
3. RoBERTa mean-pooled hidden state
4. RoBERTa 3-way sentiment probabilities

Then a small head learns how to use them.

DistilBERT (uncased Wikipedia/BookCorpus) was the second-run text tower. Lighter, fine for generic English, weaker on meme slang. In the submitted notebook it is only a load fallback.

### Why freeze the backbones?

Fine-tuning CLIP + BERT on 7k crowd labels overfit noise. Val accuracy went below majority. Frozen features + a tiny head:

- trains in seconds per epoch
- cannot destroy the pretrained space
- makes image-only / text-only ablations actually mean something

If the head cannot beat majority on frozen CLIP-L features, unfreezing will not magically find 70% accuracy on this label set.

### Why gated fusion, not just concat?

Concat is the baseline fusion. I also used:

- element-wise product (agreement)
- absolute difference (disagreement — useful when the joke is image vs text)
- a learned gate `g = sigmoid(W [img; txt])` so the head can trust image or text per sample
- CLIP image–text cosine (“does the caption match the picture?”)

Textbook concat-only version if they want it: `z = [img; txt]`.

### Why these libraries?

Each one has a job.

| Library | What I used it for |
| --- | --- |
| `pandas` / `numpy` | CSV, cleaning, arrays |
| `PIL` | Load messy meme images (truncated files, odd formats) |
| `matplotlib` / `seaborn` | Required plots: class imbalance, curves, confusion matrix |
| `torch` | Fusion head, training loop |
| `transformers` | CLIP + RoBERTa — I did not reimplement ViT |
| `sklearn` | Stratified split, `StratifiedKFold`, accuracy / F1 / log-loss, logistic regression, `StandardScaler` |
| `kaggle` | Download competition files |
| Google Drive helpers | Colab session cache so I do not re-extract 7k images every runtime |

If they say it is too heavy: the backbones are pretrained and frozen. I am not training them. sklearn LogReg is a simple linear control so I can see whether the neural head is even needed.

### Why logistic regression next to a neural head?

Control, not a second fancy model. Same frozen features, linear classifier. On this data LogReg OOF accuracy was worse (~0.33–0.35) because it spread probability mass off `positive` and accuracy punished it. The blender put 100% weight on the MLP, which itself had collapsed to majority. Extra model capacity did not help.

### Why 5-fold CV?

One 80/20 split on noisy labels is jittery. 5-fold OOF is the number I trust. Every fold’s MLP sat at **0.4467–0.4478**.

### Why class offsets / `BALANCE_WEIGHT`?

Accuracy on an imbalanced set wants you to predict `positive`. Offsets add a bias per class on the log-probabilities, tuned on OOF. With `BALANCE_WEIGHT = 0.1` they barely moved accuracy (0.4472 → 0.4474) and the test file still went 100% positive. I tried not to be a constant classifier; the metric still chose that.

### Why is Kaggle 49% if train majority is 45%?

The hidden test mix is a bit more `positive` than train (or a constant-positive file simply hits that rate). **0.49 ≈ fraction of test labelled `positive`.** That is not “the model is 49% smart”.

### Did the model overfit?

Second run looked more diverse and was worse than majority on val — that is the overfit / noise-fit.

Submitted version did not overfit in the usual sense: train and OOF both sit on majority. Even with **99.9% caption overlap** between test and `labels.csv`, the head still predicted `positive`. A memorising model would have scored far above 49%. I did not copy training labels into the test file; that would be leakage, not classification.

### Why not look up the overlapping captions and get ~99% on Kaggle?

The task is to build a multimodal classifier, not to join `test.csv` to `labels.csv`. I printed the overlap as a diagnostic. Using it as answers would invalidate the assignment.

### Why 5 classes if the model only predicts one?

That is the dataset. Memotion-style overall sentiment is subjective. Adjacent classes (`positive` vs `very_positive`, `neutral` vs `negative`) are unstable. Research on this data often struggles to beat majority even with 3 classes. I still trained 5-way because that is the spec.

---

## Notebook order (if they open it)

1. **Kaggle + Drive cache** — train zip is the 6,992 labelled set; `test.csv` / sample submission come from the competition.
2. **Stage 1** — load `labels.csv`, drop junk index, show imbalance, print majority baseline **0.447**.
3. **Stage 2** — map `image_name` → file (jpg/png/gif/webp/jpe), CLIP preprocess (resize/crop, CLIP mean/std). No random flip: it mirrors meme text.
4. **Stage 3** — `text_corrected` with OCR fallback; light clean (URLs, watermarks); tokenize twice (CLIP 77, RoBERTa 128); stratified split.
5. **Stage 4** — frozen CLIP vision, frozen CLIP text + RoBERTa, gated fusion, MLP.
6. **Stage 5** — train only the head, early stop on val **loss**, plot vs majority line.
7. **Stage 6** — val report, confusion matrix (single `positive` column), ablations, 5-fold test predictions, `submission.csv`.

Ablations I ran: majority / fusion / image-only / text-only / logistic regression. Image-only and text-only should not beat a model that already cannot leave `positive`.

---

## Questions I expect

**“Your confusion matrix is a single column. Did you fail?”**

I failed to beat the majority baseline. I did not fail to evaluate. The matrix is the correct picture of an accuracy-optimal constant classifier on this label noise.

**“Why should we give marks?”**

The rubric is six stages, not the leaderboard. Data loading, image mapping, text cleaning, two branches, fusion, training curves, metrics, and a valid `submission.csv` are all there. I also compared against the only honest baseline.

**“What would you do with more time?”**

1. Collapse to 3 classes (`positive` / `neutral` / `negative`) as a diagnostic, not for the official 5-class file.
2. Keep frozen CLIP, try a linear probe with class-balanced reporting, not class-weighted loss for accuracy.
3. Error-analyse OCR vs corrected text.
4. Ask organisers whether 99.9% overlap is intended. I still would not leak labels.

**“What is CLIP cosine for?”**

If image and caption embeddings agree, the meme is “literal”. If they disagree, the joke is often ironic. Cosine is a cheap extra scalar. It did not rescue accuracy here.

**“LayerNorm, dropout, label smoothing, OneCycle — why?”**

Standard regularisation for a small head on 7k rows: LayerNorm on CLIP features, dropout 0.4, modality dropout 0.2 (randomly drop image or text), label smoothing 0.1, AdamW, OneCycle. None of this beats a bad label signal.

**“Did you use class weights?”**

No. Accuracy + rare `very_negative` (~2%) would make the model chase the tail and hurt the leaderboard. Earlier runs taught me that. I report macro-F1 but I do not train for it.

**“CLIPModel vs CLIPVisionModel?”**

Colab’s transformers 5.x: `CLIPVisionModel` no longer exposes `.vision_model` the same way. I load full `CLIPModel` and call `.vision_model` / `.visual_projection` / `.text_model` / `.text_projection`. Engineering fix, not an accuracy trick.

**“How long does it train?”**

Backbone feature extract is the slow part (once, then cached on Drive). Head training is seconds per epoch on GPU. Early stop around a handful of epochs when val loss stops falling.

---

## Architecture (draw if needed)

```
meme image ──► CLIP ViT-L/14 (frozen) ──► 768-d image vec ──► ImageBranch
meme caption ─► CLIP text (frozen)   ──► 768-d text vec  ─┐
              ─► Twitter-RoBERTa (frozen) ► hidden + 3-way probs ─┴► TextBranch
                                                              │
                         gated fusion [img, txt, img*txt, |img-txt|, mix, cosine]
                                                              │
                                                    MLP ──► 5 logits
                                                    argmax ──► overall_sentiment
```

Trainable: ImageBranch, TextBranch, gate, MLP. Not trainable: CLIP, RoBERTa.

---

## Numbers I want in my head

- 6,992 train rows, 2,098 test rows, 5 classes
- Train mix ≈ 14.8% very_positive, **44.7% positive**, 31.5% neutral, 6.9% negative, 2.2% very_negative
- Majority baseline **0.447**
- ResNet era Kaggle **0.22**
- Mixed CLIP+DistilBERT Kaggle **0.498**, val **0.412**
- Submitted OOF **0.447**, Kaggle **~0.49**, test **100% positive**
- Test captions identical to train **99.9%**
- Images resolved **100%**, non-empty captions **99.9%**
- Blend weight MLP **1.0** (LogReg discarded)

---

## If they press on the score

If the label of a meme is something even humans disagree on, a 5-class accuracy leaderboard will sit near the majority class. My job was to build the multimodal pipeline, measure it against that baseline, and not hide a constant predictor behind a more fashionable architecture. That is what this notebook does.

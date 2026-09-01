# inner_speech_deep_learning_classification
classifying directional words (right/left/up/down- in Spanish) from neural signals using deep learning

# Comparison of Inner Speech Classification with Deep Learning Methods

## Goal

My goal for this project was to compare deep learning methods (CNN-BiLSTM and EEGNet) on inner speech classification of four directional words (right/left/up/down, in Spanish). The main reason I did this project was to gain a better technical understanding of a section of BCIs that focus on decoding neural signals into words, which can aid people who do not have the ability to speak to still be able to communicate with their peers and loved ones.

## Methods

### Data

- Source: [NEMAR on003626](https://nemar.org/dataset/on003626)
- Preprocessed data downloaded following instructions from [nemarDatasets/on003626](https://github.com/nemarDatasets/on003626)
- 10 subjects (2 subjects each had one session that was corrupted after preprocessing and were excluded)
- 5,460 total trials across all three task conditions, but only 2,236 inner speech trials (~220 trials per subject) were used for this project
- 256 Hz sampling rate (confirmed after preprocessing)
- 128 EEG channels
- 2.5 seconds of the action interval kept per epoch
- Since each subject's data spans 3 recording sessions, each subject's EEG data was z-score standardized (per channel, across that subject's own trials) to account for scale differences across sessions

### Models

- **CNN-BiLSTM** (the CNN portion is similar to EEGNet)
  - Temporal and spatial convolutional layers to learn spatiotemporal patterns
  - Bidirectional LSTM layer to learn temporal dependencies across the trial
  - fed into a final dense + softmax classification layer
- **EEGNet**
  - Implemented via `braindecode.models.EEGNet`

### Evaluation

- Within-subject evaluation, 20 repeated random 80/10/10 train/val/test splits per subject (seeds 0-19), averaged
- 4-class balanced accuracy (chance level = 0.250)

## Results

Both models showed slight overfitting to their training data.

### EEGNet — Within-Subject Results

Setup: 641 timesteps per trial, 20 repeated random 80/10/10 splits per subject, 4-class balanced accuracy (chance = 0.250)

| Subject | Trials | Seeds | Mean ± Std | Min | Max |
|---|---|---|---|---|---|
| sub-01 | 200 | 20 | 0.255 ± 0.089 | 0.100 | 0.450 |
| sub-02 | 240 | 20 | 0.250 ± 0.068 | 0.125 | 0.375 |
| sub-03 | 180 | 20 | 0.251 ± 0.119 | 0.062 | 0.537 |
| sub-04 | 240 | 20 | 0.219 ± 0.083 | 0.042 | 0.333 |
| sub-05 | 240 | 20 | 0.312 ± 0.094 | 0.125 | 0.500 |
| sub-06 | 216 | 20 | 0.281 ± 0.096 | 0.125 | 0.500 |
| sub-07 | 240 | 20 | 0.279 ± 0.076 | 0.083 | 0.417 |
| sub-08 | 200 | 20 | 0.253 ± 0.112 | 0.100 | 0.450 |
| sub-09 | 160 | 20 | 0.247 ± 0.087 | 0.062 | 0.438 |
| sub-10 | 160 | 20 | 0.275 ± 0.105 | 0.062 | 0.438 |

**Aggregate**

| Aggregation method | Balanced accuracy |
|---|---|
| Mean of per-subject means | 0.262 ± 0.024 |
| Pooled across all 200 subject-seed runs | 0.262 ± 0.097 |

### CNN-LSTM — Within-Subject Results

Setup: 641 timesteps per trial, 20 repeated random 80/10/10 splits per subject, 4-class balanced accuracy (chance = 0.250)

| Subject | Trials | Seeds | Mean ± Std | Min | Max |
|---|---|---|---|---|---|
| sub-01 | 200 | 20 | 0.283 ± 0.083 | 0.150 | 0.500 |
| sub-02 | 240 | 20 | 0.308 ± 0.084 | 0.167 | 0.500 |
| sub-03 | 180 | 20 | 0.251 ± 0.095 | 0.000 | 0.425 |
| sub-04 | 240 | 20 | 0.229 ± 0.079 | 0.125 | 0.417 |
| sub-05 | 240 | 20 | 0.219 ± 0.076 | 0.125 | 0.375 |
| sub-06 | 216 | 20 | 0.290 ± 0.065 | 0.150 | 0.400 |
| sub-07 | 240 | 20 | 0.233 ± 0.078 | 0.083 | 0.375 |
| sub-08 | 200 | 20 | 0.283 ± 0.104 | 0.150 | 0.500 |
| sub-09 | 160 | 20 | 0.291 ± 0.072 | 0.188 | 0.438 |
| sub-10 | 160 | 20 | 0.259 ± 0.101 | 0.062 | 0.438 |

**Aggregate**

| Aggregation method | Balanced accuracy |
|---|---|
| Mean of per-subject means | 0.265 ± 0.029 |
| Pooled across all 200 subject-seed runs | 0.265 ± 0.089 |

### Spatial filter interpretability

To check what each model actually learned, the spatial filter weights of a representative model (sub-01, seed 0) were plotted back onto the scalp.

<p float="left">
  <img src="images/cnn_lstm_topomap_sub01.png" width="45%" alt="CNN-LSTM mean absolute spatial weight across all filters for sub-01, seed 0, showing a peak near left temporal electrodes" />
  <img src="images/eegnet_topomap_sub01.png" width="45%" alt="EEGNet mean absolute spatial weight across all filters for sub-01, seed 0, showing weight spread fairly evenly across the scalp" />
</p>

*Left: CNN-LSTM. Right: EEGNet. Both show the mean absolute spatial filter weight per electrode, averaged across all filters, viewed from above with the nose pointing up.*

## Discussion

- Given that this project only used inner speech data (~220 trials per subject), the reduced amount of data available for the models to learn from is likely the main reason behind the overfitting, since both EEGNet and CNN-BiLSTM showed similar overfitting and end results
- Overall, these models achieved close-to-chance accuracy, which most likely means the models were probably guessing randomly (1/4 chance of getting it right)
- Deep learning models need a lot of annotated data to learn patterns properly, so given this dataset's size, building a model to learn features important for imagined speech of directional commands may not have been the best-suited approach
- However, the CNN-LSTM's spatial filters did concentrate weight on a specific region, most strongly on left temporal electrodes near the ear, suggesting the model settled on a consistent electrode subset rather than spreading attention randomly
- EEGNet's spatial weights were far more uniformly distributed across the scalp, with no single electrode region standing out to the same degree, so the two architectures do not appear to converge on the same features
- The left temporal region CNN-LSTM emphasized sits close to areas classically tied to language processing (Wernicke's area), though it also sits close to jaw muscles, so residual subvocal movement is also a possible explanation. 

## Future Direction

- As a future step, I want to  try a Random Forest classifier restricted to features from specific, hypothesis-driven channel regions instead of the full 128-channels
- Spatial thinking is primarily associated with occipital and parietal regions, since the four classes here are spatial direction words, so band power, ERP, and nonlinear features computed only on occipital-parietal channels could carry more class-relevant signal per feature than the whole-scalp approach used so far
- Separately, general inner speech processing (auditory, semantic, and motor planning and articulation) recruits areas closer to premotor cortex and Broca's and Wernicke's areas, so features from fronto-temporal channels could still be worth including, though these regions will probably give information on whether inner speech is happening at all rather than which direction was chosen
- Narrowing features to these regions manually could reduce the feature-to-trial ratio problem that likely contributed to overfitting in the deep models, given the limited ~220 trials per subject

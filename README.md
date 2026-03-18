# neuro-simulation-power
A simulation-based framework for power analysis in neuroimaging studies using realistic signal and noise models derived from pilot data.

The current implementation uses pilot-derived estimates of signal amplitude, spatial smoothness, and noise variance to generate realistic subject-level data. Simulated datasets are analyzed using the same statistical models and multiple-comparison correction procedures planned for the final study (e.g., FDR/FWE correction), enabling study-specific and model-consistent power estimation for complex neuroimaging designs.

This repository is being shared to support transparency and reproducibility for ongoing grant applications and associated work.


## Data Availability

Due to the use of sensitive neuroimaging data, raw datasets are not included in this repository. The simulation framework can be run using user-supplied data or synthetic inputs.

# Feed-Forward Predictor Evaluation

Rows: 4

| method | rows | mean PSNR | mean MS-SSIM | mean AUC | mean total s | target hits | mean target s |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned | 1 | 21.0514 | 0.85978 | 20.4061 | 0.949523 | 0 | 0.000000 |
| learned_tensor | 1 | 23.4686 | 0.90109 | 21.7372 | 0.333759 | 1 | 0.128815 |
| scratch | 1 | 22.9056 | 0.90148 | 21.6377 | 0.308444 | 1 | 0.166847 |
| tensor_prior | 1 | 25.3249 | 0.91512 | 24.1797 | 0.448505 | 1 | 0.174787 |

Target PSNR: 22.0

Verdict: equal-N evaluator completed. Interpret quality/speed only at the scope of the image split and checkpoint used for this run.

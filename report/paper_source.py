"""CVPR-format final paper; all numerical tables come from saved analysis."""
import numpy as np

NAMES={'clean':'Clean-only','pooled':'Pooled','uniform':'Uniform + guard','adaptive':'Adaptive + guard','no_guard':'Without guard','no_anchor':'Without anchors'}
def tex(text):
    return str(text).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
def pct(x):return f'{100*float(x):.2f}'

def make_paper(config,summary,category,seeds,decisions,efficiency,primary,abstract,audit):
    def row(m,scope='corrupted',b=4096):return next(r for r in summary if r['method']==m and r['scope']==scope and int(r['budget'])==b)
    def table_rows(methods,scope='corrupted'):
        return '\n'.join(' & '.join([NAMES[m]]+[pct(row(m,scope)[k]) for k in ['image_auroc','pixel_auroc','pixel_ap','fpr','recall']])+r' \\' for m in methods)
    delta=primary['delta']*100;lo,hi=np.array(primary['ci95'])*100
    abstract_tex=tex(abstract).replace(f'[{lo:+.2f}, {hi:+.2f}]',r'$['+f'{lo:+.2f}'+r',\,'+f'{hi:+.2f}'+r']$')
    verdict='met' if primary['success_criterion_met'] else 'not met'
    best=max(config['methods'][:4],key=lambda m:float(row(m)['image_auroc']))
    bottle_fallbacks=[sum(d['fallback']=='True' for d in decisions if int(d['budget'])==b and d['category']=='bottle' and d['method']=='adaptive') for b in sorted(config['budgets'])]
    bottle_note='For bottle, these counts are '+ '/'.join(map(str,bottle_fallbacks))+' out of three runs. '
    if bottle_fallbacks[-1]==3:
        bottle_note+='At the largest budget, every adaptive bank equals the clean-only bank. '
    bottle_note+='The safeguard uses a budget-specific clean reference, so extra memory can change acceptance as well as coverage.'
    source=r'''% Final course-project paper. Supplied CVPR 2026 style, not a conference acceptance claim.
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage[pagenumbers]{cvpr}
\usepackage{microtype}
\usepackage{array}
\usepackage{tabularx}
\usepackage{balance}
\definecolor{cvprblue}{rgb}{0.21,0.49,0.74}
\usepackage[breaklinks,colorlinks,allcolors=cvprblue]{hyperref}
\graphicspath{{../report/figures/}}
\hypersetup{pdftitle={Clean-Anchored PatchCore: A Controlled Fixed-Budget Robustness Study},pdfauthor={Sarthak Shukla and Yash Doshi}}
\title{Clean-Anchored PatchCore: A Controlled Fixed-Budget\\Robustness Study for Industrial Defect Detection}
\author{Sarthak Shukla \qquad Yash Doshi\\
{\small S20230020348 \qquad S20230020360}\\
Indian Institute of Information Technology Sri City\\
{\small Deep Learning and Computer Vision --- Group 23 Final Project}}
\begin{document}
\maketitle
\begin{abstract}
@ABSTRACT@
\end{abstract}

\section{Introduction}
Visual anomaly detection is useful when normal products are readily available but the space of manufacturing defects is difficult to enumerate. A detector must reject genuinely defective products while remaining stable under harmless changes in acquisition. A darkened normal bottle and a damaged bottle may both be distant from a clean feature memory, yet they require different decisions. This project studies memory construction under a fixed descriptor budget, using industrial images with image-level labels and pixel masks from MVTec AD~\cite{bergmann2019}.

PatchCore~\cite{roth2022} uses locally aggregated pretrained patch features and a representative coreset of normal descriptors. Storing more transformed normal features can improve coverage but consumes memory or displaces clean representatives. Our research question is whether explicitly preserving clean anchors and allocating remaining capacity to the weakest-covered acquisition group improves robustness over ordinary pooled augmentation.

We implement a normal-only clean-coverage safeguard: an augmented bank is accepted only if its clean validation coverage remains close to the clean-only reference. This protects a measurable proxy, not defect recall itself. Augmented descriptors may also move defective patches closer to the normal memory, so detection performance requires an independent test.

The completed study compares six variants on bottle, hazelnut and tile, with three seeds and three budgets. It includes shared score reweighting, held-out calibration, two ablations, corruption-specific results and reproducibility artifacts. The study contributes a transparent implementation of this combination and a controlled assessment within the PatchCore framework. The conclusions follow the measured comparison against pooled augmentation.

\section{Related Work}
\paragraph{Feature-memory anomaly detection.}
PatchCore combines locally aware patch descriptors with coreset sampling and nearest-neighbor scoring~\cite{roth2022}. We retain the pretrained feature-memory framework and modify the composition of the bank. The feature construction is informed by the \href{https://github.com/amazon-science/patchcore-inspection}{Amazon reference implementation}, while the code is independently implemented. Candidate subsampling and exact greedy selection mean our experiments are a controlled adaptation, not reproduction of the published full-benchmark scores.

\paragraph{Augmentation and feature adaptation.}
Santos et al.~\cite{santos2023} investigate PatchCore settings and show that augmentation can help but is not uniformly beneficial. This motivates the pooled-augmentation control: improvement over a clean-only bank alone cannot demonstrate a benefit from adaptive allocation. ReConPatch~\cite{hyun2024} learns adapted patch representations. Our backbone remains frozen, isolating changes in memory selection rather than representation learning.

\paragraph{Robustness and memory design.}
Robust AD~\cite{pemula2025} motivates evaluation under acquisition variation. Our synthetic operators are deliberately narrower than a real camera-domain benchmark. Sequential PatchCore~\cite{mao2025} is another precedent for modifying normal-memory construction. The present study targets allocation among acquisition groups under the same category-specific memory budget.

\section{Method}
\subsection{Shared representation}
Images are resized to $256\times256$ with bilinear interpolation and normalized using ImageNet statistics. A frozen TorchVision WideResNet-50-2 with ImageNet V1 weights yields layer2 and layer3 features. We unfold $3\times3$ neighborhoods, align the deeper grid to layer2 using bilinear interpolation, pool each layer to 1,024 values, and pool their concatenation to 1,024 dimensions. Each image supplies a $32\times32$ grid. Features, float32 precision and scoring are shared across methods; no backbone training or fine-tuning is performed.

\subsection{Candidate groups and clean anchors}
Normal fitting images generate groups $P_0$ (clean), $P_1$ (brightness factors 0.85 and 1.15) and $P_2$ (JPEG quality 85). After seeded sampling of at most 20,000 descriptors per group, exact duplicates are assigned to their first occurrence in group order. All corresponding methods share these candidates.

For bank $M$, define nearest distance and coverage by
\begin{equation}
 \begin{aligned}
 d(z,M)&=\min_{m\in M}\|z-m\|_2,\\
 R_g(M)&=\frac{1}{|P_g|}\sum_{z\in P_g}d(z,M)^2.
 \end{aligned}
\end{equation}
The clean baseline uses exact greedy farthest-first selection initialized at the clean candidate closest to the group mean. Anchored methods protect the first $\lfloor\rho B\rfloor$ vectors of that ordered clean bank for $\rho\in\{0.50,0.75\}$.

\subsection{Allocation and validation}
Adaptive allocation chooses the augmented group with the largest $R_g(M)$, then adds its farthest unselected candidate. Nearest distances are updated incrementally until the budget is filled. Fixed ordering breaks ties and exhausted groups are skipped. Uniform allocation instead alternates augmentation groups, retaining the same anchors and validation rule.

A candidate bank passes the clean safeguard if
\begin{equation}
 R_0^{\mathrm{val}}(M)\leq1.10\max\{R_0^{\mathrm{val}}(M_{\mathrm{clean}}),10^{-12}\}.
\end{equation}
Among passing anchor fractions, select the lowest worst augmented-group validation error, breaking ties toward more anchors. If neither passes, retain the clean-only bank. Validation uses all normal validation descriptors, not anomalous images or test outcomes.

\subsection{Controls and ablations}
The six variants are: (1) clean-only selection; (2) ordinary greedy selection from the pooled candidate groups; (3) uniform augmented-group allocation with anchors and guard; (4) adaptive allocation with anchors and guard; (5) adaptive allocation without guard, retaining the anchor search and normal-validation choice; and (6) adaptive allocation without protected anchors, retaining the guard. The last variant starts with the candidate nearest to the pooled mean and allocates over all three groups. If rejected, it falls back to the clean bank, and that fallback is reported.

\subsection{Image score and localization}
Let $q^*$ be the test patch with largest nearest-bank distance, $m^*$ its nearest bank vector, and $s^*=\|q^*-m^*\|_2$. Following PatchCore Eq.~(7)~\cite{roth2022}, the image score is
\begin{equation}
 s=\left(1-\frac{\exp(\|q^*-m^*\|_2)}{\sum_{m\in\mathcal{N}_9(m^*)}\exp(\|q^*-m\|_2)}\right)s^*.
\end{equation}
The nine bank neighbors include $m^*$, and stable softmax avoids overflow. Patch maps retain unreweighted distances, are bilinearly upsampled and smoothed with Gaussian $\sigma=4$ pixels. The earlier midterm pilot omitted image-score reweighting; its measurements are not combined with these final results.

\section{Experimental Protocol}
\subsection{Data separation and fixed settings}
The study uses bottle, hazelnut and tile from MVTec AD~\cite{bergmann2019}. Each official normal training set is shuffled with seeds 23, 42 and 61, then divided into fitting (floor of 60\%), validation (floor of 20\%) and calibration (remainder). All transformed views inherit their source split. Test images and masks are evaluated only after fitting, selection and threshold calibration. Table~\ref{tab:splits} gives counts, which are seed-invariant.
\begin{table}[t]
\centering\small
\caption{Disjoint data partitions. Test columns are normal/defective.}
\label{tab:splits}
\begin{tabular}{lrrrr}\toprule
Category & Fit & Val. & Cal. & Test\\\midrule
Bottle & 125 & 41 & 43 & 20/63\\
Hazelnut & 234 & 78 & 79 & 40/70\\
Tile & 138 & 46 & 46 & 33/84\\\bottomrule
\end{tabular}
\end{table}

We use budgets $B\in\{1024,4096,8192\}$, giving 27 category/seed/budget configurations and six variants per configuration. Configuration and implementation hashes are locked before final test evaluation. The seed-23 bottle midterm had already been observed. The proposal fixed the category, budget and corruption design, while the exact final seeds and nine-neighbor support convention are specified in the final configuration. This is not claimed as a preregistered study.

\subsection{Stress conditions and calibration}
Clean images and nine corruptions are evaluated: brightness factors 0.75/0.60/0.45, JPEG qualities 70/50/30 and Gaussian blur radii 0.5/1.0/1.5. Transformations follow resizing and apply to normal and anomalous images alike. JPEG uses Pillow 4:2:0 subsampling; blur uses Pillow's Gaussian approximation. Masks are resized with nearest-neighbor interpolation. Blur is absent from fitting and selection, testing transfer to an unseen corruption family.

Each bank has a separate threshold from disjoint clean-normal calibration scores. For $n$ values, use rank $\lceil0.95(n+1)\rceil$; if unavailable, use infinity. A score strictly greater than the threshold is anomalous. Thresholds remain fixed across corruption conditions, so the nominal 5\% target does not guarantee 5\% false positives under shift.

\subsection{Metrics and paired inference}
We report image AUROC, pixel AUROC, pixel average precision (AP), false-positive rate (FPR) and anomalous-image recall. Pixel metrics pool all $256\times256$ pixels in a category/seed/condition. Tables equally average metrics across seeds, categories and, for corrupted means, nine conditions.

The primary comparison is adaptive minus pooled mean corrupted image AUROC at $B=4096$. Following the proposal, image scores are first averaged over seeds. The paired bootstrap resamples original images within category and label, retaining all nine views and both methods together. We use 2,000 resamples and seed 20261112. This estimates conditional test-image uncertainty for these categories and fitted seeds, not uncertainty over new categories or refitted models. Success requires a positive 95\% interval and at most one percentage point loss in clean macro/seed-mean AUROC against clean-only.

\section{Results}
\subsection{Primary comparison}
Table~\ref{tab:main} reports the four main methods. Adaptive selection obtains @ADAPTIVE@\% mean corrupted image AUROC versus @POOLED@\% for pooled selection. @BEST@ has the highest corrupted mean among the main methods. The primary seed-averaged-score difference is @DELTA@ percentage points, with 95\% interval [@LOW@, @HIGH@]. The clean AUROC difference against clean-only is @CLEANDELTA@ points. Therefore the prespecified success criterion is \textbf{@VERDICT@}. Table~\ref{tab:main} averages per-seed AUROCs, giving a difference of @SEEDDELTA@ points. The primary test instead averages image scores before computing AUROC. Because AUROC is nonlinear, the aggregation orders can produce different differences, including different signs.

\begin{table*}[t]
\centering\small
\caption{Primary-budget results ($B=4096$). Percentages; equally averaged over categories and seeds. Corrupted rows additionally average nine conditions. FPR is lower-is-better; other metrics are higher-is-better.}
\label{tab:main}
\begin{tabular}{llrrrrr}\toprule
Scope & Method & Image AUROC & Pixel AUROC & Pixel AP & FPR & Recall\\\midrule
@CLEANROWS@
\midrule
@CORRUPTROWS@
\bottomrule\end{tabular}
\end{table*}

\subsection{Variation across categories and seeds}
Table~\ref{tab:categories} exposes category outcomes instead of only an aggregate. Texture and object categories can react differently to the same operator. Three seed results are also retained in the analysis files. Because they share official test images, these seeds are not independent replicas of test sampling.
\begin{table}[t]
\centering\small
\caption{Mean corrupted image AUROC at $B=4096$, averaging per-seed AUROCs (\%).}
\label{tab:categories}
\begin{tabular}{lrrrr}\toprule
Category & Clean & Pooled & Uniform & Adaptive\\\midrule
@CATEGORYROWS@
\bottomrule\end{tabular}
\end{table}

\subsection{Budget and component ablations}
Figure~\ref{fig:budgets} and Table~\ref{tab:budget} compare matched storage. The stored descriptor payload is exactly $4Bd$ bytes, or 4/16/32 MiB at $d=1024$. Larger memory can improve coverage but also changes search cost and threshold behavior. These secondary comparisons are descriptive rather than multiplicity-corrected significance tests.
\begin{table}[t]
\centering\small
\caption{Mean corrupted image AUROC (\%) by memory budget.}
\label{tab:budget}
\begin{tabular}{rrrrr}\toprule
Vectors & Clean & Pooled & Uniform & Adaptive\\\midrule
@BUDGETROWS@
\bottomrule\end{tabular}
\end{table}

\begin{table}[t]
\centering\small
\caption{Component ablations at $B=4096$ (\%). Clean/corrupted image AUROC; AP and FPR are corrupted means.}
\label{tab:ablations}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}\toprule
Variant & Clean AUC & Corr. AUC & AP & FPR\\\midrule
@ABLATIONROWS@
\bottomrule\end{tabular}
\end{table}

Table~\ref{tab:ablations} separates removal of the safeguard and protected anchors. The without-guard method retains normal-validation selection, so it is not an unvalidated or test-tuned model. The without-anchors method retains the guard, including clean fallback. At the primary budget the adaptive method falls back in @FALLBACK@ of nine category/seed runs. Across budgets 1,024/4,096/8,192, adaptive fallback counts are @BUDGETFALLBACKS@ out of nine runs each. Thus the budget comparisons also reflect whether the safeguard accepts an augmented bank. Fallback and allocation records are retained for every run. Normal coverage is a proxy: improving it does not logically imply better separation of defects from benign appearance changes.

@BOTTLEOBSERVATION@

The allocation objective uses unnormalized squared distances. A group with greater intrinsic feature variability can therefore attract most remaining memory, even if covering that variability does not improve defect ranking. This is a plausible failure mechanism, not a causal finding established by these experiments. Group-normalized coverage and separately validated calibration under acquisition shift are useful directions beyond the completed protocol.

\begin{figure*}[t]
\centering\includegraphics[width=0.96\textwidth]{budget_tradeoff.png}
\caption{Memory-budget trade-off. AUROC and false positives are averaged over the same three categories, three seeds and nine corruption conditions. Each method has identical bank payload at a given budget.}
\label{fig:budgets}
\end{figure*}

\subsection{Corruption behavior and qualitative evidence}
Figure~\ref{fig:conditions} shows the primary-budget condition profile. Adaptive FPR averages @ADAFPR@\%, versus @POOLFPR@\% for pooled selection, while recall is @ADARECALL@\% versus @POOLRECALL@\%. Ranking quality and threshold decisions are complementary: a bank can rank anomalies above normal images while shifting both score distributions past a clean-derived threshold. Thus a high AUROC is insufficient evidence of stable deployment decisions. Per-condition normal counts are limited, and severe corruption may remove visual defect evidence.

\begin{figure*}[t]
\centering\includegraphics[width=0.96\textwidth]{conditions.png}
\caption{Condition-wise primary-budget results. B denotes brightness, J JPEG quality and G blur radius. Blur is unseen during fitting/selection. All views of a source image remain paired.}
\label{fig:conditions}
\end{figure*}

Figure~\ref{fig:qualitative} uses filename-selected examples, not cases chosen for favorable method scores. The first defective image and first normal image in each category are shown; the normal image is darkened by 0.45. Shared row scales allow comparison of clean-only and adaptive distances. Both banks respond strongly near the displayed defects. Adaptive selection reduces the background response for the displayed darkened normal bottle, while normal hazelnut and tile retain substantial spatial responses. These sample observations do not establish their prevalence. A machine-readable failure table lists every false positive and false negative at the frozen thresholds.

\begin{figure*}[t]
\centering\includegraphics[width=0.80\textwidth]{qualitative.png}
\caption{Predetermined category examples, $B=4096$, seed 23. Masks show defects. Heatmaps use common scales within each row; color is not a pixel decision threshold. Original images: MVTec Software GmbH, CC BY-NC-SA 4.0, with resizing, corruption and overlays.}
\label{fig:qualitative}
\end{figure*}

\section{Efficiency and Validation}
Experiments ran on an NVIDIA GeForce RTX 2050 with 4 GB memory. TF32 was disabled and deterministic cuDNN behavior requested. Condition-level checkpoints and shared feature caches support resumable execution. Construction timings exclude feature extraction; anchored methods include both candidate fractions and validation while reusing clean anchors. The no-guard variant shares the adaptive candidate search cost.

Batch-one inference includes disk read, preprocessing, feature extraction, score/map generation and smoothing. Each run uses five warmups and ten synchronized measurements on a calibration image. At $B=4096$, adaptive inference has a median of @LATENCY@ ms across the nine run medians. These laptop measurements are indicative, not a dedicated throughput benchmark. Peak CUDA allocated memory is recorded but is not total process or reserved memory.

Eight protocol tests check direct distances, greedy uniqueness/coverage, anchors and allocation, fallback/tie rules, calibration, Eq.~(7) scoring, unanchored behavior, and exact tied-score pixel metrics. The final audit verifies @AUDITROWS@ image metric rows against raw predictions, all split and bank invariants, and all stored patch-grid shapes/finite values. An independent scikit-learn check reconstructs @PIXELROWS@ pixel metric rows for all categories and methods at $B=4096$, seed 23, clean and strongest brightness. This is a sampled pixel audit, not a complete independent rerun of feature extraction.

\section{Limitations and Conclusion}
The study covers three chosen categories, three fitting seeds, one backbone/resolution and synthetic corruptions. It does not establish full-benchmark performance or readiness for factory deployment. The normal-only safeguard cannot guarantee anomaly recall, while clean calibration cannot guarantee controlled false alarms under shift. The bootstrap is conditional on the observed categories and fitted models. The earlier pilot was already known, and no claim of prospective preregistration is made.

The final project implements and evaluates clean-anchor preservation, adaptive group allocation and normal-only validation at matched memory budgets. The primary difference is @DELTA@ points with interval [@LOW@, @HIGH@]; the success criterion is @VERDICT@. This evidence, rather than the initial expectation of improvement, determines the conclusion. Broader real-acquisition tests and separately collected normal calibration data are appropriate next steps.

\paragraph{Artifacts and attribution.}
Code, configurations, provenance, splits, calibration, per-image predictions, metrics and editable reports are provided in the \href{https://github.com/sarthak06shukla/clean-anchored-patchcore}{project repository}. Three reference adaptive banks ($B=4096$, seed 23) support direct prediction. All other banks and patch grids are retained locally and regenerable; raw datasets, pretrained weights and caches are excluded from Git history. MVTec images and derived visualizations retain CC BY-NC-SA 4.0 attribution. The pretrained backbone is used without training. This course-project study uses the supplied CVPR document style.

\clearpage
\balance
{\small\interlinepenalty=10000\bibliographystyle{ieeenat_fullname}\bibliography{references}}
\end{document}
'''
    replacements={'ABSTRACT':abstract_tex,'ADAPTIVE':pct(row('adaptive')['image_auroc']),'POOLED':pct(row('pooled')['image_auroc']),
      'BEST':NAMES[best],'DELTA':f'{delta:+.2f}','LOW':f'{lo:+.2f}','HIGH':f'{hi:+.2f}',
      'SEEDDELTA':f'{100*(float(row("adaptive")["image_auroc"])-float(row("pooled")["image_auroc"])):+.2f}',
      'ADAFPR':pct(row('adaptive')['fpr']),'POOLFPR':pct(row('pooled')['fpr']),
      'ADARECALL':pct(row('adaptive')['recall']),'POOLRECALL':pct(row('pooled')['recall']),
      'CLEANDELTA':f'{100*primary["clean_macro_seed_mean_delta_vs_clean"]:+.2f}','VERDICT':verdict,
      'CLEANROWS':'\n'.join('Clean & '+s for s in table_rows(config['methods'][:4],'clean').splitlines()),
      'CORRUPTROWS':'\n'.join('Corrupted & '+s for s in table_rows(config['methods'][:4]).splitlines()),
      'CATEGORYROWS':'\n'.join(' & '.join([cat]+[pct(next(r['image_auroc'] for r in category if int(r['budget'])==4096 and r['category']==cat and r['method']==m and r['scope']=='corrupted')) for m in config['methods'][:4]])+r' \\' for cat in config['categories']),
      'BUDGETROWS':'\n'.join(' & '.join([str(b)]+[pct(row(m,b=b)['image_auroc']) for m in config['methods'][:4]])+r' \\' for b in sorted(config['budgets'])),
      'ABLATIONROWS':'\n'.join(' & '.join([NAMES[m],pct(row(m,'clean')['image_auroc'])]+[pct(row(m)[k]) for k in ['image_auroc','pixel_ap','fpr']])+r' \\' for m in ['adaptive','no_guard','no_anchor']),
      'FALLBACK':str(sum(d['fallback']=='True' for d in decisions if int(d['budget'])==4096 and d['method']=='adaptive')),
      'BUDGETFALLBACKS':'/'.join(str(sum(d['fallback']=='True' for d in decisions if int(d['budget'])==b and d['method']=='adaptive')) for b in sorted(config['budgets'])),
      'BOTTLEOBSERVATION':bottle_note,
      'LATENCY':f'{np.median([float(r["latency_ms"]) for r in efficiency if int(r["budget"])==4096 and r["method"]=="adaptive"]):.1f}',
      'AUDITROWS':f'{audit["image_metric_rows_checked"]:,}','PIXELROWS':str(audit['pixel_metric_rows_independently_checked'])}
    for k,v in replacements.items():source=source.replace('@'+k+'@',v)
    return source

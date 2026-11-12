"""Generate final report and CVPR paper from audited, completed study artifacts."""
import csv
import argparse
import json
from pathlib import Path
import shutil
from xml.sax.saxutils import escape
import numpy as np
import build_report as layout

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'results/analysis'
NAMES={'clean':'Clean-only','pooled':'Pooled','uniform':'Uniform + guard','adaptive':'Adaptive + guard','no_guard':'Without guard','no_anchor':'Without anchors'}

def read_csv(name):return list(csv.DictReader((A/name).open()))
def pct(x):return f'{float(x)*100:.2f}'
def main():
    global A
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=ROOT/'results')
    args=parser.parse_args();results=args.results;A=results/'analysis'
    audit=json.loads((results/'audit.json').read_text());assert audit['passed'] and audit['experiments']==27
    config=json.loads((ROOT/'configs/study.json').read_text())
    summary=read_csv('summary.csv');category=read_csv('by_category.csv');seeds=read_csv('by_seed.csv')
    decisions=read_csv('decisions.csv');efficiency=read_csv('efficiency.csv');all_metrics=read_csv('all_metrics.csv')
    family_results=read_csv('by_corruption_family.csv');worst_results=read_csv('worst_condition.csv')
    primary=json.loads((A/'primary_test.json').read_text())
    def row(m,scope='corrupted',budget=4096):return next(r for r in summary if r['method']==m and r['scope']==scope and int(r['budget'])==budget)
    delta=primary['delta']*100;lo,hi=np.array(primary['ci95'])*100
    verdict='The prespecified success criterion was met.' if primary['success_criterion_met'] else 'The prespecified success criterion was not met.'
    best=max(config['methods'][:4],key=lambda m:float(row(m)['image_auroc']))
    abstract=(f'This project evaluates a fixed-budget normal-feature memory for industrial anomaly detection. Clean-Anchored PatchCore reserves clean anchors, '
        'allocates remaining capacity to the least-covered acquisition group, and applies a normal-only clean-coverage safeguard. '
        'We implement six controlled variants and evaluate bottle, hazelnut and tile from MVTec AD under clean images and nine acquisition corruptions. '
        'The completed study covers three seeds and memory budgets of 1,024, 4,096 and 8,192 vectors: 27 fitted experiment configurations and 1,620 method-condition evaluations. '
        f'At the primary 4,096-vector budget, adaptive allocation achieves {pct(row("adaptive")["image_auroc"])}% mean corrupted image AUROC '
        f'and {pct(row("adaptive")["fpr"])}% false-positive rate at frozen clean-calibration thresholds. '
        f'The prespecified paired comparison, using seed-averaged image scores, gives an adaptive-minus-pooled difference of {delta:+.2f} percentage points '
        f'with a 95% interval [{lo:+.2f}, {hi:+.2f}]. {verdict} '
        'Results, ablations, predictions, fitted banks and executable code are retained, including negative findings.')
    # Institutional report: retain midterm typography, page size, cover and table style.
    p,heading,sub,small,page,table,figure=layout.p,layout.heading,layout.sub,layout.small,layout.page,layout.table,layout.figure
    width=layout.A4[0]-104
    layout.story.append(layout.Spacer(1,36))
    p('Indian Institute of Information Technology Sri City','CenterM');p('Deep Learning and Computer Vision','CenterM')
    layout.story.append(layout.Spacer(1,47));p('Clean-Anchored<br/>PatchCore','TitleM')
    p('Fixed-Budget Memory Selection for<br/>Robust Industrial Defect Detection','CenterM')
    layout.story.append(layout.Spacer(1,31));p('<b>Final Project Report</b>','CenterM');p('Group 23','CenterM')
    layout.story.append(layout.Spacer(1,34));table(['Submitted by','Roll number'],[['Sarthak Shukla','S20230020348'],['Yash Doshi','S20230020360']],[width*.55,width*.45])
    p('November 2026','CenterM')
    page();heading('1  Abstract and completed scope');p(abstract)
    sub('1.1  Research question')
    p('At the same normal-feature memory budget, does preserving clean representatives and directing remaining capacity toward the weakest-covered acquisition group '
      'improve defect detection beyond ordinary pooled augmentation? The project separates augmentation gains from allocation gains through matched controls. '
      'The project contributes an implemented memory-selection heuristic and a controlled assessment within the established PatchCore framework.')
    sub('1.2  Final deliverables')
    table(['Deliverable','Completed scope'],[
      ['Implementation','Frozen WideResNet features, six memory selectors, image-score reweighting, maps and single-image prediction CLI.'],
      ['Experiments','Three categories x three seeds x three budgets; six methods and ten conditions for each.'],
      ['Analysis','Image/pixel AUROC, pixel AP, false positives, recall, paired bootstrap, allocation/fallback and resource measurements.'],
      ['Reproducibility','Locked protocol, filename splits, data checksums, code hashes, calibration scores, predictions, banks and patch grids.'],
      ['Documentation','This institutional report, a CVPR-format final paper, editable sources and GitHub-ready repository.']],[112,width-112])
    sub('1.3  Progress from the midterm')
    p('The midterm was a bottle-only, seed-23, 1,024-vector pilot with unreweighted image scores. The final study adds hazelnut and tile, seeds 42 and 61, '
      'the larger budgets, score reweighting and both component ablations. The pilot remains separately preserved and is not pooled with the final measurements. '
      'The final settings were locked before new test evaluations; the earlier bottle pilot had already been observed. This is a transparent continuation of a course project, not a preregistered independent replication.')
    page();heading('2  Background and problem formulation')
    p('Normal-only visual anomaly detection models defect-free training images because enumerating all manufacturing defects is impractical. '
      'PatchCore [1] represents images with locally aggregated deep patch descriptors and measures distance to a representative normal-feature memory. '
      'MVTec AD [2] supplies normal training images and labeled test images with defect masks. Our study addresses both image-level rejection and pixel-level localization.')
    sub('2.1  Acquisition changes and fixed memory')
    p('Dim lighting, compression and blur can alter normal descriptors without creating defects. Adding augmented normal views increases coverage, but a fixed memory bank '
      'forces competition between clean and transformed representatives. The question is therefore how to allocate a fixed number of descriptors. '
      'The clean-only baseline isolates the effect of adding augmentation; pooled selection is the main comparator; uniform allocation isolates the adaptive group-choice rule.')
    sub('2.2  Related research and attribution')
    p('Roth et al. introduced the PatchCore feature-memory and coreset framework [1]. The Amazon reference design [3] informs feature alignment and pooling; '
      'this project is an independent educational implementation and does not vendor upstream source. '
      'Optimizing PatchCore [6] studies model settings and augmentation; ReConPatch [7] adapts patch representations. '
      'Our backbone remains frozen, so changes arise from memory construction rather than representation training. '
      'Robust AD [8] motivates studying acquisition variation, but synthetic brightness/JPEG/blur tests do not reproduce its real acquisition-domain benchmark.')
    sub('2.3  Formal objective')
    p('For candidate group P_g and bank M, define d(z,M) as the minimum Euclidean distance from descriptor z to a bank vector. '
      'Group coverage R_g is the mean squared nearest-bank distance over that group. Adaptive allocation greedily prioritizes the augmented group with greatest R_g. '
      'This is a heuristic normal-coverage objective, not a proof of optimal coverage or defect sensitivity.')
    p('d(z,M) = min over m in M of ||z - m||_2<br/>R_g(M) = mean over z in P_g of d(z,M)^2','CodeM')
    sub('2.4  Hypothesis and success criterion')
    p('The primary comparison is mean corrupted image AUROC at B = 4,096 for adaptive versus pooled selection. Success requires a positive paired difference whose '
      '95% bootstrap interval excludes zero, together with no more than one percentage point loss in clean image AUROC versus clean-only selection. '
      'The other budgets, ablations and metrics provide secondary evidence; their many comparisons are descriptive rather than separately multiplicity-corrected claims.')
    page();heading('3  Method and implemented system')
    sub('3.1  Shared feature extraction')
    p('RGB images are resized to 256 x 256 using bilinear interpolation and normalized with ImageNet statistics. A frozen TorchVision WideResNet-50-2 [4] '
      'with IMAGENET1K_V1 weights provides layer2 and layer3 activations. A 3 x 3 neighborhood is unfolded at each location; the deeper grid is bilinearly aligned '
      'to layer2. Adaptive average pooling produces a 1,024-dimensional descriptor at each of 32 x 32 spatial locations. All methods use identical features, precision and preprocessing.')
    sub('3.2  Candidate groups and selection')
    p('The fitting groups contain clean images, brightness factors 0.85/1.15, and JPEG quality 85. Seed-controlled sampling caps each group at 20,000 descriptors '
      'before exact duplicate removal. Duplicates are assigned to their first group in clean/brightness/JPEG order. Clean and pooled banks use exact farthest-first '
      'greedy selection initialized at the candidate closest to the relevant mean; no random projection is used.')
    p('Uniform and adaptive banks reserve floor(rho B) vectors from the clean bank greedy prefix, for rho in {0.50, 0.75}. '
      'Uniform selection alternates augmentation groups; adaptive selection picks the largest mean squared coverage error, then that group\'s farthest unselected descriptor. '
      'Fixed ordering resolves ties and exhausted groups are skipped. Every accepted bank stores exactly B distinct float32 vectors.')
    sub('3.3  Clean safeguard and ablations')
    p('An anchored candidate passes when its clean-normal validation coverage is at most 1.10 times the clean-only reference coverage. Among passing candidates, '
      'the lowest worst augmented-group validation error wins; ties favor more clean anchors. If neither passes, the clean bank is used. '
      'Without-guard retains anchor selection and chooses by augmented validation coverage without rejection. Without-anchors starts at the pooled mean-nearest candidate '
      'and adaptively selects across all three groups; it retains the clean safeguard. This isolates removal of the respective component.')
    sub('3.4  Scoring and localization')
    p('Patch scores are nearest-bank Euclidean distances. The most distant patch determines the initial image score. Following PatchCore Eq. (7), this is '
      'multiplied by one minus the softmax mass assigned to its nearest representative among nine support neighbors of that representative, including itself. '
      'A numerically stable softmax is used. Patch maps retain the original distances, are bilinearly upsampled, then Gaussian-smoothed with sigma 4 pixels. '
      'This shared scoring completes the feature omitted from the midterm; the candidate cap and exact greedy implementation still make this a controlled adaptation, not published-score reproduction.')
    page();heading('4  Experimental protocol and reproducibility')
    sub('4.1  Data and splits')
    splitrows=[]
    for cat in config['categories']:
        split=json.loads((results/f'{cat}_s23_b4096/split_manifest.json').read_text())
        normal=sum(Path(n).parent.name=='good' for n in split['test'])
        splitrows.append([cat,str(len(split['fit'])),str(len(split['validation'])),str(len(split['calibration'])),f'{normal} / {len(split["test"])-normal}'])
    table(['Category','Fit','Validate','Calibrate','Test normal / defect'],splitrows,[100,65,75,75,width-315])
    p('Each official normal training set is shuffled with seeds 23, 42 or 61 and split using floors for 60% fitting and 20% validation; the remainder is calibration. '
      'Every transformed view stays with its source image. Test data are used only after banks, normal-validation choices and calibration thresholds are fixed. '
      'Filename manifests and the immutable mirror revision [5] are included. File sizes and available mirror LFS SHA-256 hashes are checked, and required images/masks are opened before fitting. '
      'Official archive checksums were not independently compared.')
    sub('4.2  Conditions and calibration')
    table(['Family','Fitting / selection','Evaluation'],[['Clean','Original resized','Original resized'],['Brightness','0.85 and 1.15','0.75, 0.60, 0.45'],['JPEG quality','85','70, 50, 30'],['Blur sigma','Absent','0.5, 1.0, 1.5']],[104,170,width-274])
    p('Transformations follow resizing; JPEG uses Pillow 4:2:0 subsampling and blur uses Pillow GaussianBlur. Resized masks use nearest-neighbor interpolation. '
      'Each method\'s image threshold is the ceil[0.95(n+1)]-th clean calibration score; an unavailable rank gives an infinite threshold. '
      'A score must strictly exceed the threshold to trigger an anomaly. The threshold is fixed across all test conditions.')
    sub('4.3  Aggregation and uncertainty')
    p('Image AUROC pools images within a category/seed/condition. Pixel AUROC and average precision pool all resized pixels in that condition. '
      'Summary tables equally average conditions, seeds and categories; clean results use only the clean condition. '
      'For the primary paired bootstrap, scores are first averaged across seeds for each original image and condition, then AUROCs are computed. '
      'Original images are resampled within category and label, retaining all nine views and both methods together. Two thousand resamples use seed 20261112. '
      'This interval is conditional on these categories and fitted seeds; it does not estimate uncertainty from new categories or training sets.')
    page();heading('5  Primary results and hypothesis test')
    small('All tabulated metrics are percentages. AUROC, AP and recall are higher-is-better; FPR is lower-is-better. Tables average per-seed metrics, while the paired test uses seed-averaged scores.')
    for scope,title in [('clean','5.1  Clean evaluation at B = 4,096'),('corrupted','5.2  Mean of nine corrupted conditions')]:
        sub(title)
        table(['Method','Image AUROC','Pixel AUROC','Pixel AP','FPR','Recall'],[[NAMES[m]]+[pct(row(m,scope)[k]) for k in ['image_auroc','pixel_auroc','pixel_ap','fpr','recall']] for m in config['methods'][:4]],[137,76,76,70,61,width-420])
    sub('5.3  Prespecified paired comparison')
    p(f'The adaptive-minus-pooled difference is <b>{delta:+.2f} percentage points</b>, with a 95% paired percentile bootstrap interval '
      f'<b>[{lo:+.2f}, {hi:+.2f}]</b>. The clean macro/seed-mean difference against clean-only is '
      f'<b>{100*primary["clean_macro_seed_mean_delta_vs_clean"]:+.2f} points</b>. <b>{verdict}</b> '
      'A confidence interval spanning zero does not establish an adaptive benefit; a negative interval favors pooled selection within this protocol.')
    p(f'The per-seed metric table gives an adaptive-minus-pooled difference of '
      f'{100*(float(row("adaptive")["image_auroc"])-float(row("pooled")["image_auroc"])):+.2f} points. '
      'That table computes AUROC for each seed and then averages; the primary comparison first averages each image\'s scores across seeds and then computes AUROC. '
      'AUROC is a nonlinear ranking measure, so these two aggregation orders need not have the same difference or sign.')
    p(f'Among the four main methods, {NAMES[best]} has the highest mean corrupted image AUROC at the primary budget. '
      f'At frozen thresholds, adaptive FPR is {pct(row("adaptive")["fpr"])}% versus {pct(row("pooled")["fpr"])}% for pooled selection, '
      f'while recall is {pct(row("adaptive")["recall"])}% versus {pct(row("pooled")["recall"])}%. '
      'High ranking quality therefore does not guarantee matched threshold decisions, and the clean calibration target does not guarantee a 5% FPR under shift.')
    page();heading('6  Category, seed and budget analysis')
    sub('6.1  Corrupted image AUROC by category')
    table(['Category']+[NAMES[m] for m in config['methods'][:4]],[[cat]+[pct(next(r['image_auroc'] for r in category if int(r['budget'])==4096 and r['category']==cat and r['method']==m and r['scope']=='corrupted')) for m in config['methods'][:4]] for cat in config['categories']],[100,90,90,106,width-386])
    sub('6.2  Corrupted image AUROC by seed')
    table(['Seed']+[NAMES[m] for m in config['methods'][:4]],[[str(s)]+[pct(next(r['image_auroc'] for r in seeds if int(r['budget'])==4096 and int(r['seed'])==s and r['method']==m and r['scope']=='corrupted')) for m in config['methods'][:4]] for s in config['seeds']],[100,90,90,106,width-386])
    sub('6.3  Fixed-budget comparison')
    table(['Vectors','Clean-only','Pooled','Uniform','Adaptive'],[[str(b)]+[pct(row(m,budget=b)['image_auroc']) for m in config['methods'][:4]] for b in sorted(config['budgets'])],[100,98,98,98,width-394])
    figure('budget_tradeoff.png',width)
    small('Figure 1. Equal-budget comparisons of mean corrupted image AUROC and false positives. Every point averages three categories, three seeds and nine conditions. Memory payload grows from 4 to 16 to 32 MiB per bank.')
    p('Changing the bank budget changes both coverage and nearest-neighbor search cost. Candidate caps and image features are held constant. '
      'Seed variation reflects training split and candidate sampling variation; the three seeds share the same official test images and must not be treated as three independent test datasets.')
    p('Adaptive clean-bank fallbacks occurred in '+
      '; '.join(f'{sum(d["fallback"]=="True" for d in decisions if int(d["budget"])==b and d["method"]=="adaptive")} of 9 runs at B = {b:,}' for b in sorted(config['budgets']))+
      '. These counts show when a budget comparison also changes whether the safeguard accepts an augmented bank.')
    page();heading('7  Ablations and acquisition robustness')
    sub('7.1  Component ablations at B = 4,096')
    table(['Method','Clean image AUROC','Corr. image AUROC','Corr. pixel AP','Corr. FPR','Corr. recall'],
      [[NAMES[m],pct(row(m,'clean')['image_auroc'])]+[pct(row(m)[k]) for k in ['image_auroc','pixel_ap','fpr','recall']] for m in ['adaptive','no_guard','no_anchor']],
      [137,76,76,70,61,width-420])
    sub('7.2  Safeguard behavior')
    drows=[]
    for m in ['uniform','adaptive','no_anchor']:
        selected=[d for d in decisions if d['method']==m and int(d['budget'])==4096]
        drows.append([NAMES[m],f'{sum(d["fallback"]=="True" for d in selected)} / {len(selected)}',
           ' / '.join(f'{np.mean([float(d[k]) for d in selected]):.0f}' for k in ['clean_allocation','brightness_allocation','jpeg_allocation'])])
    table(['Method','Clean fallbacks','Mean stored clean / bright / JPEG'],drows,[140,100,width-240])
    p('Fallback is a measured outcome of the safeguard. A rejected no-anchor candidate may produce the clean-only bank; in that case its final results describe '
      'the guarded ablation, not the rejected candidate. Removing the guard can improve or degrade detection because normal coverage is only a proxy for defect separability. '
      'Uniform and adaptive share the anchor search and safeguard, making their comparison the clearest test of the group-allocation rule.')
    bottle_fallbacks=[sum(d['fallback']=='True' for d in decisions if int(d['budget'])==b and d['category']=='bottle' and d['method']=='adaptive') for b in sorted(config['budgets'])]
    bottle_note='For bottle, adaptive fallback counts are '+ '/'.join(map(str,bottle_fallbacks))+' out of three runs at budgets 1,024/4,096/8,192. '
    if bottle_fallbacks[-1]==3:
        bottle_note+='At the largest budget, every adaptive bank therefore equals the clean-only bank. '
    p(bottle_note+'The safeguard compares against a budget-specific clean reference, so extra memory can change the acceptance decision as well as feature coverage.')
    figure('conditions.png',width)
    small('Figure 2. Primary-budget condition-wise image ranking and false positives. B denotes brightness factor, J JPEG quality and G blur sigma. '
          'All severities are evaluated on the same original images; blur is absent from fitting and selection.')
    comparisons=[]
    for family in ['brightness','jpeg','blur']:
        values={m:next(r for r in family_results if int(r['budget'])==4096 and r['method']==m and r['family']==family) for m in ['adaptive','pooled']}
        comparisons.append(f'{family}: {pct(values["adaptive"]["image_auroc"])}% versus {pct(values["pooled"]["image_auroc"])}%')
    p('Family-wise mean image AUROC for adaptive versus pooled is '+ '; '.join(comparisons)+'. Each family mean averages its three severities, three categories and three seeds.')
    page();heading('8  Qualitative examples and failure analysis')
    figure('qualitative.png',width-120)
    small('Figure 3. Predetermined examples from each category at B = 4,096 and seed 23. Each row uses a common color scale across the displayed methods. '
          'Heatmaps are distances, not calibrated pixel decisions. Original images: MVTec Software GmbH, CC BY-NC-SA 4.0; resized, corrupted and overlaid.')
    p('Examples are selected by filename rather than by favorable scores: the first defective image and first normal image in each category are shown, '
      'with the normal image subjected to brightness 0.45. Ground-truth masks expose whether elevated responses overlap defects or spread over normal structure. '
      'The same distance scale within a row supports comparison of clean-only and adaptive banks; scales can differ across categories.')
    page();heading('8  Failure analysis (continued)')
    small('Frozen-threshold decisions at B = 4,096. Values average three seeds and nine corruption conditions within each category; percentages. Lower FPR and higher recall are better.')
    error_rows=[]
    for cat in config['categories']:
        for m in config['methods'][:4]:
            r=next(r for r in category if int(r['budget'])==4096 and r['category']==cat and r['method']==m and r['scope']=='corrupted')
            error_rows.append([cat,NAMES[m],pct(r['fpr']),pct(r['recall'])])
    table(['Category','Method','False-positive rate','Defect recall'],error_rows,[95,160,120,width-375])
    p('At B = 4,096, mean worst-condition image AUROC (taking the minimum of ten conditions per category/seed before averaging) is '+
      '; '.join(NAMES[m]+': '+pct(np.mean([float(r['image_auroc']) for r in worst_results if int(r['budget'])==4096 and r['method']==m]))+'%' for m in config['methods'][:4])+'.')
    sub('8.1  Interpretation')
    p('In Figure 3, both banks place strong responses over the selected broken-bottle region, hazelnut crack and tile crack. '
      'Adaptive selection reduces the background response on the selected darkened normal bottle, while the normal hazelnut and tile retain substantial spatial responses. '
      'These observations describe the displayed samples; they do not establish how often the same behavior occurs.')
    p('A spatially concentrated heat response can support defect localization even when image-level thresholds are unreliable. Conversely, acquisition changes may '
      'raise responses across normal texture or object boundaries. Examine the recorded false positives and false negatives alongside maps; a selected figure cannot establish dataset-wide accuracy. '
      'The failure-case CSV lists all errors at the frozen threshold so the qualitative conclusions remain traceable.')
    p('The allocation rule uses unnormalized squared feature distances. A high-variability group can attract much of the residual memory even when this does not improve '
      'defect ranking. This is a plausible explanation to investigate, not a causal claim established by the present ablations.')
    sub('8.2  Limits of synthetic corruption')
    p('Severe blur or JPEG compression may erase defect evidence. Brightness factors are simple RGB changes and do not model every camera or lighting effect. '
      'The experiment measures robustness to these operators at the fixed resolution; it does not establish readiness for an industrial inspection line.')
    page();heading('9  Efficiency, validation and deployment use')
    sub('9.1  Measured resources')
    erows=[]
    for m in config['methods'][:4]:
        selected=[r for r in efficiency if int(r['budget'])==4096 and r['method']==m]
        erows.append([NAMES[m],f'{np.median([float(r["construction_seconds"]) for r in selected]):.2f}',f'{np.median([float(r["latency_ms"]) for r in selected]):.1f}','16.00'])
    table(['Method','Build median (s)','Inference median (ms)','Bank MiB'],erows,[140,120,140,width-400])
    p('The recorded hardware is an NVIDIA GeForce RTX 2050 with 4 GB device memory. Bank construction excludes feature extraction; anchored methods include '
      'both fractions and validation but reuse the clean-bank prefix. Without-guard reuses the adaptive candidate search, so its recorded construction cost is shared. '
      'Batch-one inference includes image read, feature extraction, score reweighting, upsampling and smoothing, using five warmups and ten synchronized samples '
      'on a calibration image per run. Medians above summarize nine run medians. Laptop load and caching affect timings; these are indicative measurements.')
    p('Each bank contains 4 x B x 1,024 bytes: 4, 16 or 32 MiB. Peak CUDA allocated memory is saved per run and excludes CUDA reserved memory, system RAM and '
      'the full process footprint. Shared feature caches reduce repeated extraction; condition checkpoints allow interrupted evaluation to resume.')
    sub('9.2  Validation evidence')
    p(f'Eight protocol tests passed. The final audit checked {audit["image_metric_rows_checked"]:,} image metric rows against raw predictions, '
      'split disjointness, threshold ranks, bank sizes, unique descriptors, protected anchors and fallback bank identity. '
      f'It independently recomputed {audit["pixel_metric_rows_independently_checked"]} pixel metric rows with scikit-learn from stored patch grids '
      '(all categories/methods, B = 4,096, seed 23, clean and strongest brightness). All patch-grid shapes and finite values were checked. '
      'This sampled pixel audit is not a second full feature-extraction replication of the entire study.')
    sub('9.3  Using a fitted detector')
    p('Run code/predict.py with a completed run directory, input image, method and output directory. The command emits a JSON score and threshold decision, '
      'a raw anomaly map and a PNG overlay. Select a bank from the matching product category. The visualization uses a display-only color scale and does not provide a pixel segmentation threshold.')
    page();heading('10  Conclusions, limitations and references')
    p(f'The final study completes the planned category/seed/budget matrix and both component ablations. At B = 4,096, adaptive allocation achieves '
      f'{pct(row("adaptive")["image_auroc"])}% mean corrupted image AUROC, versus {pct(row("pooled")["image_auroc"])}% for pooled selection. '
      f'The paired seed-averaged-score comparison is {delta:+.2f} points [{lo:+.2f}, {hi:+.2f}]. {verdict} '
      'The project provides a reproducible test of clean anchors, coverage-driven allocation and normal-only validation, with conclusions constrained by the measured evidence.')
    p('Limitations include three selected categories, three training seeds, synthetic corruptions, one backbone/resolution, finite calibration samples and one laptop GPU. '
      'Candidate subsampling and exact greedy selection differ from the full reference benchmark setup. The pilot was previously observed, and the final bootstrap '
      'does not resample categories or refit training sets. No defect labels were used to select banks or thresholds. '
      'Future work can test real acquisition shifts, additional product categories and threshold adaptation using separately collected normal data.')
    sub('Reproduction and artifact availability')
    p('The repository contains code, locked configuration, data provenance, split manifests, calibration, metrics, per-image scores, report figures and both editable documents. '
      'Three reference adaptive banks (B = 4,096, seed 23) are included for direct prediction. Large dataset images, pretrained weights, feature caches and the remaining banks/patch grids are excluded from Git history. '
      'All banks and patch grids remain in the local completed study and can be regenerated with the study runner. The README records commands and environment requirements. The repository is intended for educational research; '
      'MVTec image content and derived visualizations retain their original noncommercial share-alike license.')
    p('Code, editable sources and reference models: <link href="https://github.com/sarthak06shukla/clean-anchored-patchcore" color="#173c60">GitHub project repository</link>.')
    sub('References')
    refs=[('Roth et al. Towards Total Recall in Industrial Anomaly Detection. CVPR, 2022.','https://arxiv.org/abs/2106.08265'),
      ('Bergmann et al. MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. CVPR, 2019.','https://www.mvtec.com/research-teaching/datasets/mvtec-ad'),
      ('Amazon Science. PatchCore reference implementation.','https://github.com/amazon-science/patchcore-inspection'),
      ('TorchVision. Wide ResNet-50-2 pretrained model.','https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html'),
      ('foersben. MVTec AD public mirror; pinned revision and checksums in package.','https://huggingface.co/datasets/foersben/mvtec-ad'),
      ('Santos, Tran and Rippel. Optimizing PatchCore for Few/Many-Shot Anomaly Detection. arXiv:2307.10792, 2023.','https://arxiv.org/abs/2307.10792'),
      ('Hyun et al. ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection. WACV, 2024.','https://openaccess.thecvf.com/content/WACV2024/html/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.html'),
      ('Pemula, Zhang and Dabeer. Robust AD: A Real World Benchmark Dataset for Robustness in Industrial Anomaly Detection. CVPR Workshops, 2025.','https://openaccess.thecvf.com/content/CVPR2025W/VAND/papers/Pemula_Robust_AD_A_Real_World_Benchmark_Dataset_For_Robustness_in_CVPRW_2025_paper.pdf')]
    for i,(text,url) in enumerate(refs,1):small(f'[{i}] {escape(text)} <link href="{url}" color="#173c60">Source</link>.')
    def footer(canvas,doc):
        if doc.page==1:return
        w,h=layout.A4;canvas.setStrokeColor(layout.colors.HexColor('#c4d0db'));canvas.setLineWidth(.5)
        canvas.line(52,h-43,w-52,h-43);canvas.setFont('Helvetica',8);canvas.setFillColor(layout.GRAY)
        canvas.drawString(52,h-34,'DL & CV  |  Group 23');canvas.drawRightString(w-52,h-34,'Clean-Anchored PatchCore - Final Report')
        canvas.line(52,42,w-52,42);canvas.drawRightString(w-52,29,str(doc.page-1))
    out=ROOT/'group23-final-report.pdf'
    doc=layout.SimpleDocTemplate(str(out),pagesize=layout.A4,leftMargin=52,rightMargin=52,topMargin=59,bottomMargin=56,
        title='Clean-Anchored PatchCore - Final Project Report',author='Sarthak Shukla and Yash Doshi')
    doc.build(layout.story,onFirstPage=footer,onLaterPages=footer)
    # The paper shares the same data, while keeping the supplied CVPR style.
    from paper_source import make_paper
    (ROOT/'paper/main.tex').write_text(make_paper(config,summary,category,seeds,decisions,efficiency,primary,abstract,audit),encoding='utf-8')
    print(out)

if __name__=='__main__':main()

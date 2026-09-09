**FedHybridNet: Cross-Representation Learning and Federated Aggregation for Fingerprint Classification in Edge Biometric Systems**



\#Python

\#TensorFlow

\#License



Overview



FedHybridNet is a lightweight, privacy-aware hybrid framework for fingerprint pattern classification that integrates handcrafted LBP and Gabor descriptors with a CBAM-enhanced CNN through an adaptive gate fusion mechanism, trained using Federated Averaging across four edge IoT clients without sharing raw biometric data.



**Key Features**

* **Hybrid feature fusion:** LBP + Gabor handcrafted descriptors + CBAM-enhanced CNN
* **Federated learning:** FedAvg across 4 edge IoT clients (raw data never leaves local devices)
* **Explainability:** Grad-CAM visualization of discriminative fingerprint regions
* **Edge deployment:** TFLite INT8 quantization validated on physical Raspberry Pi 5 hardware
* Gaussian noise sensitivity analysis for future differential privacy guidance



**Results Summary**

|**Model**|**Training Setting**|**Acc (%)**|**Prec (%)**|**Rec (%)**|**F1 (%)<br />**|**AUC (%)**|**Params**|
|-|-|-|-|-|-|-|-|
|**FedHybridNet (Ours)**|**Federated**|**98.86**|**98.86**|**98.86**|**98.84**|**99.77**|**187,018**|
|MobileNetV2 (pretrained)|Centralized|98.86|98.84|98.84|98.84|100.00|2,422,468|
|SVM + LBP/Gabor|Centralized|100.00|100.00|100.00|100.00|100.00|N/A|
|Federated SVM (Majority Vote)|Federated|98.86|—|—|98.83|99.95|N/A|
|Plain CNN|Centralized|87.50|87.37|87.01|86.27|97.62|109,700|
|EfficientNet-B0 (pretrained)|Centralized|23.86|9.63|23.86|9.63|53.36|4,214,055|
|MobileNetV2 (scratch)|Centralized|23.86|5.97|25.00|9.63|50.00|2,422,474|
|CNN-only (no fusion)|Centralized|57.95|55.22|57.99|52.59|85.66|110,372|





**Federated Learning Results**

|**Training Scenario**|**Data Distribution**|**Rounds**|**Clients**|**Acc (%)**|**F1 (%)**|**AUC (%)**|**±Std (%)**|**Privacy**|**Data Shared<br />**|
|-|-|-|-|-|-|-|-|-|-|
|Centralized Training|Full dataset|—|1|98.86|98.84|99.77|±0.00|NO|Full dataset|
|Federated FedAvg (R=10)|IID (Equal balanced)|10|4|96.59|96.51|98.43|±0.54|YES|Weights only|
|Federated FedAvg (R=20)|IID (Equal balanced)|20|4|98.86|98.84|99.77|±0.54|YES|Weights only|
|Federated FedAvg |Mild non-IID (70% skew)<br />)|20|4|76.14|73.18|90.27|±2.31|YES|Weights only|
|Federated FedAvg |Severe non-IID (90% skew|20|4|90.91|90.69|97.31|±1.73|YES|Weights only|





**Raspberry Pi 5 Deployment**

|**Model Format**|**Acc (%)**|**Size (KB)**|**Mean Latency (ms)**|**±Std (ms)**|**Throughput (inf/s)**|**Peak RAM (MB)**|**Speedup**|
|-|-|-|-|-|-|-|-|
|Original H5|98.86|895.9|—|—|—|—|1X|
|TFLite Float32|98.86|742.3|7.96|±0.80|125.7|13|1.97X|
|TFLite INT8|98.86|226.2|4.03|±0.40|248.3|11|3.88X|





**Device:** Raspberry Pi 5 (ARM Cortex-A76, 2.4 GHz, 8 GB LPDDR4X, Raspberry Pi OS 13 Trixie 64-bit)



**Ablation Study**

|Model Configuration|Accuracy|F1-Score|Params|
|-|-|-|-|
|CNN-only (no fusion, no CBAM)|57.95%|52.59%|110,372|
|CNN + CBAM (no fusion)|87.50%|86.27%|109,700|
|CNN + Handcrafted (simple concat)|94.32%|94.18%|156,482|
|CNN + CBAM + Handcrafted (simple concat)|97.73%|97.61%|183,754|
|CNN + CBAM + Handcrafted + Adaptive Gate (FedHybridNet)|98.86%|98.84%|187,018|





**Statistical Validation**

|Metric|Run 1 (Seed=42)|Run 2 (Seed=7)|Run 3 (Seed=21)|Mean|Std|
|-|-|-|-|-|-|
|Accuracy (%)|96.59|100.00|96.59|97.73|±1.61|
|F1-Score (%)|96.51|100.00|96.51|97.67|±1.65|
|AUC-ROC (%)|99.90|100.00|99.93|99.94|±0.04|





Datasets



FedHybridNet was evaluated on the combined FVC2002 and FVC2004 benchmark datasets:



FVC2002 DATASET:    http://bias.csr.unibo.it/fvc2002/download.asp

FVC2004 DATASET:    http://bias.csr.unibo.it/fvc2004/download.asp



Dataset split:

|Split|Original Images|Augmented Images|Per Class|
|-|-|-|-|
|Train|488|2,336|584|
|Validation|90|90|22-23|
|Test|88|88|21-23|
|Total|626|2,514|-|





**Download datasets and place in the following structure:**



data/

├── FVC2002/

│   ├── class0/

│   ├── class1/

│   ├── class2/

│   └── class3/

└── FVC2004/

&#x20;   ├── class0/

&#x20;   ├── class1/

&#x20;   ├── class2/

&#x20;   └── class3/



**Installation**



git clone https://github.com/YOUR\_USERNAME/FedHybridNet.git

cd FedHybridNet



pip install -r requirements.txt



**Requirements**



tensorflow>=2.10.0

numpy>=1.21.0

opencv-python-headless>=4.5.0

scikit-learn>=1.0.0

scikit-image>=0.19.0

matplotlib>=3.5.0

seaborn>=0.11.0

psutil>=5.9.0

Pillow>=9.0.0

tqdm>=4.64.0



**Project Structure**

FedHybridNet/

├── README.md

├── requirements.txt

├── LICENSE

├── data/

│   └── README.md

├── preprocessing/

│   └── preprocess.py

├── models/

│   └── fedhybridnet.py

├── features/

│   └── feature\_extraction.py

├── federated/

│   └── fedavg.py

├── experiments/

│   ├── train\_centralized.py

│   ├── train\_federated.py

│   ├── noise\_robustness.py

│   ├── gaussian\_sensitivity.py

│   └── cross\_dataset.py

├── deployment/

│   └── tflite\_convert.py

└── notebooks/

&#x20;   └── FedHybridNet\_Demo.ipynb



**Experimental Results Files**



All experimental results are saved in JSON format:



results/evaluation/full\_results.json — centralized evaluation

results/noniid/noniid\_results.json — federated non-IID results

results/robustness/robustness\_results.json — noise robustness

results/statistical/statistical\_results.json — multi-seed validation

results/communication/communication\_results.json — communication cost

results/cross\_dataset/cross\_dataset\_results.json — cross-dataset results

results/reviewer/dp\_results.json — Gaussian noise sensitivity



**Gaussian Noise Sensitivity Analysis**

|Configuration|Accuracy|F1|AUC|Noise Level|Acc Drop|
|-|-|-|-|-|-|
|No noise (FedAvg baseline)|98.86%|98.83%|99.84%|None|0.00%|
|Gaussian (σ=0.0001)|98.86%|98.83%|99.84%|Very Low|0.00%|
|Gaussian (σ=0.001)|98.86%|98.83%|99.88%|Low|0.00%|
|Gaussian (σ=0.01)|26.14%|10.36%|50.00%|Moderate|72.72%|
|Gaussian (σ=0.1)|26.14%|10.36%|50.00%|High|72.72%|



&#x09;

Note: This experiment evaluates Gaussian weight perturbation sensitivity, not formal differential privacy.



**Communication Cost Analysis**

|Metric|Value|
|-|-|
|Float32 per client per round|730.5 KB|
|INT8 per client per round|182.6 KB|
|Total Float32 (20 rounds)|114.15 MB|
|Total INT8 (20 rounds)|28.54 MB|
|Weight transmission reduction|75.0%|
|Raw biometric data shared|0 bytes|





Citation



If you use this code or find this work helpful, please cite our paper:



@article{fedhybridnet2026,

&#x20; title={Cross-Representation Learning and Federated

&#x20;        Aggregation for Fingerprint Classification

&#x20;        in Edge Biometric Systems},

&#x20; author={Ajay S. et al.},

&#x20; journal={IEEE Access},

&#x20; year={2026},

&#x20; note={Under Review — Manuscript ID: Access-2026-36648}

}

License



This project is licensed under the MIT License. See LICENSE file for details.



Acknowledgment



This work was supported by Dayananda Sagar University, Bangalore, India. The FVC2002 and FVC2004 benchmark datasets are publicly available from the Fingerprint Verification Competition organizers at http://bias.csr.unibo.it/fvc2002/ and http://bias.csr.unibo.it/fvc2004/ respectively.



Contact



For questions, issues, or collaboration opportunities, please contact:



Author: Ajay S.

Email: ajay.s-rs-ca@dsu.edu.in

Institution: Dayananda Sagar University, Bangalore, India


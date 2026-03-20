# DADS7202-DeepLearning

# Homework :
Code and Dataset : https://drive.google.com/drive/folders/1lR14cJw8nPjcrKOdRpuiA21etvh8kiD8


# DADS7202_Group Assignment Deep Learning for "Pizza topping"

> Objective: **`What do you use to train an image classifier
> with our custom image dataset?`**

# ✨High Light
🏆 ConvNeXt-Tiny คือโมเดลที่มีประสิทธิภาพสูงสุด: จากการทดสอบเปรียบเทียบ 4 โมเดลพบว่า ConvNeXt-Tiny ให้ค่า F1-score สูงสุดที่ 81.4% โดยผลการทดสอบทางสถิติ (Dunn's Test) ยืนยันว่ามีประสิทธิภาพเหนือกว่า EfficientNetB0 อย่างมีนัยสำคัญ ($p < 0.05$) ในขณะที่กลุ่มโมเดลประสิทธิภาพสูงรองลงมาคือ ResNet18 และ MobileNet V3L ตามลำดับ

🍕 ความท้าทายของคลาส Margherita: ทุกโมเดลประสบปัญหาในการจำแนกหน้า Margherita โดยให้ค่า F1-score ต่ำสุด (เฉลี่ย 59% - 77%) เนื่องจากปัญหา Feature Overlap ของวัตถุดิบ เช่น ซอสมะเขือเทศที่มีโทนสีแดงคล้ายแผ่น Pepperoni และการกระจายตัวของชีสที่คล้ายคลึงกับหน้าอื่น รวมถึงพบ Noise ในข้อมูล (รูปที่ไม่ใช่พิซซ่า) ซึ่งต้องแก้ไขด้วยการทำ Data Cleaning และเพิ่ม Augmentation ที่เน้นจุดโฟกัสมากขึ้น

⚙️ กลยุทธ์การปรับแต่ง Classifier ที่เข้มข้น: เพื่อเพิ่มความสามารถในการเรียนรู้ลักษณะ Topping ที่ซับซ้อน ได้มีการปรับโครงสร้างส่วนท้ายของโมเดล (Fine-tuning) โดยเพิ่มความลึกของ Linear Layer (เช่น MobileNet เปลี่ยนจาก 2 เป็น 3 เลเยอร์) พร้อมใช้เทคนิค Double Dropout และการทำ Hyperparameter Sweep (Bayesian Search) เพื่อควบคุมภาวะ Overfitting และค้นหาค่าการเรียนรู้ที่เหมาะสมที่สุด

🔍 การวิเคราะห์ด้วย Grad-CAM และทางออกในอนาคต: เทคนิค Explainable AI เผยให้เห็นว่าโมเดลส่วนใหญ่โฟกัสที่ Topping หลัก (เช่น สับปะรด หรือ วงกลมสีแดง) แต่พบความผิดปกติในหน้า Seafood ที่จุดโฟกัสไปอยู่ที่ขอบภาพ จึงนำไปสู่ข้อเสนอแนะในการปรับปรุงด้วย Center-Weighted Cropping และการใช้ Advanced Loss Function เพื่อยกระดับความแม่นยำให้เสถียรยิ่งขึ้นในระดับใช้งานจริง

---

## 1. Data description 🎯

### 1.1 บทนำ

ในงานด้าน Computer Vision การจำแนกประเภทอาหาร (Food Classification) ถือเป็นโจทย์ที่มีความซับซ้อนสูง โดยเฉพาะในลักษณะงานที่เป็น Fine-grained Classification อย่างการจำแนกหน้าพิซซ่า

ความท้าทายสำคัญไม่ได้อยู่ที่โครงสร้างหลักของวัตถุซึ่งมีลักษณะเป็นวงกลมเหมือนกันทุกคลาส
แต่อยู่ที่รายละเอียดของ Texture และการซ้อนทับกันของวัตถุดิบ (Toppings)
ปัญหาที่มักพบคือ "Occlusion" หรือการที่ชีสและซอสปกคลุมวัตถุดิบหลัก
ทำให้กระบวนการดึงลักษณะเด่น (Feature Extraction) ทำได้ยากกว่าภาพวัตถุทั่วไป
จึงได้ตั้งเป้าหมายเปรียบเทียบประสิทธิภาพของ Model โครงข่ายประสาทเทียมจำนวน 4 รูปแบบ
เพื่อค้นหาโมเดลที่สามารถรับมือกับความซับซ้อนนี้ได้ดีที่สุด
ประกอบด้วย:

- EfficientNet-B0
- MobileNet V3L
- ResNet18
- ConvNeXt-Tiny

### 1.2 Exploratory data analysis: EDA

#### 1.2.1 Data Sources : มาจาก Web sites ทั้่งหมด

> ตารางที่ 1.2-1 ตารางแสดง Data Source
> | แหล่งข้อมูล | จำนวนรูปภาพ | สัดส่วน |
> | :--- | :--- | :--- |
> | alamy.com | 447 | 44.7% |
> | depositphotos.com | 220 | 22.0%|
> | shutterstock.com | 107 | 10.7% |
> | freepik.com | 33 | 3.3% |
> | อื่นๆ | 193 | 19.3% |
> | **รวม** | **1,000** | 100.0% |
#### 1.2.2 จำนวนรูปภาพในแต่ละคลาส (N = 1,000 รูปภาพ)

- Hawaiian 250 รูปภาพ
- Margherita: 250 รูปภาพ
- Pepperoni: 250 รูปภาพ
- Seafood: 250 รูปภาพ

#### 1.2.3 ขนาดรูปภาพ (ขนาดเฉลี่ย-Pixel W:1025, H:834 )

> **รูปที่ 1.2-1** กราฟ Scatter plot แสดงขนาดรูปภาพ (หน่วย Pixel)
> ![image](https://hackmd.io/_uploads/rJrWWs_5Wx.png)

#### 1.2.4 ความสว่างของรูปภาพ (Dark:507 รูปภาพ, Bright:493 รูปภาพ)

> **รูปที่ 1.2-2** กราฟความสว่างของรูปภาพ
> ![image](https://hackmd.io/_uploads/SkaP7iOqZg.png)



### 1.3 Data Balance / Imbalance

Data set balance 250 ภาพ ต่อ class

---

## 2. Data preparation 📑

### 2.1 pre-process/post process

ในขั้นตอนการจัดเก็บข้อมูลเบื้องต้น ไม่มีการทำ Manual Preprocessing
(เช่น การตัดขอบภาพด้วยมือ) ก่อนการ Train ส่วนในขั้นตอนการ Train
จะมีการทำ Automated Preprocessing ผ่านระบบ Pipeline
เพื่อปรับขนาดภาพเป็น 224x224 pixels ให้เหมาะสมกับโครงสร้างของโมเดลแต่ละชนิด

### 2.2 Train / Validation / Test Split Process

#### 2.2.1 สร้างกลุ่มข้อมูลสำหรับ Stratification

เพื่อให้การแบ่งข้อมูลมีสัดส่วนที่สมดุล จึงสร้างตัวแปรกลุ่มจากคุณลักษณะสำคัญ

- แบ่งค่า Brightness เป็น 2 กลุ่ม Dark, Bright
   → สร้างเป็นคอลัมน์ Brightness_Bin
- รวมค่า Class กับ Brightness_Bin
 → สร้างเป็นคอลัมน์ Class_Brightness_Bin เช่น Seafood_Dark

#### 2.2.2 แบ่ง Train และ Temporary Dataset

แบ่งข้อมูลครั้งแรกด้วย `train_test_split`

- Train : 60%
- Test + Valid : 40%

แบ่งข้อมูลครั้งที่ 2 ด้วย `train_test_split` ของข้อมูล Test + Valid

- Test : 20%
- Valid : 20%

ใช้ `stratify = "Class_Brightness_Bin"` เพื่อรักษาสัดส่วนของแต่ละกลุ่ม

> **ตารางที่ 2.2-1** ตารางแสดงสัดส่วนข้อมูลสุดท้าย
> | Class_Brightness_Bin | Train | Test | Valid | Total |
> |---|---|---|---|---|
> | Hawaiian_Bright | 90 | 30 | 30 | 150 |
> | Hawaiian_Dark | 60 | 20 | 20 | 100 |
> | Margherita_Bright | 68 | 23 | 23 | 114 |
> | Margherita_Dark | 82 | 27 | 27 | 136 |
> | Pepperoni_Bright | 58 | 20 | 19 | 97 |
> | Pepperoni_Dark | 92 | 30 | 31 | 153 |
> | Seafood_Bright | 79 | 27 | 26 | 132 |
> | Seafood_Dark | 71 | 23 | 24 | 118 |
> | **Total** | **600** | **200** | **200** | **1000** |

### 2.3 Data Augmentation

ในขั้นตอนการเตรียมข้อมูลสำหรับการ Train model ใช้ Transform Pipeline ของ PyTorch
เพื่อทำ Image Augmentation กับภาพหน้า Pizza ก่อนส่งเข้า `Dataset` และ `DataLoader`
เพื่อเพิ่มความหลากหลายของภาพ Pizza จึงใช้เทคนิค Image Augmentation หลักดังนี้

#### 2.3.1 Random Resized Crop

สุ่มครอปภาพและปรับขนาดเป็น  ช่วยให้โมเดลเรียนรู้ตำแหน่งของพิซซ่าที่แตกต่างกัน

- ใช้พื้นที่ของภาพระหว่าง 60% – 100%
- ปรับขนาดภาพให้ได้ 224 × 224 pixels

#### 2.3.2 Random Horizontal Flip

สุ่มกลับภาพในแนวนอน เพิ่มความหลากหลายของข้อมูล

#### 2.3.3 Color Jitter

สุ่มปรับสีของภาพเล็กน้อย ช่วยให้โมเดลทนต่อความแตกต่างของแสงและสีจากสภาพการถ่ายภาพที่ต่างกัน

- Brightness ±20%
- Contrast ±20%
- Saturation ±20%
- Hue ±5%

```python
# ตัวอย่าง Code
train_transform = T.Compose([
    T.ToImage(),
    T.RandomResizedCrop(224, scale=(0.6,1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2,0.2,0.2,0.05),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=backbone_mean, std=backbone_std),
])
```

---

## 3. Model Architecture📦

ในการศึกษาครั้งนี้ ประกอบด้วย 4 Models ที่นำมาศึกษา สำหรับ Image Classifier คือ

### 3.1 Reasons

> **ตารางที่ 3.1-1** ตารางแสดงสาเหตุในการเลือกโมเดลและตัวอย่างการประยุกต์ใช้งานโมเดล
> | Backbone Model | Advantage | Use Case |
> |---|---|---|
> | EfficientNetB0  | Balanced Performance: สมดุลระหว่างความแม่นยำ (Accuracy) และความเร็ว (Speed) ดีมากเมื่อเทียบกับขนาดโมเดล | Image Classification ทั่วไป เช่น จำแนกสินค้า, Medical Imaging เบื้องต้น เช่น แยกภาพปกติ/ผิดปกติ, Document/Image Tagging เช่น แยกประเภทเอกสาร |
> | MobileNet V3L  | ลดการใช้ทรัพยากรและประมวลผลได้รวดเร็ว (Efficiency) เหมาะสำหรับการใช้งานแบบ Real-time บนอุปกรณ์พกพา (Edge Devices) โดยยังคงความแม่นยำในการจำแนกภาพที่สูงและปรับจูนได้ง่าย | Real-time Image Classification สแกนหน้าถาดเพื่อคำนวณแคลอรี่หรือระบุประเภทได้ทันทีโดยไม่ต้องส่งภาพขึ้น Cloud |
> | ResNet18  |โครงสร้างแบบ Residual Connections ช่วยให้การไหลของค่าน้ำหนัก (Gradient) ดีขึ้นเมื่อโมเดลมีความลึก แก้ปัญหา Vanishing Gradient ทำให้เทรนง่าย มีความเสถียรสูง และสกัดลักษณะเด่น (Feature) ของภาพได้ดีเยี่ยม ถือเป็นโมเดลมาตรฐานที่เชื่อถือได้และใช้ทรัพยากรระดับกลาง  |เหมาะมากสำหรับใช้เป็นโมเดลตัวตั้งต้น (Baseline) เพื่อวัดประสิทธิภาพพื้นฐานในการทดสอบสมมติฐาน หรืองานจำแนกประเภทวัตถุที่มีรายละเอียดเฉพาะกลุ่ม เช่น การจำแนกประเภทหน้าพิซซ่า เนื่องจากโมเดลมีความลึกพอที่จะเรียนรู้ลวดลายและพื้นผิวของวัตถุดิบได้ชัดเจน โดยไม่เกิดปัญหา Overfitting ง่ายจนเกินไป  |
> | ConvNeXt-Tiny  | **High Performance**: แม่นยำระดับ Transformer แต่ทำงานเร็วและเสถียรแบบ CNN,**Modern Design**: ใช้ Large Kernel (7x7) ช่วยให้แยกแยะรายละเอียดภาพที่ซับซ้อนได้ดี,**Resource Efficient**: รุ่น Tiny น้ำหนักเบา ประมวลผลไว ไม่กินทรัพยากรเครื่อง | Real-time Recognition: ระบบคิดเงินอัตโนมัติแบบทันที |
>
### 3.2 Pre-Trained Models Performance information

Pre-trained Model Performance Comparison (ImageNet-1K)

> **ตารางที่ 3.2-1** ตารางแสดงข้อมูลทั่วไปของแต่ละโมเดล
> | Model Family | Variant | Size (MB)| Top-1 Acc| Params (M)|
> |---|---|---|---|---|
> |ResNet|ResNet-18 |~98|69.8%| 11.7M|
> |ConvNeXt-Tiny|ConvNeXt-Tiny |~114|82.5%|28.6M|
> |MobileNet|MobileNetV3-L |~22|75.3%|5.5M|
> |EfficientNet|EfficientNet-B0 |~21|77.7%|5.3M|

#### Model Performance Comparison📊

ในการวัดค่า Accuracy ใช้ค่า Backbone เป็น Baseline จากโมเดลที่ยังไม่ได้ทำ Fine-tuning
และไม่ได้จำแนก 4 คลาสของหน้า pizza แต่เป็นการ**ทำนายว่าเป็น Pizza เท่านั้น**
เพื่อใช้เปรียบเทียบกับผลลัพธ์หลังการทำ Fine-tuning เรียบร้อยแล้ว

> **ตารางที่ 3.2-2** ตารางแสดง Accuracy Score เปรียบเทียบแต่ละ Model Architecture กรณีไม่ทำ Fine-tuning และทำ Fine-tuning
> | Model | Baseline <br>(Predict Pizza) | Base line <br>(4 Pizza classes) |Fine-Tuning <br>(4 Pizza classes) |
> |:---|:---:|:---:|:---:|
> | EfficientNetB0  | 88.0% |0.0%| 72.9% ± 1.0% |
> | MobileNet | **89.5%** |0.0%| 80.8% ± 1.0% |
> | ResNet18  |86.6%  |0.0%|80.6% ± 2.1% |
> | ConvNeXt-Tiny  | **89.5%** |0.0%| **81.5% ± 1.0%** |
> 
> **หมายเหตุ: ค่า Accuracy ในคอลัมน์ Baseline ไม่ได้จำแนก 4 คลาสของ pizza
> แต่เป็นการทำนายว่าเป็น pizza เท่านั้น**

> **รูปที่ 3.2-1** ภาพตัวอย่างการทำนายของ Model ที่ไม่ได้ทำการ Fine-Tuning
> ![image](https://hackmd.io/_uploads/r1aU3j8qZl.png)

> **รูปที่ 3.2-2** ภาพตัวอย่างการทำนายของ Model ที่ทำการ Fine-Tuning
> ![image](https://hackmd.io/_uploads/r1QD3oI5Ze.png)



### 3.3 Original Backbone Model (Remove/Keep/Add)

> **ตารางที่ 3.3-1** ตารางแสดงการ remove/keep/Add ของแต่ละโมเดล
> | Original Backbone Model | Remove | Keep | Add |
> |---|---|---|---|
> | EfficientNetB0  | Classifier ทุก Layer | Features และ Avgpool Layer | Classifier Layer ทั้งชุด<br>จำนวน Layer และ Node จะใช้ WandB เป็นตัวช่วยในการหา Best Parameter |
> | MobileNet  | Classifier ทุก Layer | Features และ Avgpool Layer  | 1.เพิ่ม linear layer จาก 2 เป็น 3 และ feature width เพื่อให้ model มีพื้นที่การเรียนรู้ที่ซับซ้อนมากขึ้น, Backbone 2 Linear Layers (960 → 1280 → 1000), Fine-tune 3 Linear Layers (960 → 1560 → 640 → 4) <br> 2. Activation function เปลี่ยนจาก Hardswish เป็น function มาตรฐาน ReLu ทำการคำนวณได้เร็วและเสถียร <br> 3. เพิ่ม Dropout เป็น 2 จุด ป้องกันไม่ให้ model จำภาพเดิม หรือ overfitting และช่วยให้ F1 score เสถียรขึ้น <br> 4. ปรับ output จาก 1000 เป็น 4 ตามจำนวน class ที่ต้องการของหน้า pizza|Feature Layer |Classifier Layer  |
> | ResNet18  |Classifier ทุก Layer |Features และ Avgpool Layer   |เพิ่มในส่วน Linear Layer หรือ Fully Connected Layer (Classifier)ใหม่ เพื่อแยกคลาสวัตถุออกเป็น 4 Classes |Input+Feature Layer1-3 | Feature Extractor Layer4 + Classification Layer|
> | ConvNeXt-Tiny | Classifier ทุก Layer | Features และ Avgpool Layer | เพิ่มชั้น Classifier ชุดใหม่ที่ประกอบด้วย Linear Layer, ReLU และ Dropout โดยกำหนด Output เป็น 4 คลาสตามโจทย์ |



### 3.4 Fined Tuning Model structure and parameters setting 

> **ตารางที่ 3.4-1** ตารางแสดงการ Add/Freeze/Unfreeze ของโมเดล - EfficientNet-B0 
> |Layer Type| In-Features | Out-Features | Activation/Regularization| Freeze | Unfreeze |Remark|
> |---|---|---|---|---|---|---|
> |Feature Layer<br>Original|Original |Original|Original|✓|---|---|
> |AdaptiveAvgPool2d|- |-|---|---|✓|---|
> |Linear Layer 1|1280|**1024**|**ReLU**|---|✓|---|
> |Dropout 1| - | - |**0.5**|---|✓|---|
> |Linear Layer 2|**1024**|**1024**|**ReLU**|---|✓|---|
> |Dropout 2|-|-|**0.5**|---|✓|---|
> |Linear Layer 3|**1024**|**4**|---|---|✓|---|
>
> **ตารางที่ 3.4-2** ตารางแสดงการ Add/Freeze/Unfreeze ของโมเดล - MobileNet V3L
> |Layer Type| In-Features | Out-Features | Activation/Regularization| Freeze | Unfreeze |Remark|
> |---|---|---|---|---|---|---|
> |Feature Layer<br>Original|Original |Original|Original|✓|-|---|
> |AdaptiveAvgPool2d|- |-|---|---|✓|---|
> |Linear Layer 1|960 |1560|ReLU|---|✓|---|
> |Dropout 1| - | - |0.2|--- |✓|---|
> |Linear Layer 2|1560|640|ReLU|---|✓|---|
> |Dropout 2|-| -|0.2|---|✓|---|
> |Linear Layer 2|640|4|---|---|✓|---|

> **ตารางที่ 3.4-3** ตารางแสดงการ Add/Freeze/Unfreeze ของโมเดล ResNet18
> |Layer Type| In-Features | Out-Features | Activation/Regularization| Freeze | Unfreeze | Remark |
> |---|---|---|---|---|---|---|
> |Feature Layer<br>Original|Original |Original|Original|✓|-|---|
> |AdaptiveAvgPool2d|- |-|---|---|✓|---|
> |Linear Layer 1|64 |64|ReLu|---|✓|
> |Dropout 1| - | - |0.2 |---|✓|
> |Linear Layer 2|64|128|ReLu|---|✓|
> |Dropout 2|-|-|0.2|---|✓|
> |Linear Layer 3|128|256|4|---|✓|

> **ตารางที่ 3.4-4** ตารางแสดงการ Add/Freeze/Unfreeze ของโมเดล ConvNeXt-Tiny
> |Layer Type| In-Features | Out-Features | Activation/Regularization| Freeze | Unfreeze |Remark|
> |---|---|---|---|---|---|---|
> |Feature Layer<br>Original|Original |Original|Original|✓|-|---|
> |AdaptiveAvgPool2d|- |-|---|---|✓|---|
> |Linear Layer 1|768|**768**|ReLU|---|✓|---|
> |Dropout 1| - | - |0.3 |---|✓|---|
> |Linear Layer 2|**768**|**768**|ReLU|---|✓|---|
> |Dropout 2|-|-|0.3|---|✓|---|
> |Linear Layer 3|**768**|**4**|---|---|✓|---|

---

## 4. Training Method 🔮

### 4.1 วิธีการ Training

* โมเดลนี้ถูกออกแบบมาเพื่อ **ทำ Fine-tuning กับโมเดลที่ Pre-trained แล้ว** โดย Train เฉพาะ Classifier Layer เมื่อทำการ freeze ใน Feature layer และใช้เทคนิคการทำ Hyper-parameter tuning ด้วยเครื่องมือ W&B 

* วิธีการ Train ทำการ freeze และ unfreeze weights ของแต่ละ layer และ แต่ละ Model อ้างอิงตามตาราง 3.4-1 (EfficientNet B0) , 3.4-2 (MobileNet V3L), 3.4-3 (RestNet18) และ 3.4-4 (ConvNext-Tiny)


### 4.2 รายละเอียด Hyperparameters หลัง Fined-Tune 

#### 4.2.1 Training Process

การทำ hyperparameter sweep นี้ครอบคลุมทั้ง
**Training parameters**
-`batch_size`
-`learning_rate`
-`optimizer`
**Model architecture**
-`hidden_dim`
-`classifier_layers`
-`dropout`

การฝึกโมเดลถูกติดตามด้วยค่า Loss ทั้งในชุด Training และ Validation
ในแต่ละ epoch ระหว่างการฝึกได้มีการบันทึก **model checkpoint**
โดยเลือกโมเดลที่มีค่า **validation loss ต่ำที่สุด** เพื่อใช้เป็นโมเดลสุดท้ายสำหรับการประเมินผลบนชุด
**test dataset** วิธีนี้ช่วยให้ได้โมเดลที่มีความสามารถในการ generalize กับข้อมูลที่ไม่เคยเห็นได้ดีที่สุด
Cross-Entropy Loss สำหรับงาน Classification สามารถเขียนได้ดังนี้

$$
L = - \sum_{i=1}^{C} y_i \log(p_i)
$$

โดยที่

- \(C\) คือจำนวน class
- \(y_i\) คือค่าจริงของ class (one-hot encoded label)
- \(p_i\) คือความน่าจะเป็นที่โมเดลทำนายสำหรับ class \(i\)

สำหรับชุดข้อมูลที่มี \(N\) ตัวอย่าง ค่า loss เฉลี่ยคือ

$$
L = - \frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{C} y_{n,i} \log(p_{n,i})
$$

> **ตารางที่ 4.2-1** ตารางแสดง Sweep Configuration 
> | Configuration  | ค่าที่ใช้ | คำอธิบาย |
> |---|---|---|
> |Method | Grid, Random, Bayes|1. Grid Search : เป็นการค้นหาแบบ "ปูพรม" โดยระบบจะทดลองทุกคู่ผสม (Combination) ของพารามิเตอร์ที่กำหนดไว้ <br> 2. Random Search : เป็นการ "สุ่ม" เลือกค่าพารามิเตอร์จากช่วงที่กำหนดไว้ <br>3. Bayesian Search เป็นวิธีที่เรียนรู้จากข้อมูลในอดีต เพื่อทำนายว่า "พารามิเตอร์ชุดไหนน่าจะให้ผลลัพธ์ดีที่สุด |

> **ตารางที่ 4.2-2** ตารางแสดง Hyperparameter จาก Sweep Configuration  
> | Hyperparameter | ค่าที่ใช้ | คำอธิบาย |
> |---|---|---|
> | `batch_size` | 32, 64 | จำนวนตัวอย่างข้อมูลที่ใช้ในการคำนวณ gradient ต่อหนึ่ง iteration ของการฝึกโมเดล |
> | `dropout` | 0.2, 0.3, 0.5 | อัตราการปิด neuron แบบสุ่มในระหว่างการ train เพื่อช่วยลด overfitting |
> | `hidden_dim` | 1024, 1280, 1536 | จำนวน neuron ใน hidden layer ของ classifier ซึ่งมีผลต่อความสามารถในการเรียนรู้ของโมเดล |
> | `classifier_layers` | 1, 2 | จำนวน hidden layer ใน classifier ที่ต่อจาก backbone |
> | `learning_rate` | 0.00001 – 0.001 (log-uniform) | อัตราการปรับค่า weight ของโมเดลในแต่ละขั้นตอนของ optimizer |
> | `optimizer` | adam, sgd | อัลกอริทึมที่ใช้ในการปรับปรุง weight ของโมเดล |

> **ตารางที่ 4.2-3** ตารางแสดง Hyperparameter จาก Sweep Configuration ที่นำมาใช้งาน

* ทุกโมเดลใช้ method "Bayes" ในการ train โมเดล

> | Hyperparameter | EfficientNetB0| MobileNet | ResNet18 | ConvNeXt-Tiny |
> |---|:--:|:--:|:--:|:--:|
> | `batch_size` | 64 | 16 | 32 | 32 |
> | `dropout` | 0.5 | 0.2 | 0.3 | 0.3 |
> | `hidden_dim` | 1024 | - | 1024 | 450 |
> | `classifier_layers` | 2 | - | 1 | 2 |
> | `learning_rate` | $$4.146 \times 10^{-4}$$ | $$1.847\times 10^{-05}$$| $$2.468 \times 10^{-4}$$ | $$7.031 \times 10^{-5}$$ |
> | `optimizer` | adam |adam | adam | adam |

#### 4.2.2 เหตุผลในการเลือก Hyperparameters เหล่านี้

##### 4.2.2.1 Batch Size

`batch_size` มีผลต่อความเร็วในการ train และความเสถียรของการคำนวณ gradient  

- ค่า **มากขึ้น** → train เร็วขึ้น แต่ใช้ memory มากขึ้น  
- ค่า **น้อยลง** → gradient มี noise มากขึ้น
แต่อาจช่วย generalization ได้ดีขึ้น  

จึงทดลองที่ **32 และ 64**

##### 4.2.2.2 Dropout

`dropout` ใช้เพื่อลดปัญหา **overfitting**
โดยการสุ่มปิด neuron ระหว่างการ train  

- ค่าเล็ก → regularization น้อย  
- ค่าใหญ่ → ลด overfitting ได้มากขึ้น

##### 4.2.2.3 Hidden Dimension

`hidden_dim` กำหนด **ขนาดของ hidden layer ใน classifier**

- ค่าน้อย → โมเดลเล็ก train เร็ว  
- ค่ามาก → โมเดลมี capacity สูงขึ้น

##### 4.2.2.4 Classifier Layers

`classifier_layers` กำหนด **ความลึกของ classifier**

- layer น้อย → model simple  
- layer มาก → model มีความสามารถในการเรียนรู้ pattern ซับซ้อนมากขึ้น

##### 4.2.2.5 Learning Rate

`learning_rate` มีผลต่อ **ความเร็วและเสถียรภาพของการ train**

##### 4.2.2.6 Optimizer

`Adam`

- adaptive learning rate
- converge เร็ว

`SGD`

- generalization ดี
- มักให้ performance ดีในบางงาน

### สรุป

การทดลองหลาย configuration ช่วยให้สามารถค้นหา **ค่าพารามิเตอร์ที่เหมาะสมที่สุดสำหรับโมเดล**  
โดยพิจารณาจาก **validation performance เช่น validation loss หรือ validation accuracy**


---

## 5. Evaluation Metrics 📈

### 5.1 Metrics ที่ใช้ในการ Evaluate Model

ในรายนี้เลือกการ evaluation โดยพิจารณา F1-Score และ Confusion Matrix 

#### Evaluation Metrics

ในการประเมินประสิทธิภาพของโมเดลสำหรับการจำแนกประเภทภาพพิซซ่า
ใช้ตัวชี้วัด (evaluation metrics) หลายประเภท เพื่อสะท้อนความสามารถของโมเดลในหลายมิติ
โดยประกอบด้วย Accuracy, Precision, Recall, F1-score และ Confusion Matrix

**Accuracy**
Accuracy คือสัดส่วนของจำนวนตัวอย่างที่โมเดลทำนายถูกต้องเทียบกับจำนวนตัวอย่างทั้งหมด
ใช้เพื่อวัดประสิทธิภาพโดยรวมของโมเดล

$$
Accuracy = \frac{Number\ of\ correct\ predictions}{Total\ number\ of\ samples}
$$

**Precision**
Precision คือสัดส่วนของจำนวนตัวอย่างที่โมเดลทำนายว่าเป็นคลาสหนึ่งและถูกต้อง
เทียบกับจำนวนตัวอย่างทั้งหมดที่โมเดลทำนายว่าเป็นคลาสนั้น แสดงถึงความแม่นยำของการทำนาย

$$
Precision = \frac{TP}{TP + FP}
$$

**Recall**
Recall คือสัดส่วนของจำนวนตัวอย่างของคลาสนั้นที่โมเดลสามารถทำนายได้ถูกต้อง
เทียบกับจำนวนตัวอย่างจริงทั้งหมดของคลาสนั้น แสดงถึงความสามารถของโมเดลในการตรวจจับตัวอย่างของคลาสนั้น

$$
Recall = \frac{TP}{TP + FN}
$$

**F1-score**
F1-score เป็นค่าเฉลี่ยเชิงฮาร์มอนิกระหว่าง Precision และ Recall ใช้เพื่อประเมินสมดุลระหว่างความแม่นยำและความสามารถในการตรวจจับของโมเดล

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

**Confusion Matrix**

เป็นตารางแสดงจำนวนการทำนายถูกและผิดในแต่ละคลาส (TP, FP, TN, FN)
เพื่อวิเคราะห์ความสับสนของโมเดล โดยในงานนี้คำนวณจาก test dataset
เพื่อเปรียบเทียบโมเดลที่ผ่านการ fine-tune ทั้ง 4 โมเดล

---

## 6. Experiment result💭

### 6.1 แสดงกราฟการ Train : Overfit and Underfit

* ผลการทดลองที่แสดงผลการ Train แต่ละ Model ที่ค่า Parameter ที่ดีที่สุดโดยดูค่า Accuracy และ Loss ระหว่างการ Train และ Validation 

Train data : 
* จากกราฟ Training พบว่า **train accuracy** และ **train F1-score** เพิ่มขึ้นอย่างต่อเนื่อง แสดงว่าโมเดลสามารถเรียนรู้ลักษณะของข้อมูลได้ดี ขณะเดียวกัน **train loss** ลดลงอย่างสม่ำเสมอ

Validation data : 
* จากกราฟ Validation พบว่า **validation accuracy** และ ***validation F1-score** สอดคล้องกับกราฟ Train คือมีค่าสูงขึ้น อยู่ประมาณ 72.9%–81.5% ของแต่ละ Model ขณะที่ **validation loss** ลดลงอย่างรวดเร็วในช่วงแรกก่อนจะเริ่มทรงตัว บาง Model สังเกตเห็นค่า loss เพิ่มสูงขึ้นในช่วงท้ายทำให้เกิด overfit เช่น EfficientNet-B0 และRestNet (ทำการ stop loss ด้วย early stop point)ไม่ได้ทำการ Checkpoint ที่ตำแหน่ง loss ต่ำสุด) 

การปรับ **Hyperparameter** ทำผ่านตัวแปร `config` ซึ่งสามารถมาจากระบบทำ sweep
เช่น `wandb.sweep`
ทำให้โครงสร้างของ classifier **เปลี่ยนได้แบบ dynamic ตามค่า hyperparameter**


> **รูปที่ 6.1-1** กราฟ Accuracy เปรียบเทียบช่วงการ Train และการ Validation ของ EfficientNetB0
> ![image](https://hackmd.io/_uploads/HkH9_Sd9Wx.png)
 
> **รูปที่ 6.1-2** กราฟ Loss เปรียบเทียบช่วงการ Train และการ Validation ของ EfficientNetB0
> ![image](https://hackmd.io/_uploads/rkiJtSd5Ze.png)

> **รูปที่ 6.1-3** กราฟ Accuracy เปรียบเทียบช่วงการ Train และการ Validation ของ MobileNet
>![image](https://hackmd.io/_uploads/BJB3mKFcZx.png)

> **รูปที่ 6.1-4** กราฟ Loss เปรียบเทียบช่วงการ Train และการ Validation ของ MobileNet
>![image](https://hackmd.io/_uploads/SkoWQtYqZl.png)

> **รูปที่ 6.1-5** กราฟ Accuracy เปรียบเทียบช่วงการ Train และการ Validation ของ ResNet18
>![image](https://hackmd.io/_uploads/BJl35O_qWg.png)

> **รูปที่ 6.1-6** กราฟ Loss เปรียบเทียบช่วงการ Train และการ Validation ของ ResNet18
>![image](https://hackmd.io/_uploads/SysX6du9-g.png)

> **รูปที่ 6.1-7** กราฟ Accuracy เปรียบเทียบช่วงการ Train และการ Validation ของ ConvNeXt-Tiny
> ![image](https://hackmd.io/_uploads/Bk99YVd9Zl.png)

> **รูปที่ 6.1-8** กราฟ Loss เปรียบเทียบช่วงการ Train และการ Validation ของ ConvNeXt-Tiny
> ![image](https://hackmd.io/_uploads/Hk9z5VO5Wg.png)

### 6.2 Evaluation Metrics สำหรับ 4 Model

จาก 5 การทดลองในแต่ละ model จะได้ผล evaluation และค่าทางสถิติดังนี้

> **ตารางที่ 6.2-1** ตารางแสดง Score ทั้งหมดของทุก Model Architecture
> | Model | Precision | Recall | F1 | Accuracy |
> |:---|:---:|:---:|:---:|:---:|
> | EfficientNetB0 | 73.3% ± 1% | 72.9% ± 1% | 72.7% ± 1% | 72.9% ± 1% |
> | MobileNet | 81.1% ± 1% | 80.8% ± 1% | 80.6% ± 1%| 80.8% ± 1%|
> | ResNet18 |80.5% ± 2%  |80.4% ± 2% | 80.2% ± 2% |80.4% ± 2%|
> | ConvNeXt-Tiny |81.8% ± 1% | 81.5% ± 1%| 81.4% ± 1% | 81.5% ± 1% |

* จากผลการทดลองจะเห็นว่า Model โมเดล ConvNeXt-Tiny ได้ค่า F1 score 81.4% สูงที่สุด ที่ทำการ Train หลัง Fine-Tuned แล้ว
* ดังนั้นสรุปว่า Model : ConvNeXt-Tiny เหมาะสมที่สุด สำหรับการทำนายเพื่อแยกหน้า pizza จำนวน 4 หน้า (Hawaiian, Margherita, Pepperoni, Seafood)


> **รูปที่ 6.2-1** กราฟ Confusion Matrix ของทุก Model Architecture
> ![image](https://hackmd.io/_uploads/S1yzpKF5Wg.png)

> **ตารางที่ 6.2-2** การประเมิน F1 Score แยก Class จาก Confusion Matrix table
> | Class | EfficientNet | MobileNet | ResNet18 | ConvNeXt-Tiny |
> |:---|:---:|:---:|:---:|:---:|
> | Hawaiian | 73% |85% | 85% | 83% |
> | **Margherita** | **59%** |**75%** | **73%** | **77%** |
> | Pepperoni | 75% | 82% | 82% | 86% |
> | Seafood | 82% | 84% | 81% | 80% |

* จาก รูปที่ 6.2-1 และ Table 6.2-2 จะเห็นได้ว่าแต่ละคลาสมีประสิทธิภาพในการจำแนกที่แตกต่างกันอย่างชัดเจน โดยเฉพาะคลาส Margherita ซึ่งมีค่า F1-score ต่ำที่สุด (59%) เมื่อเทียบกับคลาสอื่นๆ
แสดงให้เห็นว่าโมเดลยังไม่สามารถเรียนรู้ลักษณะเฉพาะของคลาสนี้ได้อย่างมีประสิทธิภาพ

* เมื่อพิจารณาร่วมกับ Confusion Matrix พบว่า คลาส Margherita
มีแนวโน้มถูกทำนายผิดไปเป็นคลาสอื่นในสัดส่วนที่ค่อนข้างสูง โดยเฉพาะคลาสที่มีลักษณะใกล้เคียง
เช่น Pepperoni

* ในขณะที่คลาสอื่น เช่น Seafood และ Pepperoni มีค่า F1-score ค่อนข้างสูง
แสดงให้เห็นว่าโมเดลสามารถแยกแยะลักษณะของคลาสเหล่านี้ได้ดีกว่า
อาจเนื่องมาจากความแตกต่างของ feature ที่ชัดเจนกว่าเมื่อเทียบกับคลาสอื่น

ดังนั้น ปัญหาหลักของโมเดลในงานนี้คือการจำแนกคลาส Margherita ซึ่งอาจมีสาเหตุจาก:

- ความคล้ายคลึงของลักษณะภาพกับคลาสอื่น (feature overlap)
- ลักษณะเด่น (discriminative features) ของคลาสยังไม่ชัดเจนสำหรับโมเดล


### 6.3 Set hypothesis and p-value

* ในการประเมินประสิทธิภาพของโมเดลทั้ง 4 โมเดล (EfficientNetB0, MobileNet, ResNet18 และ ConvNeXt-Tiny) ทีมได้ทำการทดสอบซ้ำจำนวน 5 รอบ เพื่อลดความลำเอียงจากการสุ่มค่าเริ่มต้น
* การทดสอบสมมติฐานทางสถิติใช้หลักการ Non-parametric เนื่องจากขนาดตัวอย่างในแต่ละกลุ่มมีขนาดเล็ก ($n=5$) โดยมีผล evaluation F1-score ตามตารางที่ 6.3-1

**ตารางที่ 6.3-1** F1 Score สำหรับทำ Hypothesis ของแต่ละโมเดล
 | Model | ครั้งที่ 1 | ครั้งที่ 2 | ครั้งที่ 3 | ครั้งที่ 4 | ครั้งที่ 5 |
 |:---|:---:|:---:|:---:|:---:|:---:|
 | EfficientNetB0 | 72.2% | 70.9% | 73.1% | 74.3% | 72.8% |
 | MobileNet V3L | 81.3% |81.3% | 79.8% | 81.4% | 79.3% |
 | ResNet18 | 78.5% | 82.4% | 77.1% | 83.5% | 80.2% |
 | ConvNeXt-Tiny | 80.0% | 81.9% | 81.9% | 82.0% | 81.4% |
 
#### 6.3.1 การทดสอบสมมติฐาน
##### 6.3.1.1 การทดสอบ Kruskal-Wallis (Omnibus Test)
เป็นการทดสอบเพื่อตอบคำถามว่า "ในภาพรวม ทั้ง 4 โมเดลมีประสิทธิภาพต่างกันจริงหรือไม่?"
สมมติฐานหลัก (กำหนดระดับนัยสำคัญ = 0.05)

- $H_0$: มัธยฐานของค่า F1-score ของโมเดลไม่แตกต่างกัน
- $H_1$: มีอย่างน้อย 1 โมเดลที่มีมัธยฐานของค่า F1-score  แตกต่างอย่างมีนัยสำคัญ

##### 6.3.1.2 การทดสอบรายคู่ด้วย Dunn's Test (Post-hoc Test)

เป็นการทดสอบว่า "โมเดลคู่ไหนบ้างที่ต่างกันจริง?" (ทำเมื่อส่วนที่ 1 มีนัยสำคัญ)
สมมติฐานหลัก (กำหนดระดับนัยสำคัญ = 0.05)

- $H_0$: มัธยฐานของค่า F1-score  ระหว่างโมเดลคู่ที่เปรียบเทียบไม่แตกต่างกัน
- $H_1$: มัธยฐานของค่า F1-score  ระหว่างโมเดลคู่ที่เปรียบเทียบ
มีความแตกต่างกันอย่างมีนัยสำคัญทางสถิติ

#### 6.3.2 ผลการทดสอบสมติฐานและการวิเคราะห์ผลการทดสอบ

##### 6.3.2.1 ผลการทดสอบ Kruskal-Wallis
* H-statistic: 11.8914
* p-value: 0.00776

##### 6.3.2.2 ผลการทดสอบรายคู่ (Dunn's Test)

> **ตารางที่ 6.3-2** ตารางแสดงผลการทดสอบ Dunn's Test Hypothesis
>| Model | ConvNeXt-Tiny | EfficientNetB0 | MobileNet V3L | ResNet18 |
> |:---|:---:|:---:|:---:|:---:|
> | ConvNeXt-Tiny | 1.000000 | 0.006671 | 1.000000 | 1.000000 |
> | EfficientNetB0 | 0.006671 | 1.000000 | 0.170475 | 0.061779 |
> | MobileNet  | 1.000000 | 0.170475 | 1.000000 | 1.000000 |
> | ResNet18 | 1.000000 | 0.061779 | 1.000000| 1.000000 |

> **รูปที่ 6.3-1** กราฟ Box plot ของมัธยฐานของแต่ละโมเดล
> ![image](https://hackmd.io/_uploads/SJl8vlocbl.png)

---


## 7. Discussion and Conclusion📝


### 7.1 Refer 6.3 (Hypothesis)

จากการทดสอบด้วยสถิติ Kruskal-Wallis Test เพื่อเปรียบเทียบค่ามัธยฐานของ F1-Score จากทั้ง 4 โมเดล
พบว่าค่าสถิติทดสอบ $H = 11.8914$ และมีค่า $p\text{-value} = 0.00776
ซึ่งมีค่าน้อยกว่าระดับนัยสำคัญที่กำหนด (\alpha = 0.05)
สรุปผล : ปฏิเสธสมมติฐานหลัก (H_0$)

สรุปได้ว่า มี model อย่างน้อยหนึ่งชนิดที่ให้ประสิทธิภาพความแม่นยำแตกต่างจากกลุ่มอื่นอย่างมีนัยสำคัญทางสถิติ เพื่อระบุความแตกต่างอย่างเจาะจง ทีมได้ทำการทดสอบรายคู่ด้วย Dunn's Test
จากตารางผลลัพธ์สามารถจำแนกกลุ่มประสิทธิภาพได้ดังนี้

**คู่ที่มีความแตกต่างอย่างมีนัยสำคัญ ($p < 0.05$):**
* ConvNeXt-Tiny vs EfficientNetB0 : ConvNeXt-Tiny มีประสิทธิภาพสูงกว่าอย่างชัดเจน

**คู่ที่ไม่มีความแตกต่างอย่างมีนัยสำคัญ ($p > 0.05$):**
* ResNet18 vs EfficientNetB0: ResNet18 แม้ค่า $p$ จะค่อนข้างน้อยและเกือบถึงระดับนัยสำคัญ
แต่ในเชิงสถิติยังถือว่าไม่แตกต่างกันอย่างเด็ดขาด
* MobileNet vs EfficientNetB0: ไม่พบความแตกต่างอย่างมีนัยสำคัญ
* ConvNeXt-Tiny vs ResNet18 :  ค่า $p$ ที่เท่ากับ 1.00 บ่งชี้ว่าอันดับประสิทธิภาพของโมเดลกลุ่มนี้
มีความใกล้เคียงกันมากจนไม่สามารถแยกความแตกต่างได้
* ResNet18 vs MobileNet : ไม่พบความแตกต่างอย่างมีนัยสำคัญสรุปเชิงสถิติ

โมเดลสามารถแบ่งออกเป็น 2 กลุ่มหลัก คือ
* กลุ่มประสิทธิภาพสูง ประกอบด้วย ConvNeXt-Tiny, ResNet18 และ MobileNet
โดยที่ **ConvNeXt-Tiny เป็นโมเดลที่ดีที่สุด**
* กลุ่มประสิทธิภาพน้อยที่สุดในการทดลองนี้คือ EfficientNetB0 ซึ่งให้ค่าความแม่นยำต่ำที่สุด ดังนั้น **EfficientNetB0 เป็นโมเดลที่แย่ที่สุด** ในกลุ่มการทดลองครั้งนี้

จึงสรุปได้ว่า ConvNeXt-Tiny เป็นโมเดลที่เหมาะสมที่สุด สำหรับการแก้ปัญหาโจทย์จำแนกประเภทหน้าของพิซซ่าทั้ง 4 หน้า ซึ่งสอดคล้องกับการเปรียบเทียบ F1 Score จากค่าเฉลี่ยรายโมเดล

### 7.2 การวิเคราะห์ค่าผลการทดลอง 

* จากการทดลองพบประเด็นที่น่าสนใจการทำนายหน้า Margherita ให้ค่า F1 score ต่ำในทุก modelเนื่องจากมีการซ้อนทับกันขององค์ประกอบ เช่น ผักหรือชีส ซึ่งเป็นวัตถุดิบที่มักถูกนำไปใช้เป็นส่วนประกอบเสริมในพิซซ่าหน้าอื่นๆ เช่นกัน ทำให้การมี "ผัก" หรือ "ชีส" อยู่ในเกือบทุกหน้าพิซซ่า
อาจทำให้เกิดสัญญาณรบกวน **(Noise)** ในการเรียนรู้ของโมเดล อาจทำให้เกิดความสับสน
* หากโมเดลยึดติดกับลักษณะสีเขียวของผักมากเกินไป ความสับสนนี้เกิดจากลักษณะของ Color และ Texture Overlap เนื่องจากซอสมะเขือเทศใน Margherita และแผ่น Pepperoni มีโทนสีแดงที่ใกล้เคียงกัน ประกอบกับการกระจายตัวของชีสที่มีลักษณะเป็นวงขาวคล้ายกันในบางมุมกล้อง ทำให้โมเดลเกิดความสับสนในเชิงพื้นที่
* แนวทางการแก้ไข : ทำ Preprocess ของภาพให้ focus หรือมีหน้า pizza   มากที่สุด ลดผลของ noise และ/หรือเพิ่มการทำ Augmentation : RandomResizedCrop หรือ Color Jitter ,Padding & Square Resize

### 7.3 การวิเคราะห์ความสมเหตุสมผลของโมเดลด้วย Grad-CAM และ Eyeball Analysis

ในส่วนนี้เป็นการประยุกต์ใช้เทคนิค Explainable AI ผ่าน Grad-CAM เพื่อ "ส่องสมองโมเดล" ว่าโมเดล (จากภาพเป็นตัวอย่าง ConvNeXt-Tiny เนื่องจากเป็น Best Model)ใช้บริเวณใดของภาพในการตัดสินใจจำแนกประเภทหน้าพิซซ่า และเปรียบเทียบกับวิธีที่มนุษย์ใช้แยกแยะ เพื่อยืนยันว่าโมเดลเรียนรู้จากลักษณะเด่นจริง

> **รูปที่ 7.3-1** กราฟการวิเคราะห์ Grad-CAM
> ![image](https://hackmd.io/_uploads/H1bs9xj9Wl.png)


#### 7.3.1 การวิเคราะห์พฤติกรรมการตัดสินใจ

จากการทดสอบพบว่าโมเดลมีพฤติกรรมการรับรู้ที่น่าสนใจ ดังนี้:
* คลาส Hawaiian: จุดโฟกัสอยู่ที่ชิ้น "สับปะรด" คือจุดที่แตกต่างจากพิซซ่าหน้าอื่น
* คลาส Pepperoni: โมเดลแสดงจุดโฟกัส (Heatmap) อย่างชัดเจนบริเวณวัตถุดิบที่มีลักษณะเป็น "วงกลมสีแดง"
* คลาส Margherita: จุดโฟกัสส่วนใหญ่อยู่ที่บริเวณ "สีเขียวของผัก" ซึ่งเป็นองค์ประกอบหลักที่ใช้แยกแยะคลาสนี้นออกจากคลาสอื่น
* คลาส Seafood: แม้โมเดลจะทำนายได้ถูกต้องด้วยค่าความเชื่อมั่นสูงถึง 80.0%
แต่จากการวิเคราะห์ Grad-CAM พบข้อสังเกตที่ผิดปกติ (Abnormality Observation)
คือจุดโฟกัสไปกระจุกตัวอยู่บริเวณ "ขอบของรูปภาพ" แทนที่จะเป็นบริเวณกึ่งกลางของหน้าพิซซ่า
ซึ่งสะท้อนให้เห็นถึงความสำคัญของขั้นตอน Data Centering
และเป็นบทเรียนในการปรับปรุง Augmentation Strategy ในอนาคต
เพื่อให้โมเดลโฟกัสที่จุดศูนย์กลางของวัตถุได้ดียิ่งขึ้น

#### 7.3.2 การวิเคราะห์เปรียบเทียบระหว่างมนุษย์และโมเดล (Eyeball Analysis)

จากการเปรียบเทียบพบว่าโมเดลมีความพยายามเลียนแบบการมองเห็นของมนุษย์ในบางส่วน
เช่น การหาวัตถุดิบชิ้นเด่น (Toppings) อย่างไรก็ตาม ในกรณีที่จุดโฟกัสอยู่ที่ขอบภาพ
สันนิษฐานว่าเกิดจากขั้นตอน Preprocessing (Random Resized Crop)
ที่อาจทำให้จุดโฟกัสหลักของภาพถูกร่นไปอยู่ชิดขอบ ส่งผลให้โมเดลต้องเรียนรู้จากพื้นที่ส่วนที่เหลือ
ซึ่งอาจเป็นจุดที่ "เปราะบาง" หากนำไปใช้กับภาพที่มีการจัดวางองค์ประกอบแตกต่างออกไป

#### 7.3.3 แนวทางเชิงกลยุทธ์ในการปรับปรุงโมเดล (Strategic Model Improvements)

เพื่อให้โมเดลมีความแม่นยำและเสถียร (Robustness) มากยิ่งขึ้น ทีมวิจัยขอเสนอแนวทางดังนี้:

* ด้านการเตรียมข้อมูล (Data-Centric)
    * Center-Weighted Cropping: ปรับปรุงขั้นตอนการ Crop ภาพให้เน้นบริเวณกึ่งกลาง
เพื่อให้โมเดลโฟกัสที่จุดสำคัญของหน้าพิซซ่าได้ดีขึ้น
    * Class-Specific Augmentation: เพิ่มชุดข้อมูลในคลาสที่โมเดลยังสับสน (เช่น Margherita)
    ผ่านเทคนิค Data Augmentation เพื่อเพิ่มความหลากหลายของลักษณะเด่น

* ด้านสถาปัตยกรรมและการเทรน (Model-Centric):

    * Fine-tune Deeper Layers: พิจารณาการปลดล็อค (Unfreeze)
ชั้น Convolutional บล็อกท้ายๆ ของ ConvNeXt-Tiny เพื่อให้โมเดลเรียนรู้ลักษณะเฉพาะ
(Fine-grained features) ของหน้าพิซซ่าได้ลึกซึ้งกว่าเดิม
    * Advanced Loss Function: พิจารณาใช้เทคนิค Class Weighting หรือ Focal Loss
เพื่อแก้ปัญหาความไม่สมดุลของข้อมูลการเรียนรู้ (Imbalanced Data Learning)
และลดผลกระทบจากคลาสที่ทำนายได้ยาก (Hard examples)

สรุป
* ถึงแม้ว่า ConvNeXt-Tiny จะให้ผลลัพธ์ที่ดีที่สุดในเชิงสถิติ แต่การใช้ Grad-CAM ช่วยให้เราเห็นโอกาสในการพัฒนา
* โดยเฉพาะการปรับปรุงภาพอินพุตให้มีจุดโฟกัสที่สมบูรณ์จะช่วยยกระดับความแม่นยำของโมเดลสู่ระดับที่ใช้งานจริงได้อย่างมีประสิทธิภาพยิ่งขึ้น

---

## 8. References🌐
>
## Citing
>
## 👥 Members, Percent Contribution and Responsibility

|No|Pic |ID|	Name| % Contribution	|Responsibility|
|--|--|--|-----|---------------|--------------|
|1.| |6710421003|	Phongsathon Int.|25%|Finetuned Model (ResNet18),Discussed,Report Making|
|2.| |6720422003| Artittaya P.	|25%| Image collection, Fine-tune Model (ConvNeXt), Report making |
|3.| |6720422017|	Siriwat C.|25%|Fine-tune Model (MobileNet),Discussed, Report Making, Inspect Report|
|4.| |6720422030|	Noppawat C.|25%|Create master Python code<br>Finetuned Model (EfficientNet),Discussed, Report making|

## 🖇️End Credit
| ลำดับ | เว็บไซต์แหล่งที่มา | จำนวนรูปภาพ | สัดส่วน (%) |
|--|--|-----|---------------|
| 1 | www.alamy.com | 447 | 44.7% |
| 2 | depositphotos.com | 220 | 22.0% |
| 3 | www.shutterstock.com | 107 | 10.7% |
| 4 | www.freepik.com | 33 | 3.3% |
| 5 | www.dreamstime.com | 27 | 2.7% |
| 6 | www.123rf.com | 27 | 2.7% |
| 7 | www.gettyimages.com | 24 | 2.4% |
| 8 | www.canstockphoto.com | 22 | 2.2% |
| 9 | www.istockphoto.com | 22 | 2.2% |
| 10 | www.vectorstock.com | 18 | 1.8% |
| 11 | www.foodiesfeed.com | 11 | 1.1% |
| 12 | www.pexels.com | 9 | 0.9% |
| 13 | www.pixabay.com | 9 | 0.9% |
| 14 | www.unsplash.com | 7 | 0.7% |
| 15 | www.burst.shopify.com | 5 | 0.5% |
| 16 | www.kaboompics.com | 5 | 0.5% |
| 17 | www.stocksnap.io | 4 | 0.4% |
| 18 | www.gratisography.com | 3 | 0.3% |
|  | รวมทั้งสิ้น | 1000 | 100.0% |

## Reference

### ConvNeXt

Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 11976-11986).

### ResNet

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770-778).

### MobileNet

Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 1314-1324).

### EfficientNet

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In International Conference on Machine Learning (ICML) (pp. 6105-6114). PMLR.

---

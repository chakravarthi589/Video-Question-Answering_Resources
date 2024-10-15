<h1 align="center"> Video-Question-Answering (VideoQA) Resources </h1>

The Video-Question-Answering-Resources repository is a curated guide for beginners and researchers interested in the Video Question Answering (VQA) field. It provides an organized collection of the most relevant papers, models, datasets, and additional resources to help users understand and contribute to this evolving area. The repository focuses on the intersection of computer vision and natural language processing, particularly how video data can be used to answer complex questions, offering a range of materials from introductory guides to advanced research. (Last Update on 10/15/2024)

## Keywords: 
Video question answering (VideoQA), LLMs, Long video understanding, Spatial Reasoning, Temporal Reasoning, Multi-Choice QA, Open-Ended QA;

## Curators:
[ Bharatesh Chakravarthi, Ph.D](https://chakravarthi589.github.io/)
</br>
[Joseph Raj Vishal](https://github.com/joe-rabbit)

---

- [**Beginners Guide to Video-Question-Answering**](#Beginners-Guide-to-Video-Question-Answering) <br>
- [**Publications**](#Publications) <br/>
  - Survey/Review Papers
  - Conference/Journal Papers 
- [**Datasets**](#Datasets) <br>
- [**Models**](#Models) <br>
- [**Additional Resources**](#Additional-Resources) <br>

  
---
## Beginners Guide to Video Question Answering

1. **[Answering Questions from YouTube Videos with OpenAI Whisper and GPT-4 (Medium article)](https://medium.com/@mksupriya2/answering-questions-from-youtube-videos-with-openai-whisper-and-gpt-4-9a0ae11389ba)**

2. **[Try a quick example on how to use LLMs for Video Question Answering here](https://colab.research.google.com/drive/1qTUr1rYB3L3ZlFyLocWbRKg_HVfLvyvT?usp=sharing)** (Check Additional Resources for API key)
3.  **[Community  Computer Vision Course (Unit 4) MultiModal Models](https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/vlm-intro)**

---
## Publications 
### Survey/Review Papers

- A Survey on Generative AI and LLM for Video Generative Understanding, and Streaming (2024) <a href="https://arxiv.org/abs/2404.16038" target="_blank">[Paper]
- Video Question Answering: a Survey of Models and Datasets (2021) <a href="https://link.springer.com/article/10.1007/s11036-020-01730-0#ref-CR57" target="_blank">[Paper]
- A survey on VQA: Datasets and approaches (2020, ITCA) <a href="https://doi.org/10.1109/ITCA52113.2020.00069" target="_blank">[Paper]

### Conference/Journal Papers
#### 2024
- Align and Aggregate: Compositional Reasoning with Video Alignment and Answer Aggregation for Video Question Answering (**CVPR**) <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_CVPR_2024_paper.pdf" target="_blank">[Paper]
- MVBench: A Comprehensive Multi-modal Video Understanding Benchmark (**CVPR**)  <a href="https://openaccess.thecvf.com/content/CVPR2024/html/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.html" target="_blank">[Paper]
- Can I Trust Your Answer? Visually Grounded Video Question Answering (**CVPR**) <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Can_I_Trust_Your_Answer_Visually_Grounded_Video_Question_Answering_CVPR_2024_paper.pdf" target="_blank">[Paper]
- Event Graph Guided Compositional Spatial-Temporal Reasoning for Video Question Answering <a href="https://doi.org/10.1109/TIP.2024.3358726"  target="_blank">[Paper]
- LVBench: An Extreme Long Video Understanding Benchmark <a href="https://ui.adsabs.harvard.edu/link_gateway/2024arXiv240608035W/doi:10.48550/arXiv.2406.08035" target="_blank">[Paper]
- Short Film Dataset (SFD): A Benchmark for Story-Level Video Understanding <a href="https://arxiv.org/abs/2406.10221"  target="_blank">[Paper]
- Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Input <a href="https://arxiv.org/abs/2408.15542"  target="_blank">[Paper]
- CinePile: A Long Video Question Answering Dataset and Benchmark <a href="https://arxiv.org/abs/2405.08813" target="_blank">[Paper]
- Video-Language Alignment via Spatio-Temporal Graph Transformer <a href="https://arxiv.org/abs/2407.11677" target="_blank">[Paper]
- Neural-Symbolic VideoQA: Learning Compositional Spatio-Temporal Reasoning for Real-world Video Question Answering <a href="https://arxiv.org/abs/2404.04007" target="_blank">[Paper]
- VideoChat: Chat-Centric Video Understanding <a href="https://arxiv.org/abs/2305.06355" target="_blank">[Paper]
- LITA: Language Instructed Temporal-Localization Assistant <a href="https://arxiv.org/html/2403.19046v1" target="_blank">[Paper]
- Sports-QA: A Large-Scale Video Question Answering Benchmark for Complex and Professional Sports <a href="https://arxiv.org/abs/2401.01505" target="_blank">[Paper]
- Videoagent: Long-form Video Understanding with Large Language Model as Agent <a href="https://arxiv.org/abs/2403.10517" target="_blank">[Paper]
- AMEGO: Active Memory from Long EGOcentric Videos <a href="https://arxiv.org/pdf/2409.10917" target="_blank">[Paper]
- Video Instruction Tuning With Synthetic Data <a href="https://arxiv.org/abs/2410.02713" target="_blank">[Paper]

#### 2023
- Open-Vocabulary Video Question Answering: A New Benchmark for Evaluating the Generalizability of Video Question Answering Models (**CVPR**) <a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Ko_Open-vocabulary_Video_Question_Answering_A_New_Benchmark_for_Evaluating_the_ICCV_2023_paper.pdf" target="_blank">[Paper]
- ANetQA: A Large-Scale Benchmark for Fine-Grained Compositional Reasoning Over Untrimmed Videos (**CVPR**) <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_ANetQA_A_Large-Scale_Benchmark_for_Fine-Grained_Compositional_Reasoning_Over_Untrimmed_CVPR_2023_paper.pdf" target="_blank">[Paper]
- Discovering Spatio-Temporal Rationales for Video Question Answering (**ICCV**) <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Li_Discovering_Spatio-Temporal_Rationales_for_Video_Question_Answering_ICCV_2023_paper.html" target="_blank">[Paper]
- Egoschema: A Diagnostic Benchmark for Very Long-Form Video Language Understanding (**NeurIPS**) <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/90ce332aff156b910b002ce4e6880dec-Paper-Datasets_and_Benchmarks.pdf" target="_blank">[Paper]
- Visual Instruction Tuning (**NeurIPS**) <a href="https://arxiv.org/abs/2304.08485" target="_blank">[Paper]
- A Simple LLM Framework for Long-Range Video Question-Answering (Preprint) <a href="https://arxiv.org/abs/2312.17235" target="_blank">[Paper]
- A Large Cross-Modal Video Retrieval Dataset with Reading Comprehension <a href="https://arxiv.org/abs/2305.03347">[Paper]
- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning <a href="https://arxiv.org/abs/2305.06500" target="_blank">[Paper]

#### 2022
- Measuring Compositional Consistency for Video Question Answering (**CVPR**) <a href="https://arxiv.org/abs/2204.07190" target="_blank">[Paper]
- From Representation to Reasoning: Towards Both Evidence and Commonsense Reasoning for Video Question Answering (**CVPR**) <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Li_From_Representation_to_Reasoning_Towards_Both_Evidence_and_Commonsense_Reasoning_CVPR_2022_paper.pdf" target="_blank">[Paper]
- Zero-Shot Video Question Answering via Frozen Bidirectional Language Models (**NeurIPS**) <a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/00d1f03b87a401b1c7957e0cc785d0bc-Paper-Conference.pdf" target="_blank">[Paper]
- Dynamic Spatio-Temporal Modular Network for Video Question Answering <a href="https://dl.acm.org/doi/10.1145/3503161.3548061" target="_blank">[Paper]
- Ego4D:Around the World in 3000 Hours of EgoCentric Video<a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.pdf" target="_blank">[Paper]

#### 2021
- NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions (**CVPR**) <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_NExT-QA_Next_Phase_of_Question-Answering_to_Explaining_Temporal_Actions_CVPR_2021_paper.pdf" target="_blank">[Paper]
- Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling (**CVPR**) <a href="https://doi.ieeecomputersociety.org/10.1109/CVPR46437.2021.00725" target="_blank">[Paper]
- AGQA: A Benchmark for Compositional Spatio-Temporal Reasoning (**CVPR**) <a href="https://openaccess.thecvf.com/content/CVPR2021/html/Grunde-McLaughlin_AGQA_A_Benchmark_for_Compositional_Spatio-Temporal_Reasoning_CVPR_2021_paper.html" target="_blank">[Paper]
- On the Hidden Treasure of Dialog in Video Question Answering (**ICCV**) <a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Engin_On_the_Hidden_Treasure_of_Dialog_in_Video_Question_Answering_ICCV_2021_paper.pdf" target="_blank">[Paper]
- Self-Supervised Pre-training and Contrastive Representation Learning for Multiple-choice Video QA (**AAAI**) <a href="https://doi.org/10.1609/aaai.v35i14.17556" target="_blank">[Paper]
- Hierarchical Conditional Relation Networks for Multimodal Video Question Answering  <a href="https://link.springer.com/article/10.1007/s11263-021-01514-3" target="_blank">[Paper]
- TruMan: Trope Understanding in Movies and Animations <a href="https://doi.org/10.1145/3459637.3482018" target="_blank">[Paper]
- Perceiver IO: A General Architecture for Structured Inputs & Outputs<a href="https://arxiv.org/abs/2107.14795" target="_blank">[Paper]

#### 2020
- BERT Representations for Video Question Answering (**WACV**) <a href="https://openaccess.thecvf.com/content_WACV_2020/html/Yang_BERT_representations_for_Video_Question_Answering_WACV_2020_paper.html" target="_blank">[Paper]
- KnowIT VQA: Answering Knowledge-Based Questions about Videos (**AAAI**) <a href="https://doi.org/10.1609/aaai.v34i07.6713" target="_blank">[Paper]
- Divide and Conquer: Question-Guided Spatio-Temporal Contextual Attention for Video Question Answering (**AAAI**) <a href="https://doi.org/10.1609/aaai.v34i07.6766" target="_blank">[Paper]
- TVQA+: Spatio-Temporal Grounding for Video Question Answering <a href="https://aclanthology.org/2020.acl-main.730/" target="_blank">[Paper]
- Video Question Answering for Surveillance (TechRxiv - Not Peer Reviewed) <a href="https://www.techrxiv.org/users/663145/articles/675946-video-question-answering-for-surveillance" target="_blank">[Paper]
- The MSR-Video to Text Dataset with Clean Annotations <a href="https://doi.org/10.1016/j.cviu.2022.103581" target="_blank">[Paper]
-TVQA: Localized,Compositional Video Question Answering <a href="https://arxiv.org/abs/1809.01696">[Paper]


#### 2019
- EgoVQA: An Egocentric Video Question Answering Benchmark Dataset (**CVPR**) <a href="https://openaccess.thecvf.com/content_ICCVW_2019/papers/EPIC/Fan_EgoVQA_-_An_Egocentric_Video_Question_Answering_Benchmark_Dataset_ICCVW_2019_paper.pdf" target="_blank">[Paper]
- Beyond RNNs: Positional Self-Attention with Co-Attention for Video Question Answering (**AAAI**) <a href="https://doi.org/10.1609/aaai.v33i01.33018658" target="_blank">[Paper]
- Compositional Attention Networks with Two-Stream Fusion for Video Question Answering <a href="https://doi.org/10.1109/TIP.2019.2940677" target="_blank">[Paper]
- Learning to Reason with Relational Video Representation for Question Answering <a href="https://www.researchgate.net/profile/Truyen-Tran-2/publication/334388370_Neural_Reasoning_Fast_and_Slow_for_Video_Question_Answering/links/5d2c386b92851cf44085055d/Neural-Reasoning-Fast-and-Slow-for-Video-Question-Answering.pdf" target="_blank">[Paper]
- Video Question Answering with Spatio-Temporal Reasoning <a href="https://link.springer.com/article/10.1007/s11263-019-01189-x" target="_blank">[Paper]
- Spatio-Temporal Relation Reasoning for Video Question Answering <a href="https://open.library.ubc.ca/media/stream/pdf/24/1.0384578/3" target="_blank">[Paper]
- Moments in Time Dataset: one million videos for event understanding <a href="http://moments.csail.mit.edu/TPAMI.2019.2901464.pdf">[Paper]
  
#### 2018
- Multimodal Dual Attention Memory for Video Story Question Answering (**CVPR**) <a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Kyungmin_Kim_Multimodal_Dual_Attention_ECCV_2018_paper.pdf" target="_blank">[Paper]
- TVQA: Localized, Compositional Video Question Answering <a href="https://aclanthology.org/D18-1167/"  target="_blank">[Paper]
- Explore Multi-Step Reasoning in Video Question Answering <a href="https://dl.acm.org/doi/10.1145/3240508.3240563"  target="_blank">[Paper]
- Towards Automatic Learning of Procedures From Web Instructional Videos <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17344">[Paper]
- Weakly-Supervised Video Object Grounding from Text by Loss Weighting and Object Interaction <a href="http://bmvc2018.org/contents/papers/0070.pdf">[Paper]
- On the effectiveness of task granularity for transfer learning <a href="https://arxiv.org/abs/1804.09235">[Paper]

#### 2017
- A Dataset and Exploration of Models for Understanding Video Data through Fill-in-the-Blank Question Answering (**CVPR**) <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Maharaj_A_Dataset_and_CVPR_2017_paper.pdf" target="_blank">[Paper]
- MarioQA: Answering Questions by Watching Gameplay (**CVPR**) <a href="https://openaccess.thecvf.com/content_ICCV_2017/papers/Mun_MarioQA_Answering_Questions_ICCV_2017_paper.pdf" target="_blank">[Paper]
- Leveraging Video Description to Learn Video Question Answering (**AAAI**) <a href="https://doi.org/10.1609/aaai.v31i1.11238" target="_blank">[Paper]
- Video Question Answering via Gradually Refined Attention over Appearance and Motion <a href="https://dl.acm.org/doi/abs/10.1145/3123266.3123427?casa_token=TPImYXpw2zYAAAAA:-yPqs_YzwkfIBEcYHzs0EAWQcprtt0HYrEugKwiYEFfNZMvZ8WqjtjJKFOX3hVwhmIvck-QCUQbHQw" target="_blank">[Paper]
- DeepStory: Video Story QA by Deep Embedded Memory Networks <a href="https://dl.acm.org/doi/10.5555/3172077.3172168" target="_blank">[Paper]
- Video Question Answering via Hierarchical Spatio-Temporal Attention Networks <a href="https://www.ijcai.org/proceedings/2017/0492.pdf" target="_blank">[Paper]
- The "something something" video database for learning and evaluating visual common sense <a href="https://arxiv.org/abs/1706.04261">[Paper]

#### 2016
- MovieQA: Understanding Stories in Movies through Question-Answering (**CVPR**) <a href="https://openaccess.thecvf.com/content_cvpr_2016/html/Tapaswi_MovieQA_Understanding_Stories_CVPR_2016_paper.html" target="_blank">[Paper]
- MSR-VTT: A Large Video Description Dataset for Bridging Video and Language (**CVPR**) <a href="https://doi.org/10.1109/CVPR.2016.571" target="_blank">[Paper]
- TGIF: A New Dataset and Benchmark on Animated GIF Description (**CVPR**) <a href="https://openaccess.thecvf.com/content_cvpr_2016/papers/Li_TGIF_A_New_CVPR_2016_paper.pdf" target="_blank">[Paper]

---
## Datasets
| Year | Name | Key Features |
|------|------|----------|
| 2024 | [CinePile](https://ruchitrawal.github.io/cinepile/) | The **CinePile** dataset consists of **9,396 movie clips** sourced from the Movieclips YouTube channel, divided into training and testing splits of **9,248** and **148 videos**, respectively. Through a question-answer generation and filtering pipeline, the dataset produced **298,888 training points** and **4,940 test-set points**, averaging **32 questions per video scene**.|
| 2023 | [TextVR](https://github.com/callsys/TextVR) | The **TextVR** dataset is a large-scale cross-modal video retrieval dataset, containing **42,200 sentence queries** for **10,500 videos** across **eight scenario domains**, including Street View, Game, Sports, Driving, Activity, TV Show, and Cooking. |
| 2023 | [VideoChat](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) | **VideoChat** is a video-centric multimodal instruction data based on WebVid-10M. The project features a **100K video-instruction** dataset created using  human-assisted and semi-automatic annotation techniques.|
| 2022 | [Ego4D](https://ego4d-data.org/#intro) | **Ego4D** is a comprehensive egocentric video dataset comprising **3,670 hours** of daily-life activities recorded by **931 camera wearers** across **74 locations** in **9 countries**, covering various scenarios like household, outdoor, and workplace settings.  |
| 2021 | [NExTQA](https://github.com/doc-doc/NExT-QA) | The **NExT-QA** dataset comprises **5,440 videos**, split into **3,870** for training, **570** for validation, and **1,000** for testing. It features around **52,044 question-answer** pairs, with approximately **47,692** for multiple-choice QA and **52,044** for open-ended QA. The questions are divided into three main types: *causal questions* (48% of the dataset), *temporal questions (29%)*, and *descriptive questions (23%)*. |
| 2021 | [LSMDC-QA](https://sites.google.com/site/describingmovies/download?authuser=0) (Requires request access) | **LSMDC-QA** (Large Scale Movie Description Challenge) contains **118,081 short video clips** extracted from **202 movies**. It consists of **7408** clips, and evaluation is performed on a test set of **1000 videos** from movies disjoint. |
| 2019 | [Moments in Time Dataset](http://moments.csail.mit.edu/#) | **The Moments in Time dataset** consists of one million videos, each 3 seconds long, with 339 different classes.  |
| 2018 | [TVQA](https://github.com/jayleicn/TVQA/tree/master?tab=readme-ov-file) | **TVQA** is a large-scale video question-answering dataset built from six popular TV shows, including *Friends*, *The Big Bang Theory*, and *How I Met Your Mother*. It contains **152.5K QA** pairs sourced from **21.8K video clips**, covering over **460 hours** of content.|
| 2018 | [YouCook2](http://youcook2.eecs.umich.edu/) | **YouCook2** is one of the largest instructional video datasets focused on task-oriented cooking, featuring **2,000 untrimmed videos** from **89 recipes**, with an average of **22 videos** per recipe. Each video, averaging **5.26 minutes** and totaling **176 hours**, includes annotated procedure steps with their corresponding temporal boundaries. |
| 2018 | [TVQA+](https://github.com/jayleicn/TVQAplus) | **TVQA+** includes **29.4K multiple-choice questions** grounded in both temporal and spatial domains. To collect spatial groundings, a set of visual concept words—objects and people—are identified, and corresponding object regions in individual frames are annotated with bounding boxes.  |
| 2017 | [TGIF-QA](https://github.com/YunseokJANG/tgif-qa) | **TGIF-QA**, a large-scale dataset, contains **165K question-answer** pairs based on animated GIFs, testing video-based Visual Question Answering (VQA) across four question types: Repetition Count, Repeating Action, State Transition, and Frame QA.|
| 2017 | [MarioQA](https://github.com/JonghwanMun/MarioQA) | **MarioQA** is a dataset specifically designed for video-based question-answering in the context of *Super Mario Bros.* gameplay, containing over **70,000 question-answer pairs** linked to gameplay footage.|
| 2017 | [Something-Something v1 & v2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) | **Something-Something** is a collection of 220,847 labeled video clips of humans performing predefined basic actions with everyday objects. The dataset comprises **220,847 videos** divided into a training set of **168,913**, a validation set of **24,777**, and a test set of **27,157 (without labels)**, with a total of **174 unique labels**.|
| 2016 | [MSVD-QA](https://github.com/xudejing/video-question-answering?tab=readme-ov-file) | The **MSVD-QA** dataset is a Video Question Answering (VideoQA) dataset derived from the **Microsoft Research Video Description (MSVD)** dataset, which includes around **120K sentences** describing over **2,000 videos** snippets.The dataset includes **1,970 video clips** and approximately **50.5K QA pairs**. |
| 2016 | [MSRVTT-QA](https://github.com/xudejing/video-question-answering?tab=readme-ov-file) | **MSRVTT-QA** consists of **10K web video clips** with a total duration of **41.2 hours**. It spans **200k clip-sentence pairs**. Each video clip is annotated with about **20 natural sentences.** |
| 2016 | [MovieQA](https://github.com/makarandtapaswi/MovieQA_benchmark?tab=readme-ov-file) | The **MovieQA dataset** is designed for movie question answering, aimed at evaluating automatic story comprehension through both video and text. It contains nearly **15,000 multiple-choice questions** derived from over **400 movies**.|
| 2016 | [PororoQA](https://github.com/Kyung-Min/PororoQA) | The **Pororo** dataset based on children's cartoons features a simple story structure with episodes averaging **7.2 minutes**, where similar events are frequently repeated. The dataset comprises **8,834 QA pairs**, with an average of **51.66 questions per episode**, excluding ambiguous or unrelated questions. |
| 2014 | [Activity Net](http://activity-net.org/download.html) | **ActivityNet** is a large-scale video benchmark for human activity understanding. ActivityNet aims to cover a wide range of complex human activities. ActivityNet provides samples from **203 activity classes** with an average of **137 untrimmed videos** per class and **1.41 activity instances** per video, for a total of **849 video hours**. |

---
## Models
## Open Source Models
| Model Name  | Links |
|-------------|-------------------------------|
| InternVL | [Hugging Face](https://huggingface.co/OpenGVLab/InternVL2-76B) , [GitHub](https://github.com/OpenGVLab/InternVL) |
| LLaVa | [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/llava) , [GitHub](https://github.com/haotian-liu/LLaVA) |
| LITA | [GitHub](https://github.com/NVlabs/LITA)|
| End2End ChatBot |[Hugging Face](https://huggingface.co/spaces/OpenGVLab/InternVideo2-Chat-8B-HD) , [GitHub](https://github.com/OpenGVLab/Ask-Anything)|
| VideoLLAMA2 | [Hugging Face](https://huggingface.co/spaces/lixin4ever/VideoLLaMA2), [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA2) |
|FrozenBiLM | [GitHub](https://github.com/antoyang/FrozenBiLM) |
|PercieverIO | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/perceiver),[GitHub](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)|
|InstructBlipVideo |[Hugging Face](https://huggingface.co/docs/transformers/model_doc/instructblipvideo#instructblipvideo) , [GitHub](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) |

---

## Models that Require APIs
| Model Name | API Link |
|------------|----------|
| ChatGPT    | [Here](https://platform.openai.com/api-keys) |
| Gemini |[Here](https://ai.google.dev/gemini-api/docs/vision?lang=python)|
| Llama 3.2|[Here](https://docs.llama-api.com/quickstart#llama-3-2-instruct-chat-models-with-vision)|


---

## Additional-Resources

1. **[OpenAI Docs](https://platform.openai.com/docs/api-reference/introduction)**
2. **[Gemini Docs](https://ai.google.dev/gemini-api/docs)**
3. **[LLAMA Docs](https://docs.llama-api.com/quickstart)**
---

### :arrow_heading_up: [Back to Top](#Keywords)





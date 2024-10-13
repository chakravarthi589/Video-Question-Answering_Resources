<h1 align="center"> Video-Question-Answering-Resources </h1>

The Video-Question-Answering-Resources repository is a curated guide for both beginners and researchers interested in the field of Video Question Answering (VQA). It provides an organized collection of the most relevant papers, models, datasets, and additional resources to help users understand and contribute to this evolving area. The repository focuses on the intersection of computer vision and natural language processing, particularly how video data can be used to answer complex questions, offering a range of materials from introductory guides to advanced research. (Last Update on 10/11/2024)

### Keywords:

### Curators:
[ Bharatesh Chakravarthi, Ph.D](https://chakravarthi589.github.io/)
</br>
[Joseph Raj Vishal](https://github.com/joe-rabbit)

---


- [**Beginners Guide to Video-Question-Answering**](#Beginners-Guide-to-Video-Question-Answering) <br>
- [**Publications**](#Publications) <br/>
  - Survey/Review Papers
  - Conference/Journal Papers 
- [**Benchmark Datasets**](#Benchmark-Datasets) <br>
- [**Current Models on Hugging Face**](#Current-Models-on-Hugging-Face) <br>
- [**Additional Resources**](#Additional-Resourcese) <br>

  
---
## Beginners Guide to Video Question Answering

1. **[Answering Questions from YouTube Videos with OpenAI Whisper and GPT-4 (Medium article)](https://medium.com/@mksupriya2/answering-questions-from-youtube-videos-with-openai-whisper-and-gpt-4-9a0ae11389ba)**

2. **[Try a quick example on how to use LLMs for Video Question Answering here](https://colab.research.google.com/drive/1qTUr1rYB3L3ZlFyLocWbRKg_HVfLvyvT?usp=sharing)** (Check Additional Resources for API key)
3.  **[Community  Computer Vision Course (Unit 4) MultiModal Models](https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/vlm-intro)**

---
## Publications 
### Survey/Review Papers

- A Survey on Generative AI and LLM for Video Generative Understanding,and Streaming (2024) <a href="https://arxiv.org/abs/2404.16038" target="_blank">[Paper]
- Video Question Answering: a Survey of Models and Datasets (2021) <a href="https://link.springer.com/article/10.1007/s11036-020-01730-0#ref-CR57" target="_blank">[Paper]
- A survey on VQA:Datasets and approaches (2020,ITCA) <a href="https://doi.org/10.1109/ITCA52113.2020.00069" target="_blank">[Paper]

### Conference/Journal Papers
#### 2024

- **CVPR**
  - [Align and Aggregate: Compositional Reasoning with Video Alignment and Answer Aggregation for Video Question Answering](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_CVPR_2024_paper.pdf)
  - [MVBench: A Comprehensive Multi-modal Video Understanding Benchmark](https://openaccess.thecvf.com/content/CVPR2024/html/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.html)
  - [Can I Trust Your Answer? Visually Grounded Video Question Answering](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Can_I_Trust_Your_Answer_Visually_Grounded_Video_Question_Answering_CVPR_2024_paper.pdf)

- **IEEE Transactions**
  - [Event Graph Guided Compositional Spatial-Temporal Reasoning for Video Question Answering](https://doi.org/10.1109/TIP.2024.3358726)

- **Preprints**
  - [LVBench: An Extreme Long Video Understanding Benchmark](https://ui.adsabs.harvard.edu/link_gateway/2024arXiv240608035W/doi:10.48550/arXiv.2406.08035)
  - [Short Film Dataset (SFD): A Benchmark for Story-Level Video Understanding](https://arxiv.org/abs/2406.10221)
  - [Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Input](https://arxiv.org/abs/2408.15542)
  - [CinePile: A Long Video Question Answering Dataset and Benchmark](https://arxiv.org/abs/2405.08813)
  - [Video-Language Alignment via Spatio-Temporal Graph Transformer](https://arxiv.org/abs/2407.11677)
  - [Neural-Symbolic VideoQA: Learning Compositional Spatio-Temporal Reasoning for Real-world Video Question Answering](https://arxiv.org/abs/2404.04007)
  - [VideoChat: Chat-Centric Video Understanding](https://arxiv.org/abs/2305.06355)
  - [LITA: Language Instructed Temporal-Localization Assistant](https://arxiv.org/html/2403.19046v1)
  - [Sports-QA: A Large-Scale Video Question Answering Benchmark for Complex and Professional Sports](https://arxiv.org/abs/2401.01505)
  - [Videoagent: Long-form Video Understanding with Large Language Model as Agent](https://arxiv.org/abs/2403.10517)
  - [AMEGO: Active Memory from Long EGOcentric Videos](https://arxiv.org/pdf/2409.10917)

#### 2023

- **CVPR**
  - [Open-Vocabulary Video Question Answering: A New Benchmark for Evaluating the Generalizability of Video Question Answering Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Ko_Open-vocabulary_Video_Question_Answering_A_New_Benchmark_for_Evaluating_the_ICCV_2023_paper.pdf)
  - [ANetQA: A Large-Scale Benchmark for Fine-Grained Compositional Reasoning Over Untrimmed Videos](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_ANetQA_A_Large-Scale_Benchmark_for_Fine-Grained_Compositional_Reasoning_Over_Untrimmed_CVPR_2023_paper.pdf)

- **ICCV**
  - [Discovering Spatio-Temporal Rationales for Video Question Answering](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Discovering_Spatio-Temporal_Rationales_for_Video_Question_Answering_ICCV_2023_paper.html)

- **NeurIPS**
  - [Egoschema: A Diagnostic Benchmark for Very Long-Form Video Language Understanding](https://proceedings.neurips.cc/paper_files/paper/2023/file/90ce332aff156b910b002ce4e6880dec-Paper-Datasets_and_Benchmarks.pdf)
  - [Visual Instruction Tuning (NeurIPS 2023 - Oral)](https://arxiv.org/abs/2304.08485)

- **Preprints**
  - [A Simple LLM Framework for Long-Range Video Question-Answering (Preprint)](https://arxiv.org/abs/2312.17235)

#### 2022

- **CVPR**
  - [Measuring Compositional Consistency for Video Question Answering](https://arxiv.org/abs/2204.07190)
  - [From Representation to Reasoning: Towards Both Evidence and Commonsense Reasoning for Video Question Answering](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_From_Representation_to_Reasoning_Towards_Both_Evidence_and_Commonsense_Reasoning_CVPR_2022_paper.pdf)

- **NeurIPS**
  - [Zero-Shot Video Question Answering via Frozen Bidirectional Language Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/00d1f03b87a401b1c7957e0cc785d0bc-Paper-Conference.pdf)

- **ACM**
  - [Dynamic Spatio-Temporal Modular Network for Video Question Answering](https://dl.acm.org/doi/10.1145/3503161.3548061)

#### 2021

- **CVPR**
  - [NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions](https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_NExT-QA_Next_Phase_of_Question-Answering_to_Explaining_Temporal_Actions_CVPR_2021_paper.pdf)
  - [Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling](https://doi.ieeecomputersociety.org/10.1109/CVPR46437.2021.00725)
  - [AGQA: A Benchmark for Compositional Spatio-Temporal Reasoning](https://openaccess.thecvf.com/content/CVPR2021/html/Grunde-McLaughlin_AGQA_A_Benchmark_for_Compositional_Spatio-Temporal_Reasoning_CVPR_2021_paper.html)

- **ICCV**
  - [On the Hidden Treasure of Dialog in Video Question Answering](https://openaccess.thecvf.com/content/ICCV2021/papers/Engin_On_the_Hidden_Treasure_of_Dialog_in_Video_Question_Answering_ICCV_2021_paper.pdf)

- **AAAI**
  - [Self-Supervised Pre-training and Contrastive Representation Learning for Multiple-choice Video QA](https://doi.org/10.1609/aaai.v35i14.17556)

- **ACM**
  - [Hierarchical Conditional Relation Networks for Multimodal Video Question Answering](https://link.springer.com/article/10.1007/s11263-021-01514-3)
  - [TruMan: Trope Understanding in Movies and Animations](https://doi.org/10.1145/3459637.3482018)

#### 2020

- **WACV**
  - [BERT Representations for Video Question Answering](https://openaccess.thecvf.com/content_WACV_2020/html/Yang_BERT_representations_for_Video_Question_Answering_WACV_2020_paper.html)

- **AAAI**
  - [KnowIT VQA: Answering Knowledge-Based Questions about Videos](https://doi.org/10.1609/aaai.v34i07.6713)
  - [Divide and Conquer: Question-Guided Spatio-Temporal Contextual Attention for Video Question Answering](https://doi.org/10.1609/aaai.v34i07.6766)

- **ACL**
  - [TVQA+: Spatio-Temporal Grounding for Video Question Answering](https://aclanthology.org/2020.acl-main.730/)

- **Preprints**
  - [Video Question Answering for Surveillance (TechRxiv - Not Peer Reviewed)](https://www.techrxiv.org/users/663145/articles/675946-video-question-answering-for-surveillance)

- **Other**
  - [The MSR-Video to Text Dataset with Clean Annotations](https://doi.org/10.1016/j.cviu.2022.103581)


#### 2019

- **CVPR**
  - [EgoVQA: An Egocentric Video Question Answering Benchmark Dataset](https://openaccess.thecvf.com/content_ICCVW_2019/papers/EPIC/Fan_EgoVQA_-_An_Egocentric_Video_Question_Answering_Benchmark_Dataset_ICCVW_2019_paper.pdf)

- **AAAI**
  - [Beyond RNNs: Positional Self-Attention with Co-Attention for Video Question Answering](https://doi.org/10.1609/aaai.v33i01.33018658)

- **IEEE**
  - [Compositional Attention Networks with Two-Stream Fusion for Video Question Answering](https://doi.org/10.1109/TIP.2019.2940677)

- **Preprint**
  - [Learning to Reason with Relational Video Representation for Question Answering](https://www.researchgate.net/profile/Truyen-Tran-2/publication/334388370_Neural_Reasoning_Fast_and_Slow_for_Video_Question_Answering/links/5d2c386b92851cf44085055d/Neural-Reasoning-Fast-and-Slow-for-Video-Question-Answering.pdf)

- **Other**
  - [Video Question Answering with Spatio-Temporal Reasoning (Springer)](https://link.springer.com/article/10.1007/s11263-019-01189-x)
  - [Spatio-Temporal Relation Reasoning for Video Question Answering (Dissertation, University of British Columbia)](https://open.library.ubc.ca/media/stream/pdf/24/1.0384578/3)
#### 2018

- **CVPR**
  - [Multimodal Dual Attention Memory for Video Story Question Answering](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kyungmin_Kim_Multimodal_Dual_Attention_ECCV_2018_paper.pdf)

- **ACL**
  - [TVQA: Localized, Compositional Video Question Answering](https://aclanthology.org/D18-1167/)

- **ACM**
  - [Explore Multi-Step Reasoning in Video Question Answering](https://dl.acm.org/doi/10.1145/3240508.3240563)

#### 2017

- **CVPR**
  - [A Dataset and Exploration of Models for Understanding Video Data through Fill-in-the-Blank Question Answering](https://openaccess.thecvf.com/content_cvpr_2017/papers/Maharaj_A_Dataset_and_CVPR_2017_paper.pdf)
  - [MarioQA: Answering Questions by Watching Gameplay](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mun_MarioQA_Answering_Questions_ICCV_2017_paper.pdf)

- **AAAI**
  - [Leveraging Video Description to Learn Video Question Answering](https://doi.org/10.1609/aaai.v31i1.11238)

- **ACM**
  - [Video Question Answering via Gradually Refined Attention over Appearance and Motion](https://dl.acm.org/doi/abs/10.1145/3123266.3123427?casa_token=TPImYXpw2zYAAAAA:-yPqs_YzwkfIBEcYHzs0EAWQcprtt0HYrEugKwiYEFfNZMvZ8WqjtjJKFOX3hVwhmIvck-QCUQbHQw)
  - [DeepStory: Video Story QA by Deep Embedded Memory Networks](https://dl.acm.org/doi/10.5555/3172077.3172168)

- **IJCAI**
  - [Video Question Answering via Hierarchical Spatio-Temporal Attention Networks](https://www.ijcai.org/proceedings/2017/0492.pdf)

  #### 2016

- **CVPR**
  - [MovieQA: Understanding Stories in Movies through Question-Answering](https://openaccess.thecvf.com/content_cvpr_2016/html/Tapaswi_MovieQA_Understanding_Stories_CVPR_2016_paper.html)
  - [MSR-VTT: A Large Video Description Dataset for Bridging Video and Language](https://doi.org/10.1109/CVPR.2016.571)
  - [TGIF: A New Dataset and Benchmark on Animated GIF Description](https://openaccess.thecvf.com/content_cvpr_2016/papers/Li_TGIF_A_New_CVPR_2016_paper.pdf)

---
## Datasets
| Name | Features | Link |
|------|----------|------|
|  TGIF-QA    |          |      |
|  MSVD-QA    |          |      |
|  MSRVTT-QA    |          |      |
|  TVQA    |          |      |
|  TVQA+    |          |      |
|  MovieQA   |          |      |
|  EgoQA   |          |      |
|  NExTQA   |          |      |
|  LSMDC-QA   |          |      |
|  PororoQA   |          |      |
|  MarioQA  |          |      |

---
## Models
## Open Source Models
| Model Name  | Links |
|-------------|-------------------------------|
| InternVL | [Hugging Face](https://huggingface.co/OpenGVLab/InternVL2-76B) , [GitHub](https://github.com/OpenGVLab/InternVL) |
| LLaVa | [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/llava) , [GitHub](https://github.com/haotian-liu/LLaVA) |
| LITA | [GitHub](https://github.com/NVlabs/LITA)|
| End2End ChatBot |[Hugging Face](https://huggingface.co/spaces/OpenGVLab/InternVideo2-Chat-8B-HD) , [GitHub](https://github.com/OpenGVLab/Ask-Anything)|
| VideoLLAMA2 | [Hugging Face](https://github.com/DAMO-NLP-SG/VideoLLaMA2) , [GitHub](https://huggingface.co/spaces/lixin4ever/VideoLLaMA2)|

---

## Models that Require APIs
| Model Name | API Link |
|------------|----------|
| ChatGPT    | [Here](https://platform.openai.com/api-keys) |
| Gemini |[Here](https://ai.google.dev/gemini-api/docs/vision?lang=python)|
| Llama 3.2|[Here](https://docs.llama-api.com/quickstart#llama-3-2-instruct-chat-models-with-vision)|


---

## Additional Resources

1. **[OpenAI Docs](https://platform.openai.com/docs/api-reference/introduction)**
2. **[Gemini Docs](https://ai.google.dev/gemini-api/docs)**
3. **[LLAMA Docs](https://docs.llama-api.com/quickstart)**
---

### :arrow_heading_up: [Back to Top](#Keywords)





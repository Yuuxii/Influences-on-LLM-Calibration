# Influences on LLM Calibration

This repository contains the source code for the paper:  
**_Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles_**

---

## 📁 Project Structure

### 🔧 Training the Model
- `src/train.py` — Main script for training models.

### 🧪 Data Preparation & Answer Generation

#### Generate Answers
- **OpenAI-style API generation**:  
  `src/generate_answers_openai_style.py`

- **HuggingFace models**:  
  `src/generate_answers.py`

#### Post-Processing
- `src/post_processing.py` — Clean and structure generated answers.

#### Evaluate Answer Correctness using LLM-as-a-Judge
- `src/evaluate_prom_score.py`

---

## 📖 Acknowledgements

We build upon prior work in LLM calibration, particularly:

```bibtex
@inproceedings{ulmer-etal-2024-calibrating,
  title     = "Calibrating Large Language Models Using Their Generations Only",
  author    = "Ulmer, Dennis and Gubri, Martin and Lee, Hwaran and Yun, Sangdoo and Oh, Seong",
  booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month     = aug,
  year      = "2024",
  address   = "Bangkok, Thailand",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2024.acl-long.824/",
  doi       = "10.18653/v1/2024.acl-long.824",
  pages     = "15440--15459"
}

@inproceedings{tian-etal-2023-just,
  title     = "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback",
  author    = "Tian, Katherine and Mitchell, Eric and Zhou, Allan and Sharma, Archit and Rafailov, Rafael and Yao, Huaxiu and Finn, Chelsea and Manning, Christopher",
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  month     = dec,
  year      = "2023",
  address   = "Singapore",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2023.emnlp-main.330/",
  doi       = "10.18653/v1/2023.emnlp-main.330",
  pages     = "5433--5442"
}
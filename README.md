# Sentiment Analysis of Customer Product Reviews in Bengali Language

A full-stack application for sentiment analysis of customer product reviews written in Bengali language. This project combines **Node.js/Express** backend with **Python machine learning** models to classify customer sentiments.

## 📋 Overview

This project provides a web interface to analyze customer product reviews in Bengali language using advanced NLP and machine learning techniques. It determines whether a review expresses positive, negative, or neutral sentiment.

**Research Paper:** See [Resarch Paper/A Sentiment Analysis Framework for Customer Product Reviews in the Bengali Language.pdf](Resarch%20Paper/A%20Sentiment%20Analysis%20Framework%20for%20Customer%20Product%20%20Reviews%20in%20the%20Bengali%20Language.pdf) for detailed methodology and findings.

## 🛠️ Tech Stack

### Backend (Node.js/Express)

- **Express.js** - Web framework
- **EJS** - Template engine
- **Mongoose** - MongoDB ODM
- **Multer** - File upload handling
- **CSV-Parser** - CSV file parsing
- **XLSX** - Excel file support

### Machine Learning (Python)

- **TensorFlow/Keras** - Deep learning models
- **scikit-learn** - TF-IDF Vectorization and ML utilities
- **pandas** - Data manipulation
- **openpyxl** - Excel file operations

## 📁 Project Structure

```
.
├── app.js                              # Main Express application
├── package.json                        # Node.js dependencies
├── public/                             # Static files
│   ├── CSS/
│   │   └── style.css                  # Styling
│   ├── dependencys/
│   │   └── images/                    # Image assets
│   └── validations/
│       └── validate.js                # Client-side validation
├── views/                             # EJS templates
│   ├── includes/
│   │   ├── nav.ejs                   # Navigation bar
│   │   └── footer.ejs                # Footer
│   ├── layouts/
│   │   └── boilerplate.ejs           # Base template
│   └── pages/
│       ├── landing.ejs               # Home page
│       ├── aboutUs.ejs               # About page
│       ├── featureImportance.ejs     # ML feature importance visualization
│       ├── chi2Features.ejs          # Chi-square feature analysis
│       ├── output.ejs                # Analysis results
│       ├── feedBack.ejs              # User feedback
│       ├── donateUs.ejs              # Donation page
│       └── done.ejs                  # Completion page
├── pythonCodeFolder/                  # ML models and data processing
│   ├── main.py                       # Main Python script
│   ├── ml_models.py                  # ML model implementations
│   ├── requirements.txt              # Python dependencies
│   ├── NLP_project.ipynb             # Jupyter notebook with analysis
│   └── processed/                    # Processed data directory
├── utils/                             # Utility functions
│   ├── asyncAwait.js                # Async/await helpers
│   └── ExpressError.js              # Custom error handling
└── Resarch Paper/                     # Research documentation
    └── A Sentiment Analysis Framework for Customer Product Reviews in the Bengali Language.pdf
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v14 or higher)
- **Python 3.10** (required)
- **MongoDB** (for database)
- **npm** or **yarn**

### Installation

#### 1. Clone/Download the Repository

```bash
cd "Summer 2025/Sentiment Analysis of Customer Product Review in Bengali Language"
```

#### 2. Set Up Python Virtual Environment (First Time Only)

Navigate to the Python folder and create a virtual environment with Python 3.10:

```bash
cd pythonCodeFolder
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error with PowerShell, you may need to run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3. Install Python Dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

Then return to the main directory:

```bash
cd ..
```

#### 4. Install Node.js Dependencies (First Time Only)

```bash
npm install
```

#### 5. Configure Environment

Create a `.env` file in the root directory (if needed for MongoDB connection, API keys, etc.)

### Running the Application

#### First Time Setup

1. **Activate Python Virtual Environment:**

   ```bash
   cd pythonCodeFolder
   .\venv\Scripts\Activate.ps1
   cd ..
   ```

2. **Run npm install (if not done):**
   ```bash
   npm install
   ```

#### Start the Server

From the main project directory (with Python venv still activated):

```bash
node app.js
```

Or with nodemon (if installed):

```bash
nodemon app.js
```

The server will start on `http://localhost:8080/home` (or your configured port)

**Note:** The Python models will be called from the backend as needed for sentiment analysis.

## 🎥 Video Tutorials

- **[How to Run the Project](https://youtu.be/4ZZ0HLH1jk8)** - Step-by-step guide to set up and run the application
- **[Project Demo](https://youtu.be/97hXrV_eF74)** - Live demonstration of the sentiment analysis application in action

## 📊 Features

- **Sentiment Classification** - Analyze Bengali product reviews
- **Feature Importance Visualization** - Display important features in the model
- **Chi-square Analysis** - Statistical feature analysis
- **File Upload Support** - Process CSV/Excel files with reviews
- **User Feedback** - Collect user feedback on predictions
- **Responsive UI** - Mobile-friendly web interface

## 🧠 Machine Learning Models

The project includes trained ML models for Bengali sentiment analysis:

- TensorFlow/Keras neural networks for deep learning
- scikit-learn models with TF-IDF vectorization
- Feature importance analysis and visualization

Training and model details are available in:

- [pythonCodeFolder/NLP_project.ipynb](pythonCodeFolder/Google%20Colab%20Code/NLP_project.ipynb) - Jupyter notebook with full analysis
- [pythonCodeFolder/ml_models.py](pythonCodeFolder/ml_models.py) - Model implementation

## 🎯 Algorithms & Performance Metrics

The project implements and compares multiple machine learning algorithms for sentiment classification:

### Classical Machine Learning Models (TF-IDF Vectorization)

| Algorithm                     | Type           | Metrics                                                              | Notes                                                                    |
| ----------------------------- | -------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Logistic Regression**       | Linear Model   | Precision, Recall, F1-Score (Macro)                                  | Fast and interpretable; computes feature importance through coefficients |
| **Naive Bayes**               | Probabilistic  | Precision: 0.85, Recall: 0.83, F1-Score: 0.84, Test Accuracy: 73.24% | Effective baseline; includes feature log probabilities analysis          |
| **Random Forest**             | Ensemble       | Precision, Recall, F1-Score (Macro)                                  | Provides feature importances; robust to overfitting                      |
| **K-Nearest Neighbors (KNN)** | Distance-Based | Precision, Recall, F1-Score (Macro)                                  | Uses k=3 neighbors; versatile for multi-class classification             |

### Deep Learning Models (Tokenization & Padding)

| Algorithm                              | Type          | Architecture                                                                         | Metrics                             | Notes                                                     |
| -------------------------------------- | ------------- | ------------------------------------------------------------------------------------ | ----------------------------------- | --------------------------------------------------------- |
| **CNN (Convolutional Neural Network)** | Deep Learning | Embedding (64D) → Conv1D (128 filters, kernel=5) → GlobalMaxPooling1D → Dense layers | Precision, Recall, F1-Score (Macro) | Captures local patterns in text; batch_size=32, epochs=25 |
| **LSTM**                               | RNN           | Embedding (64D) → LSTM (64 units) → Dense                                            | Currently Inactive                  | Can be enabled for sequential pattern learning            |

### Evaluation Metrics

All models are evaluated using:

- **Precision (Macro)**: Average precision across all sentiment classes
- **Recall (Macro)**: Average recall across all sentiment classes
- **F1-Score (Macro)**: Harmonic mean of precision and recall
- **Test/Validation Accuracy**: Overall accuracy on the test set

The macro-averaged metrics ensure balanced evaluation across all three sentiment classes (Positive, Negative, Neutral).

### Feature Analysis

- **Chi-Square Analysis**: Applied to top discriminative features for Logistic Regression
- **Feature Importance**: Extracted from:
  - Tree-based models (Random Forest): Feature importances
  - Linear models (Logistic Regression): Coefficient magnitudes
  - Probabilistic models (Naive Bayes): Feature log probabilities

## 📝 Usage

1. **Navigate to the web application** - Open `http://localhost:3000`
2. **Input Review** - Enter a Bengali language product review
3. **Analyze** - Click analyze to get sentiment prediction
4. **View Results** - See sentiment classification (positive/negative/neutral) along with confidence scores
5. **Export Results** - Download analysis results as CSV/Excel

## 📚 Research Paper

For detailed information about the methodology, dataset, models, and results, refer to:

**[A Sentiment Analysis Framework for Customer Product Reviews in the Bengali Language](Resarch%20Paper/A%20Sentiment%20Analysis%20Framework%20for%20Customer%20Product%20%20Reviews%20in%20the%20Bengali%20Language.pdf)**

This paper provides:

- Literature review of sentiment analysis in Bengali
- Dataset description and preprocessing techniques
- Model architecture and training methodology
- Performance evaluation and results
- Discussion and conclusions

## 🔧 Development

### API Endpoints

Check `app.js` for available endpoints. Key routes typically include:

- `GET /` - Landing page
- `GET /aboutUs` - About page
- `POST /analyze` - Analyze sentiment
- `GET /featureImportance` - View feature importance
- `GET /chi2Features` - Chi-square analysis
- `POST /feedback` - Submit user feedback

### Adding New Features

1. Add backend routes in `app.js`
2. Create corresponding EJS templates in `views/pages/`
3. Add frontend validation in `public/validations/validate.js`
4. Update Python models if needed in `pythonCodeFolder/ml_models.py`

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the ISC License - see the [package.json](package.json) for details.

## 👥 Authors

Md. Zubair Rahman

Developed as part of a research project on Bengali Language NLP.

## 📧 Contact & Support

For questions or issues, please refer to the research paper or contact the project maintainers.

## 🙏 Acknowledgments

- Research paper contributors
- Bengali NLP community
- Open-source libraries (TensorFlow, scikit-learn, Express.js)

---

**Last Updated:** February 2026

For more information about the sentiment analysis framework and methodology, please refer to the research paper in the `Research Paper/` directory.

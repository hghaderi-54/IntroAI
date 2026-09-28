# Titanic Classification for Beginners

A step-by-step introduction to machine learning classification for Hamid, available in English and Farsi. Both notebooks teach the same lesson with runnable Python cells, short exercises, and solutions.

## Choose your language

- **English:** [Notebook](Titanic_Classification_Beginner_EN.ipynb) · [Open in Colab](https://colab.research.google.com/github/hghaderi-54/IntroAI/blob/main/Titanic/beginner/Titanic_Classification_Beginner_EN.ipynb)
- **فارسی:** [دفترچه](Titanic_Classification_Beginner_FA.ipynb) · [بازکردن در Colab](https://colab.research.google.com/github/hghaderi-54/IntroAI/blob/main/Titanic/beginner/Titanic_Classification_Beginner_FA.ipynb)

## Run the notebook

1. Open a notebook in Colab, or download its `.ipynb` file and import it into Kaggle or Jupyter.
2. Run code cells from top to bottom with **Shift + Enter**.
3. Try each exercise before reading its solution.

No local CSV is required. The notebook uses `titanic.csv` from the current working directory when available; otherwise, it loads the [IntroAI Titanic dataset](https://raw.githubusercontent.com/SharifiZarchi/IntroAI/main/Session_05/Titanic/titanic.csv). Enable Internet in Kaggle for the online fallback. For an attached Kaggle dataset, change `LOCAL_CSV` to its `/kaggle/input/...` path.

For local Jupyter, install `pandas`, `scikit-learn`, `matplotlib`, and `ipython` in your notebook environment. Colab and Kaggle normally include these libraries.

## Learning sequence

Data → X/y → train/test split → create model → fit → predict → evaluate → predict a new passenger.

You will learn why X is two-dimensional and y is one-dimensional, encode one categorical feature, fill missing ages using only the training median, train a simple LogisticRegression classifier, and evaluate it with accuracy and a confusion matrix.

Both versions were executed successfully using the online dataset: 891 passengers, with 78.2% test accuracy for the fixed split, compared with a 61.5% majority-class baseline. Results can vary with library versions or data changes. This is an introductory learning exercise, not a competitive Kaggle solution.

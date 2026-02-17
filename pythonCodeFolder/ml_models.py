# import sys
# import json
# import os
# import pandas as pd
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# def main():
#     try:
#         raw_input = sys.stdin.read()
#         if not raw_input:
#             raise ValueError("No input received via stdin.")

#         data = json.loads(raw_input)

#         input_texts = data.get("inputTexts", [])
#         tfidf_data = data.get("tfidfValues", [])
#         token_data = data.get("tokenizedData", [])
#         labels = data.get("labels", [])
#         algo_ids = data.get("algoList", [])
#         feature_names = data.get("featureNames", [])
#         request_feature_importance = data.get("requestFeatureImportance", False)
#         request_chi2 = data.get("requestChi2", False)  # new flag

#         if not (input_texts and labels and algo_ids):
#             raise ValueError("Missing one or more required input arrays.")

#         predictions = {"Text": [str(t) for t in input_texts]}
#         summary = {}
#         feature_importance = {}
#         chi2_features = {}

#         if any(id in [1, 2, 3, 4] for id in algo_ids):
#             X = pd.DataFrame(tfidf_data)
#             y = pd.Series(labels)

#             X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
#             predict_input = X.iloc[:len(input_texts)]

#             models = {
#                 1: ("Logistic", LogisticRegression(max_iter=1000)),
#                 2: ("NaiveBayes", MultinomialNB()),
#                 3: ("RandomForest", RandomForestClassifier()),
#                 4: ("KNN", KNeighborsClassifier(n_neighbors=3))
#             }

#             for algo_id in algo_ids:
#                 if algo_id in models:
#                     name, model = models[algo_id]
#                     try:
#                         model.fit(X_train, y_train)
#                         preds = model.predict(predict_input)
#                         predictions[name] = preds.tolist()
#                         summary[name] = pd.Series(preds).value_counts().to_dict()

#                         if request_feature_importance:
#                             if hasattr(model, "feature_importances_"):
#                                 importances = model.feature_importances_
#                                 feature_importance[name] = sorted(
#                                     zip(feature_names, importances),
#                                     key=lambda x: x[1], reverse=True
#                                 )[:20]
#                             elif hasattr(model, "coef_"):
#                                 coef = model.coef_
#                                 if coef.shape[0] == 1:
#                                     feature_importance[name] = sorted(
#                                         zip(feature_names, coef[0]),
#                                         key=lambda x: abs(x[1]),
#                                         reverse=True
#                                     )[:20]
#                                 else:
#                                     class_dict = {}
#                                     for i, class_coef in enumerate(coef):
#                                         class_dict[f"class_{i}"] = sorted(
#                                             zip(feature_names, class_coef),
#                                             key=lambda x: abs(x[1]),
#                                             reverse=True
#                                         )[:20]
#                                     feature_importance[name] = class_dict
#                             elif hasattr(model, "feature_log_prob_"):  # NaiveBayes
#                                 probs = model.feature_log_prob_
#                                 class_dict = {}
#                                 for i, prob_row in enumerate(probs):
#                                     class_dict[f"class_{i}"] = sorted(
#                                         zip(feature_names, prob_row),
#                                         key=lambda x: abs(x[1]),
#                                         reverse=True
#                                     )[:20]
#                                 feature_importance[name] = class_dict

#                         # Chi-squared only for Logistic Regression
#                         if request_chi2 and algo_id == 1:
#                             selector = SelectKBest(score_func=chi2, k=20)
#                             selector.fit(X, y)
#                             scores = selector.scores_
#                             mask = selector.get_support()
#                             selected = [(f, round(s, 4)) for f, s, m in zip(feature_names, scores, mask) if m]
#                             selected.sort(key=lambda x: x[1], reverse=True)
#                             chi2_features[name] = selected

#                     except Exception as model_error:
#                         predictions[name] = ["❌ Error"]
#                         summary[name] = {"error": str(model_error)}

#         # --- Deep learning models unchanged ---
#         if any(id in [5, 6] for id in algo_ids):
#             token_df = pd.DataFrame(token_data)
#             if token_df.empty or 'padded' not in token_df.columns:
#                 raise ValueError("Tokenized data missing or improperly formatted.")

#             X_pad = token_df['padded'].apply(lambda x: list(map(int, str(x).split())))
#             max_len = max(X_pad.apply(len))
#             padded = pad_sequences(X_pad, maxlen=max_len)

#             y_cat = to_categorical(labels, num_classes=3)
#             X_train_pad, X_pred_pad, y_train_cat, _ = train_test_split(
#                 padded, y_cat, test_size=0.2, random_state=42
#             )

#             label_check = [label.argmax() for label in y_train_cat]
#             if len(set(label_check)) < 2:
#                 raise ValueError("Insufficient class diversity in training data.")

#             for algo_id in algo_ids:
#                 if algo_id == 5:
#                     model = Sequential([
#                         Embedding(input_dim=5000, output_dim=64, input_length=max_len),
#                         LSTM(64, dropout=0.3, recurrent_dropout=0.3),
#                         Dense(3, activation="softmax")
#                     ])
#                     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#                     model.fit(X_train_pad, y_train_cat, validation_split=0.2, batch_size=32, epochs=30, verbose=0)
#                     preds = model.predict(X_pred_pad, verbose=0)
#                     predicted_classes = preds.argmax(axis=1)
#                     predictions["LSTM"] = predicted_classes.tolist()[:len(input_texts)]
#                     summary["LSTM"] = pd.Series(predicted_classes).value_counts().to_dict()

#                 elif algo_id == 6:
#                     model = Sequential([
#                         Embedding(input_dim=5000, output_dim=64, input_length=max_len),
#                         Conv1D(64, 5, activation="relu"),
#                         GlobalMaxPooling1D(),
#                         Dense(3, activation="softmax")
#                     ])
#                     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#                     model.fit(X_train_pad, y_train_cat, validation_split=0.2, batch_size=32, epochs=30, verbose=0)
#                     preds = model.predict(X_pred_pad, verbose=0)
#                     predicted_classes = preds.argmax(axis=1)
#                     predictions["CNN"] = predicted_classes.tolist()[:len(input_texts)]
#                     summary["CNN"] = pd.Series(predicted_classes).value_counts().to_dict()

#         df_result = pd.DataFrame(predictions)
#         output = {
#             "table": df_result.to_dict(orient="records"),
#             "summary": summary,
#             "featureImportance": feature_importance,
#             "chi2Features": chi2_features
#         }
#         print(json.dumps(output, ensure_ascii=False))

#     except Exception as e:
#         sys.stderr.write("Error: " + str(e) + "\n")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()











import sys
import json
import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input:
            raise ValueError("No input received via stdin.")

        data = json.loads(raw_input)

        input_texts = data.get("inputTexts", [])
        tfidf_data = data.get("tfidfValues", [])
        token_data = data.get("tokenizedData", [])
        labels = data.get("labels", [])
        algo_ids = data.get("algoList", [])
        feature_names = data.get("featureNames", [])
        request_feature_importance = data.get("requestFeatureImportance", False)
        request_chi2 = data.get("requestChi2", False)

        if not (input_texts and labels and algo_ids):
            raise ValueError("Missing one or more required input arrays.")

        predictions = {"Text": [str(t) for t in input_texts]}
        summary = {}
        feature_importance = {}
        chi2_features = {}

        # ===== Classical ML Models =====
        if any(id in [1, 2, 3, 4] for id in algo_ids):
            X = pd.DataFrame(tfidf_data)
            y = pd.Series(labels)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            predict_input = X.iloc[:len(input_texts)]

            models = {
                1: ("Logistic", LogisticRegression(max_iter=1000)),
                2: ("NaiveBayes", MultinomialNB()),
                3: ("RandomForest", RandomForestClassifier()),
                4: ("KNN", KNeighborsClassifier(n_neighbors=3))
            }

            for algo_id in algo_ids:
                if algo_id in models:
                    name, model = models[algo_id]
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(predict_input)
                        predictions[name] = preds.tolist()

                        # Evaluation on test set
                        test_preds = model.predict(X_test)
                        precision = precision_score(y_test, test_preds, average="macro", zero_division=0)
                        recall = recall_score(y_test, test_preds, average="macro", zero_division=0)
                        f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)

                        summary[name] = {
                            "counts": pd.Series(preds).value_counts().to_dict(),
                            "precision": round(precision, 4),
                            "recall": round(recall, 4),
                            "f1_score": round(f1, 4)
                        }

                        # Feature importance
                        if request_feature_importance:
                            if hasattr(model, "feature_importances_"):
                                importances = model.feature_importances_
                                feature_importance[name] = sorted(
                                    zip(feature_names, importances),
                                    key=lambda x: x[1], reverse=True
                                )[:20]
                            elif hasattr(model, "coef_"):
                                coef = model.coef_
                                if coef.shape[0] == 1:
                                    feature_importance[name] = sorted(
                                        zip(feature_names, coef[0]),
                                        key=lambda x: abs(x[1]),
                                        reverse=True
                                    )[:20]
                                else:
                                    class_dict = {}
                                    for i, class_coef in enumerate(coef):
                                        class_dict[f"class_{i}"] = sorted(
                                            zip(feature_names, class_coef),
                                            key=lambda x: abs(x[1]),
                                            reverse=True
                                        )[:20]
                                    feature_importance[name] = class_dict
                            elif hasattr(model, "feature_log_prob_"):
                                probs = model.feature_log_prob_
                                class_dict = {}
                                for i, prob_row in enumerate(probs):
                                    class_dict[f"class_{i}"] = sorted(
                                        zip(feature_names, prob_row),
                                        key=lambda x: abs(x[1]),
                                        reverse=True
                                    )[:20]
                                feature_importance[name] = class_dict

                        # Chi² for Logistic Regression
                        if request_chi2 and algo_id == 1:
                            selector = SelectKBest(score_func=chi2, k=20)
                            selector.fit(X, y)
                            scores = selector.scores_
                            mask = selector.get_support()
                            selected = [(f, round(s, 4)) for f, s, m in zip(feature_names, scores, mask) if m]
                            selected.sort(key=lambda x: x[1], reverse=True)
                            chi2_features[name] = selected

                    except Exception as model_error:
                        predictions[name] = ["❌ Error"]
                        summary[name] = {"error": str(model_error)}

        # ===== Deep Learning Models =====
        if any(id in [5, 6] for id in algo_ids):
            token_df = pd.DataFrame(token_data)
            if token_df.empty or 'padded' not in token_df.columns:
                raise ValueError("Tokenized data missing or improperly formatted.")

            X_pad = token_df['padded'].apply(lambda x: list(map(int, str(x).split())))
            max_len = max(X_pad.apply(len))
            padded = pad_sequences(X_pad, maxlen=max_len)

            y_cat = to_categorical(labels, num_classes=3)
            X_train_pad, X_test_pad, y_train_cat, y_test_cat = train_test_split(
                padded, y_cat, test_size=0.2, random_state=42
            )

            y_test_labels = [label.argmax() for label in y_test_cat]
            if len(set(y_test_labels)) < 2:
                raise ValueError("Insufficient class diversity in training data.")

            for algo_id in algo_ids:
                # if algo_id == 5:
                #     model = Sequential([
                #         Embedding(input_dim=5000, output_dim=64, input_length=max_len),
                #         LSTM(64, dropout=0.3, recurrent_dropout=0.3),
                #         Dense(3, activation="softmax")
                #     ])
                #     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
                #     model.fit(X_train_pad, y_train_cat, validation_split=0.2, batch_size=32, epochs=30, verbose=0)
                #     preds = model.predict(X_test_pad, verbose=0)
                #     predicted_classes = preds.argmax(axis=1)

                #     precision = precision_score(y_test_labels, predicted_classes, average="macro", zero_division=0)
                #     recall = recall_score(y_test_labels, predicted_classes, average="macro", zero_division=0)
                #     f1 = f1_score(y_test_labels, predicted_classes, average="macro", zero_division=0)

                #     predictions["LSTM"] = predicted_classes.tolist()[:len(input_texts)]
                #     summary["LSTM"] = {
                #         "counts": pd.Series(predicted_classes).value_counts().to_dict(),
                #         "precision": round(precision, 4),
                #         "recall": round(recall, 4),
                #         "f1_score": round(f1, 4)
                #     }

                if algo_id == 6:
                    model = Sequential([
                        Embedding(input_dim=5000, output_dim=64, input_length=max_len),
                        Conv1D(64, 5, activation="relu"),
                        GlobalMaxPooling1D(),
                        Dense(3, activation="softmax")
                    ])
                    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
                    model.fit(X_train_pad, y_train_cat, validation_split=0.2, batch_size=32, epochs=30, verbose=0)
                    preds = model.predict(X_test_pad, verbose=0)
                    predicted_classes = preds.argmax(axis=1)

                    precision = precision_score(y_test_labels, predicted_classes, average="macro", zero_division=0)
                    recall = recall_score(y_test_labels, predicted_classes, average="macro", zero_division=0)
                    f1 = f1_score(y_test_labels, predicted_classes, average="macro", zero_division=0)

                    predictions["CNN"] = predicted_classes.tolist()[:len(input_texts)]
                    summary["CNN"] = {
                        "counts": pd.Series(predicted_classes).value_counts().to_dict(),
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1_score": round(f1, 4)
                    }

        # ===== Output =====
        df_result = pd.DataFrame(predictions)
        output = {
            "table": df_result.to_dict(orient="records"),
            "summary": summary,
            "featureImportance": feature_importance,
            "chi2Features": chi2_features
        }
        print(json.dumps(output, ensure_ascii=False))

    except Exception as e:
        sys.stderr.write("Error: " + str(e) + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()


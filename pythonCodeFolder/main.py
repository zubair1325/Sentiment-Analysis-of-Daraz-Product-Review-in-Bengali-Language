def main():
    import pandas as pd
    import re
    import sys
    import io
    import csv

    # ====== READ FILE FROM STDIN ======
    raw = sys.stdin.buffer.read()

    # Try to detect file type by trying both parsers
    try:
        df = pd.read_csv(io.BytesIO(raw), quoting=csv.QUOTE_ALL, engine='python', encoding='utf-8')
    except:
        try:
            df = pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            raise ValueError("Unsupported file type or corrupted content.") from e

    # ====== CLEANING ======
    df['content'] = df['content'].astype(str).str.replace(r'[\r\n]+', ' ', regex=True)

    def normalize_numerals(text):
        english_to_bangla_digits = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
        return text.translate(english_to_bangla_digits)

    def extract_only_bangla(text):
        if isinstance(text, str):
            return re.sub(r"[^\u0980-\u09FF\s]", "", text)
        return ""

    df['content'] = df['content'].apply(normalize_numerals)
    df['cleaned_text'] = df['content'].apply(extract_only_bangla)
    df['cleaned_text'] = df['cleaned_text'].str.strip()
    df = df[df['cleaned_text'].astype(bool)]

    # ====== LABEL ENCODING ======
    if 'Label' in df.columns:
        label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        df['encoded_label'] = df['Label'].str.lower().map(label_map)

    # ====== TF-IDF VECTORIZATION ======
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(max_features=8000)
    X = tfidf.fit_transform(df['cleaned_text'])
    tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

    # ====== TOKENIZATION & PADDING ======
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    vocab_size = 5000
    oov_token = "<OOV>"
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(df['cleaned_text'])

    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    # ====== EXPORTS ======
    df_export = pd.DataFrame({
        'original_text': df['cleaned_text'],
        'tokenized': sequences,
        'padded': [' '.join(map(str, p)) for p in padded_sequences]
    })

    # Include encoded label if available
    if 'encoded_label' in df.columns:
        df_export['label'] = df['encoded_label']

    # Write output to stdout as CSV
    output_buffer = io.StringIO()
    df_export.to_csv(output_buffer, index=False)
    sys.stdout.buffer.write(output_buffer.getvalue().encode('utf-8'))

if __name__ == "__main__":
    main()
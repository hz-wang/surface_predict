def predict_labels(model, texts, dict, MAX_SEN_LENGTH):
    sen2text = []
    textID = 0
    sens_all = []
    for text in texts:
        textID += 1
        sens = split_long_sentence(text, MAX_SEN_LENGTH)
        for sen in sens:
            sen2text.append(textID)
            sens_all.append(sen)
    sequences = []
    for sen in sens_all:
        seq = text_to_sequence(sen, dict)
        sequences.append(seq)
    X = pad_sequences(sequences, maxlen=MAX_SEN_LENGTH)
    preds = model.predict(X, batch_size=100, verbose=0)
    labels = []
    label_temp = []
    for i in range(len(preds)):
        sen = sens_all[i]
        pred = preds[i]
        pred = pred[len(pred) - len(sen):]
        label = preds2labels_simple(pred)
        if i == 0:
            label_temp = label
        elif sen2text[i] == sen2text[i - 1]:
            label_temp += label
        else:
            labels.append(label_temp)
            label_temp = label
        if (i + 1) == len(preds):
            labels.append(label_temp)
    return labels
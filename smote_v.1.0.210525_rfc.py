import pandas as pd
from numpy import asarray, mean
import time
from math import ceil, sqrt as sq
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap as LSC
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn import metrics as skm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as TTS, \
    StratifiedKFold as SKF, GridSearchCV as GSCV, cross_val_score as cvs
from sklearn.ensemble import RandomForestClassifier as sRFC

df = pd.read_csv('cancer_data_processed.csv')
y, X = df.iloc[:, -1], df.drop("Biopsy", axis=1)

plt.rcParams["font.sans-serif"] = "Helvetica"

# config these 3 sizes
plt.rcParams["font.size"] = 42
width, height = 6, 6

start = time.process_time()


def steps(model, X_train, y_train, X_test, y_test, no, w=width, h=height):

    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    y_clsf = model.predict(X_test)

    [[tn, fp], [fn, tp]] = skm.confusion_matrix(y_test, y_clsf)
    cm = skm.confusion_matrix(y_test, y_clsf, labels=model.classes_)

    # plot the confusion matrix
    redmap = LSC.from_list('mycmap', ['white', 'red'])
    fig, ax = plt.subplots(figsize=(w, h))

    per_by_true = [tn / (tn + fp), fp / (tn + fp),
                   fn / (fn + tp), tp / (fn + tp)]
    cm_n = [[tn / (tn + fp), fp / (tn + fp)], [fn / (fn + tp), tp / (fn + tp)]]

    group_counts = ['{0: 0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.1%}'.format(value) for value in per_by_true]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = asarray(labels).reshape(2, 2)
    sns.heatmap(cm_n, annot=labels, fmt='', cmap=redmap, cbar=False)

    plt.subplots_adjust(left=0.225, bottom=0.225)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(str(no) + ".png")

    spec_0, spec_1 = tp / (tp + fn), tn / (tn + fp)
    # recall_1, recall_0 = spec_0, spec_1
    print("Positive recall, negative recall")
    print(spec_0, spec_1)
    spec_w = (spec_0 * (tn + fp) + spec_1 * (tp + fn)) / (tn + fp + fn + tp)

    thescores = [skm.accuracy_score(y_test, y_clsf),
                 skm.precision_score(y_test, y_clsf, zero_division=0),
                 spec_0, spec_w, skm.f1_score(y_test, y_clsf),
                 skm.fbeta_score(y_test, y_clsf, beta=2),
                 skm.roc_auc_score(y_test, y_pred)]

    return thescores


def main(X_in, y_in):

    num_folds = 7

    # Let's use the same split here
    X_train, X_test, y_train, y_test = TTS(
        X_in, y_in, test_size=0.2, random_state=42)
    # Standardize the data first: only standardize the training set
    # and then transform the test set the same magnitude
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    # X_train.shape vs X_test.shape <=> (686, 28) vs (172, 28)

    # Define the K-fold Cross Validator
    kf = SKF(n_splits=num_folds)

    nobio, bio = y_train.value_counts()[0], y_train.value_counts()[1]
    equal_size = ceil(sq(nobio * bio))  # geom mean of the 0 & 1 sizes

    nth_srfc = sRFC(random_state=42)
    smt_srfc = make_pipeline(SMOTE(random_state=42), sRFC(random_state=42))
    nms_srfc = make_pipeline(NearMiss(), sRFC(random_state=42))
    smn_srfc = make_pipeline(
        SMOTE(sampling_strategy={1: equal_size}, random_state=42),
        NearMiss(sampling_strategy={0: equal_size}), sRFC(random_state=42))

    print("Recall performance without tuning:")
    print("No over/under-sampling")
    print(mean(cvs(nth_srfc, X_train, y_train, scoring="recall", cv=kf)))
    print("SMOTE the positive")
    print(mean(cvs(smt_srfc, X_train, y_train, scoring="recall", cv=kf)))
    print("Near Miss the negative")
    print(mean(cvs(nms_srfc, X_train, y_train, scoring="recall", cv=kf)))
    print("Both SMOTE & Near Miss to the geometric mean")
    print(mean(cvs(smn_srfc, X_train, y_train, scoring="recall", cv=kf)))

    params = {'n_estimators': range(10, 191, 10),
              'max_depth': range(5, 51, 10),
              'max_features': [None, "sqrt", "log2"],
              'min_samples_split': range(2, 11, 10)}

    print("Tuning the hyper-parameters...")

    new_params = {'randomforestclassifier__' + key: params[
        key] for key in params}
    grid_srf = GSCV(smn_srfc, param_grid=new_params, cv=kf,
                    scoring='recall', return_train_score=True, n_jobs=-1)

    grid_srf.fit(X_train, y_train)
    selparams = grid_srf.best_params_
    print(selparams)
    print("Recall score on the training set given the above params:")
    print(grid_srf.best_score_)
    y_pred_smn = grid_srf.predict(X_test)
    print("Recall score on the test set given the above params:")
    print(skm.recall_score(y_test, y_pred_smn))

    print("Tuning done! Re-train the models using the 'best' parameters...")

    # then we train the models given different data manipulation again
    # using the "best" parameters we got a moment ago

    rfc_n_est = 'randomforestclassifier__n_estimators'
    rfc_min_n = 'randomforestclassifier__min_samples_split'
    rfc_max_features = 'randomforestclassifier__max_features'
    rfc_max_depth = 'randomforestclassifier__max_depth'

    srfc_nth = sRFC(n_estimators=selparams[rfc_n_est],
                    min_samples_split=selparams[rfc_min_n],
                    max_features=selparams[rfc_max_features],
                    max_depth=selparams[rfc_max_depth], random_state=42)

    srfc_smt = make_pipeline(SMOTE(random_state=42), sRFC(
        n_estimators=selparams[rfc_n_est],
        min_samples_split=selparams[rfc_min_n],
        max_features=selparams[rfc_max_features],
        max_depth=selparams[rfc_max_depth], random_state=42))

    srfc_nms = make_pipeline(NearMiss(), sRFC(
        n_estimators=selparams[rfc_n_est],
        min_samples_split=selparams[rfc_min_n],
        max_features=selparams[rfc_max_features],
        max_depth=selparams[rfc_max_depth], random_state=42))

    srfc_smn = make_pipeline(
        SMOTE(sampling_strategy={1: equal_size}, random_state=42),
        NearMiss(sampling_strategy={0: equal_size}),
        sRFC(n_estimators=selparams[rfc_n_est],
             min_samples_split=selparams[rfc_min_n],
             max_features=selparams[rfc_max_features],
             max_depth=selparams[rfc_max_depth], random_state=42))

    thelist = [[] for i in range(4)]
    modellist = [srfc_nth, srfc_smt, srfc_nms, srfc_smn]

    for i in range(len(modellist)):
        thelist[i] = steps(modellist[i], X_train, y_train, X_test, y_test, i)

    rfc_names = ["Accuracy", "Precision", "Recall",
                 "Specificity", "F1", "F2", "AUC"]

    # write the (LaTeX) table ready text files

    with open("rfc_scores.txt", "a+") as f:

        f.write("For the test set (n = 172),\n")
        f.write("{:11s} {:8s} {:8s} {:8s} {:8s} \n".format(
            "", "nothing", "SMOTE", "NearMiss", "both"))

        for i in range(7):
            f.write("{:11s} {:.6f} {:.6f} {:.6f} {:.6f} \n".format(
                rfc_names[i], thelist[0][i], thelist[1][i],
                thelist[2][i], thelist[3][i]))

        f.write("{:s} & {:s} & {:s} & {:s} & {:s} \\\\ \n".format(
            "Over-/Under-sampling?", "nothing", "SMOTE", "Near miss", "both"))
        f.write("\\hline\n")

        for i in range(7):
            f.write("{:s} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\ \n".format(
                rfc_names[i], thelist[0][i] * 100, thelist[1][i] * 100,
                thelist[2][i] * 100, thelist[3][i] * 100))


main(X, y)
end = time.process_time()
print("Elapsed time: " + str(end - start) + "s.")

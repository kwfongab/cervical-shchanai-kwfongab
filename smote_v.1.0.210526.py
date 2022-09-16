from pandas import read_csv, DataFrame as DF
from numpy import append as npadd, array, mean
from math import ceil, sqrt as sq
from matplotlib import pyplot as plt
from collections import Counter
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import auc, roc_curve, confusion_matrix as CM, \
    accuracy_score as AC, precision_recall_fscore_support as PRFS, \
    fbeta_score as fb
from sklearn.model_selection import train_test_split as TTS

# train:validate:test = 60:20:20

df = read_csv('cancer_data_processed.csv')
y, X = df.iloc[:, -1], df.drop("Biopsy", axis=1)
print('Original dataset shape %s' % Counter(y))


def result_writing(trues, preds, txtf):
    a_m = AC(trues, preds)
    txtf.write("\nAccuracy: \t\t\t\t {:.6f}".format(a_m))

    (p_m, r_m, f1_m) = PRFS(trues, preds, average="macro")[0:3]
    txtf.write("\nPrecision, macro: \t\t {:.6f}".format(p_m))
    txtf.write("\nRecall, macro: \t\t\t {:.6f}".format(r_m))
    txtf.write("\nF1 score, macro: \t\t {:.6f}".format(f1_m))
    f2_m = fb(trues, preds, beta=2, average="macro")
    txtf.write("\nF2 score, macro: \t\t {:.6f}".format(f2_m))

    (p_w, r_w, f1_w) = PRFS(trues, preds, average="weighted")[0:3]
    txtf.write("\nPrecision, weighted: \t {:.6f}".format(p_w))
    txtf.write("\nRecall, weighted: \t\t {:.6f}".format(r_w))
    txtf.write("\nF1 score, weighted: \t {:.6f}".format(f1_w))
    f2_w = fb(trues, preds, beta=2, average="weighted")
    txtf.write("\nF2 score, weighted: \t {:.6f}".format(f2_w))

    tn, fp, fn, tp = CM(trues, preds).ravel()
    spec_0, spec_1 = tp / (tp + fn), tn / (tn + fp)
    spec_macro = (spec_0 + spec_1) / 2
    spec_weighted = (spec_0 * (tn + fp) +
                     spec_1 * (tp + fn)) / (tn + fp + fn + tp)

    txtf.write("\nRecall, +ve only: \t\t {:.6f}".format(spec_0))
    txtf.write("\nSpecificity, marco: \t {:.6f}".format(spec_macro))
    txtf.write("\nSpecificity, weighted: \t {:.6f}".format(spec_weighted))
    txtf.write("\nFall-out, macro: \t\t {:.6f}".format(1 - spec_macro))
    txtf.write("\nFall-out, weighted: \t {:.6f}".format(1 - spec_weighted))
    txtf.write("\n\n")

    return a_m, p_w, spec_0, f1_w, f2_w, spec_weighted


def fit_to_roc(model, X_train, y_train, X_test, y_test, keyword):
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    filename = "smote_first_output_" + str(model)[:8].upper() + ".txt"
    txtf = open(filename, "a+")
    txtf.write("\n\nResult using " + str(model) +
               " about " + str(keyword) + ":\n")
    a, p, r, f, g, s = result_writing(y_test, y_predicted, txtf)
    txtf.close()
    y_scores = model.predict_proba(X_test)
    fpr, tpr, thr = roc_curve(y_test, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)
    return roc_auc, a, p, r, f, g, s


def all_models(X_train, y_train, X_test, y_test, kw):

    a, b, c, d = X_train, y_train, X_test, y_test

    # Gaussian Naive Bayes
    nb = GaussianNB()
    ra_nb, a_nb, p_nb, r_nb, f_nb, g_nb, s_nb = fit_to_roc(nb, a, b, c, d, kw)

    # Decision Trees
    dt = tree.DecisionTreeClassifier(random_state=42)
    ra_dt, a_dt, p_dt, r_dt, f_dt, g_dt, s_dt = fit_to_roc(dt, a, b, c, d, kw)

    # Use LBFGS solver for Logistic regression w/ L2 penalty
    lr = LR(max_iter=1000)
    ra_lr, a_lr, p_lr, r_lr, f_lr, g_lr, s_lr = fit_to_roc(lr, a, b, c, d, kw)

    # Support Vector Machine
    sv = SVC(gamma='auto', probability=True, random_state=42)
    ra_sv, a_sv, p_sv, r_sv, f_sv, g_sv, s_sv = fit_to_roc(sv, a, b, c, d, kw)

    # K nearest neighbor
    nn = KNC()
    ra_nn, a_nn, p_nn, r_nn, f_nn, g_nn, s_nn = fit_to_roc(nn, a, b, c, d, kw)

    return [ra_nb, ra_dt, ra_lr, ra_sv, ra_nn, a_nb, a_dt, a_lr, a_sv, a_nn,
            p_nb, p_dt, p_lr, p_sv, p_nn, r_nb, r_dt, r_lr, r_sv, r_nn,
            f_nb, f_dt, f_lr, f_sv, f_nn, g_nb, g_dt, g_lr, g_sv, g_nn,
            s_nb, s_dt, s_lr, s_sv, s_nn]


def workflow(maxcount, X_in, y_in):
    count = 0

    samplemethod = ["SMOTE", "Near Miss", "SMOTE + Near Miss"]
    namelist = ["GAUSSIAN", "DECISION", "LOGISTIC",
                "PIPELINE", "KNEIGHBO"]
    params = ["Accuracy", "Recall (+ve)", "Specificity",
              "Precision", "F1 score", "F2 score", "AUC score"]

    upb, modelchoice, relatedmetrics = \
        len(samplemethod), len(namelist), len(params)
    umrtotal = upb * modelchoice * relatedmetrics

    # a__, p__, r__, f__, g__, s__
    # Accuracy, Precision, Recall, F1, F2, Specificity

    # _g_, _d_, _l_, _s_, _k_
    # Gaussian Naive Bayes, Decision Trees, Logistic Regression, SVM, kNN

    # __u, __d, __b
    # SMOTE, NearMiss, Both

    gnb_u, gnb_d, gnb_b, dt_u, dt_d, dt_b, lr_u, lr_d, lr_b, \
    svm_u, svm_d, svm_b, knn_u, knn_d, knn_b, \
    agu, agd, agb, adu, add, adb, alu, ald, alb, asu, asd, asb, aku, akd, \
    akb, pgu, pgd, pgb, pdu, pdd, pdb, plu, pld, plb, psu, psd, psb, pku, \
    pkd, pkb, rgu, rgd, rgb, rdu, rdd, rdb, rlu, rld, rlb, rsu, rsd, rsb, \
    rku, rkd, rkb, fgu, fgd, fgb, fdu, fdd, fdb, flu, fld, flb, fsu, fsd, \
    fsb, fku, fkd, fkb, ggu, ggd, ggb, gdu, gdd, gdb, glu, gld, glb, gsu, \
    gsd, gsb, gku, gkd, gkb, sgu, sgd, sgb, sdu, sdd, sdb, slu, sld, slb, \
    ssu, ssd, ssb, sku, skd, skb, = ([] for i in range(umrtotal))

    while count < maxcount:

        X_train, X_test, y_train, y_test = TTS(X_in, y_in, test_size=0.2)

        # Standardize the data first
        # Only standardize the training set
        # and then transform the test set the same magnitude
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        # Oversampling
        smt_up = SMOTE(random_state=42)
        X_smote, y_smote = smt_up.fit_resample(X_train, y_train)

        # Undersampling
        nms_down = NearMiss()
        X_nms, y_nms = nms_down.fit_resample(X_train, y_train)

        # Both (?)
        nobio, bio = y_train.value_counts()[0], y_train.value_counts()[1]
        equal_size = ceil(sq(nobio * bio))  # geom mean of the 0 & 1 sizes
        pipe_smn = make_pipeline(
            SMOTE(sampling_strategy={1: equal_size}, random_state=42),
            NearMiss(sampling_strategy={0: equal_size}))
        X_smn, y_smn = pipe_smn.fit_resample(X_train, y_train)

        thedata = [npadd(X_smn[i], y_smn[i]) for i in range(len(X_smn))]
        DF(thedata).to_csv("Rebalanced_dataset_{0:2d}_with_{1:s}.csv".format(
            count, "SMOTE_&_Near_Miss"), index=False)

        # random forest to be included in each sample manipulation (?)
        stat_smote = all_models(X_smote, y_smote, X_test, y_test, "SMOTE")
        stat_nms = all_models(X_nms, y_nms, X_test, y_test, "Near Miss")
        stat_smn = all_models(
            X_smn, y_smn, X_test, y_test, "SMOTE + Near Miss")

        [x.append(y) for x, y in zip(
            [gnb_u, dt_u, lr_u, svm_u, knn_u, agu, adu, alu, asu, aku,
             pgu, pdu, plu, psu, pku, rgu, rdu, rlu, rsu, rku,
             fgu, fdu, flu, fsu, fku, ggu, gdu, glu, gsu, gku,
             sgu, sdu, slu, ssu, sku], stat_smote)]
        [x.append(y) for x, y in zip(
            [gnb_d, dt_d, lr_d, svm_d, knn_d, agd, add, ald, asd, akd,
             pgd, pdd, pld, psd, pkd, rgd, rdd, rld, rsd, rkd,
             fgd, fdd, fld, fsd, fkd, ggd, gdd, gld, gsd, gkd,
             sgd, sdd, sld, ssd, skd], stat_nms)]
        [x.append(y) for x, y in zip(
            [gnb_b, dt_b, lr_b, svm_b, knn_b, agb, adb, alb, asb, akb,
             pgb, pdb, plb, psb, pkb, rgb, rdb, rlb, rsb, rkb,
             fgb, fdb, flb, fsb, fkb, ggb, gdb, glb, gsb, gkb,
             sgb, sdb, slb, ssb, skb], stat_smn)]
        count += 1

    data = array(
        [agu, adu, alu, asu, aku, rgu, rdu, rlu, rsu, rku, sgu, sdu,
         slu, ssu, sku, pgu, pdu, plu, psu, pku, fgu, fdu, flu, fsu,
         fku, ggu, gdu, glu, gsu, gku, gnb_u, dt_u, lr_u, svm_u, knn_u,
         agd, add, ald, asd, akd, rgd, rdd, rld, rsd, rkd, sgd, sdd,
         sld, ssd, skd, pgd, pdd, pld, psd, pkd, fgd, fdd, fld, fsd,
         fkd, ggd, gdd, gld, gsd, gkd, gnb_d, dt_d, lr_d, svm_d, knn_d,
         agb, adb, alb, asb, akb, rgb, rdb, rlb, rsb, rkb, sgb, sdb,
         slb, ssb, skb, pgb, pdb, plb, psb, pkb, fgb, fdb, flb, fsb,
         fkb, ggb, gdb, glb, gsb, gkb, gnb_b, dt_b, lr_b, svm_b, knn_b])

    means = mean(data, axis=1)
    means = means.tolist()

    # Fill in the average of all the parameters of interest
    for i in range(len(namelist)):
        with open("smote_first_output_{:s}.txt".format(
                namelist[i]), "a+") as txt:
            txt.write("\nOn average over {:d} times,".format(maxcount))
            for j in range(len(samplemethod)):
                txt.write("\nafter preprocessing the data with {:s},".format(
                    samplemethod[j]))
                l = i + len(namelist) * len(params) * j
                for k in range(len(params)):
                    txt.write("\n{0:13s} {1:.6f}".format(
                        params[k] + ":", means[l + len(namelist) * k]))

    return gnb_u, gnb_d, gnb_b, dt_u, dt_d, dt_b, lr_u, lr_d, lr_b,\
           svm_u, svm_d, svm_b, knn_u, knn_d, knn_b


gnb_u, gnb_d, gnb_b, dt_u, dt_d, dt_b, lr_u, lr_d, lr_b, \
svm_u, svm_d, svm_b, knn_u, knn_d, knn_b = workflow(100, X, y)

up_df = DF({"Gaussian Naive Bayes": gnb_u, "Decision Tree": dt_u,
            "Logistic Regression": lr_u,
            "SVM": svm_u, "kNN (k = 5)": knn_u})

down_df = DF({"Gaussian Naive Bayes": gnb_d, "Decision Tree": dt_d,
              "Logistic Regression": lr_d,
              "SVM": svm_d, "kNN (k = 5)": knn_d})

bal_df = DF({"Gaussian Naive Bayes": gnb_b, "Decision Tree": dt_b,
             "Logistic Regression": lr_b,
             "SVM": svm_b, "kNN (k = 5)": knn_b})

plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["font.size"] = 12

fig, [axu, axd, axb] = plt.subplots(1, 3, sharey=True)

xticks = [1, 2, 3, 4, 5]
ax_xticks = [r"GNB", r"Tree", r"LR", r"SVM", r"kNN"]

axu.boxplot(up_df)
axd.boxplot(down_df)
axb.boxplot(bal_df)

for axes in [axu, axd, axb]:
    axes.set_xticks(xticks)
    axes.set_xticklabels(ax_xticks)

plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["font.size"] = 12

plt.show()

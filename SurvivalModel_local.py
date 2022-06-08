### SURVIVAL ANALYSIS ###

### code inspired from https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html ###

### to install scikit-survival, first install CMake: https://cmake.org/ then pip install scikit-survival ###
### you also might need Visual Studio Build tools at https://visualstudio.microsoft.com/visual-cpp-build-tools/ ###

# classic imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import copy
from eli5.sklearn import PermutationImportance

# more imports
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, ShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import FitFailedWarning

# from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import (
    load_breast_cancer,
    load_flchain,
    load_gbsg2,
    load_veterans_lung_cancer,
)

# from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
# from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from sksurv.preprocessing import OneHotEncoder, encode_categorical

from sksurv.svm import FastSurvivalSVM

# from sksurv.kernels import clinical_kernel
from sksurv.svm import FastKernelSurvivalSVM

# from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator

# from sksurv.util import Surv

sns.set_style("whitegrid")


###############################################


class SurvivalModel:
    """
    General object for performing survival analyses with different estimators and
    retrieving the coefficients associated with each feature
    """

    def __init__(self, X_train=None, y_train=None, random_state=42):
        if X_train is None and y_train is None:
            X_train, y_train = load_breast_cancer()
            self.status_str = "e.tdm"
            self.time_to_event_str = "t.tdm"
        elif X_train is not None and y_train is not None:
            if isinstance(y_train, pd.DataFrame):
                self.status_str = y_train.columns[0]
                self.time_to_event_str = y_train.columns[1]
                y_train = y_train.to_numpy()
            elif isinstance(y_train, np.ndarray):
                self.status_str = y_train.dtype.names[0]
                self.time_to_event_str = y_train.dtype.names[1]
            else:
                raise ValueError("y_train is not a DataFrame or a numpy array")
        else:
            raise ValueError("Only one array passed, two are needed")
        self.X_train = X_train
        self.y_train = y_train
        self.seed = random_state
        self.X_train_ohe = OneHotEncoder().fit_transform(self.X_train)
        self.X_test_ohe = None
        self.estimator = None
        self.coeffs = None
        self.score = None
        self.result = None
        self.pred_surv = None
        self.features_scores = None
        self.parsimonious = None

    def __str__(self):
        return (
            f"Instance of SurvivalModel class with the following attributes: \n"
            f"Training: {self.X_train.shape[0]} samples and {self.X_train.shape[1]} features \n"
            f"Targets: {len(self.y_train.dtype.names)} columns named {self.status_str} and {self.time_to_event_str} \n"
            f"Estimator: {self.estimator} with random seed {self.seed} and score {self.score} \n"
        )

    def __repr__(self):
        return (
            f"Instance of SurvivalModel class with the following attributes: \n"
            f"Training: {self.X_train.shape[0]} samples and {self.X_train.shape[1]} features \n"
            f"Targets: {len(self.y_train.dtype.names)} columns named {self.status_str} and {self.time_to_event_str} \n"
            f"Estimator: {self.estimator} with random seed {self.seed} and score {self.score}"
        )

    def fit_cox_ph(self, mode="full"):
        """Fits a Cox Proportional Hazard model to the dataset
        syntax: your_model = SurvivalModel(X_train, y_train).fit_cox_ph(mode='full/parsimonious/elastic-net')
        mode='full' will build a complete model (unpenalized)
        mode='parsimonious' will select the best k features and
        mode='elastic-net' will apply double regularization
        model.coeffs: coefficients for features in final model
        model.score: concordance index (interpret like AUROC)
        model.result: concordance_index, concording_pairs, discording_pairs
        model.feature_scores: predictivity for each feature
        """

        def helper_fn(X, y):
            n_features = X.shape[1]
            scores = np.empty(n_features)
            m = CoxPHSurvivalAnalysis()
            for j in range(n_features):
                Xj = X[:, j : j + 1]
                m.fit(Xj, y)
                scores[j] = m.score(Xj, y)
            return scores

        if mode == "full":
            self.estimator = CoxPHSurvivalAnalysis()
            self.estimator.fit(self.X_train_ohe, self.y_train)
            self.coeffs = pd.Series(
                self.estimator.coef_, index=self.X_train_ohe.columns
            )
            self.score = self.estimator.score(self.X_train_ohe, self.y_train)
            self.features_scores = pd.Series(
                self.fit_score_features(), index=self.X_train_ohe.columns
            ).sort_values(ascending=False)

        elif mode == "parsimonious":
            pipe = Pipeline(
                [
                    ("encode", OneHotEncoder()),
                    ("select", SelectKBest(helper_fn, k=3)),
                    ("model", CoxPHSurvivalAnalysis()),
                ]
            )
            param_grid = {"select__k": np.arange(1, self.X_train_ohe.shape[1] + 1)}
            cv = KFold(n_splits=3, random_state=self.seed, shuffle=True)
            gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv)
            gcv.fit(self.X_train, self.y_train)
            results = pd.DataFrame(gcv.cv_results_).sort_values(
                by="mean_test_score", ascending=False
            )
            self.parsimonious = results.loc[:, ~results.columns.str.endswith("_time")]
            pipe.set_params(**gcv.best_params_)
            pipe.fit(self.X_train_ohe, self.y_train)

            enc, trans, estim = [s[1] for s in pipe.steps]
            self.estimator = estim
            self.coeffs = pd.Series(
                self.estimator.coef_, index=self.X_train_ohe.columns
            )
            self.score = self.estimator.score(self.X_train_ohe, self.y_train)
            self.features_scores = pd.Series(
                estim.coef_, index=enc.encoded_columns_[trans.get_support()]
            ).sort_values(ascending=False)

        elif mode == "elastic-net":
            self.estimator = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01)
            self.estimator.fit(self.X_train_ohe, self.y_train)
            coeff_net = pd.DataFrame(
                self.estimator.coef_,
                index=self.X_train_ohe.columns,
                columns=np.round(self.estimator.alphas_, 5),
            )

            self.plot_coefficients(coeff_net, n_highlight=5)
            # self.features_scores = pd.Series(self.estimator.coef_, index=self.X_train_ohe.columns).sort_values(ascending=False)

            coxnet_pipe = make_pipeline(
                StandardScaler(),
                CoxnetSurvivalAnalysis(
                    l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100
                ),
            )
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FitFailedWarning)
            coxnet_pipe.fit(self.X_train_ohe, self.y_train)

            estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
            cv = KFold(n_splits=5, shuffle=True, random_state=0)
            gcv = GridSearchCV(
                make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
                param_grid={
                    "coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]
                },
                cv=cv,
                error_score=0.5,
                n_jobs=4,
            ).fit(self.X_train_ohe, self.y_train)

            cv_results = pd.DataFrame(gcv.cv_results_)

            alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
            mean = cv_results.mean_test_score
            std = cv_results.std_test_score

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(alphas, mean)
            ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
            ax.set_xscale("log")
            ax.set_ylabel("concordance index")
            ax.set_xlabel("alpha")
            best_alpha = gcv.best_params_["coxnetsurvivalanalysis__alphas"][0]
            ax.axvline(best_alpha, c="C1")
            ax.axhline(0.5, color="grey", linestyle="--")
            ax.grid(True)

            best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
            best_coefs = pd.DataFrame(
                best_model.coef_,
                index=self.X_train_ohe.columns,
                columns=["coefficient"],
            )

            non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
            print("Number of non-zero coefficients: {}".format(non_zero))

            non_zero_coefs = best_coefs.query("coefficient != 0")
            coef_order = non_zero_coefs.abs().sort_values("coefficient").index
            self.coeffs = non_zero_coefs.loc[coef_order]

            _, ax = plt.subplots(figsize=(6, 8))
            non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
            ax.set_xlabel("coefficient")
            ax.grid(True)

            self.estimator = make_pipeline(
                StandardScaler(),
                CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True),
            )
            self.estimator.set_params(**gcv.best_params_)
            self.estimator.fit(self.X_train_ohe, self.y_train)
            self.features_scores = non_zero_coefs.squeeze()

        prediction = self.estimator.predict(self.X_train_ohe)
        result = concordance_index_censored(
            self.y_train[self.status_str],
            self.y_train[self.time_to_event_str],
            prediction,
        )
        self.result = pd.DataFrame(
            result[0:3],
            index=["concordance_index", "concording_pairs", "discording_pairs"],
        )

    def fit_rf(
        self,
        n_estimators=1000,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
    ):
        self.estimator = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=self.seed,
        )
        self.estimator.fit(self.X_train_ohe, self.y_train)
        self.score = self.estimator.score(self.X_train_ohe, self.y_train)
        # self.coeffs = pd.Series(self.estimator.coef_, index=self.X_ohe.columns) #TODO: does not work as .coef has wrong size
        self.features_scores = pd.Series(
            self.fit_score_features(), index=self.X_train_ohe.columns
        ).sort_values(ascending=False)

    def fit_svm(self):
        self.estimator = FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=0)
        pass

    def fit_kernel_svm(self):
        self.estimator = FastKernelSurvivalSVM(
            optimizer="rbtree", kernel="precomputed", random_state=self.seed
        )
        pass

    def fit_score_features(self):
        if self.estimator is not None:
            dummy_estimator = copy.deepcopy(self.estimator)
            try:
                n_features = self.X_train_ohe.shape[1]
                scores = np.empty(n_features)
                for j in range(n_features):
                    xj = self.X_train_ohe.iloc[:, j : j + 1]
                    dummy_estimator.fit(xj, self.y_train)
                    scores[j] = dummy_estimator.score(xj, self.y_train)
                return scores
            except:
                perm = PermutationImportance(
                    dummy_estimator, n_iter=15, random_state=self.seed
                )
                perm.fit(self.X_train_ohe, self.y_train)
                scores = perm.feature_importances
                return scores
        else:
            raise ValueError("no estimator selected. Fit a model first.")

    def predict(self, X_test, max_time=1000, plot=True):
        """Predicts survival time for the test set"""

        self.X_test_ohe = OneHotEncoder().fit_transform(X_test)
        self.pred_surv = self.estimator.predict_survival_function(self.X_test_ohe)
        if plot:
            time_points = np.arange(1, max_time)
            for i, surv_func in enumerate(self.pred_surv):
                plt.step(
                    time_points,
                    surv_func(time_points),
                    where="post",
                    label="Sample %d" % (i + 1),
                )
            plt.ylabel("est. probability of survival $\hat{S}(t)$")
            plt.xlabel("time $t$")
            plt.legend(loc="best")

        return self.pred_surv

    def evaluate(self, test_size=0.2):
        """Evaluate the quality of the model with cross-validation"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train, test_size=test_size, random_state=self.seed,
        )
        num_columns = [col for col in X_train.columns if is_numeric_dtype(X_train[col])]

        # imputing data
        imputer = SimpleImputer().fit(X_train.loc[:, num_columns])
        X_test_imputed = imputer.transform(X_test.loc[:, num_columns])

        # does the observed time of the test data lies within the observed time range of the training data?
        y_events = y_train[y_train[self.status_str]]
        train_min, train_max = (
            y_events[self.time_to_event_str].min(),
            y_events[self.time_to_event_str].max(),
        )
        y_events = y_test[y_test[self.status_str]]
        test_min, test_max = (
            y_events[self.time_to_event_str].min(),
            y_events[self.time_to_event_str].max(),
        )
        assert (
            train_min <= test_min < test_max < train_max
        ), "time range or test data is not within time range of training data."

        times = np.percentile(
            self.y_train[self.time_to_event_str], np.linspace(5, 81, 15)
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, col in enumerate(num_columns):
            self.plot_cumulative_dynamic_auc(
                y_train,
                y_test,
                X_test_imputed[:, i],
                col,
                times,
                color="C{}".format(i),
                axis=ax,
            )
            ret = concordance_index_ipcw(
                y_train, y_test, X_test_imputed[:, i], tau=times[-1]
            )  # TODO: what do we do with the concordance index?

        plt.savefig("Cumulative_Dynamic_AUC.png")

    def plot_cumulative_dynamic_auc(
        self, y_train, y_test, risk_score, label, times, color=None, axis=None
    ):
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_score, times)
        if axis is None:
            fig, axis = plt.subplots(figsize=(9, 6))
        axis.plot(times, auc, marker="o", color=color, label=label)
        axis.set_xlabel("time from enrollment")
        axis.set_ylabel("time-dependent AUC")
        axis.axhline(mean_auc, color=color, linestyle="--")
        axis.legend()


    def evaluate_predictions(self, test_size=0.2):
        va_x_train, va_x_test, va_y_train, va_y_test = train_test_split(
            self.X_train,
            self.y_train,
            test_size=test_size,
            stratify=self.y_train["Status"],
            random_state=0,
        )
        cph = make_pipeline(OneHotEncoder(), self.estimator)
        cph.fit(va_x_train, va_y_train)

        va_times = np.arange(8, 184, 7)
        cph_risk_scores = cph.predict(va_x_test)
        cph_auc, cph_mean_auc = cumulative_dynamic_auc(
            va_y_train, va_y_test, cph_risk_scores, va_times
        )

        plt.plot(va_times, cph_auc, marker="o")
        plt.axhline(cph_mean_auc, linestyle="--")
        plt.xlabel("time from enrollment")
        plt.ylabel("time-dependent AUC")
        plt.grid(True)
        plt.savefig("Cross-validation.png")

    def plot_coefficients(self, coefs, n_highlight):
        _, ax = plt.subplots(figsize=(9, 6))
        n_features = coefs.shape[0]
        alphas = coefs.columns
        for row in coefs.itertuples():
            ax.semilogx(alphas, row[1:], ".-", label=row.Index)

        alpha_min = alphas.min()
        top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
        for name in top_coefs.index:
            coef = coefs.loc[name, alpha_min]
            plt.text(
                alpha_min,
                coef,
                name + "   ",
                horizontalalignment="right",
                verticalalignment="center",
            )

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.grid(True)
        ax.set_xlabel("alpha")
        ax.set_ylabel("coefficient")

    def plot_data(self, feature: str = None):
        time, survival_prob = kaplan_meier_estimator(
            self.y_train[:, 0], self.y_train[:, 1]
        )
        fig, ax = plt.subplots(2, 1, figsize=(9, 6))
        ax[0].step(time, survival_prob, where="post")
        ax[0].set_ylabel("est. probability of survival $\hat{S}(t)$")
        ax[0].set_xlabel("time $t$")

        if feature is not None:
            for value in self.X_train[feature].unique():
                mask = self.X_train[feature] == value
                time_treatment, sp = kaplan_meier_estimator(
                    self.y_train[self.status_str][mask],
                    self.y_train[self.time_to_event_str][mask],
                )
                ax[1].step(
                    time_treatment,
                    sp,
                    where="post",
                    label="%s (n = %d)" % (value, mask.sum()),
                )

            ax[1].set_ylabel("est. probability of survival $\hat{S}(t)$")
            ax[1].set_xlabel("time $t$")
            ax[1].legend(loc="best")
        fig.savefig(f"KM_{feature}.png")


#######################
# print("creating model")
# model = SurvivalModel()

# print("plotting data")
# model.plot_data(feature="Celltype")

# print("fitting full cox")
# model.fit_cox_ph()

# print("fitting parsimonious")
# model.fit_cox_ph(mode="parsimonious")

# print("fitting elastic-net")
# model.fit_cox_ph(mode="elastic-net")

# print("fitting rf")
# model.fit_rf()

# print("fitting svm")
# model.fit_svm()

# print("fitting kernel svm")
# model.fit_kernel_svm()

# print("fitting and scoring features")
# model.fit_score_features()

# print("predicting")
# model.predict()

# print("evaluating")
# model.evaluate()

# print("evaluating predictions")
# model.evaluate_predictions()

# print("plotting coefficients")
# model.plot_coefficients()


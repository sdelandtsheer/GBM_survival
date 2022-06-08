
from SurvivalModel import SurvivalModel

model = SurvivalModel()
model.plot_data(feature="er")
model.fit_cox_ph()

dummy_test = model.X_train_ohe.iloc[0:1, :]

pred = model.predict(dummy_test)


model.fit_cox_ph(mode="parsimonious")
model.fit_cox_ph(mode="elastic-net")
model.fit_rf()
model.fit_score_features()
model.plot_coefficients(model.coeffs, 5)
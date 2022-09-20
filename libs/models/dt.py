import joblib

from libs.models.base import EcgModel


class RandomForestEcgModel(EcgModel):
    rf = None
    model_file = "model.pkl"

    def restore(self):
        self.rf = joblib.load(self.model_file)


    def predict(self, x):
        return self.rf.predict(x)

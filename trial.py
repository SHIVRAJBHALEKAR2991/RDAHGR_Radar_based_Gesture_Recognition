import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers

# import your custom classes
from definations import (
    TEA_ME, TEA_MTA, CT_Module, two_plus_oneDConv,
    Cross_MSECA_Module, ArcFace
)

# Load model architecture
with open("exp_results/exp1/exp_1_mesca_early_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Load model with custom objects (including L2)
model = model_from_json(
    loaded_model_json,
    custom_objects={
        "TEA_ME": TEA_ME,
        "TEA_MTA": TEA_MTA,
        "CT_Module": CT_Module,
        "two_plus_oneDConv": two_plus_oneDConv,
        "Cross_MSECA_Module": Cross_MSECA_Module,
        "ArcFace": ArcFace,
        "L2": regularizers.L2,   # ✅ add this line
    }
)

# Load weights
model.load_weights("exp_results/exp1/exp_1_mesca_early_weights.h5")

print("✅ Model loaded successfully with weights!")

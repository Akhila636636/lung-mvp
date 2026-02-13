def load_model():
    # placeholder for real model loading
    print("Model loader ready")

def predict(input_path):
    """
    Dummy prediction function.
    Later this will load a real model.
    """

    result = {
        "probability": 0.73,
        "report": "Mock nodule detected. Risk: Medium.",
        "meta": {
            "source": input_path
        }
    }

    return result

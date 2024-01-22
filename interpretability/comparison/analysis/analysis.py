class Analysis:
    def __init__(self, run_name, filepath):
        self.run_name = run_name
        self.filepath = filepath

    def load_wrapper(self, filepath):
        # Throw a warning
        return None

    def get_model_output(self):
        return None

    def compute_FPs(self, latents, inputs):
        return None

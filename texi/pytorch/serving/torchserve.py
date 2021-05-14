# -*- coding: utf-8 -*-

from ts.torch_handler.base_handler import BaseHandler


class TransformerHandler(BaseHandler):
    def __init__(self) -> None:
        super().__init__()
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, ctx):
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError(f"Missing model file: {model_pt_path}")
        self.model = torch.jit.load(model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        print(data)
        return data

    def inference(self, data, *args, **kwargs):
        return

    def postprocess(self, data):
        return

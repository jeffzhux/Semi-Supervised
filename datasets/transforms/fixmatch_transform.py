
from datasets.transforms.build import build_transform

class FixMatchTransform():
    def __init__(self, weak, strong):
        self.weak = build_transform(weak)
        self.strong = build_transform(strong)

    def __call__(self, x):
        return self.weak(x), self.strong(x)
